#!/usr/bin/env python3
# Hierarchical cluster PPM with global budget packing.
import argparse, collections, gzip, json, math, os, pickle, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch

MASK64 = (1 << 64) - 1

def hash_seq(seq, base: int = 11400714819323198485) -> int:
    h = 1469598103934665603
    for x in seq:
        h = (h * base + (int(x) + 1)) & MASK64
    return h

def find_token_files(path: str) -> List[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    exts = {".bin", ".npy", ".npz", ".pt", ".pth"}
    return [f for f in sorted(p.rglob("*")) if f.is_file() and f.suffix.lower() in exts]

def _load_one_file(path: Path) -> np.ndarray:
    s = path.suffix.lower()
    if s == ".npy":
        return np.asarray(np.load(path, mmap_mode="r"), dtype=np.int32)
    if s == ".npz":
        z = np.load(path)
        if len(z.files) != 1:
            raise ValueError(f"{path}: expected single array in npz, got {z.files}")
        return np.asarray(z[z.files[0]], dtype=np.int32)
    if s in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().astype(np.int32, copy=False)
        if isinstance(obj, dict):
            for k in ("tokens", "data", "ids", "input_ids"):
                if k in obj:
                    v = obj[k]
                    if isinstance(v, torch.Tensor):
                        return v.detach().cpu().numpy().astype(np.int32, copy=False)
                    return np.asarray(v, dtype=np.int32)
        raise ValueError(f"{path}: unsupported torch object")
    if s == ".bin":
        return np.fromfile(path, dtype=np.uint16).astype(np.int32, copy=False)
    raise ValueError(f"Unsupported file type: {path}")

class TokenCorpus:
    def __init__(self, files: List[Path]):
        self.files = files
        self.lengths = []
        total = 0
        for f in files:
            n = int(_load_one_file(f).shape[0])
            self.lengths.append(n)
            total += n
        self.total_tokens = total
        self.prefix = np.cumsum([0] + self.lengths)

    def read_range(self, start: int, end: int) -> np.ndarray:
        start = max(0, int(start))
        end = min(int(end), self.total_tokens)
        if end <= start:
            return np.empty((0,), dtype=np.int32)
        out = np.empty((end - start,), dtype=np.int32)
        pos = 0
        for i, f in enumerate(self.files):
            fs, fe = self.prefix[i], self.prefix[i + 1]
            if fe <= start or fs >= end:
                continue
            arr = _load_one_file(f)
            local_s = max(0, start - fs)
            local_e = min(self.lengths[i], end - fs)
            chunk = np.asarray(arr[local_s:local_e], dtype=np.int32)
            out[pos:pos + len(chunk)] = chunk
            pos += len(chunk)
        return out

def maybe_load_sentencepiece(tokenizer_path: Optional[str]):
    if not tokenizer_path:
        return None
    try:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_path)
        return sp
    except Exception:
        return None

def approx_eval_bytes(tokens: np.ndarray, sp) -> int:
    if sp is None:
        return int(tokens.shape[0] * 3)
    total = 0
    for t in tokens.tolist():
        piece = sp.id_to_piece(int(t))
        if piece.startswith("▁"):
            piece = " " + piece[1:]
        total += len(piece.encode("utf-8"))
    return total

@dataclass
class TableEntry:
    total: int
    entries: List[Tuple[int, int]]

class SparseHashTable:
    def __init__(self):
        self.map: Dict[int, TableEntry] = {}
    def get(self, key: int) -> Optional[TableEntry]:
        return self.map.get(key)
    def set(self, key: int, entry: TableEntry):
        self.map[key] = entry
    def __len__(self):
        return len(self.map)

def build_similarity_embeddings(train_tokens: np.ndarray, vocab_size: int, window: int, emb_dim: int, device: str):
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    x = torch.from_numpy(train_tokens.astype(np.int64, copy=False)).to(dev)
    co = torch.zeros((vocab_size, vocab_size), dtype=torch.float32, device=dev)
    for d in range(1, window + 1):
        w = 1.0 / d
        a = x[:-d]
        b = x[d:]
        idx = a * vocab_size + b
        vals = torch.full((idx.numel(),), w, dtype=torch.float32, device=dev)
        co.view(-1).index_add_(0, idx, vals)
        idx2 = b * vocab_size + a
        co.view(-1).index_add_(0, idx2, vals)
    row = co.sum(dim=1, keepdim=True).clamp_min_(1.0)
    col = co.sum(dim=0, keepdim=True).clamp_min_(1.0)
    total = co.sum().clamp_min_(1.0)
    pmi = torch.log((co * total + 1e-6) / (row * col + 1e-6))
    ppmi = torch.clamp(pmi, min=0.0)
    q = min(max(emb_dim, 2), min(ppmi.shape) - 1)
    U, S, _ = torch.pca_lowrank(ppmi, q=q, center=False)
    emb = U[:, :emb_dim] * torch.sqrt(S[:emb_dim]).unsqueeze(0)
    emb = emb / emb.norm(dim=1, keepdim=True).clamp_min_(1e-6)
    return emb.detach()

def kmeans_gpu(emb: torch.Tensor, k: int, iters: int = 25, seed: int = 42):
    g = torch.Generator(device=emb.device)
    g.manual_seed(seed)
    n = emb.shape[0]
    perm = torch.randperm(n, generator=g, device=emb.device)
    centroids = emb[perm[:k]].clone()
    for _ in range(iters):
        d = emb.pow(2).sum(1, keepdim=True) - 2.0 * emb @ centroids.t() + centroids.pow(2).sum(1).unsqueeze(0)
        assign = torch.argmin(d, dim=1)
        new_centroids = torch.zeros_like(centroids)
        counts = torch.bincount(assign, minlength=k).to(emb.dtype).unsqueeze(1)
        new_centroids.index_add_(0, assign, emb)
        mask = counts.squeeze(1) > 0
        new_centroids[mask] /= counts[mask]
        new_centroids[~mask] = centroids[~mask]
        centroids = new_centroids
    return assign

class BudgetPacker:
    def __init__(self, budget_bytes: int):
        self.budget = int(budget_bytes)
        self.used = 0
        self.candidates = []
        self.tables: Dict[str, SparseHashTable] = {}
    def add(self, table_name: str, key: int, entries: List[Tuple[int, int]], total: int, item_bytes: int, value_bias: float, prefixes: List[int]):
        prev = 0
        retained = 0
        for pref in prefixes:
            if pref <= prev or pref > len(entries):
                continue
            sub = entries[:pref]
            retained_now = sum(cnt for _, cnt in sub)
            gain = (retained_now - retained) * (1.0 + 0.02 * pref) * value_bias
            full_cost = 16 + pref * item_bytes
            old_cost = 0 if prev == 0 else 16 + prev * item_bytes
            delta = full_cost - old_cost
            self.candidates.append((gain / max(1, delta), table_name, key, total, sub, pref, item_bytes))
            prev = pref
            retained = retained_now
    def pack(self):
        self.candidates.sort(reverse=True, key=lambda x: x[0])
        chosen_len, chosen_payload = {}, {}
        for _, table_name, key, total, sub, pref, item_bytes in self.candidates:
            cur = chosen_len.get((table_name, key), 0)
            if pref <= cur:
                continue
            full_cost = 16 + pref * item_bytes
            old_cost = 0 if cur == 0 else 16 + cur * item_bytes
            delta = full_cost - old_cost
            if self.used + delta > self.budget:
                continue
            self.used += delta
            chosen_len[(table_name, key)] = pref
            chosen_payload[(table_name, key)] = (total, sub)
            self.tables.setdefault(table_name, SparseHashTable())
        for (table_name, key), (total, sub) in chosen_payload.items():
            self.tables[table_name].set(key, TableEntry(total=total, entries=sub))
        return self.tables

class HierModel:
    def __init__(self, vocab_size: int, levels: List[int]):
        self.vocab_size = vocab_size
        self.level_sizes = levels
        self.level_assignments = [np.zeros(vocab_size, dtype=np.int32) for _ in levels]
        self.level_tables: List[Dict[int, SparseHashTable]] = []
        self.exact_tables: Dict[int, SparseHashTable] = {}
        self.leaf_tables: Dict[int, SparseHashTable] = {}
        self.prev_leaf_tables: Dict[int, SparseHashTable] = {}
        self.level_unigrams = [np.ones(k, dtype=np.float64) / k for k in levels]
        self.token_unigram = np.ones(vocab_size, dtype=np.float64) / vocab_size
        self.token_given_leaf_unigram = np.ones(vocab_size, dtype=np.float64) / vocab_size
        self.lambda_leaf = 0.72
        self.lambda_exact = 0.18
        self.lambda_base = 0.10
        self.base_alpha = 0.02
        self.level_weights = []
    def save(self, path: str):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = {
            "vocab_size": self.vocab_size,
            "level_sizes": self.level_sizes,
            "level_assignments": self.level_assignments,
            "level_tables": [{o: t.map for o, t in level.items()} for level in self.level_tables],
            "exact_tables": {o: t.map for o, t in self.exact_tables.items()},
            "leaf_tables": {o: t.map for o, t in self.leaf_tables.items()},
            "prev_leaf_tables": {leaf: t.map for leaf, t in self.prev_leaf_tables.items()},
            "level_unigrams": self.level_unigrams,
            "token_unigram": self.token_unigram,
            "token_given_leaf_unigram": self.token_given_leaf_unigram,
            "lambda_leaf": self.lambda_leaf,
            "lambda_exact": self.lambda_exact,
            "lambda_base": self.lambda_base,
            "base_alpha": self.base_alpha,
            "level_weights": self.level_weights,
        }
        with gzip.open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    def score_token(self, hist_tokens: np.ndarray, hist_levels: List[np.ndarray], target: int) -> float:
        p_exact = 0.0
        if self.exact_tables:
            used = 0
            for order, table in self.exact_tables.items():
                if len(hist_tokens) < order:
                    continue
                ent = table.get(hash_seq(hist_tokens[-order:]))
                if ent is None:
                    continue
                used += 1
                for tok, cnt in ent.entries:
                    if tok == target:
                        p_exact += cnt / ent.total
                        break
            if used:
                p_exact /= used
        p_levels = 1.0
        leaf = int(self.level_assignments[-1][target])
        for li, weight in enumerate(self.level_weights):
            c = int(self.level_assignments[li][target])
            p = 0.0
            used = 0
            for order, table in self.level_tables[li].items():
                if len(hist_levels[li]) < order:
                    continue
                ent = table.get(hash_seq(hist_levels[li][-order:]))
                if ent is None:
                    continue
                used += 1
                for cc, cnt in ent.entries:
                    if cc == c:
                        p += cnt / ent.total
                        break
            if used:
                p /= used
            else:
                p = float(self.level_unigrams[li][c])
            p_levels *= max(p, 1e-9) ** weight
        p_leaf_tok = 0.0
        used = 0
        for order, table in self.leaf_tables.items():
            if len(hist_tokens) < order:
                continue
            ent = table.get(hash_seq(list(hist_tokens[-order:]) + [leaf]))
            if ent is None:
                continue
            used += 1
            for tok, cnt in ent.entries:
                if tok == target:
                    p_leaf_tok += cnt / ent.total
                    break
        if used:
            p_leaf_tok /= used
        prev = int(hist_tokens[-1]) if len(hist_tokens) else -1
        p_prev_leaf = 0.0
        if prev >= 0 and leaf in self.prev_leaf_tables:
            ent = self.prev_leaf_tables[leaf].get(hash_seq([prev, leaf]))
            if ent is not None:
                for tok, cnt in ent.entries:
                    if tok == target:
                        p_prev_leaf = cnt / ent.total
                        break
        p_tok_given_leaf = 0.6 * p_prev_leaf + 0.3 * p_leaf_tok + 0.1 * float(self.token_given_leaf_unigram[target])
        p_base = (1.0 - self.base_alpha) * float(self.token_unigram[target]) + self.base_alpha / self.vocab_size
        p = self.lambda_exact * p_exact + self.lambda_leaf * (p_levels * p_tok_given_leaf) + self.lambda_base * p_base
        return max(p, 1e-12)
    def evaluate(self, tokens: np.ndarray, tokenizer_path: Optional[str]):
        sp = maybe_load_sentencepiece(tokenizer_path)
        eval_bytes = approx_eval_bytes(tokens, sp)
        level_histories = [assign[tokens] for assign in self.level_assignments]
        max_ctx = 1
        for level in self.level_tables:
            if level:
                max_ctx = max(max_ctx, max(level.keys()))
        if self.exact_tables:
            max_ctx = max(max_ctx, max(self.exact_tables.keys()))
        if self.leaf_tables:
            max_ctx = max(max_ctx, max(self.leaf_tables.keys()))
        nll_bits = 0.0
        n = 0
        start = time.time()
        for i in range(1, len(tokens)):
            s = max(0, i - max_ctx)
            p = self.score_token(tokens[s:i], [h[s:i] for h in level_histories], int(tokens[i]))
            nll_bits += -math.log2(p)
            n += 1
            if n % 100000 == 0:
                print(f"[eval] {n:,} tokens, bits/token={nll_bits/n:.4f}, elapsed={time.time()-start:.1f}s, val_bpb={nll_bits/max(1, eval_bytes):.4f}")
        return {
            "eval_tokens": n,
            "eval_bytes": eval_bytes,
            "nll_bits": nll_bits,
            "bits_per_token": nll_bits / max(1, n),
            "val_bpb": nll_bits / max(1, eval_bytes),
            "eval_seconds": time.time() - start,
        }

def fit_model(args, train_tokens: np.ndarray):
    levels = [int(x) for x in args.cluster_levels.split(",") if x.strip()]
    t0 = time.time()
    emb = build_similarity_embeddings(train_tokens, args.vocab_size, args.cluster_window, args.cluster_emb_dim, args.device)
    token_to_level = [kmeans_gpu(emb, k, args.kmeans_iters, args.seed + i).cpu().numpy().astype(np.int32, copy=False) for i, k in enumerate(levels)]
    model = HierModel(args.vocab_size, levels)
    model.level_assignments = token_to_level
    model.lambda_leaf = args.lambda_leaf
    model.lambda_exact = args.lambda_exact
    model.lambda_base = args.lambda_base
    model.base_alpha = args.base_alpha
    if args.level_weights:
        w = [float(x) for x in args.level_weights.split(",") if x.strip()]
        s = sum(w)
        model.level_weights = [x / s for x in w]
    else:
        raw = np.linspace(1.0, 2.0, len(levels))
        raw = raw / raw.sum()
        model.level_weights = raw.tolist()
    token_counts = np.bincount(train_tokens, minlength=args.vocab_size).astype(np.float64)
    model.token_unigram = (token_counts + 1.0) / (token_counts.sum() + args.vocab_size)
    for li, k in enumerate(levels):
        ids = token_to_level[li][train_tokens]
        cnt = np.bincount(ids, minlength=k).astype(np.float64)
        model.level_unigrams[li] = (cnt + 1.0) / (cnt.sum() + k)
    leaf_assign = token_to_level[-1]
    token_given_leaf = np.zeros(args.vocab_size, dtype=np.float64)
    for leaf in range(levels[-1]):
        mask = (leaf_assign == leaf)
        denom = token_counts[mask].sum()
        if denom > 0:
            token_given_leaf[mask] = token_counts[mask] / denom
    token_given_leaf[token_given_leaf == 0] = 1.0 / args.vocab_size
    model.token_given_leaf_unigram = token_given_leaf
    level_histories = [a[train_tokens] for a in token_to_level]
    raw_level = [{o: {} for o in range(1, args.cluster_max_order + 1)} for _ in levels]
    raw_exact = {o: {} for o in range(1, args.exact_max_order + 1)}
    raw_leaf = {o: {} for o in range(1, args.leaf_token_max_order + 1)}
    raw_prev_leaf = {leaf: {} for leaf in range(levels[-1])}
    print(f"[cluster] building hierarchical raw tables on {len(train_tokens):,} tokens")
    for i in range(1, len(train_tokens)):
        tok = int(train_tokens[i]); leaf = int(level_histories[-1][i])
        for li in range(len(levels)):
            h = level_histories[li]
            target = int(h[i])
            for order in range(1, args.cluster_max_order + 1):
                if i < order: break
                key = hash_seq(h[i-order:i]); d = raw_level[li][order]
                ctr = d.get(key)
                if ctr is None: ctr = collections.Counter(); d[key] = ctr
                ctr[target] += 1
        for order in range(1, args.exact_max_order + 1):
            if i < order: break
            key = hash_seq(train_tokens[i-order:i]); d = raw_exact[order]
            ctr = d.get(key)
            if ctr is None: ctr = collections.Counter(); d[key] = ctr
            ctr[tok] += 1
        for order in range(1, args.leaf_token_max_order + 1):
            if i < order: break
            key = hash_seq(list(train_tokens[i-order:i]) + [leaf]); d = raw_leaf[order]
            ctr = d.get(key)
            if ctr is None: ctr = collections.Counter(); d[key] = ctr
            ctr[tok] += 1
        prev = int(train_tokens[i-1]); key = hash_seq([prev, leaf]); d = raw_prev_leaf[leaf]
        ctr = d.get(key)
        if ctr is None: ctr = collections.Counter(); d[key] = ctr
        ctr[tok] += 1
        if i % 1000000 == 0:
            print(f"[cluster] processed {i:,}/{len(train_tokens):,}")
    fit_seconds = time.time() - t0
    packer = BudgetPacker(int(args.target_size_mb * 1024 * 1024))
    cluster_prefixes = [1, 2, 4, 8, 16, 32]
    token_prefixes = [1, 2, 4, 8, 12, 16, 24, 32]
    model.level_tables = []
    for li, k in enumerate(levels):
        level_map = {}
        bias = 1.5 + 0.25 * (len(levels) - li)
        for order in range(1, args.cluster_max_order + 1):
            table_name = f"cluster_l{li}_o{order}"
            for key, ctr in raw_level[li][order].items():
                packer.add(table_name, key, ctr.most_common(min(args.cluster_topk, k)), int(sum(ctr.values())), 4, bias * (1.0 + 0.04 * order), [p for p in cluster_prefixes if p <= min(args.cluster_topk, k)])
            print(f"[extract] level={li+1} order={order}: candidate_contexts={len(raw_level[li][order]):,}")
            level_map[order] = SparseHashTable()
        model.level_tables.append(level_map)
    for order in range(1, args.exact_max_order + 1):
        table_name = f"exact_o{order}"
        for key, ctr in raw_exact[order].items():
            packer.add(table_name, key, ctr.most_common(args.exact_topk), int(sum(ctr.values())), 8, 0.9 + 0.05 * order, [p for p in token_prefixes if p <= args.exact_topk])
        print(f"[extract] exact order={order}: candidate_contexts={len(raw_exact[order]):,}")
        model.exact_tables[order] = SparseHashTable()
    for order in range(1, args.leaf_token_max_order + 1):
        table_name = f"leaf_o{order}"
        for key, ctr in raw_leaf[order].items():
            packer.add(table_name, key, ctr.most_common(args.leaf_token_topk), int(sum(ctr.values())), 8, 1.4 + 0.1 * order, [p for p in token_prefixes if p <= args.leaf_token_topk])
        print(f"[extract] leaf-token order={order}: candidate_contexts={len(raw_leaf[order]):,}")
        model.leaf_tables[order] = SparseHashTable()
    for leaf in range(levels[-1]):
        table_name = f"prevleaf_{leaf}"
        for key, ctr in raw_prev_leaf[leaf].items():
            packer.add(table_name, key, ctr.most_common(args.token_given_leaf_topk), int(sum(ctr.values())), 8, 1.8, [p for p in token_prefixes if p <= args.token_given_leaf_topk])
        model.prev_leaf_tables[leaf] = SparseHashTable()
    packed = packer.pack()
    for li in range(len(levels)):
        for order in range(1, args.cluster_max_order + 1):
            model.level_tables[li][order] = packed.get(f"cluster_l{li}_o{order}", SparseHashTable())
    for order in range(1, args.exact_max_order + 1):
        model.exact_tables[order] = packed.get(f"exact_o{order}", SparseHashTable())
    for order in range(1, args.leaf_token_max_order + 1):
        model.leaf_tables[order] = packed.get(f"leaf_o{order}", SparseHashTable())
    for leaf in range(levels[-1]):
        model.prev_leaf_tables[leaf] = packed.get(f"prevleaf_{leaf}", SparseHashTable())
    metrics = {
        "fit_seconds": fit_seconds,
        "estimated_model_size_bytes": packer.used,
        "estimated_model_size_mb": packer.used / (1024 * 1024),
        "budget_fill_ratio": packer.used / max(1, packer.budget),
        "graceful_unused_bytes": packer.budget - packer.used,
        "cluster_levels": args.cluster_levels,
    }
    return model, metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--tokenizer_path", type=str, default=None)
    ap.add_argument("--vocab_size", type=int, default=1024)
    ap.add_argument("--cluster_levels", type=str, default="8,32")
    ap.add_argument("--cluster_window", type=int, default=8)
    ap.add_argument("--cluster_emb_dim", type=int, default=16)
    ap.add_argument("--cluster_max_order", type=int, default=10)
    ap.add_argument("--leaf_token_max_order", type=int, default=2)
    ap.add_argument("--exact_max_order", type=int, default=2)
    ap.add_argument("--cluster_topk", type=int, default=16)
    ap.add_argument("--leaf_token_topk", type=int, default=16)
    ap.add_argument("--token_given_leaf_topk", type=int, default=16)
    ap.add_argument("--exact_topk", type=int, default=12)
    ap.add_argument("--lambda_leaf", type=float, default=0.72)
    ap.add_argument("--lambda_exact", type=float, default=0.18)
    ap.add_argument("--lambda_base", type=float, default=0.10)
    ap.add_argument("--base_alpha", type=float, default=0.02)
    ap.add_argument("--level_weights", type=str, default="")
    ap.add_argument("--kmeans_iters", type=int, default=30)
    ap.add_argument("--max_train_tokens", type=int, default=10_000_000)
    ap.add_argument("--val_tokens", type=int, default=500_000)
    ap.add_argument("--target_size_mb", type=float, default=16.0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_model", type=str, default="./ppm_hier_model.pkl.gz")
    ap.add_argument("--save_metrics", type=str, default="./ppm_hier_metrics.json")
    args = ap.parse_args()
    files = find_token_files(args.data_path)
    if not files:
        raise SystemExit(f"No token files found under {args.data_path}")
    print(f"[data] found {len(files)} token files")
    corpus = TokenCorpus(files)
    print(f"[data] total_tokens={corpus.total_tokens:,}")
    train_end = corpus.total_tokens - args.val_tokens
    train_start = max(0, train_end - args.max_train_tokens)
    val_start = train_end
    val_end = min(corpus.total_tokens, val_start + args.val_tokens)
    print(f"[data] train_range=({train_start}, {train_end}), val_range=({val_start}, {val_end})")
    t0 = time.time()
    train_tokens = corpus.read_range(train_start, train_end)
    val_tokens = corpus.read_range(val_start, val_end)
    print(f"[data] loaded train={len(train_tokens):,}, val={len(val_tokens):,} in {time.time()-t0:.1f}s")
    model, fit_metrics = fit_model(args, train_tokens)
    eval_metrics = model.evaluate(val_tokens, args.tokenizer_path)
    result = {**fit_metrics, **eval_metrics, "train_tokens": int(len(train_tokens)), "val_tokens_total": int(len(val_tokens))}
    print("[result] " + json.dumps(result, indent=2, sort_keys=True))
    model.save(args.save_model)
    with open(args.save_metrics, "w", encoding="utf-8") as f:
        json.dump(result, f)
    print(json.dumps(result))
if __name__ == "__main__":
    main()
