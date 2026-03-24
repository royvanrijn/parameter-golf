import argparse
import collections
import gzip
import json
import math
import os
import pickle
import time
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
        arr = np.load(path, mmap_mode="r")
        return np.asarray(arr, dtype=np.int32)
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
            arr = _load_one_file(f)
            n = int(arr.shape[0])
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
            fs = self.prefix[i]
            fe = self.prefix[i + 1]
            if fe <= start or fs >= end:
                continue
            local_s = max(0, start - fs)
            local_e = min(self.lengths[i], end - fs)
            arr = _load_one_file(f)
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


def build_similarity_embeddings(
    train_tokens: np.ndarray,
    vocab_size: int,
    window: int,
    emb_dim: int,
    device: str,
):
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    x = torch.from_numpy(train_tokens.astype(np.int64, copy=False)).to(dev)
    left = torch.zeros((vocab_size, vocab_size), dtype=torch.float32, device=dev)
    right = torch.zeros((vocab_size, vocab_size), dtype=torch.float32, device=dev)
    for d in range(1, window + 1):
        w = 1.0 / d
        cur = x[:-d]
        prev = x[d:]
        idx_l = cur * vocab_size + prev
        vals = torch.full((idx_l.numel(),), w, dtype=torch.float32, device=dev)
        left.view(-1).index_add_(0, idx_l, vals)
        nxt = x[d:]
        idx_r = cur * vocab_size + nxt
        right.view(-1).index_add_(0, idx_r, vals)

    m = torch.cat([left, right], dim=1)
    row = m.sum(dim=1, keepdim=True).clamp_min_(1.0)
    col = m.sum(dim=0, keepdim=True).clamp_min_(1.0)
    total = m.sum().clamp_min_(1.0)
    pmi = torch.log((m * total + 1e-6) / (row * col + 1e-6))
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
        d = (
            emb.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * emb @ centroids.t()
            + centroids.pow(2).sum(dim=1).unsqueeze(0)
        )
        assign = torch.argmin(d, dim=1)
        new_centroids = torch.zeros_like(centroids)
        counts = torch.bincount(assign, minlength=k).to(emb.dtype).unsqueeze(1)
        new_centroids.index_add_(0, assign, emb)
        mask = counts.squeeze(1) > 0
        new_centroids[mask] /= counts[mask]
        new_centroids[~mask] = centroids[~mask]
        centroids = new_centroids
    return assign, centroids


def build_hierarchy(
    emb: torch.Tensor,
    levels: List[int],
    kmeans_iters: int,
    seed: int,
) -> List[np.ndarray]:
    hier = []
    current_emb = emb
    for i, k in enumerate(levels):
        assign, _ = kmeans_gpu(current_emb, k, iters=kmeans_iters, seed=seed + i)
        hier.append(assign.detach().cpu().numpy().astype(np.int32, copy=False))
    return hier


class BudgetPacker:
    def __init__(self, budget_bytes: int):
        self.budget = int(budget_bytes)
        self.used = 0
        self.heap = []
        self.tables: Dict[str, SparseHashTable] = {}
        self.stats = collections.Counter()
        self.ctx_selected = 0
        self.entry_selected = 0

    def add_context_candidates(
        self,
        table_name: str,
        key: int,
        counter: collections.Counter,
        topk: int,
        prefix_sizes: List[int],
        value_item_bytes: int,
        value_bias: float = 1.0,
    ):
        entries = counter.most_common(topk)
        if not entries:
            return
        total = int(sum(counter.values()))
        table = self.tables.setdefault(table_name, SparseHashTable())
        # Store best prefixes first, then upgrades. Cost is incremental.
        prev_len = 0
        retained = 0
        for pref in prefix_sizes:
            if pref > len(entries) or pref <= prev_len:
                continue
            sub = [(int(tok), int(cnt)) for tok, cnt in entries[:pref]]
            retained_now = sum(cnt for _, cnt in sub)
            inc_gain = float(retained_now - retained) * (1.0 + 0.01 * pref) * value_bias
            header = 16 if prev_len == 0 else 0
            inc_cost = header + (pref - prev_len) * value_item_bytes
            if inc_cost <= 0:
                continue
            self.heap.append((
                inc_gain / inc_cost,
                table_name,
                key,
                total,
                sub,
                pref,
                inc_cost,
                retained_now,
            ))
            prev_len = pref
            retained = retained_now

    def pack(self):
        self.heap.sort(reverse=True, key=lambda x: x[0])
        best_len: Dict[Tuple[str, int], int] = {}
        best_payload: Dict[Tuple[str, int], Tuple[int, List[Tuple[int, int]]]] = {}
        for score, table_name, key, total, sub, pref, inc_cost, retained in self.heap:
            cur = best_len.get((table_name, key), 0)
            if pref <= cur:
                continue
            # Full cost of switching from cur to pref.
            full_cost = 16 + pref * (4 if table_name.startswith("cluster") else 8)
            if table_name.startswith("cluster"):
                full_cost = 16 + pref * 4
            elif table_name.startswith("exact"):
                full_cost = 16 + pref * 8
            else:
                full_cost = 16 + pref * 8
            current_cost = 0
            if cur > 0:
                if table_name.startswith("cluster"):
                    current_cost = 16 + cur * 4
                else:
                    current_cost = 16 + cur * 8
            delta = full_cost - current_cost
            if self.used + delta > self.budget:
                continue
            self.used += delta
            best_len[(table_name, key)] = pref
            best_payload[(table_name, key)] = (total, sub)

        for (table_name, key), (total, sub) in best_payload.items():
            self.tables[table_name].set(key, TableEntry(total=total, entries=sub))
            self.ctx_selected += 1
            self.entry_selected += len(sub)
            self.stats[table_name] += 16 + len(sub) * (4 if table_name.startswith("cluster") else 8)
        return self.tables

    def metrics(self):
        return {
            "estimated_model_size_bytes": int(self.used),
            "estimated_model_size_mb": float(self.used / (1024 * 1024)),
            "budget_fill_ratio": float(self.used / max(1, self.budget)),
            "graceful_unused_bytes": float(self.budget - self.used),
            "selected_contexts": float(self.ctx_selected),
            "selected_entries": float(self.entry_selected),
            **{f"budget_{k}_bytes": int(v) for k, v in self.stats.items()},
        }


class HierClusterModel:
    def __init__(self, vocab_size: int, levels: List[int]):
        self.vocab_size = vocab_size
        self.level_sizes = list(levels)
        self.num_levels = len(levels)
        self.cluster_of_levels: List[np.ndarray] = [np.zeros(vocab_size, dtype=np.int32) for _ in levels]
        self.token_unigram = np.ones(vocab_size, dtype=np.float64) / vocab_size
        self.token_given_leaf_unigram = np.ones(vocab_size, dtype=np.float64) / vocab_size
        self.level_unigrams: List[np.ndarray] = [np.ones(k, dtype=np.float64) / k for k in levels]

        self.level_tables: List[Dict[int, SparseHashTable]] = []
        self.leaf_token_tables: Dict[int, SparseHashTable] = {}
        self.exact_tables: Dict[int, SparseHashTable] = {}
        self.token_given_leaf_prev_tables: Dict[int, SparseHashTable] = {}

        self.level_weights = []
        self.lambda_leaf = 0.55
        self.lambda_exact = 0.25
        self.lambda_base = 0.20
        self.base_alpha = 0.02

    def save(self, path: str):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = {
            "vocab_size": self.vocab_size,
            "level_sizes": self.level_sizes,
            "cluster_of_levels": self.cluster_of_levels,
            "token_unigram": self.token_unigram,
            "token_given_leaf_unigram": self.token_given_leaf_unigram,
            "level_unigrams": self.level_unigrams,
            "level_weights": self.level_weights,
            "lambda_leaf": self.lambda_leaf,
            "lambda_exact": self.lambda_exact,
            "lambda_base": self.lambda_base,
            "base_alpha": self.base_alpha,
            "level_tables": [
                {order: table.map for order, table in level.items()}
                for level in self.level_tables
            ],
            "leaf_token_tables": {order: table.map for order, table in self.leaf_token_tables.items()},
            "exact_tables": {order: table.map for order, table in self.exact_tables.items()},
            "token_given_leaf_prev_tables": {
                leaf: table.map for leaf, table in self.token_given_leaf_prev_tables.items()
            },
        }
        with gzip.open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def score_token(self, history_tokens: np.ndarray, history_levels: List[np.ndarray], target: int) -> float:
        p_exact = 0.0
        if self.exact_tables:
            for order, table in self.exact_tables.items():
                if len(history_tokens) < order:
                    continue
                key = hash_seq(history_tokens[-order:])
                ent = table.get(key)
                if ent is None or ent.total <= 0:
                    continue
                for tok, cnt in ent.entries:
                    if tok == target:
                        p_exact += cnt / ent.total
                        break
            p_exact /= max(1, len(self.exact_tables))

        leaf = int(self.cluster_of_levels[-1][target])
        p_leaf = 1.0
        for li, level_size in enumerate(self.level_sizes):
            c = int(self.cluster_of_levels[li][target])
            level_tables = self.level_tables[li]
            p_c = 0.0
            used = 0
            for order, table in level_tables.items():
                if len(history_levels[li]) < order:
                    continue
                key = hash_seq(history_levels[li][-order:])
                ent = table.get(key)
                if ent is None or ent.total <= 0:
                    continue
                used += 1
                for cc, cnt in ent.entries:
                    if cc == c:
                        p_c += cnt / ent.total
                        break
            if used > 0:
                p_c /= used
            else:
                p_c = float(self.level_unigrams[li][c])
            p_leaf *= max(p_c, 1e-9) ** self.level_weights[li]

        p_tok_leaf_short = 0.0
        if self.leaf_token_tables:
            for order, table in self.leaf_token_tables.items():
                if len(history_tokens) < order:
                    continue
                key = hash_seq(list(history_tokens[-order:]) + [leaf])
                ent = table.get(key)
                if ent is None or ent.total <= 0:
                    continue
                for tok, cnt in ent.entries:
                    if tok == target:
                        p_tok_leaf_short += cnt / ent.total
                        break
            p_tok_leaf_short /= max(1, len(self.leaf_token_tables))

        prev_token = int(history_tokens[-1]) if len(history_tokens) > 0 else -1
        p_tok_leaf_prev = 0.0
        if prev_token >= 0 and leaf in self.token_given_leaf_prev_tables:
            table = self.token_given_leaf_prev_tables[leaf]
            key = hash_seq([prev_token, leaf])
            ent = table.get(key)
            if ent is not None and ent.total > 0:
                for tok, cnt in ent.entries:
                    if tok == target:
                        p_tok_leaf_prev = cnt / ent.total
                        break

        p_tok_leaf_base = float(self.token_given_leaf_unigram[target])
        p_tok_given_leaf = 0.55 * p_tok_leaf_prev + 0.35 * p_tok_leaf_short + 0.10 * p_tok_leaf_base
        p_hier = p_leaf * p_tok_given_leaf

        p_base = (1.0 - self.base_alpha) * float(self.token_unigram[target]) + self.base_alpha / self.vocab_size
        p = self.lambda_exact * p_exact + self.lambda_leaf * p_hier + self.lambda_base * p_base
        return max(p, 1e-12)

    def evaluate(self, tokens: np.ndarray, tokenizer_path: Optional[str], progress_every: int = 100000):
        sp = maybe_load_sentencepiece(tokenizer_path)
        eval_bytes = approx_eval_bytes(tokens, sp)
        histories = [np.asarray([int(level[t]) for t in tokens], dtype=np.int32) for level in self.cluster_of_levels]
        max_ctx = 0
        for lt in self.level_tables:
            if lt:
                max_ctx = max(max_ctx, max(lt.keys()))
        if self.exact_tables:
            max_ctx = max(max_ctx, max(self.exact_tables.keys()))
        if self.leaf_token_tables:
            max_ctx = max(max_ctx, max(self.leaf_token_tables.keys()))
        nll_bits = 0.0
        n = 0
        start = time.time()
        for i in range(1, len(tokens)):
            s = max(0, i - max_ctx)
            h_tokens = tokens[s:i]
            h_levels = [hist[s:i] for hist in histories]
            p = self.score_token(h_tokens, h_levels, int(tokens[i]))
            nll_bits += -math.log2(p)
            n += 1
            if progress_every and n % progress_every == 0:
                print(f"[eval] {n:,} tokens, bits/token={nll_bits/n:.4f}, elapsed={time.time()-start:.1f}s, val_bpb={nll_bits/max(1, eval_bytes):.4f}")
        return {
            "eval_tokens": n,
            "eval_bytes": eval_bytes,
            "nll_bits": nll_bits,
            "bits_per_token": nll_bits / max(1, n),
            "val_bpb": nll_bits / max(1, eval_bytes),
            "eval_seconds": time.time() - start,
        }


def fit_model(args, train_tokens: np.ndarray) -> Tuple[HierClusterModel, Dict[str, float]]:
    levels = [int(x) for x in args.cluster_levels.split(",") if x.strip()]
    if not levels:
        raise ValueError("cluster_levels must be non-empty")
    t0 = time.time()
    emb = build_similarity_embeddings(
        train_tokens=train_tokens,
        vocab_size=args.vocab_size,
        window=args.cluster_window,
        emb_dim=args.cluster_emb_dim,
        device=args.device,
    )
    hierarchy = build_hierarchy(emb, levels, args.kmeans_iters, args.seed)
    fit_embed_seconds = time.time() - t0

    model = HierClusterModel(vocab_size=args.vocab_size, levels=levels)
    model.cluster_of_levels = [h.astype(np.int32, copy=False) for h in hierarchy]
    model.level_weights = [float(x) for x in np.asarray(args.level_weights.split(","), dtype=np.float64)]
    if len(model.level_weights) != len(levels):
        # fallback: deeper levels matter more
        raw = np.linspace(1.0, 2.0, len(levels))
        raw = raw / raw.sum()
        model.level_weights = raw.tolist()
    else:
        s = sum(model.level_weights)
        model.level_weights = [x / s for x in model.level_weights]

    model.lambda_leaf = args.lambda_leaf
    model.lambda_exact = args.lambda_exact
    model.lambda_base = args.lambda_base
    model.base_alpha = args.base_alpha

    token_counts = np.bincount(train_tokens, minlength=args.vocab_size).astype(np.float64)
    model.token_unigram = (token_counts + 1.0) / (token_counts.sum() + args.vocab_size)

    leaf_of_token = model.cluster_of_levels[-1]
    leaf_count = levels[-1]
    leaf_ids = leaf_of_token[train_tokens]
    leaf_counts = np.bincount(leaf_ids, minlength=leaf_count).astype(np.float64)
    model.level_unigrams = []
    for li, k in enumerate(levels):
        ids = model.cluster_of_levels[li][train_tokens]
        cnt = np.bincount(ids, minlength=k).astype(np.float64)
        model.level_unigrams.append((cnt + 1.0) / (cnt.sum() + k))

    token_given_leaf_uni = np.zeros(args.vocab_size, dtype=np.float64)
    for leaf in range(leaf_count):
        mask = (leaf_of_token == leaf)
        denom = token_counts[mask].sum()
        if denom > 0:
            token_given_leaf_uni[mask] = token_counts[mask] / denom
    token_given_leaf_uni[token_given_leaf_uni == 0.0] = 1.0 / args.vocab_size
    model.token_given_leaf_unigram = token_given_leaf_uni

    level_histories = [h[train_tokens] if False else model.cluster_of_levels[li][train_tokens] for li in range(len(levels))]

    raw_level_tables = [{o: {} for o in range(1, args.cluster_max_order + 1)} for _ in levels]
    raw_exact_tables = {o: {} for o in range(1, args.exact_max_order + 1)}
    raw_leaf_token_tables = {o: {} for o in range(1, args.leaf_token_max_order + 1)}
    raw_tglp = {leaf: {} for leaf in range(leaf_count)}

    n = len(train_tokens)
    print(f"[cluster] building hierarchical raw tables on {n:,} tokens")
    for i in range(1, n):
        tok = int(train_tokens[i])
        leaf = int(level_histories[-1][i])

        for li in range(len(levels)):
            h = level_histories[li]
            target_cluster = int(h[i])
            for order in range(1, args.cluster_max_order + 1):
                if i < order:
                    break
                key = hash_seq(h[i - order:i])
                d = raw_level_tables[li][order]
                ctr = d.get(key)
                if ctr is None:
                    ctr = collections.Counter()
                    d[key] = ctr
                ctr[target_cluster] += 1

        for order in range(1, args.exact_max_order + 1):
            if i < order:
                break
            key = hash_seq(train_tokens[i - order:i])
            d = raw_exact_tables[order]
            ctr = d.get(key)
            if ctr is None:
                ctr = collections.Counter()
                d[key] = ctr
            ctr[tok] += 1

        for order in range(1, args.leaf_token_max_order + 1):
            if i < order:
                break
            key = hash_seq(list(train_tokens[i - order:i]) + [leaf])
            d = raw_leaf_token_tables[order]
            ctr = d.get(key)
            if ctr is None:
                ctr = collections.Counter()
                d[key] = ctr
            ctr[tok] += 1

        prev_token = int(train_tokens[i - 1])
        key = hash_seq([prev_token, leaf])
        d = raw_tglp[leaf]
        ctr = d.get(key)
        if ctr is None:
            ctr = collections.Counter()
            d[key] = ctr
        ctr[tok] += 1

        if i % 1_000_000 == 0:
            print(f"[cluster] processed {i:,}/{n:,}")

    fit_seconds = time.time() - t0

    budget = int(args.target_size_mb * 1024 * 1024)
    packer = BudgetPacker(budget)

    cluster_prefixes = [1, 2, 4, 8, 16, 32]
    exact_prefixes = [1, 2, 4, 8, 12, 16, 24, 32]
    leaf_tok_prefixes = [1, 2, 4, 8, 12, 16, 24, 32]

    # Level tables: smaller alphabets at upper levels get higher bias.
    model.level_tables = []
    for li, k in enumerate(levels):
        level_map = {}
        bias = 1.4 + 0.3 * (len(levels) - li)
        for order in range(1, args.cluster_max_order + 1):
            table_name = f"cluster_l{li}_o{order}"
            for key, ctr in raw_level_tables[li][order].items():
                packer.add_context_candidates(
                    table_name=table_name,
                    key=key,
                    counter=ctr,
                    topk=min(args.cluster_topk, k),
                    prefix_sizes=[p for p in cluster_prefixes if p <= min(args.cluster_topk, k)],
                    value_item_bytes=4,
                    value_bias=bias * (1.0 + 0.03 * order),
                )
            level_map[order] = SparseHashTable()
            print(f"[extract] level={li+1} order={order}: candidate_contexts={len(raw_level_tables[li][order]):,}")
        model.level_tables.append(level_map)

    for order in range(1, args.exact_max_order + 1):
        table_name = f"exact_o{order}"
        for key, ctr in raw_exact_tables[order].items():
            packer.add_context_candidates(
                table_name=table_name,
                key=key,
                counter=ctr,
                topk=args.exact_topk,
                prefix_sizes=[p for p in exact_prefixes if p <= args.exact_topk],
                value_item_bytes=8,
                value_bias=0.9 + 0.05 * order,
            )
        model.exact_tables[order] = SparseHashTable()
        print(f"[extract] exact order={order}: candidate_contexts={len(raw_exact_tables[order]):,}")

    for order in range(1, args.leaf_token_max_order + 1):
        table_name = f"leaf_tok_o{order}"
        for key, ctr in raw_leaf_token_tables[order].items():
            packer.add_context_candidates(
                table_name=table_name,
                key=key,
                counter=ctr,
                topk=args.leaf_token_topk,
                prefix_sizes=[p for p in leaf_tok_prefixes if p <= args.leaf_token_topk],
                value_item_bytes=8,
                value_bias=1.45 + 0.12 * order,
            )
        model.leaf_token_tables[order] = SparseHashTable()
        print(f"[extract] leaf-token order={order}: candidate_contexts={len(raw_leaf_token_tables[order]):,}")

    for leaf in range(leaf_count):
        table_name = f"tglp_leaf{leaf}"
        for key, ctr in raw_tglp[leaf].items():
            packer.add_context_candidates(
                table_name=table_name,
                key=key,
                counter=ctr,
                topk=args.token_given_leaf_topk,
                prefix_sizes=[p for p in leaf_tok_prefixes if p <= args.token_given_leaf_topk],
                value_item_bytes=8,
                value_bias=1.8,
            )
        model.token_given_leaf_prev_tables[leaf] = SparseHashTable()

    tables = packer.pack()

    # Install packed tables into model.
    for li in range(len(levels)):
        for order in range(1, args.cluster_max_order + 1):
            model.level_tables[li][order] = tables.get(f"cluster_l{li}_o{order}", SparseHashTable())
    for order in range(1, args.exact_max_order + 1):
        model.exact_tables[order] = tables.get(f"exact_o{order}", SparseHashTable())
    for order in range(1, args.leaf_token_max_order + 1):
        model.leaf_token_tables[order] = tables.get(f"leaf_tok_o{order}", SparseHashTable())
    for leaf in range(leaf_count):
        model.token_given_leaf_prev_tables[leaf] = tables.get(f"tglp_leaf{leaf}", SparseHashTable())

    extract_seconds = time.time() - t0 - fit_seconds
    metrics = {
        "fit_embed_seconds": fit_embed_seconds,
        "fit_seconds": fit_seconds,
        "extract_seconds": extract_seconds,
        "cluster_levels": args.cluster_levels,
        **packer.metrics(),
    }
    return model, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--tokenizer_path", type=str, default=None)
    ap.add_argument("--vocab_size", type=int, default=1024)
    ap.add_argument("--cluster_levels", type=str, default="16,64")
    ap.add_argument("--cluster_window", type=int, default=8)
    ap.add_argument("--cluster_emb_dim", type=int, default=16)
    ap.add_argument("--cluster_max_order", type=int, default=8)
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
    eval_metrics = model.evaluate(val_tokens, tokenizer_path=args.tokenizer_path)

    result = {
        **fit_metrics,
        **eval_metrics,
        "train_tokens": int(len(train_tokens)),
        "val_tokens_total": int(len(val_tokens)),
        "cluster_window": args.cluster_window,
        "cluster_emb_dim": args.cluster_emb_dim,
        "cluster_max_order": args.cluster_max_order,
        "leaf_token_max_order": args.leaf_token_max_order,
        "exact_max_order": args.exact_max_order,
        "cluster_topk": args.cluster_topk,
        "leaf_token_topk": args.leaf_token_topk,
        "token_given_leaf_topk": args.token_given_leaf_topk,
        "exact_topk": args.exact_topk,
        "lambda_leaf": args.lambda_leaf,
        "lambda_exact": args.lambda_exact,
        "lambda_base": args.lambda_base,
        "base_alpha": args.base_alpha,
    }
    print("[result] " + json.dumps(result, indent=2, sort_keys=True))
    model.save(args.save_model)
    with open(args.save_metrics, "w", encoding="utf-8") as f:
        json.dump(result, f)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
