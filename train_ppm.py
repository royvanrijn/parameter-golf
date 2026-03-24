#!/usr/bin/env python3
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
    files = [f for f in sorted(p.rglob("*")) if f.is_file() and f.suffix.lower() in exts]
    return files


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
        # OpenAI parameter golf dataset shards are uint16 token ids.
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


def topk_counter(counter: collections.Counter, k: int) -> List[Tuple[int, int]]:
    if not counter:
        return []
    return counter.most_common(k)


@dataclass
class TableEntry:
    total: int
    entries: List[Tuple[int, int]]  # item -> count


class SparseHashTable:
    def __init__(self):
        self.map: Dict[int, TableEntry] = {}

    def get(self, key: int) -> Optional[TableEntry]:
        return self.map.get(key)

    def set(self, key: int, entry: TableEntry):
        self.map[key] = entry

    def __len__(self):
        return len(self.map)


class ClusterPPMModel:
    def __init__(self, vocab_size: int, num_clusters: int):
        self.vocab_size = vocab_size
        self.num_clusters = num_clusters
        self.cluster_of = np.zeros(vocab_size, dtype=np.uint8)
        self.cluster_unigram = np.ones(num_clusters, dtype=np.float64) / num_clusters
        self.token_unigram = np.ones(vocab_size, dtype=np.float64) / vocab_size
        self.token_in_cluster_unigram = np.ones(vocab_size, dtype=np.float64) / vocab_size
        self.cluster_tables: Dict[int, SparseHashTable] = {}
        self.exact_tables: Dict[int, SparseHashTable] = {}
        self.token_given_cluster_prev: Dict[int, SparseHashTable] = {}  # key=(prev_token, cluster)
        self.l_exact = 0.30
        self.l_cluster = 0.55
        self.l_base = 0.15
        self.base_alpha = 0.02

    def save(self, path: str):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = {
            "vocab_size": self.vocab_size,
            "num_clusters": self.num_clusters,
            "cluster_of": self.cluster_of,
            "cluster_unigram": self.cluster_unigram,
            "token_unigram": self.token_unigram,
            "token_in_cluster_unigram": self.token_in_cluster_unigram,
            "l_exact": self.l_exact,
            "l_cluster": self.l_cluster,
            "l_base": self.l_base,
            "base_alpha": self.base_alpha,
            "cluster_tables": {
                order: table.map for order, table in self.cluster_tables.items()
            },
            "exact_tables": {
                order: table.map for order, table in self.exact_tables.items()
            },
            "token_given_cluster_prev": {
                cluster: table.map for cluster, table in self.token_given_cluster_prev.items()
            },
        }
        with gzip.open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def score_token(self, history_tokens: np.ndarray, history_clusters: np.ndarray, target: int) -> float:
        # exact-token short-context mixture
        p_exact = 0.0
        for order, table in self.exact_tables.items():
            if len(history_tokens) < order:
                continue
            key = hash_seq(history_tokens[-order:])
            ent = table.get(key)
            if ent is None or ent.total <= 0:
                continue
            for tok, cnt in ent.entries:
                if tok == target:
                    p_exact += (cnt / ent.total) / len(self.exact_tables)
                    break

        # cluster -> token path
        c = int(self.cluster_of[target])
        p_cluster = 0.0
        used_orders = 0
        for order, table in self.cluster_tables.items():
            if len(history_clusters) < order:
                continue
            key = hash_seq(history_clusters[-order:])
            ent = table.get(key)
            if ent is None or ent.total <= 0:
                continue
            used_orders += 1
            for cc, cnt in ent.entries:
                if cc == c:
                    p_cluster += cnt / ent.total
                    break
        if used_orders > 0:
            p_cluster /= used_orders
        else:
            p_cluster = float(self.cluster_unigram[c])

        prev_token = int(history_tokens[-1]) if len(history_tokens) > 0 else -1
        tg = self.token_given_cluster_prev.get(c)
        p_tok_given_cluster = 0.0
        if tg is not None and prev_token >= 0:
            key = hash_seq([prev_token, c])
            ent = tg.get(key)
            if ent is not None and ent.total > 0:
                for tok, cnt in ent.entries:
                    if tok == target:
                        p_tok_given_cluster = cnt / ent.total
                        break
        if p_tok_given_cluster <= 0.0:
            p_tok_given_cluster = float(self.token_in_cluster_unigram[target])

        p_hier = p_cluster * p_tok_given_cluster
        p_base = (1.0 - self.base_alpha) * float(self.token_unigram[target]) + self.base_alpha / self.vocab_size
        p = self.l_exact * p_exact + self.l_cluster * p_hier + self.l_base * p_base
        return max(p, 1e-12)

    def evaluate(self, tokens: np.ndarray, tokenizer_path: Optional[str] = None, progress_every: int = 100000):
        sp = maybe_load_sentencepiece(tokenizer_path)
        eval_bytes = approx_eval_bytes(tokens, sp)
        nll_bits = 0.0
        n = 0
        start = time.time()
        cluster_hist = np.asarray([int(self.cluster_of[t]) for t in tokens], dtype=np.int32)
        max_ctx = max([0] + list(self.exact_tables.keys()) + list(self.cluster_tables.keys()))
        for i in range(1, len(tokens)):
            hist_s = max(0, i - max_ctx)
            p = self.score_token(tokens[hist_s:i], cluster_hist[hist_s:i], int(tokens[i]))
            nll_bits += -math.log2(p)
            n += 1
            if progress_every and (n % progress_every == 0):
                bpt = nll_bits / n
                bpb = nll_bits / max(1, eval_bytes)
                print(f"[eval] {n:,} tokens, bits/token={bpt:.4f}, elapsed={time.time()-start:.1f}s, val_bpb={bpb:.4f}")
        return {
            "eval_tokens": n,
            "eval_bytes": eval_bytes,
            "nll_bits": nll_bits,
            "bits_per_token": nll_bits / max(1, n),
            "val_bpb": nll_bits / max(1, eval_bytes),
            "eval_seconds": time.time() - start,
        }


def build_similarity_embeddings(
    train_tokens: np.ndarray,
    vocab_size: int,
    window: int,
    emb_dim: int,
    device: str,
):
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    x = torch.from_numpy(train_tokens.astype(np.int64, copy=False)).to(dev)
    left = torch.zeros((vocab_size, vocab_size), dtype=torch.float32, device=dev)
    right = torch.zeros((vocab_size, vocab_size), dtype=torch.float32, device=dev)
    n = x.numel()
    for d in range(1, window + 1):
        w = 1.0 / d
        src = x[d:]
        prev = x[:-d]
        nxt = x[d:]
        cur = x[:-d]
        idx = cur * vocab_size + prev
        vals = torch.full((idx.numel(),), w, dtype=torch.float32, device=dev)
        left.view(-1).index_add_(0, idx, vals)
        idx2 = cur * vocab_size + nxt
        right.view(-1).index_add_(0, idx2, vals)

    m = torch.cat([left, right], dim=1)  # [V, 2V]
    row = m.sum(dim=1, keepdim=True).clamp_min_(1.0)
    col = m.sum(dim=0, keepdim=True).clamp_min_(1.0)
    total = m.sum().clamp_min_(1.0)
    pmi = torch.log((m * total + 1e-6) / (row * col + 1e-6))
    ppmi = torch.clamp(pmi, min=0.0)
    q = min(emb_dim, min(ppmi.shape) - 1)
    U, S, V = torch.pca_lowrank(ppmi, q=q, center=False)
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


def estimate_entry_size(num_items: int) -> int:
    # 8 bytes key + 4 bytes total + 1 byte used + per item (2 tok + 2 cnt) rounded
    return 16 + 8 * num_items


def extract_counter_table(
    raw: Dict[int, collections.Counter],
    topk: int,
    budget_bytes: int,
    prefer_large: bool = True,
) -> Tuple[SparseHashTable, int, int]:
    items = []
    for key, ctr in raw.items():
        entries = [(int(tok), int(cnt)) for tok, cnt in ctr.most_common(topk)]
        if not entries:
            continue
        total = int(sum(ctr.values()))
        size = estimate_entry_size(len(entries))
        utility = sum(cnt for _, cnt in entries) * (1.0 + 0.02 * min(len(entries), topk))
        if prefer_large:
            utility *= (1.0 + 0.03 * len(entries))
        items.append((utility / max(1, size), key, total, entries, size))
    items.sort(reverse=True, key=lambda x: x[0])
    table = SparseHashTable()
    used = 0
    selected = 0
    for _, key, total, entries, size in items:
        if used + size > budget_bytes:
            continue
        table.set(key, TableEntry(total=total, entries=entries))
        used += size
        selected += len(entries)
    return table, used, selected


def fit_cluster_ppm(args, train_tokens: np.ndarray) -> Tuple[ClusterPPMModel, Dict[str, float]]:
    t0 = time.time()
    emb = build_similarity_embeddings(
        train_tokens,
        vocab_size=args.vocab_size,
        window=args.cluster_window,
        emb_dim=args.cluster_emb_dim,
        device=args.device,
    )
    assign, centroids = kmeans_gpu(emb, args.num_clusters, iters=args.kmeans_iters, seed=args.seed)
    cluster_of = assign.detach().cpu().numpy().astype(np.uint8, copy=False)
    train_clusters = cluster_of[train_tokens]

    model = ClusterPPMModel(vocab_size=args.vocab_size, num_clusters=args.num_clusters)
    model.cluster_of[:] = cluster_of
    model.l_exact = args.lambda_exact
    model.l_cluster = args.lambda_cluster
    model.l_base = args.lambda_base
    model.base_alpha = args.base_alpha

    token_counts = np.bincount(train_tokens, minlength=args.vocab_size).astype(np.float64)
    cluster_counts = np.bincount(train_clusters, minlength=args.num_clusters).astype(np.float64)
    model.token_unigram = (token_counts + 1.0) / (token_counts.sum() + args.vocab_size)
    model.cluster_unigram = (cluster_counts + 1.0) / (cluster_counts.sum() + args.num_clusters)

    token_in_cluster = np.zeros(args.vocab_size, dtype=np.float64)
    for c in range(args.num_clusters):
        mask = (cluster_of == c)
        denom = token_counts[mask].sum()
        if denom <= 0:
            continue
        token_in_cluster[mask] = token_counts[mask] / denom
    token_in_cluster[token_in_cluster == 0.0] = 1.0 / max(1, args.vocab_size)
    model.token_in_cluster_unigram = token_in_cluster

    raw_cluster_tables = {o: {} for o in range(1, args.cluster_max_order + 1)}
    raw_exact_tables = {o: {} for o in range(1, args.exact_max_order + 1)}
    raw_tgc = {c: {} for c in range(args.num_clusters)}

    n = len(train_tokens)
    chunk_log = 1_000_000
    print(f"[cluster] building raw tables on {n:,} tokens")
    for i in range(1, n):
        tok = int(train_tokens[i])
        c = int(train_clusters[i])

        for order in range(1, args.cluster_max_order + 1):
            if i < order:
                break
            key = hash_seq(train_clusters[i - order:i])
            ctr = raw_cluster_tables[order].get(key)
            if ctr is None:
                ctr = collections.Counter()
                raw_cluster_tables[order][key] = ctr
            ctr[c] += 1

        for order in range(1, args.exact_max_order + 1):
            if i < order:
                break
            key = hash_seq(train_tokens[i - order:i])
            ctr = raw_exact_tables[order].get(key)
            if ctr is None:
                ctr = collections.Counter()
                raw_exact_tables[order][key] = ctr
            ctr[tok] += 1

        prev_tok = int(train_tokens[i - 1])
        key = hash_seq([prev_tok, c])
        ctr = raw_tgc[c].get(key)
        if ctr is None:
            ctr = collections.Counter()
            raw_tgc[c][key] = ctr
        ctr[tok] += 1

        if i % chunk_log == 0:
            print(f"[cluster] processed {i:,}/{n:,}")

    fit_seconds = time.time() - t0

    budget = int(args.target_size_mb * 1024 * 1024)
    # Split budget across components. Exact small, cluster medium, token-given-cluster large.
    budget_cluster = int(budget * 0.28)
    budget_exact = int(budget * 0.22)
    budget_tgc = budget - budget_cluster - budget_exact

    cluster_used = exact_used = tgc_used = 0
    selected_contexts = 0
    selected_entries = 0

    for order in range(1, args.cluster_max_order + 1):
        table, used, sel = extract_counter_table(raw_cluster_tables[order], args.cluster_topk, max(1, budget_cluster // args.cluster_max_order))
        model.cluster_tables[order] = table
        cluster_used += used
        selected_contexts += len(table)
        selected_entries += sel
        print(f"[extract] cluster order={order}: candidate_contexts={len(raw_cluster_tables[order]):,}, selected={len(table):,}")

    for order in range(1, args.exact_max_order + 1):
        table, used, sel = extract_counter_table(raw_exact_tables[order], args.exact_topk, max(1, budget_exact // args.exact_max_order))
        model.exact_tables[order] = table
        exact_used += used
        selected_contexts += len(table)
        selected_entries += sel
        print(f"[extract] exact order={order}: candidate_contexts={len(raw_exact_tables[order]):,}, selected={len(table):,}")

    per_cluster_budget = max(1, budget_tgc // args.num_clusters)
    for c in range(args.num_clusters):
        table, used, sel = extract_counter_table(raw_tgc[c], args.token_given_cluster_topk, per_cluster_budget)
        model.token_given_cluster_prev[c] = table
        tgc_used += used
        selected_contexts += len(table)
        selected_entries += sel

    extract_seconds = time.time() - t0 - fit_seconds

    metrics = {
        "fit_seconds": fit_seconds,
        "extract_seconds": extract_seconds,
        "cluster_budget_bytes": cluster_used,
        "exact_budget_bytes": exact_used,
        "token_given_cluster_budget_bytes": tgc_used,
        "estimated_model_size_bytes": cluster_used + exact_used + tgc_used,
        "estimated_model_size_mb": (cluster_used + exact_used + tgc_used) / (1024 * 1024),
        "selected_contexts": float(selected_contexts),
        "selected_entries": float(selected_entries),
    }
    return model, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--tokenizer_path", type=str, default=None)
    ap.add_argument("--vocab_size", type=int, default=1024)
    ap.add_argument("--num_clusters", type=int, default=64)
    ap.add_argument("--cluster_window", type=int, default=8)
    ap.add_argument("--cluster_emb_dim", type=int, default=16)
    ap.add_argument("--kmeans_iters", type=int, default=30)
    ap.add_argument("--cluster_max_order", type=int, default=6)
    ap.add_argument("--exact_max_order", type=int, default=2)
    ap.add_argument("--cluster_topk", type=int, default=8)
    ap.add_argument("--exact_topk", type=int, default=8)
    ap.add_argument("--token_given_cluster_topk", type=int, default=12)
    ap.add_argument("--target_size_mb", type=float, default=16.0)
    ap.add_argument("--max_train_tokens", type=int, default=15_000_000)
    ap.add_argument("--val_tokens", type=int, default=500_000)
    ap.add_argument("--lambda_exact", type=float, default=0.30)
    ap.add_argument("--lambda_cluster", type=float, default=0.55)
    ap.add_argument("--lambda_base", type=float, default=0.15)
    ap.add_argument("--base_alpha", type=float, default=0.02)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_model", type=str, default="./ppm_cluster_model.pkl.gz")
    ap.add_argument("--save_metrics", type=str, default="./ppm_cluster_metrics.json")
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

    model, fit_metrics = fit_cluster_ppm(args, train_tokens)
    eval_metrics = model.evaluate(val_tokens, tokenizer_path=args.tokenizer_path)

    result = {
        **fit_metrics,
        **eval_metrics,
        "train_tokens": int(len(train_tokens)),
        "val_tokens_total": int(len(val_tokens)),
        "num_clusters": args.num_clusters,
        "cluster_window": args.cluster_window,
        "cluster_emb_dim": args.cluster_emb_dim,
        "cluster_max_order": args.cluster_max_order,
        "exact_max_order": args.exact_max_order,
        "cluster_topk": args.cluster_topk,
        "exact_topk": args.exact_topk,
        "token_given_cluster_topk": args.token_given_cluster_topk,
        "lambda_exact": args.lambda_exact,
        "lambda_cluster": args.lambda_cluster,
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
