#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import collections
import gc
import gzip
import json
import math
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Counter, DefaultDict, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import sentencepiece as spm  # type: ignore
except Exception:
    spm = None

MASK64 = (1 << 64) - 1
HASH_BASE = 1315423911


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# -----------------------------
# Data loading
# -----------------------------


def discover_token_files(data_path: str) -> List[Path]:
    root = Path(data_path)
    if root.is_file():
        return [root]
    exts = {".bin", ".npy", ".npz", ".pt", ".pth"}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    if not files:
        raise FileNotFoundError(f"No token files found under {data_path!r}")
    return files


class TokenFile:
    def __init__(self, path: Path, vocab_size: int):
        self.path = path
        self.vocab_size = vocab_size
        self._arr = self._load(path)
        self.length = int(self._arr.shape[0])

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            arr = np.load(path, mmap_mode="r")
        elif suffix == ".npz":
            z = np.load(path, mmap_mode="r")
            if len(z.files) != 1:
                raise ValueError(f"Expected exactly one array in {path}")
            arr = z[z.files[0]]
        elif suffix in {".pt", ".pth"}:
            import torch
            arr = torch.load(path, map_location="cpu")
            if hasattr(arr, "numpy"):
                arr = arr.numpy()
            arr = np.asarray(arr)
        elif suffix == ".bin":
            size = path.stat().st_size
            if size % 2 != 0:
                raise ValueError(f"Expected even byte length for uint16 .bin file: {path}")
            arr = np.memmap(path, dtype=np.uint16, mode="r")
        else:
            raise ValueError(f"Unsupported file type: {path}")
        arr = np.asarray(arr)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        return arr

    def slice(self, start: int, end: int) -> np.ndarray:
        return np.asarray(self._arr[start:end], dtype=np.int64)


class TokenCorpus:
    def __init__(self, files: Sequence[Path], vocab_size: int):
        self.files = [TokenFile(p, vocab_size) for p in files]
        self.vocab_size = vocab_size
        self.lengths = [f.length for f in self.files]
        self.prefix = [0]
        for n in self.lengths:
            self.prefix.append(self.prefix[-1] + n)
        self.total_tokens = self.prefix[-1]

    def locate(self, global_idx: int) -> Tuple[int, int]:
        file_idx = bisect.bisect_right(self.prefix, global_idx) - 1
        local_idx = global_idx - self.prefix[file_idx]
        return file_idx, local_idx

    def get_range(self, start: int, end: int) -> np.ndarray:
        if start < 0 or end < start or end > self.total_tokens:
            raise ValueError(f"Bad range [{start}, {end}) total={self.total_tokens}")
        if start == end:
            return np.empty((0,), dtype=np.int64)
        f0, l0 = self.locate(start)
        f1, l1 = self.locate(end - 1)
        if f0 == f1:
            return self.files[f0].slice(l0, l1 + 1)
        parts = [self.files[f0].slice(l0, self.files[f0].length)]
        for fi in range(f0 + 1, f1):
            parts.append(self.files[fi].slice(0, self.files[fi].length))
        parts.append(self.files[f1].slice(0, l1 + 1))
        return np.concatenate(parts, axis=0)


# -----------------------------
# Hashing
# -----------------------------


def hash_history_view(history: np.ndarray, order: int) -> int:
    h = 1469598103934665603
    start = history.shape[0] - order
    for i in range(start, history.shape[0]):
        h = (h * HASH_BASE + (int(history[i]) + 1)) & MASK64
    return h


def hash_context_rows(ctx: np.ndarray, order: int) -> np.ndarray:
    out = np.zeros((ctx.shape[0],), dtype=np.uint64)
    for r in range(ctx.shape[0]):
        out[r] = np.uint64(hash_history_view(ctx[r], order))
    return out


# -----------------------------
# Model
# -----------------------------


@dataclass
class ContextCandidate:
    order: int
    ctx_hash: int
    entries: List[Tuple[int, int]]  # sorted by count desc initially, later token asc when stored
    full_total: int
    full_unique: int


class SparsePPM:
    def __init__(
        self,
        vocab_size: int,
        max_order: int,
        topk_per_context: int,
        min_count: int,
        discount: float = 0.75,
        base_alpha: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.max_order = max_order
        self.topk_per_context = topk_per_context
        self.min_count = min_count
        self.discount = discount
        self.base_alpha = base_alpha

        self.unigram = np.zeros((vocab_size,), dtype=np.int64)
        self.total_unigrams = 0
        self.continuation_unigram = np.ones((vocab_size,), dtype=np.int64)
        self.total_continuations = int(self.continuation_unigram.sum())

        # final model: order -> ctx_hash -> (tokens, counts, full_total, full_unique)
        self.orders: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray, int, int]]] = {}

    def fit_raw(self, train_tokens: np.ndarray, chunk_tokens: int = 1_000_000, report_every_chunks: int = 1) -> Dict[int, Counter[int]]:
        n = train_tokens.shape[0]
        if n <= self.max_order:
            raise ValueError(f"Need > max_order tokens, got {n}")
        eprint(f"[ppm] fitting raw counts on {n:,} tokens, max_order={self.max_order}")
        self.unigram = np.bincount(train_tokens.astype(np.int64), minlength=self.vocab_size).astype(np.int64)
        self.total_unigrams = int(self.unigram.sum())

        pair_counters: Dict[int, collections.Counter[int]] = {
            order: collections.Counter() for order in range(1, self.max_order + 1)
        }

        num_chunks = 0
        for start in range(0, n, chunk_tokens):
            end = min(n, start + chunk_tokens + self.max_order)
            block = train_tokens[start:end]
            if block.shape[0] <= self.max_order:
                continue
            num_chunks += 1
            for order in range(1, self.max_order + 1):
                if block.shape[0] <= order:
                    continue
                ctx = np.lib.stride_tricks.sliding_window_view(block, order + 1)
                prev = ctx[:, :order]
                nxt = ctx[:, order].astype(np.uint64)
                h = hash_context_rows(prev, order)
                pair = h * np.uint64(self.vocab_size) + nxt
                uniq, counts = np.unique(pair, return_counts=True)
                cnt = pair_counters[order]
                for u, c in zip(uniq.tolist(), counts.tolist()):
                    cnt[u] += int(c)
            if num_chunks % report_every_chunks == 0:
                eprint(f"[ppm] processed chunk {num_chunks}, tokens up to {min(start + chunk_tokens, n):,}/{n:,}")
                gc.collect()

        continuation = np.zeros((self.vocab_size,), dtype=np.int64)
        for packed in pair_counters.get(1, {}).keys():
            tok = int(packed % self.vocab_size)
            continuation[tok] += 1
        continuation += 1
        self.continuation_unigram = continuation
        self.total_continuations = int(continuation.sum())
        return pair_counters

    def base_prob(self, token: int) -> float:
        p_cont = self.continuation_unigram[token] / self.total_continuations
        p_uni = (self.unigram[token] + self.base_alpha) / (self.total_unigrams + self.base_alpha * self.vocab_size)
        return 0.85 * p_cont + 0.15 * p_uni

    @staticmethod
    def _context_bytes(entry_count: int) -> int:
        # 8 hash + 4 total + 4 unique + tokens/count arrays (2+2 each)
        return 16 + entry_count * 4

    def _candidate_score(self, order: int, kept_counts: List[int], full_total: int, full_unique: int) -> float:
        kept_sum = float(sum(kept_counts))
        if kept_sum <= 0.0 or full_total <= 0:
            return 0.0
        coverage = kept_sum / full_total
        order_bonus = 1.0 + 0.22 * order
        uniq_penalty = 1.0 / (1.0 + 0.08 * math.log1p(max(0, full_unique - len(kept_counts))))
        return kept_sum * coverage * order_bonus * uniq_penalty

    def _build_candidates(self, raw_pair_counters: Dict[int, Counter[int]], temp_topk_per_context: int) -> Tuple[Dict[int, Dict[int, ContextCandidate]], Dict[str, float]]:
        candidates: Dict[int, Dict[int, ContextCandidate]] = {o: {} for o in range(1, self.max_order + 1)}
        initial_contexts = 0
        initial_entries = 0
        for order in range(1, self.max_order + 1):
            contexts: DefaultDict[int, Dict[int, int]] = collections.defaultdict(dict)
            for packed, c in raw_pair_counters[order].items():
                if c < self.min_count:
                    continue
                ctx_hash = int(packed // self.vocab_size)
                tok = int(packed % self.vocab_size)
                contexts[ctx_hash][tok] = int(c)
            eprint(f"[extract] order={order}: candidate_contexts={len(contexts):,}")
            initial_contexts += len(contexts)
            for ctx_hash, token_counts in contexts.items():
                items = sorted(token_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:temp_topk_per_context]
                if not items:
                    continue
                candidates[order][ctx_hash] = ContextCandidate(
                    order=order,
                    ctx_hash=ctx_hash,
                    entries=items,
                    full_total=int(sum(token_counts.values())),
                    full_unique=len(token_counts),
                )
                initial_entries += len(items)
        return candidates, {
            "initial_contexts_after_min_count": float(initial_contexts),
            "initial_entries_after_filtering": float(initial_entries),
            "candidate_contexts": float(sum(len(v) for v in candidates.values())),
        }

    def extract_from_raw(self, raw_pair_counters: Dict[int, Counter[int]], target_size_bytes: int, temp_topk_per_context: int) -> Dict[str, float]:
        candidates, stats = self._build_candidates(raw_pair_counters, temp_topk_per_context=temp_topk_per_context)

        base_size = int(self.unigram.nbytes + self.continuation_unigram.nbytes + 4096)
        remaining = max(0, target_size_bytes - base_size)

        # Prefix options let us pack the budget almost exactly.
        prefix_items: List[Tuple[float, float, int, int, int, int]] = []
        # (utility_per_byte, utility, order, ctx_hash, keep_len, bytes_cost)
        for order in range(1, self.max_order + 1):
            for ctx_hash, cand in candidates[order].items():
                max_keep = min(self.topk_per_context, len(cand.entries))
                running_counts: List[int] = []
                s = 0
                for i in range(max_keep):
                    s += cand.entries[i][1]
                    running_counts.append(s)
                for keep_len in range(1, max_keep + 1):
                    util = self._candidate_score(order, [c for _, c in cand.entries[:keep_len]], cand.full_total, cand.full_unique)
                    bytes_cost = self._context_bytes(keep_len)
                    prefix_items.append((util / max(1, bytes_cost), util, order, ctx_hash, keep_len, bytes_cost))

        prefix_items.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_keep: Dict[Tuple[int, int], int] = {}
        best_util: Dict[Tuple[int, int], float] = {}
        used = 0
        for upb, util, order, ctx_hash, keep_len, bytes_cost in prefix_items:
            key = (order, ctx_hash)
            cur = best_keep.get(key, 0)
            cur_size = self._context_bytes(cur) if cur > 0 else 0
            delta = bytes_cost - cur_size
            if delta <= 0 or used + delta > remaining:
                continue
            prev_util = best_util.get(key, 0.0)
            if util <= prev_util:
                continue
            best_keep[key] = keep_len
            best_util[key] = util
            used += delta

        # Greedy leftover fill with single-entry upgrades.
        leftovers = remaining - used
        if leftovers > 0:
            upgrades: List[Tuple[float, int, int, int, int]] = []
            for order in range(1, self.max_order + 1):
                for ctx_hash, cand in candidates[order].items():
                    key = (order, ctx_hash)
                    cur = best_keep.get(key, 0)
                    max_keep = min(self.topk_per_context, len(cand.entries))
                    if cur >= max_keep:
                        continue
                    next_keep = cur + 1
                    old_util = self._candidate_score(order, [c for _, c in cand.entries[:cur]], cand.full_total, cand.full_unique) if cur > 0 else 0.0
                    new_util = self._candidate_score(order, [c for _, c in cand.entries[:next_keep]], cand.full_total, cand.full_unique)
                    delta_util = max(0.0, new_util - old_util)
                    delta_size = self._context_bytes(next_keep) - (self._context_bytes(cur) if cur > 0 else 0)
                    upgrades.append((delta_util / max(1, delta_size), order, ctx_hash, next_keep, delta_size))
            upgrades.sort(reverse=True)
            for upb, order, ctx_hash, next_keep, delta_size in upgrades:
                key = (order, ctx_hash)
                cur = best_keep.get(key, 0)
                if next_keep != cur + 1:
                    continue
                if delta_size <= leftovers:
                    best_keep[key] = next_keep
                    leftovers -= delta_size
                    if leftovers <= 0:
                        break

        final_orders: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray, int, int]]] = {o: {} for o in range(1, self.max_order + 1)}
        selected_contexts = 0
        selected_entries = 0
        used_size = base_size
        for order in range(1, self.max_order + 1):
            for ctx_hash, cand in candidates[order].items():
                keep_len = best_keep.get((order, ctx_hash), 0)
                if keep_len <= 0:
                    continue
                kept = cand.entries[:keep_len]
                kept = sorted(kept, key=lambda kv: kv[0])
                tokens = np.array([t for t, _ in kept], dtype=np.uint16)
                counts = np.array([min(65535, c) for _, c in kept], dtype=np.uint16)
                final_orders[order][ctx_hash] = (tokens, counts, cand.full_total, cand.full_unique)
                selected_contexts += 1
                selected_entries += keep_len
                used_size += self._context_bytes(keep_len)
        self.orders = final_orders
        stats.update({
            "selected_contexts": float(selected_contexts),
            "selected_entries": float(selected_entries),
            "estimated_model_size_bytes": float(min(used_size, target_size_bytes)),
            "estimated_model_size_mb": float(min(used_size, target_size_bytes) / (1024 * 1024)),
            "budget_fill_ratio": float(min(1.0, used_size / max(1, target_size_bytes))),
            "graceful_unused_bytes": float(max(0, target_size_bytes - used_size)),
            "target_size_mb": float(target_size_bytes / (1024 * 1024)),
            "temp_topk_per_context": float(temp_topk_per_context),
            "topk_per_context": float(self.topk_per_context),
        })
        return stats

    def _ctx_prob(self, ctx_data: Tuple[np.ndarray, np.ndarray, int, int], token: int, lower_prob: float) -> float:
        tokens, counts, full_total, full_unique = ctx_data
        if full_total <= 0:
            return lower_prob
        idx = int(np.searchsorted(tokens, token))
        c = int(counts[idx]) if idx < tokens.shape[0] and int(tokens[idx]) == token else 0
        d = min(self.discount, full_total / max(1.0, full_unique))
        bow = min(0.999999, (d * full_unique) / full_total)
        p = bow * lower_prob
        if c > 0:
            p += max(c - d, 0.0) / full_total
        return max(1e-12, p)

    def prob_next(self, history: np.ndarray, token: int) -> float:
        p = self.base_prob(token)
        max_order = min(self.max_order, history.shape[0])
        for order in range(1, max_order + 1):
            ctx_hash = hash_history_view(history, order)
            ctx_data = self.orders.get(order, {}).get(ctx_hash)
            if ctx_data is None:
                continue
            p = self._ctx_prob(ctx_data, token, p)
        return max(1e-12, p)

    def evaluate(self, tokens: np.ndarray, byte_lengths: Optional[np.ndarray] = None, log_every: int = 100_000) -> Dict[str, float]:
        total_nll_bits = 0.0
        total_tok = 0
        total_bytes = 0
        start_time = time.time()
        for i in range(self.max_order, tokens.shape[0]):
            history = tokens[max(0, i - self.max_order):i]
            tgt = int(tokens[i])
            p = self.prob_next(history, tgt)
            total_nll_bits += -math.log2(p)
            total_tok += 1
            if byte_lengths is not None:
                total_bytes += int(byte_lengths[tgt])
            if total_tok % log_every == 0:
                elapsed = time.time() - start_time
                bpt = total_nll_bits / total_tok
                msg = f"[eval] {total_tok:,} tokens, bits/token={bpt:.4f}, elapsed={elapsed:.1f}s"
                if total_bytes > 0:
                    msg += f", val_bpb={total_nll_bits / total_bytes:.4f}"
                eprint(msg)
        out = {
            "bits_per_token": total_nll_bits / total_tok,
            "nll_bits": total_nll_bits,
            "eval_tokens": float(total_tok),
        }
        if total_bytes > 0:
            out["val_bpb"] = total_nll_bits / total_bytes
            out["eval_bytes"] = float(total_bytes)
        return out

    def save(self, path: str) -> None:
        payload = {
            "vocab_size": self.vocab_size,
            "max_order": self.max_order,
            "topk_per_context": self.topk_per_context,
            "min_count": self.min_count,
            "discount": self.discount,
            "base_alpha": self.base_alpha,
            "unigram": self.unigram,
            "continuation_unigram": self.continuation_unigram,
            "total_unigrams": self.total_unigrams,
            "total_continuations": self.total_continuations,
            "orders": {
                order: {
                    h: {
                        "tokens": tokens,
                        "counts": counts,
                        "full_total": full_total,
                        "full_unique": full_unique,
                    }
                    for h, (tokens, counts, full_total, full_unique) in order_dict.items()
                }
                for order, order_dict in self.orders.items()
            },
        }
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with gzip.open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


# -----------------------------
# Utilities
# -----------------------------


def infer_byte_lengths(vocab_size: int, tokenizer_path: Optional[str]) -> Optional[np.ndarray]:
    if not tokenizer_path:
        return None
    if spm is None:
        eprint("[warn] sentencepiece not installed; cannot compute byte lengths for val_bpb")
        return None
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    lengths = np.ones((vocab_size,), dtype=np.int32)
    for i in range(vocab_size):
        try:
            piece = sp.id_to_piece(i)
            lengths[i] = max(1, len(piece.encode("utf-8")))
        except Exception:
            lengths[i] = 1
    return lengths


def choose_train_val_ranges(total_tokens: int, val_tokens: int, max_train_tokens: Optional[int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    if total_tokens <= val_tokens + 10:
        raise ValueError(f"Not enough tokens: total={total_tokens} val={val_tokens}")
    val_start = total_tokens - val_tokens
    val_range = (val_start, total_tokens)
    if max_train_tokens is None or max_train_tokens >= val_start:
        train_range = (0, val_start)
    else:
        train_range = (max(0, val_start - max_train_tokens), val_start)
    return train_range, val_range


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Token-level sparse PPM with conservative extraction and discounted backoff.")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--vocab_size", type=int, default=1024)
    p.add_argument("--tokenizer_path", type=str, default=None)
    p.add_argument("--max_order", type=int, default=4)
    p.add_argument("--topk_per_context", type=int, default=8)
    p.add_argument("--temp_topk_per_context", type=int, default=32)
    p.add_argument("--min_count", type=int, default=2)
    p.add_argument("--discount", type=float, default=0.75)
    p.add_argument("--base_alpha", type=float, default=0.1)
    p.add_argument("--chunk_tokens", type=int, default=1_000_000)
    p.add_argument("--max_train_tokens", type=int, default=5_000_000)
    p.add_argument("--val_tokens", type=int, default=500_000)
    p.add_argument("--target_size_mb", type=float, default=16.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--save_model", type=str, default="./ppm_model.pkl.gz")
    p.add_argument("--save_metrics", type=str, default="./ppm_metrics.json")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    files = discover_token_files(args.data_path)
    eprint(f"[data] found {len(files)} token files")
    corpus = TokenCorpus(files, vocab_size=args.vocab_size)
    eprint(f"[data] total_tokens={corpus.total_tokens:,}")

    train_range, val_range = choose_train_val_ranges(corpus.total_tokens, args.val_tokens, args.max_train_tokens)
    eprint(f"[data] train_range={train_range}, val_range={val_range}")

    t0 = time.time()
    train_tokens = corpus.get_range(*train_range)
    val_tokens = corpus.get_range(*val_range)
    eprint(f"[data] loaded train={train_tokens.shape[0]:,}, val={val_tokens.shape[0]:,} in {time.time()-t0:.1f}s")

    bad = int((train_tokens >= args.vocab_size).sum() + (val_tokens >= args.vocab_size).sum())
    if bad:
        raise ValueError(f"Found {bad} token ids >= vocab_size={args.vocab_size}")

    byte_lengths = infer_byte_lengths(args.vocab_size, args.tokenizer_path)

    model = SparsePPM(
        vocab_size=args.vocab_size,
        max_order=args.max_order,
        topk_per_context=args.topk_per_context,
        min_count=args.min_count,
        discount=args.discount,
        base_alpha=args.base_alpha,
    )

    fit_start = time.time()
    raw = model.fit_raw(train_tokens, chunk_tokens=args.chunk_tokens)
    fit_seconds = time.time() - fit_start

    extract_start = time.time()
    extract_stats = model.extract_from_raw(
        raw_pair_counters=raw,
        target_size_bytes=int(args.target_size_mb * 1024 * 1024),
        temp_topk_per_context=args.temp_topk_per_context,
    )
    extract_seconds = time.time() - extract_start

    eval_start = time.time()
    metrics = model.evaluate(val_tokens, byte_lengths=byte_lengths)
    eval_seconds = time.time() - eval_start

    metrics.update({
        **extract_stats,
        "fit_seconds": fit_seconds,
        "extract_seconds": extract_seconds,
        "eval_seconds": eval_seconds,
        "train_tokens": int(train_tokens.shape[0]),
        "val_tokens_total": int(val_tokens.shape[0]),
        "max_order": args.max_order,
        "min_count": args.min_count,
        "discount": args.discount,
        "base_alpha": args.base_alpha,
    })
    eprint("[result] " + json.dumps(metrics, indent=2, sort_keys=True))

    model.save(args.save_model)
    parent = os.path.dirname(args.save_metrics)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.save_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
