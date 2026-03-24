#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import collections
import gc
import gzip
import json
import math
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Counter, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import sentencepiece as spm  # type: ignore
except Exception:
    spm = None


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
# Hashing / stats
# -----------------------------


def hash_context(ctx: np.ndarray, order: int, base: np.uint64 = np.uint64(1315423911)) -> np.ndarray:
    h = np.zeros((ctx.shape[0],), dtype=np.uint64)
    for i in range(order):
        h = h * base + ctx[:, i].astype(np.uint64) + np.uint64(1)
    return h


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


# -----------------------------
# Model structures
# -----------------------------


@dataclass
class EntryStats:
    token: int
    count: int
    score: float


@dataclass
class ContextStats:
    order: int
    ctx_hash: int
    entries: List[EntryStats]
    total_count: int
    utility: float
    utility_per_byte: float
    size_bytes: int


@dataclass
class OrderStats:
    contexts: Dict[int, Counter[int]]


class SparsePPM:
    def __init__(
        self,
        vocab_size: int,
        max_order: int,
        topk_per_context: int,
        min_count: int,
        lambdas: Optional[List[float]] = None,
    ):
        self.vocab_size = vocab_size
        self.max_order = max_order
        self.topk_per_context = topk_per_context
        self.min_count = min_count
        self.unigram = np.zeros((vocab_size,), dtype=np.int64)
        self.total_unigrams = 0
        self.orders: Dict[int, OrderStats] = {}
        if lambdas is None:
            raw = [1.0] + [0.7 ** (max_order - i) for i in range(1, max_order + 1)]
            s = sum(raw)
            lambdas = [x / s for x in raw]
        self.lambdas = lambdas

    def fit_raw(
        self,
        train_tokens: np.ndarray,
        chunk_tokens: int = 1_000_000,
        report_every_chunks: int = 1,
    ) -> Dict[int, Counter[int]]:
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
                h = hash_context(prev, order)
                pair = h * np.uint64(self.vocab_size) + nxt
                uniq, counts = np.unique(pair, return_counts=True)
                cnt = pair_counters[order]
                for u, c in zip(uniq.tolist(), counts.tolist()):
                    cnt[u] += int(c)
            if num_chunks % report_every_chunks == 0:
                eprint(f"[ppm] processed chunk {num_chunks}, tokens up to {end:,}/{n:,}")
                gc.collect()
        return pair_counters

    def _base_prob(self, token: int, alpha: float = 0.1) -> float:
        return (self.unigram[token] + alpha) / (self.total_unigrams + alpha * self.vocab_size)

    def extract_from_raw(
        self,
        raw_pair_counters: Dict[int, Counter[int]],
        target_size_bytes: Optional[int] = None,
        temp_topk_per_context: Optional[int] = None,
        max_contexts_per_order: Optional[int] = None,
    ) -> Dict[str, float]:
        temp_topk = temp_topk_per_context or max(self.topk_per_context * 4, self.topk_per_context)
        all_contexts: List[ContextStats] = []
        initial_context_count = 0
        initial_entry_count = 0

        for order in range(1, self.max_order + 1):
            contexts: DefaultDict[int, Counter[int]] = collections.defaultdict(collections.Counter)
            for packed, c in raw_pair_counters[order].items():
                if c < self.min_count:
                    continue
                ctx_hash = int(packed // self.vocab_size)
                token = int(packed % self.vocab_size)
                contexts[ctx_hash][token] = int(c)
            initial_context_count += len(contexts)

            local_contexts: List[ContextStats] = []
            for ctx_hash, ctr in contexts.items():
                # keep larger temporary fanout first
                items = ctr.items()
                if len(ctr) > temp_topk:
                    items = ctr.most_common(temp_topk)
                else:
                    items = list(items)
                total = int(sum(c for _, c in items))
                if total <= 0:
                    continue
                entries: List[EntryStats] = []
                utility = 0.0
                for token, count in items:
                    p_base = self._base_prob(token)
                    p_ctx = count / total
                    gain = max(0.0, math.log2(max(p_ctx, 1e-12) / max(p_base, 1e-12)))
                    score = count * gain
                    if score <= 0.0:
                        continue
                    entries.append(EntryStats(token=int(token), count=int(count), score=float(score)))
                    utility += score
                if not entries:
                    continue
                entries.sort(key=lambda e: (-e.score, -e.count, e.token))
                if len(entries) > self.topk_per_context:
                    entries = entries[: self.topk_per_context]
                    total = int(sum(e.count for e in entries))
                    utility = float(sum(e.score for e in entries))
                size_bytes = 8 + len(entries) * 4
                utility_per_byte = utility / max(size_bytes, 1)
                local_contexts.append(
                    ContextStats(
                        order=order,
                        ctx_hash=ctx_hash,
                        entries=entries,
                        total_count=total,
                        utility=utility,
                        utility_per_byte=utility_per_byte,
                        size_bytes=size_bytes,
                    )
                )
                initial_entry_count += len(entries)

            local_contexts.sort(key=lambda c: (-c.utility_per_byte, -c.utility, -c.order, c.ctx_hash))
            if max_contexts_per_order is not None:
                local_contexts = local_contexts[:max_contexts_per_order]
            all_contexts.extend(local_contexts)
            eprint(f"[extract] order={order}: candidate_contexts={len(local_contexts):,}")

        all_contexts.sort(key=lambda c: (-c.utility_per_byte, -c.utility, -c.order, c.ctx_hash))
        chosen: List[ContextStats]
        if target_size_bytes is None:
            chosen = all_contexts
        else:
            size = self.unigram.nbytes
            chosen = []
            for ctx in all_contexts:
                if size + ctx.size_bytes > target_size_bytes:
                    continue
                chosen.append(ctx)
                size += ctx.size_bytes

        chosen_by_order: Dict[int, Dict[int, Counter[int]]] = {order: {} for order in range(1, self.max_order + 1)}
        for ctx in chosen:
            chosen_by_order[ctx.order][ctx.ctx_hash] = collections.Counter({e.token: e.count for e in ctx.entries})
        self.orders = {order: OrderStats(contexts=ctxs) for order, ctxs in chosen_by_order.items() if ctxs}

        estimated = self.estimated_size_bytes()
        return {
            "candidate_contexts": float(len(all_contexts)),
            "selected_contexts": float(sum(len(v.contexts) for v in self.orders.values())),
            "selected_entries": float(sum(len(c) for o in self.orders.values() for c in o.contexts.values())),
            "initial_contexts_after_min_count": float(initial_context_count),
            "initial_entries_after_filtering": float(initial_entry_count),
            "estimated_model_size_bytes": float(estimated),
            "estimated_model_size_mb": float(estimated / (1024 * 1024)),
        }

    def prob_next(self, history: np.ndarray, target: int, alpha: float = 0.1) -> float:
        p = self.lambdas[0] * self._base_prob(target, alpha=alpha)
        for order in range(1, self.max_order + 1):
            if history.shape[0] < order:
                continue
            ctx = history[-order:].reshape(1, order)
            h = int(hash_context(ctx, order)[0])
            stats = self.orders.get(order)
            if stats is None:
                continue
            ctr = stats.contexts.get(h)
            if not ctr:
                continue
            total = sum(ctr.values())
            bucket = len(ctr)
            p_order = (ctr.get(target, 0) + alpha) / (total + alpha * max(bucket, 1))
            p += self.lambdas[order] * p_order
        return max(float(p), 1e-12)

    def evaluate(
        self,
        tokens: np.ndarray,
        byte_lengths: Optional[np.ndarray] = None,
        log_every: int = 1_000_000,
        history_cap: int = 256,
    ) -> Dict[str, float]:
        if tokens.shape[0] <= self.max_order:
            raise ValueError("Validation set too small")
        total_nll_bits = 0.0
        total_tok = 0
        total_bytes = 0
        start_time = time.time()
        for i in range(self.max_order, tokens.shape[0]):
            history = tokens[:i] if i < history_cap else tokens[i - history_cap:i]
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
        bits_per_token = total_nll_bits / total_tok
        out = {
            "bits_per_token": bits_per_token,
            "nll_bits": total_nll_bits,
            "eval_tokens": float(total_tok),
        }
        if total_bytes > 0:
            out["val_bpb"] = total_nll_bits / total_bytes
            out["eval_bytes"] = float(total_bytes)
        return out

    def estimated_size_bytes(self) -> int:
        size = self.unigram.nbytes
        for stats in self.orders.values():
            for ctr in stats.contexts.values():
                size += 8
                size += len(ctr) * 4
        return size

    def save(self, path: str) -> None:
        payload = {
            "vocab_size": self.vocab_size,
            "max_order": self.max_order,
            "topk_per_context": self.topk_per_context,
            "min_count": self.min_count,
            "lambdas": self.lambdas,
            "unigram": self.unigram,
            "total_unigrams": self.total_unigrams,
            "orders": {
                order: {h: dict(ctr) for h, ctr in stats.contexts.items()}
                for order, stats in self.orders.items()
            },
        }
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


# -----------------------------
# Main
# -----------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Token-level sparse PPM / interpolated n-gram baseline with big-build then extract/prune.")
    p.add_argument("--data_path", type=str, required=True, help="Path to token shard directory or file (.bin/.npy/.npz/.pt)")
    p.add_argument("--vocab_size", type=int, default=1024)
    p.add_argument("--tokenizer_path", type=str, default=None, help="Optional SentencePiece model for approximate val_bpb")
    p.add_argument("--max_order", type=int, default=4)
    p.add_argument("--topk_per_context", type=int, default=8, help="Final kept continuations per context")
    p.add_argument("--temp_topk_per_context", type=int, default=32, help="Temporary larger fanout before extraction")
    p.add_argument("--min_count", type=int, default=2)
    p.add_argument("--chunk_tokens", type=int, default=1_000_000)
    p.add_argument("--max_train_tokens", type=int, default=5_000_000)
    p.add_argument("--val_tokens", type=int, default=500_000)
    p.add_argument("--target_size_mb", type=float, default=16.0, help="Extraction target size. Set <=0 to disable budgeted extraction")
    p.add_argument("--max_contexts_per_order", type=int, default=0, help="Optional cap before global extraction; 0 disables")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--save_model", type=str, default="/mnt/data/ppm_model.pkl.gz")
    p.add_argument("--save_metrics", type=str, default="/mnt/data/ppm_metrics.json")
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
    )

    fit_start = time.time()
    raw_pair_counters = model.fit_raw(train_tokens, chunk_tokens=args.chunk_tokens)
    fit_seconds = time.time() - fit_start

    extract_start = time.time()
    target_size_bytes = None if args.target_size_mb <= 0 else int(args.target_size_mb * 1024 * 1024)
    extraction = model.extract_from_raw(
        raw_pair_counters,
        target_size_bytes=target_size_bytes,
        temp_topk_per_context=args.temp_topk_per_context,
        max_contexts_per_order=None if args.max_contexts_per_order <= 0 else args.max_contexts_per_order,
    )
    extract_seconds = time.time() - extract_start

    eval_start = time.time()
    metrics = model.evaluate(val_tokens, byte_lengths=byte_lengths)
    eval_seconds = time.time() - eval_start
    metrics.update(extraction)
    metrics.update(
        {
            "fit_seconds": fit_seconds,
            "extract_seconds": extract_seconds,
            "eval_seconds": eval_seconds,
            "train_tokens": int(train_tokens.shape[0]),
            "val_tokens_total": int(val_tokens.shape[0]),
            "max_order": args.max_order,
            "topk_per_context": args.topk_per_context,
            "temp_topk_per_context": args.temp_topk_per_context,
            "min_count": args.min_count,
            "target_size_mb": args.target_size_mb,
            "estimated_model_size_mb": model.estimated_size_bytes() / (1024 * 1024),
        }
    )

    eprint("[result] " + json.dumps(metrics, indent=2, sort_keys=True))
    model.save(args.save_model)
    with open(args.save_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
