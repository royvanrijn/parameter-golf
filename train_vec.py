#!/usr/bin/env python3
"""Train lightweight token vector embeddings from local tokenized corpora.

The script is designed for very fast experimentation on the parameter-golf token files:
- learns token embeddings from local co-occurrence statistics
- optionally refines with a tiny attention-style context block
- evaluates embedding quality with intrinsic + predictive metrics
- saves an artifact that can be reused as a pre/post step around another model
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def find_token_files(path: str) -> list[Path]:
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
    def __init__(self, files: list[Path]):
        self.files = files
        self.lengths: list[int] = []
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
            out[pos : pos + len(chunk)] = chunk
            pos += len(chunk)
        return out


@dataclass
class TinyContextAttention:
    """A tiny offset-attention module used only for refinement/generation.

    weights[d-1] corresponds to offset d in [1..window].
    """

    weights: np.ndarray  # shape [window], float32

    def normalized(self) -> np.ndarray:
        x = self.weights.astype(np.float64)
        x = x - x.max()
        e = np.exp(x)
        z = max(e.sum(), 1e-12)
        return (e / z).astype(np.float32)


class VectorEmbeddingModel:
    def __init__(
        self,
        embeddings: np.ndarray,
        vocab_size: int,
        dim: int,
        window: int,
        config: dict[str, Any],
        attention: TinyContextAttention | None = None,
    ):
        self.embeddings = embeddings.astype(np.float32, copy=False)
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.window = int(window)
        self.config = dict(config)
        self.attention = attention
        self._normalized = self._normalize(self.embeddings)

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(x, axis=1, keepdims=True)
        denom = np.clip(denom, 1e-6, None)
        return x / denom

    def encode_tokens(self, token_ids: np.ndarray) -> np.ndarray:
        idx = np.asarray(token_ids, dtype=np.int64)
        return self.embeddings[idx]

    def decode_vectors(self, vectors: np.ndarray, topk: int = 1) -> tuple[np.ndarray, np.ndarray]:
        v = np.asarray(vectors, dtype=np.float32)
        if v.ndim == 1:
            v = v[None, :]
        v = self._normalize(v)
        sim = v @ self._normalized.T
        k = max(1, min(topk, self.vocab_size))
        idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        row = np.arange(sim.shape[0])[:, None]
        scores = sim[row, idx]
        order = np.argsort(-scores, axis=1)
        idx = idx[row, order]
        scores = scores[row, order]
        return idx.astype(np.int32), scores.astype(np.float32)

    def context_vector(self, context_tokens: np.ndarray) -> np.ndarray:
        ctx = np.asarray(context_tokens, dtype=np.int64)
        if ctx.size == 0:
            return np.zeros((self.dim,), dtype=np.float32)
        ctx = ctx[-self.window :]
        mat = self.embeddings[ctx]
        if self.attention is None:
            return mat.mean(axis=0)
        w = self.attention.normalized()
        use = w[: mat.shape[0]][::-1]
        use = use / max(use.sum(), 1e-12)
        return (mat * use[:, None]).sum(axis=0)

    def generate(self, prompt_tokens: np.ndarray, steps: int = 20, temperature: float = 1.0, topk: int = 20) -> np.ndarray:
        out = np.asarray(prompt_tokens, dtype=np.int32).copy()
        if out.size == 0:
            raise ValueError("prompt_tokens must be non-empty")
        t = max(float(temperature), 1e-4)
        for _ in range(int(steps)):
            q = self.context_vector(out)
            q = self._normalize(q[None, :])[0]
            logits = (q @ self._normalized.T) / t
            k = max(1, min(int(topk), self.vocab_size))
            top_idx = np.argpartition(-logits, kth=k - 1)[:k]
            top_logits = logits[top_idx]
            top_logits = top_logits - np.max(top_logits)
            probs = np.exp(top_logits)
            probs /= max(probs.sum(), 1e-12)
            nxt = int(np.random.choice(top_idx, p=probs))
            out = np.append(out, np.int32(nxt))
        return out

    def save(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = {
            "embeddings": self.embeddings,
            "vocab_size": self.vocab_size,
            "dim": self.dim,
            "window": self.window,
            "config": self.config,
            "attention_weights": None if self.attention is None else self.attention.weights,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "VectorEmbeddingModel":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        attn_w = payload.get("attention_weights")
        attn = None if attn_w is None else TinyContextAttention(np.asarray(attn_w, dtype=np.float32))
        return cls(
            embeddings=np.asarray(payload["embeddings"], dtype=np.float32),
            vocab_size=int(payload["vocab_size"]),
            dim=int(payload["dim"]),
            window=int(payload["window"]),
            config=dict(payload.get("config", {})),
            attention=attn,
        )


def build_cooc_ppmi_embeddings(
    tokens: np.ndarray,
    vocab_size: int,
    dim: int,
    window: int,
    device: torch.device,
    smooth: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    x = torch.from_numpy(tokens.astype(np.int64, copy=False)).to(device)
    co = torch.zeros((vocab_size, vocab_size), dtype=torch.float32, device=device)
    for d in range(1, window + 1):
        w = 1.0 / (d**smooth)
        a = x[:-d]
        b = x[d:]
        idx = a * vocab_size + b
        vals = torch.full((idx.numel(),), w, dtype=torch.float32, device=device)
        co.view(-1).index_add_(0, idx, vals)
        idx2 = b * vocab_size + a
        co.view(-1).index_add_(0, idx2, vals)
    row = co.sum(dim=1, keepdim=True).clamp_min_(1.0)
    col = co.sum(dim=0, keepdim=True).clamp_min_(1.0)
    total = co.sum().clamp_min_(1.0)
    pmi = torch.log((co * total + 1e-6) / (row * col + 1e-6))
    ppmi = torch.clamp(pmi, min=0.0)
    q = min(max(dim, 2), min(ppmi.shape) - 1)
    U, S, _ = torch.pca_lowrank(ppmi, q=q, center=False)
    emb = U[:, :dim] * torch.sqrt(S[:dim]).unsqueeze(0)
    emb = F.normalize(emb, dim=1)
    sparsity = float((co == 0).float().mean().item())
    return emb.detach(), {"cooc_sparsity": sparsity, "cooc_total_mass": float(total.item())}


def refine_with_tiny_attention(
    init_emb: torch.Tensor,
    tokens: np.ndarray,
    vocab_size: int,
    window: int,
    steps: int,
    batch_size: int,
    lr: float,
    neg_k: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, TinyContextAttention, dict[str, float]]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    emb = torch.nn.Parameter(init_emb.clone().to(device))
    offset_logits = torch.nn.Parameter(torch.zeros((window,), device=device))
    opt = torch.optim.Adam([emb, offset_logits], lr=lr)

    x = torch.from_numpy(tokens.astype(np.int64, copy=False)).to(device)
    n = x.numel()
    if n < window + 2:
        return init_emb, TinyContextAttention(np.zeros((window,), dtype=np.float32)), {"attn_refine_steps": 0.0}

    for _ in range(steps):
        pos = torch.randint(window, n - 1, (batch_size,), device=device, generator=g)
        target = x[pos]
        offsets = torch.arange(1, window + 1, device=device)
        ctx_idx = pos[:, None] - offsets[None, :]
        ctx = x[ctx_idx]
        w = F.softmax(offset_logits, dim=0)
        ctx_vec = (emb[ctx] * w[None, :, None]).sum(dim=1)
        pos_score = (ctx_vec * emb[target]).sum(dim=1)

        neg = torch.randint(0, vocab_size, (batch_size, neg_k), device=device, generator=g)
        neg_emb = emb[neg]
        neg_score = (ctx_vec[:, None, :] * neg_emb).sum(dim=2)

        pos_loss = F.softplus(-pos_score).mean()
        neg_loss = F.softplus(neg_score).mean()
        reg = 1e-4 * (emb.pow(2).mean() + offset_logits.pow(2).mean())
        loss = pos_loss + neg_loss + reg

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    out_emb = F.normalize(emb.detach(), dim=1)
    attn = TinyContextAttention(offset_logits.detach().cpu().numpy().astype(np.float32))
    return out_emb, attn, {"attn_refine_steps": float(steps)}


def evaluate_embeddings(
    model: VectorEmbeddingModel,
    val_tokens: np.ndarray,
    samples: int,
    neg_k: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(val_tokens)
    if n < model.window + 2:
        return {
            "val_pair_cos_pos": 0.0,
            "val_pair_cos_neg": 0.0,
            "val_pair_nce_acc": 0.0,
            "val_next_top1": 0.0,
            "val_next_top5": 0.0,
        }

    k = min(samples, n - model.window - 1)
    pos_idx = rng.integers(model.window, n - 1, size=k)

    emb = model._normalized
    pos_cos = []
    neg_cos = []
    nce_correct = 0
    top1 = 0
    top5 = 0

    for i in pos_idx:
        target = int(val_tokens[i])
        context = val_tokens[i - model.window : i]
        q = model.context_vector(context)
        q = q / max(np.linalg.norm(q), 1e-6)

        pos = float(q @ emb[target])
        pos_cos.append(pos)

        neg_ids = rng.integers(0, model.vocab_size, size=neg_k)
        neg_vals = emb[neg_ids] @ q
        neg_cos.extend(neg_vals.tolist())
        if pos > float(np.max(neg_vals)):
            nce_correct += 1

        logits = emb @ q
        pred5 = np.argpartition(-logits, kth=4)[:5]
        if target in pred5:
            top5 += 1
        if target == int(np.argmax(logits)):
            top1 += 1

    return {
        "val_pair_cos_pos": float(np.mean(pos_cos)),
        "val_pair_cos_neg": float(np.mean(neg_cos) if neg_cos else 0.0),
        "val_pair_nce_acc": float(nce_correct / max(1, k)),
        "val_next_top1": float(top1 / max(1, k)),
        "val_next_top5": float(top5 / max(1, k)),
    }


def print_examples(model: VectorEmbeddingModel, train_tokens: np.ndarray, num_examples: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    counts = np.bincount(train_tokens, minlength=model.vocab_size)
    common = np.argsort(-counts)
    print("[examples] nearest-neighbor tokens by cosine")
    shown = 0
    for tok in common:
        if counts[tok] == 0:
            continue
        nn_idx, nn_score = model.decode_vectors(model.embeddings[tok], topk=6)
        idx = nn_idx[0].tolist()
        sc = nn_score[0].tolist()
        pairs = [f"{t}:{s:.3f}" for t, s in zip(idx[1:], sc[1:])]
        print(f"  token={int(tok):4d} freq={int(counts[tok]):7d} -> {', '.join(pairs)}")
        shown += 1
        if shown >= num_examples:
            break

    if len(train_tokens) > model.window + 5:
        s = int(rng.integers(model.window, len(train_tokens) - 5))
        prompt = train_tokens[s - model.window : s]
        gen = model.generate(prompt, steps=10, temperature=0.8, topk=16)
        print(f"[examples] generate prompt(last {model.window})={prompt.tolist()}")
        print(f"[examples] generated={gen.tolist()}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--vocab_size", type=int, default=1024)
    ap.add_argument("--max_train_tokens", type=int, default=2_000_000)
    ap.add_argument("--val_tokens", type=int, default=100_000)
    ap.add_argument("--vec_dim", type=int, default=16)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--cooc_smooth", type=float, default=1.0)
    ap.add_argument("--use_attention_refine", action="store_true")
    ap.add_argument("--attn_steps", type=int, default=200)
    ap.add_argument("--attn_batch_size", type=int, default=2048)
    ap.add_argument("--attn_lr", type=float, default=0.05)
    ap.add_argument("--neg_k", type=int, default=16)
    ap.add_argument("--metric_samples", type=int, default=8000)
    ap.add_argument("--example_count", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--save_model", type=str, default="./vec_model.pkl")
    ap.add_argument("--save_metrics", type=str, default="./vec_metrics.json")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dev = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    files = find_token_files(args.data_path)
    if not files:
        raise SystemExit(f"No token files found under {args.data_path}")
    print(f"[data] found {len(files)} token files")

    corpus = TokenCorpus(files)
    print(f"[data] total_tokens={corpus.total_tokens:,}")

    train_end = max(0, corpus.total_tokens - args.val_tokens)
    train_start = max(0, train_end - args.max_train_tokens)
    val_start = train_end
    val_end = min(corpus.total_tokens, val_start + args.val_tokens)

    t0 = time.time()
    train_tokens = corpus.read_range(train_start, train_end)
    val_tokens = corpus.read_range(val_start, val_end)
    load_s = time.time() - t0

    if len(train_tokens) < args.window + 2:
        raise SystemExit("Not enough train tokens for requested context window")

    print(
        f"[data] loaded train={len(train_tokens):,}, val={len(val_tokens):,} "
        f"in {load_s:.2f}s | train_range=({train_start},{train_end}) val_range=({val_start},{val_end})"
    )

    fit_t0 = time.time()
    emb, fit_info = build_cooc_ppmi_embeddings(
        tokens=train_tokens,
        vocab_size=args.vocab_size,
        dim=args.vec_dim,
        window=args.window,
        device=dev,
        smooth=args.cooc_smooth,
    )

    attention = None
    if args.use_attention_refine:
        emb, attention, attn_info = refine_with_tiny_attention(
            init_emb=emb,
            tokens=train_tokens,
            vocab_size=args.vocab_size,
            window=args.window,
            steps=args.attn_steps,
            batch_size=args.attn_batch_size,
            lr=args.attn_lr,
            neg_k=args.neg_k,
            device=dev,
            seed=args.seed,
        )
        fit_info.update(attn_info)

    model = VectorEmbeddingModel(
        embeddings=emb.detach().cpu().numpy(),
        vocab_size=args.vocab_size,
        dim=args.vec_dim,
        window=args.window,
        config=vars(args),
        attention=attention,
    )

    metrics = evaluate_embeddings(
        model=model,
        val_tokens=val_tokens,
        samples=args.metric_samples,
        neg_k=args.neg_k,
        seed=args.seed,
    )
    fit_seconds = time.time() - fit_t0

    model.save(args.save_model)
    payload = {
        "train_tokens": int(len(train_tokens)),
        "val_tokens": int(len(val_tokens)),
        "fit_seconds": float(fit_seconds),
        "device_used": str(dev),
        **fit_info,
        **metrics,
    }
    parent = os.path.dirname(args.save_metrics)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.save_metrics, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print("[metrics] " + json.dumps(payload, indent=2, sort_keys=True))
    print_examples(model, train_tokens, num_examples=args.example_count, seed=args.seed)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
