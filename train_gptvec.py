#!/usr/bin/env python3
"""Train a GPT-style sequence model in vector-embedding space.

Workflow:
1) Run `train_vec.py` quickly to build a token<->vector embedding artifact.
2) Load that artifact and map token ids to vectors.
3) Train a causal transformer over vectors (predict next-token vector).
4) Decode predictions back to tokens via nearest-neighbor in vector space.

This intentionally acts like a wrapper around a learned token embedding space:
  token -> vec -> GPT(vec) -> vec -> token
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from train_vec import TokenCorpus, VectorEmbeddingModel, find_token_files


@dataclass
class Hyperparameters:
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    vec_dim: int = int(os.environ.get("VEC_DIM", 16))
    vec_window: int = int(os.environ.get("VEC_WINDOW", 8))
    vec_max_train_tokens: int = int(os.environ.get("VEC_MAX_TRAIN_TOKENS", 500_000))
    vec_val_tokens: int = int(os.environ.get("VEC_VAL_TOKENS", 50_000))
    vec_metric_samples: int = int(os.environ.get("VEC_METRIC_SAMPLES", 4000))
    vec_model_path: str = os.environ.get("VEC_MODEL_PATH", "./artifacts/gptvec_model.pkl")
    vec_metrics_path: str = os.environ.get("VEC_METRICS_PATH", "./artifacts/gptvec_metrics.json")

    model_dim: int = int(os.environ.get("MODEL_DIM", 256))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 6))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 4))
    dropout: float = float(os.environ.get("DROPOUT", 0.0))
    seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", 128))
    batch_size: int = int(os.environ.get("BATCH_SIZE", 32))
    train_tokens_limit: int = int(os.environ.get("TRAIN_TOKENS_LIMIT", 2_000_000))
    iterations: int = int(os.environ.get("ITERATIONS", 3000))
    lr: float = float(os.environ.get("LR", 3e-4))
    weight_decay: float = float(os.environ.get("WEIGHT_DECAY", 0.01))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 1.0))
    log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 100))
    seed: int = int(os.environ.get("SEED", 1337))
    device: str = os.environ.get("DEVICE", "cuda")

    save_checkpoint: str = os.environ.get("SAVE_CHECKPOINT", "./artifacts/gptvec_checkpoint.pt")


class VecCausalTransformer(nn.Module):
    def __init__(self, vec_dim: int, model_dim: int, num_heads: int, num_layers: int, mlp_mult: int, dropout: float, max_seq_len: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.input_proj = nn.Linear(vec_dim, model_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, model_dim))

        ff_dim = model_dim * mlp_mult
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)
        self.output_proj = nn.Linear(model_dim, vec_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, vec_dim]
        b, t, _ = x.shape
        if t > self.max_seq_len:
            raise ValueError(f"Sequence length {t} exceeds max_seq_len={self.max_seq_len}")
        h = self.input_proj(x) + self.pos_emb[:, :t, :]
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.backbone(h, mask=mask)
        h = self.norm(h)
        return self.output_proj(h)


def parse_args() -> Hyperparameters:
    parser = argparse.ArgumentParser(description="Train GPT over token vector embeddings via train_vec delegate")
    defaults = Hyperparameters()
    for field_name, field_def in defaults.__dataclass_fields__.items():
        val = getattr(defaults, field_name)
        arg = f"--{field_name}"
        if isinstance(val, bool):
            parser.add_argument(arg, action="store_true", default=val)
        else:
            parser.add_argument(arg, type=type(val), default=val)
    ns = parser.parse_args()
    return Hyperparameters(**vars(ns))


def run_train_vec(args: Hyperparameters) -> None:
    os.makedirs(Path(args.vec_model_path).parent, exist_ok=True)
    os.makedirs(Path(args.vec_metrics_path).parent, exist_ok=True)
    cmd = [
        sys.executable,
        "train_vec.py",
        "--data_path",
        args.data_path,
        "--vocab_size",
        str(args.vocab_size),
        "--max_train_tokens",
        str(args.vec_max_train_tokens),
        "--val_tokens",
        str(args.vec_val_tokens),
        "--vec_dim",
        str(args.vec_dim),
        "--window",
        str(args.vec_window),
        "--metric_samples",
        str(args.vec_metric_samples),
        "--seed",
        str(args.seed),
        "--save_model",
        args.vec_model_path,
        "--save_metrics",
        args.vec_metrics_path,
    ]
    print("[gptvec] running delegate:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def sample_batch(tokens: torch.Tensor, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = tokens.numel() - seq_len - 1
    if max_start <= 0:
        raise RuntimeError("Token stream too short for configured seq_len")
    ix = torch.randint(0, max_start, (batch_size,), device=device)
    windows = torch.stack([tokens[i : i + seq_len + 1] for i in ix.tolist()], dim=0)
    return windows[:, :-1], windows[:, 1:]


@torch.no_grad()
def token_accuracy_from_vectors(pred_vec: torch.Tensor, target_tok: torch.Tensor, embedding_table: torch.Tensor) -> float:
    # pred_vec: [B,T,D], target_tok: [B,T], embedding_table: [V,D]
    p = F.normalize(pred_vec.reshape(-1, pred_vec.shape[-1]), dim=1)
    emb = F.normalize(embedding_table, dim=1)
    logits = p @ emb.t()
    pred_ids = logits.argmax(dim=1)
    acc = (pred_ids == target_tok.reshape(-1)).float().mean().item()
    return float(acc)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[gptvec] device={device}")

    t0 = time.time()
    run_train_vec(args)
    vec_model = VectorEmbeddingModel.load(args.vec_model_path)
    if vec_model.dim != args.vec_dim:
        raise RuntimeError(f"Vector dim mismatch: vec_model.dim={vec_model.dim}, expected={args.vec_dim}")

    files = find_token_files(args.data_path)
    if not files:
        raise SystemExit(f"No token files found under {args.data_path}")
    corpus = TokenCorpus(files)

    limit = min(corpus.total_tokens, args.train_tokens_limit)
    start = max(0, corpus.total_tokens - limit)
    train_tokens_np = corpus.read_range(start, corpus.total_tokens).astype(np.int64)
    train_tokens = torch.from_numpy(train_tokens_np).to(device)
    print(f"[gptvec] loaded train tokens={train_tokens.numel():,}")

    emb_table = torch.tensor(vec_model.embeddings, dtype=torch.float32, device=device)
    model = VecCausalTransformer(
        vec_dim=args.vec_dim,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_mult=args.mlp_mult,
        dropout=args.dropout,
        max_seq_len=args.seq_len,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    train_start = time.time()
    for step in range(1, args.iterations + 1):
        in_tok, tgt_tok = sample_batch(train_tokens, args.batch_size, args.seq_len, device=device)
        in_vec = emb_table[in_tok]
        tgt_vec = emb_table[tgt_tok]

        pred_vec = model(in_vec)
        loss = F.mse_loss(pred_vec, tgt_vec)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()

        if step % args.log_every == 0 or step == 1 or step == args.iterations:
            with torch.no_grad():
                acc = token_accuracy_from_vectors(pred_vec, tgt_tok, emb_table)
            print(f"[gptvec][step {step:6d}/{args.iterations}] loss={loss.item():.6f} token_top1={acc:.4f}")

    elapsed = time.time() - train_start
    total_elapsed = time.time() - t0

    os.makedirs(Path(args.save_checkpoint).parent, exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "args": vars(args),
        "vec_model_path": args.vec_model_path,
        "vec_metrics_path": args.vec_metrics_path,
        "train_seconds": elapsed,
        "total_seconds": total_elapsed,
    }
    torch.save(ckpt, args.save_checkpoint)

    summary = {
        "checkpoint": args.save_checkpoint,
        "vec_model": args.vec_model_path,
        "vec_metrics": args.vec_metrics_path,
        "train_tokens": int(train_tokens.numel()),
        "iterations": args.iterations,
        "train_seconds": float(elapsed),
        "total_seconds": float(total_elapsed),
        "device": str(device),
    }
    print("[gptvec] " + json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
