#!/usr/bin/env python3
"""Train a GPT-style language model with optional pretrained vec features.

Clean split:
- `train_vec.py` produces the vec artifact.
- `train_gptvec.py` consumes that artifact and trains a standard LM objective.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from train_vec import TokenCorpus, find_token_files
from vec_model import get_vec_table, load_vec_artifact


@dataclass
class Hyperparameters:
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))

    vec_path: str = os.environ.get("VEC_PATH", "./artifacts/gptvec_model.pkl")
    use_vec_input: bool = bool(int(os.environ.get("USE_VEC_INPUT", "1")))
    use_hybrid_embed: bool = bool(int(os.environ.get("USE_HYBRID_EMBED", "1")))
    freeze_vec: bool = bool(int(os.environ.get("FREEZE_VEC", "1")))
    vec_proj_dim: int = int(os.environ.get("VEC_PROJ_DIM", 256))
    aux_vec_loss_weight: float = float(os.environ.get("AUX_VEC_LOSS_WEIGHT", 0.0))

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


class GPTVecLM(nn.Module):
    def __init__(self, args: Hyperparameters, vec_table: torch.Tensor | None):
        super().__init__()
        self.args = args
        self.max_seq_len = args.seq_len

        self.token_embedding = nn.Embedding(args.vocab_size, args.model_dim)

        self.use_vec_input = bool(args.use_vec_input and vec_table is not None)
        self.use_hybrid_embed = bool(args.use_hybrid_embed)

        if vec_table is not None:
            if args.freeze_vec:
                self.register_buffer("vec_table", vec_table)
                self.vec_embedding = None
            else:
                self.vec_table = None
                self.vec_embedding = nn.Embedding.from_pretrained(vec_table, freeze=False)
            vec_dim = int(vec_table.shape[1])
            proj_in = args.vec_proj_dim if args.vec_proj_dim > 0 else vec_dim
            self.vec_preproj = nn.Identity() if proj_in == vec_dim else nn.Linear(vec_dim, proj_in)
            self.vec_proj = nn.Linear(proj_in, args.model_dim)
            self.vec_head = nn.Linear(args.model_dim, vec_dim) if args.aux_vec_loss_weight > 0 else None
        else:
            self.vec_table = None
            self.vec_embedding = None
            self.vec_preproj = None
            self.vec_proj = None
            self.vec_head = None

        self.pos_emb = nn.Parameter(torch.zeros(1, args.seq_len, args.model_dim))

        ff_dim = args.model_dim * args.mlp_mult
        enc_layer = nn.TransformerEncoderLayer(
            d_model=args.model_dim,
            nhead=args.num_heads,
            dim_feedforward=ff_dim,
            dropout=args.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(enc_layer, num_layers=args.num_layers)
        self.norm = nn.LayerNorm(args.model_dim)
        self.lm_head = nn.Linear(args.model_dim, args.vocab_size, bias=False)

    def lookup_vec(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.vec_embedding is not None:
            return self.vec_embedding(tokens)
        if self.vec_table is not None:
            return self.vec_table[tokens]
        raise RuntimeError("Vec lookup requested but no vec table is configured")

    def forward(self, x_tok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, t = x_tok.shape
        if t > self.max_seq_len:
            raise ValueError(f"Sequence length {t} exceeds max_seq_len={self.max_seq_len}")

        tok_h = self.token_embedding(x_tok)
        h = tok_h
        if self.use_vec_input:
            vec_h = self.vec_proj(self.vec_preproj(self.lookup_vec(x_tok)))
            h = h + vec_h if self.use_hybrid_embed else vec_h

        h = h + self.pos_emb[:, :t, :]
        mask = torch.triu(torch.ones(t, t, device=x_tok.device, dtype=torch.bool), diagonal=1)
        h = self.backbone(h, mask=mask)
        h = self.norm(h)
        logits = self.lm_head(h)
        vec_pred = self.vec_head(h) if self.vec_head is not None else None
        return logits, vec_pred


def parse_args() -> Hyperparameters:
    parser = argparse.ArgumentParser(description="Train GPT with standard CE loss and optional vec-side inputs")
    defaults = Hyperparameters()
    for field_name in defaults.__dataclass_fields__:
        val = getattr(defaults, field_name)
        arg = f"--{field_name}"
        if isinstance(val, bool):
            parser.add_argument(arg, type=int, choices=[0, 1], default=int(val))
        else:
            parser.add_argument(arg, type=type(val), default=val)
    ns = parser.parse_args()
    values = vars(ns)
    for k, v in list(values.items()):
        if isinstance(getattr(defaults, k), bool):
            values[k] = bool(v)
    return Hyperparameters(**values)


def sample_batch(tokens: torch.Tensor, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = tokens.numel() - seq_len - 1
    if max_start <= 0:
        raise RuntimeError("Token stream too short for configured seq_len")
    ix = torch.randint(0, max_start, (batch_size,), device=device)
    windows = torch.stack([tokens[i : i + seq_len + 1] for i in ix.tolist()], dim=0)
    return windows[:, :-1], windows[:, 1:]


@torch.no_grad()
def token_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return float((preds == targets).float().mean().item())


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[gptvec] device={device}")

    vec_table_t = None
    vec_dim = None
    if args.use_vec_input or args.aux_vec_loss_weight > 0:
        payload = load_vec_artifact(args.vec_path)
        vec_table_np = get_vec_table(payload)
        if vec_table_np.shape[0] != args.vocab_size:
            raise RuntimeError(
                f"Vec vocab mismatch: table has {vec_table_np.shape[0]}, args.vocab_size={args.vocab_size}"
            )
        vec_table_t = torch.tensor(vec_table_np, dtype=torch.float32, device=device)
        vec_dim = int(vec_table_t.shape[1])

    files = find_token_files(args.data_path)
    if not files:
        raise SystemExit(f"No token files found under {args.data_path}")
    corpus = TokenCorpus(files)

    limit = min(corpus.total_tokens, args.train_tokens_limit)
    start = max(0, corpus.total_tokens - limit)
    train_tokens_np = corpus.read_range(start, corpus.total_tokens).astype(np.int64)
    train_tokens = torch.from_numpy(train_tokens_np).to(device)
    print(f"[gptvec] loaded train tokens={train_tokens.numel():,}")

    model = GPTVecLM(args, vec_table_t).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    train_start = time.time()
    for step in range(1, args.iterations + 1):
        in_tok, tgt_tok = sample_batch(train_tokens, args.batch_size, args.seq_len, device=device)
        logits, vec_pred = model(in_tok)
        ce_loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), tgt_tok.reshape(-1))

        aux_loss = torch.zeros((), device=device)
        if args.aux_vec_loss_weight > 0:
            if vec_pred is None or vec_table_t is None:
                raise RuntimeError("Aux vec loss requested but vec head/table is unavailable")
            tgt_vec = vec_table_t[tgt_tok]
            aux_loss = F.mse_loss(vec_pred, tgt_vec)

        loss = ce_loss + args.aux_vec_loss_weight * aux_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()

        if step % args.log_every == 0 or step == 1 or step == args.iterations:
            with torch.no_grad():
                acc = token_accuracy(logits, tgt_tok)
            print(
                f"[gptvec][step {step:6d}/{args.iterations}] "
                f"loss={loss.item():.6f} ce={ce_loss.item():.6f} "
                f"aux={aux_loss.item():.6f} token_top1={acc:.4f}"
            )

    elapsed = time.time() - train_start

    os.makedirs(Path(args.save_checkpoint).parent, exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "args": vars(args),
        "vec_path": args.vec_path,
        "vec_dim": vec_dim,
        "train_seconds": elapsed,
    }
    torch.save(ckpt, args.save_checkpoint)

    summary = {
        "checkpoint": args.save_checkpoint,
        "vec_path": args.vec_path,
        "train_tokens": int(train_tokens.numel()),
        "iterations": args.iterations,
        "train_seconds": float(elapsed),
        "device": str(device),
    }
    print("[gptvec] " + json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
