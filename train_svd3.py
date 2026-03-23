"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import math
import os
import pickle
import random
import subprocess
import sys
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from svdgpt_artifact import (
    dequantize_state_dict_int8 as dequantize_state_dict_int8_np,
    export_np_dtype,
    quantize_state_dict_int8 as quantize_state_dict_int8_np,
    svd_rank_for_name,
)

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    svd_phase1_start = int(os.environ.get("SVD_PHASE1_START", 0))
    svd_phase1_every = int(os.environ.get("SVD_PHASE1_EVERY", 50))
    svd_phase1_mix_start = float(os.environ.get("SVD_PHASE1_MIX_START", 0.01))
    svd_phase1_mix_end = float(os.environ.get("SVD_PHASE1_MIX_END", 0.5))
    # Defaults to legacy SVD_PHASE2_START for backwards compatibility with the svd2-style launch command.
    svd_transition_step = int(os.environ.get("SVD_TRANSITION_STEP", os.environ.get("SVD_PHASE2_START", 4000)))
    svd_transition_from_step0 = bool(int(os.environ.get("SVD_TRANSITION_FROM_STEP0", "0")))
    svd_balance_factors = bool(int(os.environ.get("SVD_BALANCE_FACTORS", "1")))
    svd_rank_fc = int(os.environ.get("SVD_RANK_FC", 64))
    svd_rank_proj = int(os.environ.get("SVD_RANK_PROJ", 64))
    svd_rank_attn_proj = int(os.environ.get("SVD_RANK_ATTN_PROJ", 64))
    svd_use_attn_proj = bool(int(os.environ.get("SVD_USE_ATTN_PROJ", "1")))
    # Also SVD-project c_q/c_k/c_v during training and export.
    # This frees artifact bytes (they become low-rank factors instead of full int8 matrices)
    # which can be reinvested in higher MLP ranks.
    svd_use_qkv = bool(int(os.environ.get("SVD_USE_QKV", "1")))
    svd_rank_qk = int(os.environ.get("SVD_RANK_QK", 128))
    svd_rank_v = int(os.environ.get("SVD_RANK_V", 128))
    svd_export_float_dtype = os.environ.get("SVD_EXPORT_FLOAT_DTYPE", "float16")
    svd_method = os.environ.get("SVD_METHOD", "randomized").lower()
    svd_power_iters = int(os.environ.get("SVD_POWER_ITERS", 1))
    # Use separate iters for transition SVD (often can be cheaper than phase-1 refresh).
    svd_power_iters_phase2 = int(os.environ.get("SVD_POWER_ITERS_PHASE2", 0))
    svd_oversample = int(os.environ.get("SVD_OVERSAMPLE", 8))
    svd_layer_stride = int(os.environ.get("SVD_LAYER_STRIDE", 1))

# -----------------------------
# MUON OPTIMIZER
# -----------------------------
#
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


def low_rank_approx(arr: Tensor, rank: int, args: Hyperparameters, niter: int | None = None) -> Tensor:
    if args.svd_method == "full":
        u, s, vh = torch.linalg.svd(arr, full_matrices=False)
        return (u[:, :rank] * s[:rank]) @ vh[:rank, :]

    power = int(args.svd_power_iters) if niter is None else niter
    q = min(arr.shape[0], arr.shape[1], rank + max(1, args.svd_oversample))
    q = max(rank, q)
    u, s, v = torch.svd_lowrank(arr, q=q, niter=max(0, power))
    return (u[:, :rank] * s[:rank]) @ v[:, :rank].T


def truncated_svd_weight(weight: Tensor, rank: int, args: Hyperparameters, niter: int | None = None) -> tuple[Tensor, float]:
    out_dim, in_dim = weight.shape
    max_rank = min(out_dim, in_dim)
    rank = max(1, min(int(rank), max_rank))
    if rank >= max_rank:
        return weight.detach().clone(), 0.0
    arr = weight.detach().float()
    low_rank = low_rank_approx(arr, rank, args, niter=niter)
    residual = float((arr - low_rank).norm() / (arr.norm() + 1e-8))
    return low_rank.to(dtype=weight.dtype), residual


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & zlib compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)

def quantize_state_dict_int8(state_dict: dict[str, Tensor], args: Hyperparameters):
    np_state = {
        name: np.ascontiguousarray(
            (t.detach().to(dtype=torch.float32) if t.dtype == torch.bfloat16 else t.detach())
            .to("cpu")
            .numpy()
        )
        for name, t in state_dict.items()
    }
    return quantize_state_dict_int8_np(
        np_state,
        keep_float_fp32_name_patterns=INT8_KEEP_FLOAT_FP32_NAME_PATTERNS,
        svd_rank_lookup=lambda name: svd_rank_for_name(name, args),
        svd_export_dtype=export_np_dtype(args.svd_export_float_dtype),
    )

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    np_state = dequantize_state_dict_int8_np(obj)
    out: dict[str, Tensor] = {}
    for name, arr in np_state.items():
        out[name] = torch.from_numpy(np.ascontiguousarray(arr)).to("cpu")
    return out


# -----------------------------
# DATA LOADING
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class FactorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(1, min(rank, in_features, out_features))
        self.a = nn.Parameter(torch.empty(out_features, self.rank))
        self.b = nn.Parameter(torch.empty(self.rank, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.b, a=math.sqrt(5))

    def set_from_dense_weight(self, weight: Tensor, args: Hyperparameters, niter: int | None = None, balanced: bool = True) -> float:
        arr = weight.detach().float()
        max_rank = min(arr.shape[0], arr.shape[1], self.rank)
        low_rank, residual = truncated_svd_weight(arr, max_rank, args, niter=niter)
        if args.svd_method == "full":
            u, s, vh = torch.linalg.svd(low_rank, full_matrices=False)
            u, s, vh = u[:, :max_rank], s[:max_rank], vh[:max_rank, :]
        else:
            q = min(low_rank.shape[0], low_rank.shape[1], max_rank + max(1, args.svd_oversample))
            q = max(max_rank, q)
            u, s, v = torch.svd_lowrank(low_rank, q=q, niter=max(0, int(args.svd_power_iters_phase2)))
            vh = v[:, :max_rank].T
            u, s = u[:, :max_rank], s[:max_rank]
        if balanced:
            sroot = torch.sqrt(torch.clamp(s, min=1e-12))
            a = u * sroot
            b = sroot[:, None] * vh
        else:
            a = u * s
            b = vh
        self.a.data.copy_(a.to(self.a.dtype).to(self.a.device))
        self.b.data.copy_(b.to(self.b.dtype).to(self.b.device))
        return residual

    def forward(self, x: Tensor) -> Tensor:
        y = F.linear(x, self.b.to(x.dtype), None)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(y, self.a.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )

        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = torch.relu(self.fc(x))
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []

        # First half stores skips; second half reuses them in reverse order.
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)

        x = self.final_norm(x).reshape(-1, x.size(-1))
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")


def iter_named_svd_linears(
    model: GPT,
    args: Hyperparameters,
    step_1idx: int,
    use_stride: bool = True,
) -> list[tuple[str, CastedLinear, int]]:
    layer_stride = max(1, int(args.svd_layer_stride))
    period = max(1, int(args.svd_phase1_every))
    anchor = int(args.svd_phase1_start)
    period_idx = max(step_1idx - anchor, 0) // period
    layer_offset = period_idx % layer_stride
    out: list[tuple[str, CastedLinear, int]] = []
    for layer_idx, block in enumerate(model.blocks):
        if use_stride and layer_stride > 1 and layer_idx % layer_stride != layer_offset:
            continue
        out.append((f"blocks.{layer_idx}.mlp.fc", block.mlp.fc, args.svd_rank_fc))
        out.append((f"blocks.{layer_idx}.mlp.proj", block.mlp.proj, args.svd_rank_proj))
        if args.svd_use_attn_proj:
            out.append((f"blocks.{layer_idx}.attn.proj", block.attn.proj, args.svd_rank_attn_proj))
        if args.svd_use_qkv:
            out.append((f"blocks.{layer_idx}.attn.c_q", block.attn.c_q, args.svd_rank_qk))
            out.append((f"blocks.{layer_idx}.attn.c_k", block.attn.c_k, args.svd_rank_qk))
            out.append((f"blocks.{layer_idx}.attn.c_v", block.attn.c_v, args.svd_rank_v))
    return out


def get_module_by_name(root: nn.Module, name: str) -> nn.Module:
    module: nn.Module = root
    for part in name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def set_module_by_name(root: nn.Module, name: str, module: nn.Module) -> None:
    parts = name.split(".")
    parent = get_module_by_name(root, ".".join(parts[:-1])) if len(parts) > 1 else root
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = module
    else:
        setattr(parent, leaf, module)


@dataclass
class SVD3AuxStats:
    num_layers: int
    avg_residual: float
    aux_loss: Tensor


class SVD3AuxManager:
    def __init__(self, model: GPT, args: Hyperparameters):
        self.model = model
        self.args = args
        self.cached_wr: dict[str, Tensor] = {}
        self.cached_residual: dict[str, float] = {}
        self.active = False
        self.layer_names: set[str] = set()
        self.aux_loss = torch.zeros((), device=next(model.parameters()).device)
        self.handles = []
        for name, module in model.named_modules():
            if isinstance(module, CastedLinear):
                self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple[Tensor, ...], output: Tensor):
            if not self.active or name not in self.layer_names:
                return
            wr = self.cached_wr.get(name)
            if wr is None:
                return
            x = inputs[0]
            dense = output
            proj = F.linear(x, wr.to(x.dtype), None)
            self.aux_loss = self.aux_loss + F.mse_loss(dense, proj, reduction="mean")
        return hook

    @torch.no_grad()
    def refresh_targets(self, step_1idx: int) -> SVD3AuxStats:
        selected = iter_named_svd_linears(self.model, self.args, step_1idx, use_stride=True)
        self.layer_names = {name for name, _, _ in selected}
        residual_sum = 0.0
        for name, module, rank in selected:
            wr, residual = truncated_svd_weight(module.weight, rank, self.args, niter=self.args.svd_power_iters)
            self.cached_wr[name] = wr.detach()
            self.cached_residual[name] = residual
            residual_sum += residual
        avg_residual = residual_sum / max(len(selected), 1)
        return SVD3AuxStats(num_layers=len(selected), avg_residual=avg_residual, aux_loss=self.aux_loss.detach())

    def begin_step(self, active: bool) -> None:
        self.active = active
        self.aux_loss = torch.zeros((), device=next(self.model.parameters()).device)


def svd3_phase1_mix(step_1idx: int, args: Hyperparameters, transition_step: int) -> float:
    if step_1idx < args.svd_phase1_start:
        return 0.0
    denom = max(transition_step - args.svd_phase1_start, 1)
    progress = min(max((step_1idx - args.svd_phase1_start) / denom, 0.0), 1.0)
    mix = args.svd_phase1_mix_start + progress * (args.svd_phase1_mix_end - args.svd_phase1_mix_start)
    return max(0.0, min(float(mix), 1.0))


def svd3_is_phase1_refresh_step(step_1idx: int, args: Hyperparameters) -> bool:
    return (
        step_1idx >= args.svd_phase1_start
        and args.svd_phase1_every > 0
        and (step_1idx - args.svd_phase1_start) % args.svd_phase1_every == 0
    )


@torch.no_grad()
def transition_to_factorized(model: GPT, args: Hyperparameters, step_1idx: int) -> tuple[int, float]:
    converted = 0
    residual_sum = 0.0
    for name, dense, rank in iter_named_svd_linears(model, args, step_1idx, use_stride=False):
        fact = FactorizedLinear(
            in_features=dense.in_features,
            out_features=dense.out_features,
            rank=rank,
            bias=dense.bias is not None,
        ).to(device=dense.weight.device, dtype=torch.float32)
        residual = fact.set_from_dense_weight(
            dense.weight,
            args=args,
            niter=args.svd_power_iters_phase2,
            balanced=args.svd_balance_factors,
        )
        if dense.bias is not None and fact.bias is not None:
            fact.bias.data.copy_(dense.bias.detach().float())
        set_module_by_name(model, name, fact)
        converted += 1
        residual_sum += residual
    return converted, residual_sum / max(converted, 1)


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    model: nn.Module = DDP(base_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else base_model

    def build_optimizers() -> tuple[list[torch.optim.Optimizer], Muon]:
        block_named_params = list(base_model.blocks.named_parameters())
        matrix_params = [
            p
            for name, p in block_named_params
            if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        scalar_params = [
            p
            for name, p in block_named_params
            if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        if base_model.skip_weights.numel() > 0:
            scalar_params.append(base_model.skip_weights)
        token_lr_local = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
        optimizer_tok_local = torch.optim.Adam(
            [{"params": [base_model.tok_emb.weight], "lr": token_lr_local, "base_lr": token_lr_local}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizer_muon_local = Muon(
            matrix_params,
            lr=args.matrix_lr,
            momentum=args.muon_momentum,
            backend_steps=args.muon_backend_steps,
        )
        for group in optimizer_muon_local.param_groups:
            group["base_lr"] = args.matrix_lr
        optimizer_scalar_local = torch.optim.Adam(
            [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        opts: list[torch.optim.Optimizer] = [optimizer_tok_local, optimizer_muon_local, optimizer_scalar_local]
        if base_model.lm_head is not None:
            optimizer_head = torch.optim.Adam(
                [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
                betas=(args.beta1, args.beta2),
                eps=args.adam_eps,
                fused=True,
            )
            opts.insert(1, optimizer_head)
        return opts, optimizer_muon_local

    optimizers, optimizer_muon = build_optimizers()
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(
        f"svd3_training: phase1_start={args.svd_phase1_start} transition_step={args.svd_transition_step} "
        f"transition_from_step0={args.svd_transition_from_step0} "
        f"phase1_every={args.svd_phase1_every} "
        f"phase1_mix_start={args.svd_phase1_mix_start:.3f} phase1_mix_end={args.svd_phase1_mix_end:.3f} "
        f"balance_factors={args.svd_balance_factors} "
        f"rank_fc={args.svd_rank_fc} rank_proj={args.svd_rank_proj} "
        f"rank_attn_proj={args.svd_rank_attn_proj} use_attn_proj={args.svd_use_attn_proj} "
        f"use_qkv={args.svd_use_qkv} rank_qk={args.svd_rank_qk} rank_v={args.svd_rank_v} "
        f"method={args.svd_method} power_iters={args.svd_power_iters} power_iters_phase2={args.svd_power_iters_phase2} "
        f"oversample={args.svd_oversample} "
        f"layer_stride={args.svd_layer_stride} "
    )
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
    svd_aux = SVD3AuxManager(base_model, args)
    transition_step = 0 if args.svd_transition_from_step0 else max(int(args.svd_transition_step), 1)
    transitioned = False

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        # Use stable_step_ms if captured; otherwise fall back to running average.
        # This prevents growing SVD overhead from causing the warmdown to fire early.
        eff_step_ms = stable_step_ms if stable_step_ms is not None else elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * eff_step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------
    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Captured once after the model has warmed up (SVD not yet running) so the
    # warmdown schedule isn't thrown off by the growing per-step SVD overhead.
    stable_step_ms: float | None = None

    first_svd_start = min([s for s in [args.svd_phase1_start, transition_step] if s > 0] or [300])
    STABLE_STEP_CAPTURE = max(100, min(300, first_svd_start - 20))

    step = 0
    if transition_step == 0 and not transitioned:
        converted, avg_residual = transition_to_factorized(base_model, args, step_1idx=0)
        optimizers, optimizer_muon = build_optimizers()
        transitioned = True
        log0(f"svd3_transition step:0 converted_layers:{converted} avg_residual:{avg_residual:.4f}")
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        step_1idx = step + 1
        svd_mix = svd3_phase1_mix(step_1idx, args, transition_step=transition_step)
        aux_active = (not transitioned) and step_1idx >= args.svd_phase1_start
        if aux_active and svd3_is_phase1_refresh_step(step_1idx, args):
            svd_t0 = time.perf_counter()
            stats = svd_aux.refresh_targets(step_1idx)
            log0(
                f"svd3_refresh step:{step_1idx} layers:{stats.num_layers} "
                f"avg_residual:{stats.avg_residual:.4f} lambda:{svd_mix:.4f} "
                f"took_ms:{1000.0 * (time.perf_counter() - svd_t0):.1f}",
                console=False,
            )

        if (not transitioned) and step_1idx >= transition_step:
            trans_t0 = time.perf_counter()
            converted, avg_residual = transition_to_factorized(base_model, args, step_1idx)
            optimizers, optimizer_muon = build_optimizers()
            zero_grad_all()
            transitioned = True
            svd_aux.begin_step(active=False)
            log0(
                f"svd3_transition step:{step_1idx} converted_layers:{converted} "
                f"avg_residual:{avg_residual:.4f} "
                f"took_ms:{1000.0 * (time.perf_counter() - trans_t0):.1f}",
            )

        svd_aux.begin_step(active=aux_active and not transitioned)
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            aux_prev = svd_aux.aux_loss
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            aux_delta = svd_aux.aux_loss - aux_prev if svd_aux.active else torch.zeros_like(loss)
            total_loss = loss + svd_mix * aux_delta if svd_aux.active else loss
            (total_loss * grad_scale).backward()
        train_loss /= grad_accum_steps
        aux_loss_scalar = float((svd_aux.aux_loss / grad_accum_steps).detach().item()) if svd_aux.active else 0.0

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)

        # Capture a stable per-step time estimate before phase-2 SVD overhead starts.
        if stable_step_ms is None and step == STABLE_STEP_CAPTURE:
            stable_step_ms = approx_training_time_ms / step
            log0(f"stable_step_ms captured: {stable_step_ms:.2f}ms at step {step}", console=False)

        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"aux_loss:{aux_loss_scalar:.5f} svd_lambda:{svd_mix:.4f} "
                f"transitioned:{int(transitioned)} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed int8+zlib artifact and validate the round-tripped weights.

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict(), args)
    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_serialized_bytes = len(quant_raw)
    if master_process:
        with open("final_model.int8_svd.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8_svd.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"serialized_model_int8_svd_zlib:{quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} svd_factor_bytes:{quant_stats['svd_factor_bytes']} "
            f"raw_pickle:{quant_serialized_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8_svd+zlib: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8_svd.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_state = pickle.loads(zlib.decompress(quant_blob_disk))
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_svd_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_svd_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
