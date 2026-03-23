#!/usr/bin/env python3
import argparse
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse eval + model from your existing script
from train_gpt import GPT, eval_val, Hyperparameters

# ---------------------------
# Utils
# ---------------------------

def svd_approx(W, rank):
    # fast approximate SVD (much faster than full SVD)
    try:
        U, S, V = torch.svd_lowrank(W, q=rank, niter=2)
        return (U * S) @ V.T
    except Exception:
        # fallback
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        return (U[:, :rank] * S[:rank]) @ Vh[:rank, :]


def estimate_int8_zlib_size(state_dict):
    import zlib
    total = 0
    for k, v in state_dict.items():
        if not v.is_floating_point():
            b = v.numpy().tobytes()
        else:
            t = v.float().cpu()
            if t.ndim == 2:
                scale = t.abs().max(dim=1, keepdim=True).values + 1e-8
                q = torch.clamp((t / scale).round(), -127, 127).to(torch.int8)
                b = q.numpy().tobytes()
            else:
                b = t.numpy().tobytes()
        total += len(zlib.compress(b))
    return total


def apply_svd_config(state_dict, mlp_rank, attn_rank):
    sd = {}
    for k, v in state_dict.items():
        if v.ndim == 2 and v.is_floating_point():
            if "mlp" in k:
                r = min(mlp_rank, min(v.shape))
                sd[k] = svd_approx(v.float(), r)
            elif any(x in k for x in ["attn.c_q", "attn.c_k", "attn.proj"]):
                r = min(attn_rank, min(v.shape))
                sd[k] = svd_approx(v.float(), r)
            else:
                sd[k] = v
        else:
            sd[k] = v
    return sd

# ---------------------------
# Full eval + size (reuse train_gpt logic)
# ---------------------------

from train_gpt import (
    quantize_state_dict_int8,
    dequantize_state_dict_int8,
    eval_val,
    build_sentencepiece_luts,
    load_validation_tokens,
    Hyperparameters,
)
import sentencepiece as spm
import zlib, io, glob
from pathlib import Path


def evaluate_model(state_dict):
    args = Hyperparameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model exactly like train_gpt
    model = GPT(
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

    model.load_state_dict(state_dict, strict=True)

    # tokenizer + validation setup
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    val_loss, val_bpb = eval_val(
        args,
        model,
        rank=0,
        world_size=1,
        device=device,
        grad_accum_steps=1,
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
    )

    return val_loss, val_bpb


def compute_size(state_dict):
    quant_obj, _ = quantize_state_dict_int8(state_dict)
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    raw = buf.getvalue()
    return len(zlib.compress(raw))


# ---------------------------
# Search
# ---------------------------

def run_search(input_path, max_combos=20):
    print("Loading checkpoint...")
    base_sd = torch.load(input_path, map_location="cpu")

    # Reconstruct model with same hyperparameters as checkpoint
    h = Hyperparameters()
    model = GPT(
        num_layers=h.num_layers,
        model_dim=h.model_dim,
        num_heads=h.num_heads,
        num_kv_heads=h.num_kv_heads,
        mlp_mult=h.mlp_mult,
        tie_embeddings=h.tie_embeddings,
        tied_embed_init_std=h.tied_embed_init_std,
        logit_softcap=h.logit_softcap,
        rope_base=h.rope_base,
        qk_gain_init=h.qk_gain_init,
    )

    mlp_ranks = [512, 384, 320, 256, 192]
    attn_ranks = [512, 384, 320, 256]

    results = []
    combos = []

    for m in mlp_ranks:
        for a in attn_ranks:
            combos.append((m, a))

    combos = combos[:max_combos]

    for mlp_r, attn_r in combos:
        print(f"\nTesting mlp_rank={mlp_r} attn_rank={attn_r}")
        t0 = time.time()

        sd = apply_svd_config(base_sd, mlp_r, attn_r)

        size_bytes = compute_size(sd)

        val_loss, val_bpb = evaluate_model(sd)

        dt = time.time() - t0

        print(f"  size={size_bytes/1e6:.2f}MB val_bpb={val_bpb:.4f} time={dt:.2f}s")

        results.append({
            "mlp_rank": mlp_r,
            "attn_rank": attn_r,
            "size": size_bytes,
            "val_bpb": val_bpb
        })

    print("\n=== BEST UNDER 16MB ===")
    best = None
    for r in results:
        if r["size"] < 16_000_000:
            if best is None or r["val_bpb"] < best["val_bpb"]:
                best = r

    if best:
        print(best)
    else:
        print("No config under 16MB")

# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--max-combos", type=int, default=20)
    args = parser.parse_args()

    run_search(args.input, args.max_combos)
