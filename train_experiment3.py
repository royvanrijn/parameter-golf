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
    # W: [m, n]
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_r = U[:, :rank]
    S_r = S[:rank]
    V_r = Vh[:rank, :]
    return (U_r * S_r) @ V_r


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
# Search
# ---------------------------

def run_search(input_path, max_combos=20):
    print("Loading checkpoint...")
    base_sd = torch.load(input_path, map_location="cpu")

    args = Hyperparameters()
    model = GPT(args)

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

        size_bytes = estimate_int8_zlib_size(sd)

        model.load_state_dict(sd, strict=False)
        val_loss, val_bpb = eval_val(model)

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
