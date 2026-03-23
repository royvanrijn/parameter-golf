#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import io
import time
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import sentencepiece as spm
import torch
from torch import Tensor

from train_gpt import (
    GPT,
    CastedLinear,
    Hyperparameters,
    build_sentencepiece_luts,
    eval_val,
    load_validation_tokens,
    quantize_state_dict_int8,
    dequantize_state_dict_int8,
    restore_low_dim_params_to_fp32,
)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

BIG_FLOAT_TENSOR_MAX_NUMEL_KEEP = 65_536
CLIP_PERCENTILE = 99.99984
CLIP_Q = CLIP_PERCENTILE / 100.0
DEFAULT_FAMILY_ORDER = [
    "mlp.fc",
    "mlp.proj",
    "attn.c_q",
    "attn.c_k",
    "attn.proj",
]


# -----------------------------------------------------------------------------
# Family helpers
# -----------------------------------------------------------------------------

def family_of(name: str) -> str | None:
    if "mlp.fc.weight" in name:
        return "mlp.fc"
    if "mlp.proj.weight" in name:
        return "mlp.proj"
    if "attn.c_q.weight" in name:
        return "attn.c_q"
    if "attn.c_k.weight" in name:
        return "attn.c_k"
    if "attn.proj.weight" in name:
        return "attn.proj"
    return None


def is_large_2d_float_tensor(name: str, t: Tensor) -> bool:
    return bool(t.is_floating_point() and t.ndim == 2 and t.numel() > BIG_FLOAT_TENSOR_MAX_NUMEL_KEEP)


def orig_dtype_name(t: Tensor) -> str:
    return str(t.dtype).removeprefix("torch.")


# -----------------------------------------------------------------------------
# Bit packing helpers
# -----------------------------------------------------------------------------

def signed_to_unsigned(q: np.ndarray, bits: int) -> np.ndarray:
    qmax = (1 << (bits - 1)) - 1
    return (q.astype(np.int16) + qmax).astype(np.uint16)


def unsigned_to_signed(u: np.ndarray, bits: int) -> np.ndarray:
    qmax = (1 << (bits - 1)) - 1
    return (u.astype(np.int16) - qmax).astype(np.int16)


def pack_values(values: np.ndarray, bits: int) -> bytes:
    values = np.asarray(values, dtype=np.uint16).reshape(-1)
    if bits == 8:
        return values.astype(np.uint8, copy=False).tobytes()
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint16)
    bitplanes = ((values[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    return np.packbits(bitplanes.reshape(-1), bitorder="big").tobytes()


def unpack_values(payload: bytes, count: int, bits: int) -> np.ndarray:
    if bits == 8:
        return np.frombuffer(payload, dtype=np.uint8, count=count).astype(np.uint16)
    packed = np.frombuffer(payload, dtype=np.uint8)
    bits_arr = np.unpackbits(packed, bitorder="big")
    bits_arr = bits_arr[: count * bits].reshape(count, bits)
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint16)
    return (bits_arr.astype(np.uint16) << shifts[None, :]).sum(axis=1, dtype=np.uint16)


# -----------------------------------------------------------------------------
# Custom low-bit codec for selected matrices only
# -----------------------------------------------------------------------------

def quantize_matrix_per_row_bits(t: Tensor, bits: int) -> tuple[np.ndarray, np.ndarray, Tensor, Tensor]:
    assert t.ndim == 2
    maxq = (1 << (bits - 1)) - 1
    t32 = t.float().cpu().contiguous()
    clip_abs = torch.quantile(t32.abs(), CLIP_Q, dim=1)
    clip_abs = torch.clamp(clip_abs, min=1.0 / maxq)
    clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
    q = torch.clamp(torch.round(clipped / clip_abs[:, None] * maxq), -maxq, maxq).to(torch.int16)
    scales = (clip_abs / maxq).to(torch.float16)
    recon = q.float() * scales.float()[:, None]
    row_mse = ((recon - t32) ** 2).mean(dim=1)
    return q.numpy(), scales.numpy(), recon, row_mse


def encode_large_2d_tensor(name: str, t: Tensor, bits: int, outlier_frac: float) -> dict[str, Any]:
    q_np, scales_np, _recon, row_mse = quantize_matrix_per_row_bits(t, bits)
    rows, cols = q_np.shape
    entry: dict[str, Any] = {
        "kind": "q2d_packed",
        "shape": [rows, cols],
        "orig_dtype": orig_dtype_name(t),
        "bits": bits,
        "scales": torch.from_numpy(scales_np.copy()),
        "payload": pack_values(signed_to_unsigned(q_np, bits).reshape(-1), bits),
        "outlier_frac": 0.0,
    }

    if bits < 8 and outlier_frac > 0.0 and rows > 0:
        outlier_count = min(rows, max(1, int(round(rows * outlier_frac))))
        topk = torch.topk(row_mse, k=outlier_count, largest=True).indices.cpu().numpy().astype(np.int32)
        q8_np, s8_np, _recon8, _ = quantize_matrix_per_row_bits(t[topk.tolist(), :], 8)
        entry["outlier_rows"] = torch.from_numpy(topk.copy())
        entry["outlier_scales"] = torch.from_numpy(s8_np.copy())
        entry["outlier_bits"] = 8
        entry["outlier_payload"] = pack_values(signed_to_unsigned(q8_np, 8).reshape(-1), 8)
        entry["outlier_frac"] = float(outlier_frac)

    return entry


def decode_large_2d_tensor(entry: dict[str, Any]) -> Tensor:
    rows, cols = entry["shape"]
    bits = int(entry["bits"])
    dtype = getattr(torch, entry["orig_dtype"])
    u = unpack_values(entry["payload"], rows * cols, bits)
    q = unsigned_to_signed(u, bits).reshape(rows, cols)
    scales = entry["scales"].float().cpu().numpy().reshape(rows, 1)
    out = torch.from_numpy((q.astype(np.float32) * scales).astype(np.float32)).to(dtype=dtype)

    if "outlier_rows" in entry:
        idx = entry["outlier_rows"].cpu().numpy().astype(np.int64)
        obits = int(entry["outlier_bits"])
        ou = unpack_values(entry["outlier_payload"], idx.size * cols, obits)
        oq = unsigned_to_signed(ou, obits).reshape(idx.size, cols)
        oscales = entry["outlier_scales"].float().cpu().numpy().reshape(idx.size, 1)
        orec = torch.from_numpy((oq.astype(np.float32) * oscales).astype(np.float32)).to(dtype=dtype)
        out[idx.tolist(), :] = orec

    return out.contiguous()


# -----------------------------------------------------------------------------
# Hybrid artifact: exact train_gpt int8 baseline + custom overrides for <8 bit
# -----------------------------------------------------------------------------

def clone_quant_obj(obj: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in obj.items():
        if isinstance(v, dict):
            out[k] = dict(v)
        else:
            out[k] = v
    return out


def build_hybrid_artifact(state_dict: dict[str, Tensor], policy: dict[str, dict[str, float | int]]) -> dict[str, Any]:
    base_obj, _stats = quantize_state_dict_int8(state_dict)
    obj = clone_quant_obj(base_obj)
    mixed_bits: dict[str, Any] = {}

    for name, tensor in state_dict.items():
        fam = family_of(name)
        t = tensor.detach().cpu().contiguous()
        if not (fam in policy and is_large_2d_float_tensor(name, t)):
            continue

        bits = int(policy[fam]["bits"])
        outlier_frac = float(policy[fam]["outlier_frac"])
        if bits >= 8:
            continue  # keep exact train_gpt baseline path

        mixed_bits[name] = encode_large_2d_tensor(name, t, bits, outlier_frac)

        # Remove this tensor from the train_gpt baseline object so only custom path remains.
        if "quantized" in obj:
            obj["quantized"].pop(name, None)
        if "scales" in obj:
            obj["scales"].pop(name, None)
        if "dtypes" in obj:
            obj["dtypes"].pop(name, None)
        if "passthrough" in obj:
            obj["passthrough"].pop(name, None)
        if "qmeta" in obj:
            obj["qmeta"].pop(name, None)
        if "passthrough_orig_dtypes" in obj:
            obj["passthrough_orig_dtypes"].pop(name, None)

    if mixed_bits:
        obj["mixed_bits"] = mixed_bits
    return obj


def decode_hybrid_artifact(obj: dict[str, Any]) -> dict[str, Tensor]:
    base_only = {k: v for k, v in obj.items() if k != "mixed_bits"}
    sd = dequantize_state_dict_int8(base_only)
    for name, entry in obj.get("mixed_bits", {}).items():
        sd[name] = decode_large_2d_tensor(entry)
    return sd


def compute_artifact_size_bytes(obj: dict[str, Any]) -> int:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return len(zlib.compress(buf.getvalue(), level=9))


# -----------------------------------------------------------------------------
# Policy helpers
# -----------------------------------------------------------------------------

def clone_policy(policy: dict[str, dict[str, float | int]]) -> dict[str, dict[str, float | int]]:
    return {k: {"bits": int(v["bits"]), "outlier_frac": float(v["outlier_frac"])} for k, v in policy.items()}


def policy_key(policy: dict[str, dict[str, float | int]]) -> tuple[tuple[str, int, float], ...]:
    return tuple(sorted((k, int(v["bits"]), float(v["outlier_frac"])) for k, v in policy.items()))


def format_mb(n: int) -> str:
    return f"{n / 1_000_000:.2f}MB"


def initial_policy(base_sd: dict[str, Tensor]) -> dict[str, dict[str, float | int]]:
    present = {family_of(name) for name, t in base_sd.items() if family_of(name) and is_large_2d_float_tensor(name, t)}
    return {fam: {"bits": 8, "outlier_frac": 0.0} for fam in DEFAULT_FAMILY_ORDER if fam in present}


def compress_candidates(policy: dict[str, dict[str, float | int]]) -> list[tuple[str, dict[str, dict[str, float | int]]]]:
    out: list[tuple[str, dict[str, dict[str, float | int]]]] = []
    for fam, cfg in policy.items():
        bits = int(cfg["bits"])
        if bits > 6:
            nxt = clone_policy(policy)
            nxt[fam]["bits"] = bits - 1
            nxt[fam]["outlier_frac"] = 0.0
            out.append((f"compress {fam} {bits}->{bits-1}", nxt))
    return out


def refine_candidates(policy: dict[str, dict[str, float | int]], outlier_fracs: list[float]) -> list[tuple[str, dict[str, dict[str, float | int]]]]:
    out: list[tuple[str, dict[str, dict[str, float | int]]]] = []
    for fam, cfg in policy.items():
        bits = int(cfg["bits"])
        frac = float(cfg["outlier_frac"])

        if bits < 8:
            nxt = clone_policy(policy)
            nxt[fam]["bits"] = bits + 1
            nxt[fam]["outlier_frac"] = 0.0
            out.append((f"upgrade {fam} {bits}->{bits+1}", nxt))

        if bits < 8:
            bigger = [x for x in outlier_fracs if x > frac]
            if bigger:
                nxt = clone_policy(policy)
                nxt[fam]["outlier_frac"] = min(bigger)
                out.append((f"outliers {fam} {frac:.3f}->{min(bigger):.3f}", nxt))
    return out


# -----------------------------------------------------------------------------
# Evaluation using train_gpt path
# -----------------------------------------------------------------------------

@dataclass
class EvalResult:
    policy: dict[str, dict[str, float | int]]
    size_bytes: int
    val_loss: float
    val_bpb: float
    elapsed_sec: float
    artifact: dict[str, Any] | None = None
    decoded_sd: dict[str, Tensor] | None = None


class Evaluator:
    def __init__(self):
        self.args = Hyperparameters()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grad_accum_steps = 8  # match world_size=1 setup from train_gpt

        self.sp = spm.SentencePieceProcessor(model_file=self.args.tokenizer_path)
        self.val_tokens = load_validation_tokens(self.args.val_files, self.args.train_seq_len)
        self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = build_sentencepiece_luts(
            self.sp, self.args.vocab_size, self.device
        )

        self.model = GPT(
            vocab_size=self.args.vocab_size,
            num_layers=self.args.num_layers,
            model_dim=self.args.model_dim,
            num_heads=self.args.num_heads,
            num_kv_heads=self.args.num_kv_heads,
            mlp_mult=self.args.mlp_mult,
            tie_embeddings=self.args.tie_embeddings,
            tied_embed_init_std=self.args.tied_embed_init_std,
            logit_softcap=self.args.logit_softcap,
            rope_base=self.args.rope_base,
            qk_gain_init=self.args.qk_gain_init,
        ).to(self.device).bfloat16()
        for module in self.model.modules():
            if isinstance(module, CastedLinear):
                module.float()
        restore_low_dim_params_to_fp32(self.model)

    def evaluate_policy(self, base_sd: dict[str, Tensor], policy: dict[str, dict[str, float | int]], keep_artifact: bool = False) -> EvalResult:
        t0 = time.perf_counter()
        artifact = build_hybrid_artifact(base_sd, policy)
        decoded_sd = decode_hybrid_artifact(artifact)
        size_bytes = compute_artifact_size_bytes(artifact)

        self.model.load_state_dict(decoded_sd, strict=True)
        val_loss, val_bpb = eval_val(
            self.args,
            self.model,
            rank=0,
            world_size=1,
            device=self.device,
            grad_accum_steps=self.grad_accum_steps,
            val_tokens=self.val_tokens,
            base_bytes_lut=self.base_bytes_lut,
            has_leading_space_lut=self.has_leading_space_lut,
            is_boundary_token_lut=self.is_boundary_token_lut,
        )
        elapsed = time.perf_counter() - t0
        return EvalResult(
            policy=clone_policy(policy),
            size_bytes=size_bytes,
            val_loss=float(val_loss),
            val_bpb=float(val_bpb),
            elapsed_sec=elapsed,
            artifact=artifact if keep_artifact else None,
            decoded_sd=decoded_sd if keep_artifact else None,
        )


# -----------------------------------------------------------------------------
# Search
# -----------------------------------------------------------------------------

def print_result(prefix: str, r: EvalResult) -> None:
    print(
        f"{prefix} size={format_mb(r.size_bytes)} val_loss={r.val_loss:.4f} "
        f"val_bpb={r.val_bpb:.4f} time={r.elapsed_sec:.2f}s policy={r.policy}"
    )


def run_search(input_path: str, target_bytes: int, max_compress_steps: int, max_refine_steps: int, outlier_fracs: list[float], write_best: str | None):
    print("Loading checkpoint...")
    base_sd = torch.load(input_path, map_location="cpu")
    evaluator = Evaluator()
    cache: dict[tuple[tuple[str, int, float], ...], EvalResult] = {}

    def evaluate(policy: dict[str, dict[str, float | int]], keep_artifact: bool = False) -> EvalResult:
        key = policy_key(policy)
        if key in cache and not keep_artifact:
            return cache[key]
        r = evaluator.evaluate_policy(base_sd, policy, keep_artifact=keep_artifact)
        if not keep_artifact:
            cache[key] = r
        return r

    current = initial_policy(base_sd)
    baseline = evaluate(current)
    print_result("BASELINE", baseline)
    best_under = baseline if baseline.size_bytes <= target_bytes else None

    for step in range(max_compress_steps):
        if baseline.size_bytes <= target_bytes:
            break
        candidates = compress_candidates(current)
        if not candidates:
            break

        best_move_res: EvalResult | None = None
        best_move_desc: str | None = None
        best_score: float | None = None
        base_size = baseline.size_bytes
        base_bpb = baseline.val_bpb

        print(f"\n[compress step {step+1}] evaluating {len(candidates)} candidates...")
        for desc, cand in candidates:
            r = evaluate(cand)
            print_result(f"  {desc}", r)
            bytes_saved = base_size - r.size_bytes
            if bytes_saved <= 0:
                continue
            delta_bpb = r.val_bpb - base_bpb
            score = delta_bpb / bytes_saved
            if best_score is None or score < best_score or (abs(score - best_score) < 1e-18 and r.size_bytes < (best_move_res.size_bytes if best_move_res else 10**18)):
                best_score = score
                best_move_res = r
                best_move_desc = desc

        if best_move_res is None:
            print("No compression move reduced size; stopping compression phase.")
            break

        print(f"CHOSEN {best_move_desc}")
        baseline = best_move_res
        current = clone_policy(best_move_res.policy)
        if baseline.size_bytes <= target_bytes and (
            best_under is None
            or baseline.val_bpb < best_under.val_bpb
            or (baseline.val_bpb == best_under.val_bpb and baseline.val_loss < best_under.val_loss)
        ):
            best_under = baseline

    if baseline.size_bytes <= target_bytes:
        best_under = baseline
        for step in range(max_refine_steps):
            candidates = refine_candidates(current, outlier_fracs)
            if not candidates:
                break
            better: EvalResult | None = None
            better_desc: str | None = None
            print(f"\n[refine step {step+1}] evaluating {len(candidates)} candidates...")
            for desc, cand in candidates:
                r = evaluate(cand)
                if r.size_bytes > target_bytes:
                    print_result(f"  {desc} (OVER)", r)
                    continue
                print_result(f"  {desc}", r)
                if (r.val_bpb < baseline.val_bpb or (r.val_bpb == baseline.val_bpb and r.val_loss < baseline.val_loss)) and (
                    better is None or r.val_bpb < better.val_bpb or (r.val_bpb == better.val_bpb and r.val_loss < better.val_loss)
                ):
                    better = r
                    better_desc = desc
            if better is None:
                print("No under-budget refinement improved validation; stopping refine phase.")
                break
            print(f"CHOSEN {better_desc}")
            baseline = better
            current = clone_policy(better.policy)
            best_under = better

    print("\n=== FINAL BEST UNDER TARGET ===")
    if best_under is None:
        print("No policy reached target bytes.")
        return

    print_result("BEST", best_under)
    if write_best:
        final = evaluate(best_under.policy, keep_artifact=True)
        buf = io.BytesIO()
        torch.save(final.artifact, buf)
        with open(write_best, "wb") as f:
            f.write(zlib.compress(buf.getvalue(), level=9))
        torch.save(final.decoded_sd, write_best + ".decoded.pt")
        print(f"Wrote compressed artifact to {write_best}")
        print(f"Wrote decoded checkpoint to {write_best}.decoded.pt")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc mixed-bit search using exact train_gpt 8-bit baseline path")
    parser.add_argument("--input", required=True, help="Path to final_model.pt from train_gpt")
    parser.add_argument("--target-bytes", type=int, default=16_000_000)
    parser.add_argument("--max-compress-steps", type=int, default=6)
    parser.add_argument("--max-refine-steps", type=int, default=4)
    parser.add_argument("--outlier-fracs", type=str, default="0.01,0.02,0.05")
    parser.add_argument("--write-best", type=str, default="")
    args = parser.parse_args()

    outlier_fracs = [float(x) for x in args.outlier_fracs.split(",") if x.strip()]
    run_search(
        input_path=args.input,
        target_bytes=args.target_bytes,
        max_compress_steps=args.max_compress_steps,
        max_refine_steps=args.max_refine_steps,
        outlier_fracs=outlier_fracs,
        write_best=(args.write_best or None),
    )


if __name__ == "__main__":
    main()
