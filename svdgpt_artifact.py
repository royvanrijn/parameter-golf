from __future__ import annotations

import numpy as np
from collections.abc import Callable

SVD_TARGET_SUFFIX_TO_RANK_ATTR = {
    ".mlp.fc.weight": "svd_rank_fc",
    ".mlp.proj.weight": "svd_rank_proj",
    ".attn.proj.weight": "svd_rank_attn_proj",
}

INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = np.float16
INT8_PER_ROW_SCALE_DTYPE = np.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def is_svd_target_name(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in SVD_TARGET_SUFFIX_TO_RANK_ATTR)


def svd_rank_for_name(name: str, args) -> int:
    if name.endswith("mlp.fc.weight"):
        return int(args.svd_rank_fc)
    if name.endswith("mlp.proj.weight"):
        return int(args.svd_rank_proj)
    if getattr(args, "svd_use_attn_proj", True) and name.endswith("attn.proj.weight"):
        return int(args.svd_rank_attn_proj)

    if getattr(args, "svd_use_qkv", False):
        if name.endswith("attn.c_q.weight"):
            return int(args.svd_rank_qk)
        if name.endswith("attn.c_k.weight"):
            return int(args.svd_rank_qk)
        if name.endswith("attn.c_v.weight"):
            return int(args.svd_rank_v)

    if ".attn.c_qkv." in name:
        return args.svd_rank_qkv

    return 0


def export_np_dtype(dtype_name: str) -> np.dtype:
    name = dtype_name.lower()
    if name == "float16":
        return np.float16
    if name == "float32":
        return np.float32
    raise ValueError(f"Unsupported SVD_EXPORT_FLOAT_DTYPE={dtype_name}")


def keep_float_array(
    name: str,
    arr: np.ndarray,
    passthrough_orig_dtypes: dict[str, str],
    keep_float_fp32_name_patterns: tuple[str, ...],
) -> np.ndarray:
    if any(pattern in name for pattern in keep_float_fp32_name_patterns):
        return np.ascontiguousarray(arr.astype(np.float32, copy=False))
    if str(arr.dtype) in {"float32", "bfloat16"}:
        passthrough_orig_dtypes[name] = str(arr.dtype)
        return np.ascontiguousarray(arr.astype(INT8_KEEP_FLOAT_STORE_DTYPE, copy=False))
    return np.ascontiguousarray(np.array(arr, copy=True))


def quantize_float_array(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f32 = np.ascontiguousarray(arr.astype(np.float32, copy=False))
    if f32.ndim == 2:
        clip_abs = np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float32, copy=False)
        q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8, copy=False)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE, copy=False))

    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT8_CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs / 127.0 if clip_abs > 0.0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -127, 127).astype(np.int8, copy=False)
    return np.ascontiguousarray(q), scale


def svd_factorize_array(arr: np.ndarray, rank: int, store_dtype: np.dtype) -> dict[str, np.ndarray | str | tuple[int, ...] | int]:
    f32 = np.ascontiguousarray(arr.astype(np.float32, copy=False))
    out_dim, in_dim = f32.shape
    max_rank = min(out_dim, in_dim)
    rank = max(1, min(int(rank), max_rank))
    u, s, vh = np.linalg.svd(f32, full_matrices=False)

    u = np.ascontiguousarray(u[:, :rank].astype(np.float32, copy=False))
    s = np.ascontiguousarray(s[:rank].astype(store_dtype, copy=False))
    vh = np.ascontiguousarray(vh[:rank, :].astype(np.float32, copy=False))
    u_q, u_scale = quantize_float_array(u)
    vh_q, vh_scale = quantize_float_array(vh)

    return {
        "shape": f32.shape,
        "rank": rank,
        "u_q": u_q,
        "u_scale": u_scale,
        "vh_q": vh_q,
        "vh_scale": vh_scale,
        "s": s,
        "dtype": str(arr.dtype),
    }


def reconstruct_svd_factorized_array(spec: dict[str, object]) -> np.ndarray:
    u_q = np.asarray(spec["u_q"], dtype=np.int8)
    u_scale = np.asarray(spec["u_scale"], dtype=np.float32)
    if u_scale.ndim > 0:
        u = u_q.astype(np.float32) * u_scale.reshape((u_q.shape[0],) + (1,) * (u_q.ndim - 1))
    else:
        u = u_q.astype(np.float32) * float(u_scale)

    vh_q = np.asarray(spec["vh_q"], dtype=np.int8)
    vh_scale = np.asarray(spec["vh_scale"], dtype=np.float32)
    if vh_scale.ndim > 0:
        vh = vh_q.astype(np.float32) * vh_scale.reshape((vh_q.shape[0],) + (1,) * (vh_q.ndim - 1))
    else:
        vh = vh_q.astype(np.float32) * float(vh_scale)

    s = np.asarray(spec["s"], dtype=np.float32)
    return np.ascontiguousarray((u * s[None, :]) @ vh)


def quantize_state_dict_int8(
    state: dict[str, np.ndarray],
    *,
    keep_float_fp32_name_patterns: tuple[str, ...],
    svd_rank_lookup: Callable[[str], int],
    svd_export_dtype: np.dtype,
) -> tuple[dict[str, object], dict[str, int]]:
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    svd_factors: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        (
            "param_count",
            "num_tensors",
            "num_float_tensors",
            "num_nonfloat_tensors",
            "baseline_tensor_bytes",
            "int8_payload_bytes",
            "svd_factor_bytes",
        ),
        0,
    )
    for name, arr_in in state.items():
        arr = np.ascontiguousarray(np.array(arr_in, copy=False))
        stats["param_count"] += int(arr.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.nbytes)

        rank = int(svd_rank_lookup(name))
        if rank > 0 and arr.ndim == 2 and np.issubdtype(arr.dtype, np.floating):
            stats["num_float_tensors"] += 1
            spec = svd_factorize_array(arr, rank, svd_export_dtype)
            svd_factors[name] = spec
            svd_bytes = (
                spec["u_q"].nbytes
                + spec["u_scale"].nbytes
                + spec["vh_q"].nbytes
                + spec["vh_scale"].nbytes
                + spec["s"].nbytes
            )
            stats["svd_factor_bytes"] += int(svd_bytes)
            stats["int8_payload_bytes"] += int(svd_bytes)
            continue

        if not np.issubdtype(arr.dtype, np.floating):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = arr
            stats["int8_payload_bytes"] += int(arr.nbytes)
            continue

        if int(arr.size) <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_array(name, arr, passthrough_orig_dtypes, keep_float_fp32_name_patterns)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_array(arr)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(arr.dtype)
        stats["int8_payload_bytes"] += int(q.nbytes + s.nbytes)

    obj: dict[str, object] = {
        "__quant_format__": "int8_plus_svd_v2",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
        "svd_factors": svd_factors,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(quant_obj: dict[str, object]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    qmeta = quant_obj.get("qmeta", {})
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    for name, q in quant_obj["quantized"].items():
        q_np = np.asarray(q, dtype=np.int8)
        dtype_name = str(quant_obj["dtypes"][name])
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        if qmeta.get(name, {}).get("scheme") == "per_row" or scale.ndim > 0:
            out_arr = q_np.astype(np.float32) * scale.reshape((q_np.shape[0],) + (1,) * (q_np.ndim - 1))
        else:
            out_arr = q_np.astype(np.float32) * float(scale)
        out[name] = np.ascontiguousarray(out_arr.astype(np.dtype(dtype_name), copy=False))
    for name, spec in quant_obj.get("svd_factors", {}).items():
        dtype_name = str(spec["dtype"])
        out[name] = reconstruct_svd_factorized_array(spec).astype(np.dtype(dtype_name), copy=False)
    for name, arr in quant_obj["passthrough"].items():
        out_arr = np.ascontiguousarray(np.array(arr, copy=True))
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out[name] = out_arr.astype(np.dtype(orig_dtype), copy=False)
        else:
            out[name] = out_arr
    return out
