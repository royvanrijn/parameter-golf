from __future__ import annotations

from collections.abc import Callable

import numpy as np

INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = np.float16
INT8_PER_ROW_SCALE_DTYPE = np.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def is_svd_target_name(name: str) -> bool:
    """Compatibility helper for older callers.

    The current svd3 model is trained directly with factorized Linear layers, so the
    export pipeline no longer needs an extra dense->SVD conversion pass.
    """
    return False



def svd_rank_for_name(name: str, args) -> int:
    """Compatibility helper for older callers.

    We keep the function/signature so train code does not need to change, but dense
    SVD export is intentionally disabled for the current factorized model.
    """
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



def quantize_state_dict_int8(
    state: dict[str, np.ndarray],
    *,
    keep_float_fp32_name_patterns: tuple[str, ...],
    svd_rank_lookup: Callable[[str], int],
    svd_export_dtype: np.dtype,
) -> tuple[dict[str, object], dict[str, int]]:
    del svd_rank_lookup
    del svd_export_dtype

    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        (
            "param_count",
            "num_tensors",
            "num_float_tensors",
            "num_nonfloat_tensors",
            "baseline_tensor_bytes",
            "int8_payload_bytes",
        ),
        0,
    )
    for name, arr_in in state.items():
        arr = np.ascontiguousarray(np.array(arr_in, copy=False))
        stats["param_count"] += int(arr.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.nbytes)

        if not np.issubdtype(arr.dtype, np.floating):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = arr
            stats["int8_payload_bytes"] += int(arr.nbytes)
            continue

        if int(arr.size) <= INT8_KEEP_FLOAT_MAX_NUMEL:
            stats["num_float_tensors"] += 1
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
        "__quant_format__": "int8_v3",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
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

    for name, arr in quant_obj["passthrough"].items():
        out_arr = np.ascontiguousarray(np.array(arr, copy=True))
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out[name] = out_arr.astype(np.dtype(orig_dtype), copy=False)
        else:
            out[name] = out_arr

    return out
