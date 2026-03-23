from __future__ import annotations

import argparse
import heapq
import io
import json
import math
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0
INT8_PER_ROW_SCALE_DTYPE = torch.float16
DEFAULT_PREDICTIVE_PATTERNS = (
    "attn.c_k.weight",
    "attn.c_q.weight",
    "attn.proj.weight",
    "mlp.fc.weight",
    "mlp.proj.weight",
)
DEFAULT_MIN_NUMEL = 131072  # skip small tensors for predictive path


# -----------------------------
# Quantization helpers
# -----------------------------

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    """Matches the train_gpt.py checkpoint quantization policy."""
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def dequantize_tensor(q: Tensor, scales: Tensor, dtype_name: str) -> Tensor:
    dtype = getattr(torch, dtype_name)
    if scales.ndim > 0:
        s = scales.to(dtype=torch.float32)
        return (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
    return (q.float() * float(scales.item())).to(dtype=dtype).contiguous()


# -----------------------------
# Predictive coding helpers
# -----------------------------

def zigzag_encode(v: int) -> int:
    return (v << 1) ^ (v >> 31)


def zigzag_decode(u: int) -> int:
    return (u >> 1) ^ -(u & 1)


def residual_stream_left_row_major(q2d: np.ndarray) -> list[int]:
    rows, cols = q2d.shape
    residuals: list[int] = []
    for r in range(rows):
        prev = 0
        row = q2d[r]
        for c in range(cols):
            actual = int(row[c])
            residuals.append(actual - prev)
            prev = actual
    return residuals


def q2d_from_residuals_left_row_major(rows: int, cols: int, residuals: list[int]) -> np.ndarray:
    rec = np.empty((rows, cols), dtype=np.int8)
    idx = 0
    for r in range(rows):
        prev = 0
        for c in range(cols):
            actual = prev + int(residuals[idx])
            rec[r, c] = actual
            prev = actual
            idx += 1
    if idx != rows * cols:
        raise ValueError("residual length mismatch")
    return rec


# -----------------------------
# Huffman helpers
# -----------------------------

@dataclass
class HuffmanEncoded:
    payload: bytes
    num_bits: int
    code_lengths: list[tuple[int, int]]


class _BitWriter:
    def __init__(self) -> None:
        self.buf = bytearray()
        self.acc = 0
        self.bits = 0
        self.total_bits = 0

    def write(self, code: int, length: int) -> None:
        self.total_bits += length
        for shift in range(length - 1, -1, -1):
            bit = (code >> shift) & 1
            self.acc = (self.acc << 1) | bit
            self.bits += 1
            if self.bits == 8:
                self.buf.append(self.acc)
                self.acc = 0
                self.bits = 0

    def finish(self) -> tuple[bytes, int]:
        if self.bits:
            self.acc <<= 8 - self.bits
            self.buf.append(self.acc)
            self.acc = 0
            self.bits = 0
        return bytes(self.buf), self.total_bits


class _BitReader:
    def __init__(self, payload: bytes, num_bits: int) -> None:
        self.payload = payload
        self.num_bits = num_bits
        self.bit_pos = 0

    def read_bit(self) -> int:
        if self.bit_pos >= self.num_bits:
            raise EOFError("out of bits")
        byte = self.payload[self.bit_pos // 8]
        bit_in_byte = 7 - (self.bit_pos % 8)
        bit = (byte >> bit_in_byte) & 1
        self.bit_pos += 1
        return bit


def _build_code_lengths(symbols: list[int]) -> dict[int, int]:
    freq: dict[int, int] = {}
    for s in symbols:
        freq[s] = freq.get(s, 0) + 1
    if not freq:
        return {0: 1}

    heap: list[tuple[int, int, Any]] = []
    for sym in sorted(freq):
        heap.append((freq[sym], sym, ("leaf", sym)))
    heapq.heapify(heap)

    if len(heap) == 1:
        _, _, leaf = heap[0]
        return {leaf[1]: 1}

    while len(heap) > 1:
        f1, m1, n1 = heapq.heappop(heap)
        f2, m2, n2 = heapq.heappop(heap)
        merged = ("node", n1, n2)
        heapq.heappush(heap, (f1 + f2, min(m1, m2), merged))

    _, _, root = heap[0]
    lengths: dict[int, int] = {}

    def walk(node: Any, depth: int) -> None:
        kind = node[0]
        if kind == "leaf":
            lengths[node[1]] = max(depth, 1)
            return
        walk(node[1], depth + 1)
        walk(node[2], depth + 1)

    walk(root, 0)
    return lengths


def _canonical_codes(code_lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    items = sorted(((length, sym) for sym, length in code_lengths.items()), key=lambda x: (x[0], x[1]))
    code = 0
    prev_len = 0
    table: dict[int, tuple[int, int]] = {}
    for length, sym in items:
        code <<= length - prev_len
        table[sym] = (code, length)
        code += 1
        prev_len = length
    return table


def huffman_encode(symbols: list[int]) -> HuffmanEncoded:
    lengths = _build_code_lengths(symbols)
    codebook = _canonical_codes(lengths)
    w = _BitWriter()
    for s in symbols:
        code, length = codebook[s]
        w.write(code, length)
    payload, num_bits = w.finish()
    ordered_lengths = sorted(((sym, lengths[sym]) for sym in lengths), key=lambda x: x[0])
    return HuffmanEncoded(payload=payload, num_bits=num_bits, code_lengths=ordered_lengths)


def huffman_decode(enc: HuffmanEncoded, expected_symbols: int) -> list[int]:
    lengths = {sym: length for sym, length in enc.code_lengths}
    codebook = _canonical_codes(lengths)
    decode_map: dict[tuple[int, int], int] = {(code, length): sym for sym, (code, length) in codebook.items()}
    max_len = max(lengths.values()) if lengths else 1

    out: list[int] = []
    reader = _BitReader(enc.payload, enc.num_bits)
    while len(out) < expected_symbols:
        code = 0
        for length in range(1, max_len + 1):
            bit = reader.read_bit()
            code = (code << 1) | bit
            key = (code, length)
            if key in decode_map:
                out.append(decode_map[key])
                break
        else:
            raise ValueError("invalid Huffman stream")
    return out


# -----------------------------
# Tensor codecs
# -----------------------------

def encode_baseline_zlib(q: Tensor) -> dict[str, Any]:
    payload = zlib.compress(q.cpu().contiguous().numpy().tobytes(), level=9)
    return {"codec": "baseline_zlib", "payload": payload}


def decode_baseline_zlib(record: dict[str, Any], shape: tuple[int, ...]) -> Tensor:
    raw = zlib.decompress(record["payload"])
    arr = np.frombuffer(raw, dtype=np.int8).reshape(shape)
    return torch.from_numpy(arr.copy())


def encode_predictive_huffman_fast(q2d: Tensor) -> dict[str, Any]:
    q_np = q2d.cpu().numpy()
    residuals = residual_stream_left_row_major(q_np)
    symbols = [zigzag_encode(v) for v in residuals]
    enc = huffman_encode(symbols)
    decoded_symbols = huffman_decode(enc, expected_symbols=q_np.size)
    decoded_residuals = [zigzag_decode(u) for u in decoded_symbols]
    q_roundtrip = q2d_from_residuals_left_row_major(q_np.shape[0], q_np.shape[1], decoded_residuals)
    if not np.array_equal(q_roundtrip, q_np):
        raise ValueError("predictive+huffman roundtrip failed")
    return {
        "codec": "predictive_huffman_fast",
        "scan": "row_major",
        "predictor": "left",
        "payload": enc.payload,
        "num_bits": enc.num_bits,
        "code_lengths": enc.code_lengths,
    }


def decode_predictive_huffman_fast(record: dict[str, Any], shape: tuple[int, int]) -> Tensor:
    rows, cols = shape
    enc = HuffmanEncoded(
        payload=record["payload"],
        num_bits=int(record["num_bits"]),
        code_lengths=[(int(s), int(l)) for s, l in record["code_lengths"]],
    )
    symbols = huffman_decode(enc, expected_symbols=rows * cols)
    residuals = [zigzag_decode(u) for u in symbols]
    q = q2d_from_residuals_left_row_major(rows, cols, residuals)
    return torch.from_numpy(q.copy())


def encoded_codec_size_bytes(record: dict[str, Any]) -> int:
    payload = int(len(record.get("payload", b"")))
    meta_bits = 0
    if record["codec"] == "predictive_huffman_fast":
        meta_bits += 32  # num_bits
        meta_bits += len(record["code_lengths"]) * 24  # symbol + length
    return payload + math.ceil(meta_bits / 8)


def should_try_predictive(name: str, q: Tensor, patterns: tuple[str, ...], min_numel: int) -> bool:
    if q.ndim != 2:
        return False
    if q.numel() < min_numel:
        return False
    return any(pattern in name for pattern in patterns)


# -----------------------------
# State-dict serialization
# -----------------------------

def encode_state_dict_export(
    state_dict: dict[str, Tensor],
    predictive_patterns: tuple[str, ...],
    min_numel: int,
    verbose: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tensors: dict[str, Any] = {}
    report: dict[str, Any] = {
        "per_tensor": {},
        "totals": {"float32_raw": 0, "quantized_raw": 0, "quantized_zlib": 0, "predictive_huffman": 0},
        "settings": {
            "predictive_patterns": list(predictive_patterns),
            "min_numel": int(min_numel),
            "predictive_mode": "row_major_left_only",
        },
    }

    for idx, name in enumerate(sorted(state_dict), start=1):
        t0 = time.perf_counter()
        t = state_dict[name].detach().cpu().contiguous()
        if verbose:
            print(f"[{idx:03d}] {name} shape={tuple(t.shape)} dtype={t.dtype}", flush=True)

        if not t.is_floating_point():
            payload = zlib.compress(t.numpy().tobytes(), level=9)
            tensors[name] = {
                "kind": "passthrough",
                "dtype": str(t.dtype).removeprefix("torch."),
                "shape": tuple(int(x) for x in t.shape),
                "payload": payload,
            }
            sz = len(payload)
            report["per_tensor"][name] = {
                "float32_raw": 0,
                "quantized_raw": 0,
                "quantized_zlib": sz,
                "predictive_huffman": sz,
                "selected": "passthrough",
                "elapsed_sec": time.perf_counter() - t0,
            }
            report["totals"]["quantized_zlib"] += sz
            report["totals"]["predictive_huffman"] += sz
            continue

        q, scales = quantize_float_tensor(t)
        dtype_name = str(t.dtype).removeprefix("torch.")
        scale_bytes = tensor_nbytes(scales)
        q_raw_bytes = tensor_nbytes(q)

        baseline = encode_baseline_zlib(q)
        baseline_size = encoded_codec_size_bytes(baseline)

        best_record = baseline
        best_size = baseline_size
        pred_size = baseline_size

        predictive_tried = should_try_predictive(name, q, predictive_patterns, min_numel)
        if predictive_tried:
            candidate = encode_predictive_huffman_fast(q)
            size = encoded_codec_size_bytes(candidate)
            pred_size = size
            if size < best_size:
                best_record = candidate
                best_size = size

        tensors[name] = {
            "kind": "quantized",
            "orig_dtype": dtype_name,
            "shape": tuple(int(x) for x in q.shape),
            "scales": scales,
            "scale_dtype": str(scales.dtype).removeprefix("torch."),
            "codec": best_record,
        }

        float32_raw = int(t.numel()) * 4
        quantized_raw = q_raw_bytes + scale_bytes
        quantized_zlib = baseline_size + scale_bytes
        predictive_huffman = pred_size + scale_bytes

        report["per_tensor"][name] = {
            "float32_raw": float32_raw,
            "quantized_raw": quantized_raw,
            "quantized_zlib": quantized_zlib,
            "predictive_huffman": predictive_huffman,
            "selected": best_record["codec"],
            "predictive_tried": predictive_tried,
            "elapsed_sec": time.perf_counter() - t0,
        }
        report["totals"]["float32_raw"] += float32_raw
        report["totals"]["quantized_raw"] += quantized_raw
        report["totals"]["quantized_zlib"] += quantized_zlib
        report["totals"]["predictive_huffman"] += predictive_huffman

    obj = {
        "__quant_format__": "int8_predictive_huffman_fast_v1",
        "int8_clip_percentile": INT8_CLIP_PERCENTILE,
        "tensor_order": sorted(state_dict),
        "tensors": tensors,
        "settings": report["settings"],
    }
    return obj, report


def decode_state_dict_export(obj: dict[str, Any]) -> dict[str, Tensor]:
    if obj.get("__quant_format__") != "int8_predictive_huffman_fast_v1":
        raise ValueError(f"unsupported format: {obj.get('__quant_format__')}")

    out: dict[str, Tensor] = {}
    for name in obj["tensor_order"]:
        rec = obj["tensors"][name]
        shape = tuple(int(x) for x in rec["shape"])

        if rec["kind"] == "passthrough":
            raw = zlib.decompress(rec["payload"])
            np_dtype = np.dtype(getattr(np, rec["dtype"]))
            arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
            out[name] = torch.from_numpy(arr.copy())
            continue

        codec = rec["codec"]
        if codec["codec"] == "baseline_zlib":
            q = decode_baseline_zlib(codec, shape)
        elif codec["codec"] == "predictive_huffman_fast":
            q = decode_predictive_huffman_fast(codec, (shape[0], shape[1]))
        else:
            raise ValueError(f"unknown codec: {codec['codec']}")

        out[name] = dequantize_tensor(q, rec["scales"], rec["orig_dtype"])

    return out


# -----------------------------
# Utility / CLI
# -----------------------------

def extract_state_dict(ckpt: Any) -> dict[str, Tensor]:
    if isinstance(ckpt, dict) and all(isinstance(v, Tensor) for v in ckpt.values()):
        return ckpt
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            maybe = ckpt.get(key)
            if isinstance(maybe, dict) and all(isinstance(v, Tensor) for v in maybe.values()):
                return maybe
    raise ValueError("could not find a tensor state_dict in checkpoint")


def print_report(report: dict[str, Any]) -> None:
    header = f"{'tensor':60s} {'f32':>10s} {'q_raw':>10s} {'q+zlib':>10s} {'pred+huff':>10s} {'sel':>22s}"
    print(header)
    print("-" * len(header))
    for name in sorted(report["per_tensor"]):
        rec = report["per_tensor"][name]
        print(
            f"{name[:60]:60s} "
            f"{rec['float32_raw']:10d} {rec['quantized_raw']:10d} {rec['quantized_zlib']:10d} {rec['predictive_huffman']:10d} "
            f"{rec['selected']:>22s}"
        )
    totals = report["totals"]
    print("-" * len(header))
    print(
        f"{'TOTAL':60s} "
        f"{totals['float32_raw']:10d} {totals['quantized_raw']:10d} {totals['quantized_zlib']:10d} {totals['predictive_huffman']:10d} {'-':>22s}"
    )


def roundtrip_verify(original: dict[str, Tensor], restored: dict[str, Tensor]) -> None:
    if set(original) != set(restored):
        raise ValueError("roundtrip key mismatch")
    for name in original:
        a = original[name].detach().cpu()
        b = restored[name].detach().cpu()
        if a.dtype.is_floating_point:
            if a.shape != b.shape:
                raise ValueError(f"shape mismatch for {name}")
        else:
            if not torch.equal(a, b):
                raise ValueError(f"non-float tensor mismatch for {name}")


def parse_patterns(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast export for train_gpt checkpoints with selective predictive Huffman codec.")
    parser.add_argument("--input", type=Path, required=True, help="Input checkpoint path")
    parser.add_argument("--output", type=Path, default=Path("final_model.int8.ph.fast.pt"), help="Output export path")
    parser.add_argument("--report-json", type=Path, default=None, help="Optional JSON path for byte report")
    parser.add_argument(
        "--predictive-patterns",
        type=str,
        default=",".join(DEFAULT_PREDICTIVE_PATTERNS),
        help="Comma-separated tensor-name substrings eligible for fast predictive coding",
    )
    parser.add_argument(
        "--min-numel",
        type=int,
        default=DEFAULT_MIN_NUMEL,
        help="Only try predictive coding on 2D tensors with at least this many elements",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable per-tensor progress logging")
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location="cpu")
    state_dict = extract_state_dict(ckpt)
    patterns = parse_patterns(args.predictive_patterns)

    t0 = time.perf_counter()
    export_obj, report = encode_state_dict_export(
        state_dict,
        predictive_patterns=patterns,
        min_numel=args.min_numel,
        verbose=not args.quiet,
    )
    encode_sec = time.perf_counter() - t0

    buf = io.BytesIO()
    torch.save(export_obj, buf)
    args.output.write_bytes(buf.getvalue())

    t1 = time.perf_counter()
    loaded_obj = torch.load(args.output, map_location="cpu")
    restored = decode_state_dict_export(loaded_obj)
    roundtrip_verify(state_dict, restored)
    decode_sec = time.perf_counter() - t1

    report["elapsed_sec"] = {"encode": encode_sec, "decode_verify": decode_sec, "total": encode_sec + decode_sec}
    print_report(report)
    print(f"encode_sec={encode_sec:.3f} decode_verify_sec={decode_sec:.3f} total_sec={encode_sec + decode_sec:.3f}")
    print(f"written_bytes={args.output.stat().st_size}")

    if args.report_json is not None:
        args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
