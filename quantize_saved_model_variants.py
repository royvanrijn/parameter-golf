"""Quantize an already-trained full model checkpoint with train_gpt_comp variants.

Example:
    python quantize_saved_model_variants.py --input final_model.pt --output-dir quant_variants
"""

from __future__ import annotations

import argparse
import io
import zlib
from pathlib import Path

import torch

from train_gpt_comp import (
    LOG_COMPAND_MU,
    dequantize_state_dict_nonlinear,
    parse_quant_benchmark_methods,
    quant_benchmark_method_cfgs,
    quantize_state_dict_nonlinear,
    quantize_state_dict_nonlinear_mixed,
)


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return {k: v.detach().to("cpu").contiguous() for k, v in obj.items()}
    if isinstance(obj, dict) and isinstance(obj.get("state_dict"), dict):
        sd = obj["state_dict"]
        if all(isinstance(v, torch.Tensor) for v in sd.values()):
            return {k: v.detach().to("cpu").contiguous() for k, v in sd.items()}
    raise ValueError(
        "Unsupported checkpoint format. Expected a raw state_dict or dict with 'state_dict'."
    )


def run_variants(state_dict: dict[str, torch.Tensor], output_dir: Path, validate_roundtrip: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    method_cfgs = quant_benchmark_method_cfgs()
    method_names = parse_quant_benchmark_methods()

    print(f"Loaded {len(state_dict)} tensors. Running {len(method_names)} methods...")
    for method_name in method_names:
        cfg = method_cfgs.get(method_name)
        if cfg is None:
            print(f"[skip] unknown method: {method_name}")
            continue

        if cfg.get("kind") == "mixed":
            quant_obj, _ = quantize_state_dict_nonlinear_mixed(
                state_dict,
                default_cfg=dict(cfg["default"]),
                overrides=tuple(cfg["overrides"]),
                mu=LOG_COMPAND_MU,
            )
            cfg_desc = "mixed"
        else:
            quant_obj, _ = quantize_state_dict_nonlinear(
                state_dict,
                bits=int(cfg["bits"]),
                nonlinear=str(cfg["nonlinear"]),
                granularity=str(cfg["granularity"]),
                mu=LOG_COMPAND_MU,
            )
            cfg_desc = (
                f"bits={cfg['bits']} nonlinear={cfg['nonlinear']} "
                f"granularity={cfg['granularity']}"
            )

        raw_buf = io.BytesIO()
        torch.save(quant_obj, raw_buf)
        blob = zlib.compress(raw_buf.getvalue(), level=9)
        out_path = output_dir / f"{method_name}.ptz"
        out_path.write_bytes(blob)

        line = f"[ok] {method_name:36s} bytes={len(blob):9d} cfg={cfg_desc}"
        if validate_roundtrip:
            roundtrip_obj = torch.load(io.BytesIO(zlib.decompress(blob)), map_location="cpu")
            roundtrip_state = dequantize_state_dict_nonlinear(roundtrip_obj)
            if set(roundtrip_state.keys()) != set(state_dict.keys()):
                raise RuntimeError(f"Roundtrip mismatch for {method_name}: key sets differ")
            line += " roundtrip=pass"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("final_model.pt"), help="Path to full-precision model checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("quant_variants"), help="Directory to store compressed quantized artifacts")
    parser.add_argument(
        "--validate-roundtrip",
        action="store_true",
        help="Decompress/dequantize each artifact and verify state_dict keys",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input checkpoint not found: {args.input}. "
            "For QAT runs this is typically written as final_model.pt on rank 0."
        )

    state_dict = load_state_dict(args.input)
    run_variants(state_dict, args.output_dir, validate_roundtrip=args.validate_roundtrip)
    print(f"Done. Artifacts written to: {args.output_dir}")
    print(
        "Tip: choose methods with QUANT_BENCHMARK_METHODS, e.g. "
        "QUANT_BENCHMARK_METHODS=linear_int6_per_row,mix_embed_head_int8_proj_int6_col"
    )


if __name__ == "__main__":
    main()
