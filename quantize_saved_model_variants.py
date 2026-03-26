"""Quantize an already-trained full model checkpoint with train_gpt_comp variants.

Example:
    python quantize_saved_model_variants.py --input final_model.pt --output-dir quant_variants
"""

from __future__ import annotations

import argparse
import io
import math
import zlib
from pathlib import Path

import sentencepiece as spm
import torch

from train_gpt_comp import (
    GPT,
    Hyperparameters,
    LOG_COMPAND_MU,
    build_sentencepiece_luts,
    dequantize_state_dict_nonlinear,
    load_validation_tokens,
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


def eval_val_metrics(
    args: Hyperparameters,
    model: torch.nn.Module,
    device: torch.device,
    val_tokens: torch.Tensor,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    max_batches: int,
) -> tuple[float, float]:
    local_batch_tokens = args.val_batch_size
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must be >= TRAIN_SEQ_LEN for single-process eval, "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        batches = 0
        for batch_seq_start in range(0, total_seqs, local_batch_seqs):
            if max_batches > 0 and batches >= max_batches:
                break
            batch_seq_end = min(batch_seq_start + local_batch_seqs, total_seqs)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                batch_loss = model(x, y).detach()

            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()
            batches += 1

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def build_eval_context(device: torch.device) -> tuple[Hyperparameters, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    args = Hyperparameters()
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    return args, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


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
    parser.add_argument(
        "--roundtrip-val-bpb",
        action="store_true",
        help="Run actual roundtrip validation (val_loss/val_bpb) for each method using Hyperparameters() data paths",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=8,
        help="Max validation batches for --roundtrip-val-bpb (0 means full validation)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input checkpoint not found: {args.input}. "
            "For QAT runs this is typically written as final_model.pt on rank 0."
        )

    state_dict = load_state_dict(args.input)
    run_variants(state_dict, args.output_dir, validate_roundtrip=args.validate_roundtrip)

    if args.roundtrip_val_bpb:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hp, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_eval_context(device)
        model = GPT(hp).to(device)
        model.load_state_dict(state_dict, strict=True)
        fp_val_loss, fp_val_bpb = eval_val_metrics(
            hp,
            model,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            max_batches=args.max_val_batches,
        )
        print(f"[fp] val_loss={fp_val_loss:.6f} val_bpb={fp_val_bpb:.6f}")

        method_cfgs = quant_benchmark_method_cfgs()
        for method_name in parse_quant_benchmark_methods():
            cfg = method_cfgs.get(method_name)
            if cfg is None:
                continue
            artifact = (args.output_dir / f"{method_name}.ptz").read_bytes()
            roundtrip_obj = torch.load(io.BytesIO(zlib.decompress(artifact)), map_location="cpu")
            roundtrip_state = dequantize_state_dict_nonlinear(roundtrip_obj)
            model.load_state_dict(roundtrip_state, strict=True)
            q_val_loss, q_val_bpb = eval_val_metrics(
                hp,
                model,
                device,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                max_batches=args.max_val_batches,
            )
            print(f"[rt] {method_name:36s} val_loss={q_val_loss:.6f} val_bpb={q_val_bpb:.6f}")

    print(f"Done. Artifacts written to: {args.output_dir}")
    print(
        "Tip: choose methods with QUANT_BENCHMARK_METHODS, e.g. "
        "QUANT_BENCHMARK_METHODS=linear_int6_per_row,mix_embed_head_int8_proj_int6_col"
    )


if __name__ == "__main__":
    main()
