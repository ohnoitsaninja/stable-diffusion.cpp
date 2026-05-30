#!/usr/bin/env python3
"""Extract the source/cache contract for the Bonsai GemLite INT1 dominant GEMM.

This is a diagnostic helper for the experimental Bonsai INT1 fork path. It does
not participate in stable-diffusion.cpp runtime execution.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_nested_cache(cache: dict[str, Any], family: str, key: str) -> dict[str, Any] | None:
    entries = cache.get(family)
    if not isinstance(entries, dict):
        return None
    value = entries.get(key)
    return value if isinstance(value, dict) else None


def find_neighbor_keys(cache: dict[str, Any], family: str, n: int, k: int, limit: int = 16) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    entries = cache.get(family)
    if not isinstance(entries, dict):
        return out
    for key, value in entries.items():
        try:
            parsed = ast.literal_eval(key)
        except Exception:
            continue
        if not isinstance(parsed, tuple) or len(parsed) < 6:
            continue
        if parsed[1] == n or parsed[2] == k:
            out.append({"key": key, "config": value})
            if len(out) >= limit:
                break
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", required=True, type=Path)
    parser.add_argument("--generated-cache", type=Path)
    parser.add_argument("--m", type=int, default=1536)
    parser.add_argument("--n", type=int, default=27648)
    parser.add_argument("--k", type=int, default=3072)
    parser.add_argument("--type-id", type=int, default=101)
    parser.add_argument("--linear", default="single_transformer_blocks.0.attn.to_qkv_mlp_proj")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    quant_path = args.pack_dir / "quantization_config.json"
    pack_cache_path = args.pack_dir / "gemlite_autotune.json"
    quant = load_json(quant_path)
    pack_cache = load_json(pack_cache_path) if pack_cache_path.exists() else {}

    group_size = int(quant["group_size"])
    elements_per_sample = int(quant["packing_bitwidth"]) // int(quant["bits"])
    family = "GEMM" if args.m > 64 else ("GEMM_SPLITK" if args.m > 1 else "GEMV_REVSPLITK")
    key = str((args.m, args.n, args.k, group_size, elements_per_sample, args.type_id))

    generated_cache: dict[str, Any] = {}
    if args.generated_cache and args.generated_cache.exists():
        generated_cache = load_json(args.generated_cache)

    result = {
        "linear": args.linear,
        "shape": {"M": args.m, "N": args.n, "K": args.k},
        "quantization_config_path": str(quant_path),
        "pack_autotune_path": str(pack_cache_path),
        "generated_cache_path": str(args.generated_cache) if args.generated_cache else None,
        "bits": int(quant["bits"]),
        "group_size": group_size,
        "packing_bitwidth": int(quant["packing_bitwidth"]),
        "elements_per_sample": elements_per_sample,
        "input_dtype": quant.get("input_dtype"),
        "output_dtype": quant.get("output_dtype"),
        "type_id": args.type_id,
        "source_derived": {
            "matmul_family": family,
            "cache_key": key,
            "large_batch_rule": "GemLite core.py get_matmul_type returns GEMM when batch_size > 64.",
            "packed_layout": "GemLite bitpack.py packs over rows; W_q is [K / elements_per_sample, N].",
            "dequant_mode": "GemLite utils.py W_group_mode=4 uses tl.fma(bit, scale, zero).",
            "k_prune_rule": "GemLite gemm_kernels.py clamps BLOCK_SIZE_K to group_size for non-block metadata loads.",
        },
        "pack_cache_has_key": get_nested_cache(pack_cache, family, key) is not None,
        "pack_cache_config": get_nested_cache(pack_cache, family, key),
        "generated_cache_has_key": get_nested_cache(generated_cache, family, key) is not None,
        "generated_cache_config": get_nested_cache(generated_cache, family, key),
        "pack_cache_neighbors": find_neighbor_keys(pack_cache, family, args.n, args.k),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({
        "out": str(args.out),
        "family": family,
        "cache_key": key,
        "pack_cache_has_key": result["pack_cache_has_key"],
        "generated_cache_has_key": result["generated_cache_has_key"],
        "generated_cache_config": result["generated_cache_config"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
