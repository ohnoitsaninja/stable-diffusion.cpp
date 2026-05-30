#!/usr/bin/env python3
"""Generate a Lens-Turbo transformer safetensors -> native slot map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open


TOP_LEVEL_LOADED = {
    "img_in.weight",
    "img_in.bias",
    "txt_norm.0.weight",
    "txt_norm.1.weight",
    "txt_norm.2.weight",
    "txt_norm.3.weight",
    "txt_in.weight",
    "txt_in.bias",
    "norm_out.linear.weight",
    "norm_out.linear.bias",
    "proj_out.weight",
    "proj_out.bias",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def module_and_slot(name: str) -> tuple[str, str, str]:
    if name.startswith("transformer_blocks."):
        parts = name.split(".", 2)
        if len(parts) != 3:
            return "LensTransformer2DModel.transformer_blocks", "unmapped", "unknown"
        block = int(parts[1])
        suffix = parts[2]
        module = f"LensTransformer2DModel.transformer_blocks[{block}].{suffix.rsplit('.', 1)[0]}"
        slot = f"LensTransformerBlock[{block}].{suffix}"
        status = "block0-load-smoke" if block == 0 else "streamed-block-load"
        return module, slot, status
    if name in TOP_LEVEL_LOADED:
        return f"LensTransformer2DModel.{name.rsplit('.', 1)[0]}", f"LensTransformerParams.{name}", "top-level-load"
    if name.startswith("time_text_embed."):
        return f"LensTransformer2DModel.{name.rsplit('.', 1)[0]}", f"LensTransformerParams.{name}", "fixture-export-only"
    if name.startswith("pos_embed."):
        return f"LensTransformer2DModel.{name.rsplit('.', 1)[0]}", f"LensTransformerParams.{name}", "computed-not-loaded"
    return f"LensTransformer2DModel.{name.rsplit('.', 1)[0]}", f"LensTransformerParams.{name}", "not-loaded-by-current-smoke"


def main() -> int:
    args = parse_args()
    transformer_dir = Path(args.transformer_dir)
    config = json.loads((transformer_dir / "config.json").read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for shard in sorted(transformer_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in sorted(handle.keys()):
                tensor_slice = handle.get_slice(name)
                module, slot, status = module_and_slot(name)
                rows.append(
                    {
                        "safetensors_tensor_name": name,
                        "shape": list(tensor_slice.get_shape()),
                        "dtype": tensor_slice.get_dtype(),
                        "module": module,
                        "sd_cpp_parameter_slot": slot,
                        "loaded_unloaded_status": status,
                        "shard": shard.name,
                    }
                )

    expected_layers = int(config["num_layers"])
    block_indices = sorted(
        {
            int(row["safetensors_tensor_name"].split(".")[1])
            for row in rows
            if str(row["safetensors_tensor_name"]).startswith("transformer_blocks.")
        }
    )
    if block_indices != list(range(expected_layers)):
        raise RuntimeError(f"expected transformer blocks 0..{expected_layers - 1}, got {block_indices[:3]}..{block_indices[-3:]}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as out:
        out.write("# Lens Transformer Weight Map\n\n")
        out.write("## Config\n\n")
        for key in [
            "num_layers",
            "inner_dim",
            "enc_hidden_dim",
            "num_attention_heads",
            "attention_head_dim",
            "in_channels",
            "out_channels",
            "patch_size",
            "axes_dims_rope",
        ]:
            out.write(f"- `{key}`: `{config.get(key)}`\n")
        out.write(f"- `tensor_count`: `{len(rows)}`\n")
        out.write(f"- `shards`: `{', '.join(sorted({row['shard'] for row in rows}))}`\n\n")
        out.write("| safetensors tensor name | shape | dtype | LensTransformer2DModel module | sd.cpp parameter slot | loaded/unloaded status |\n")
        out.write("| --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            out.write(
                f"| `{row['safetensors_tensor_name']}` | `{row['shape']}` | `{row['dtype']}` | "
                f"`{row['module']}` | `{row['sd_cpp_parameter_slot']}` | `{row['loaded_unloaded_status']}` |\n"
            )

    print(f"wrote {output} tensors={len(rows)} layers={expected_layers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
