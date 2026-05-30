#!/usr/bin/env python3
"""Inspect Lens-Turbo transformer safetensors headers without downloading shards."""

from __future__ import annotations

import json
import struct

import requests
from huggingface_hub import hf_hub_url


FILES = [
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
]

KEYS = [
    "img_in.weight",
    "img_in.bias",
    "txt_in.weight",
    "txt_in.bias",
    "txt_norm.0.weight",
    "txt_norm.1.weight",
    "txt_norm.2.weight",
    "txt_norm.3.weight",
    "time_text_embed.timestep_embedder.linear_1.weight",
    "time_text_embed.timestep_embedder.linear_2.weight",
    "transformer_blocks.0.attn.img_qkv.weight",
    "transformer_blocks.0.attn.txt_qkv.weight",
    "transformer_blocks.0.attn.to_out.0.weight",
    "transformer_blocks.0.attn.to_add_out.weight",
    "transformer_blocks.0.img_mlp.w1.weight",
    "transformer_blocks.0.img_mlp.w2.weight",
    "transformer_blocks.0.img_mlp.w3.weight",
    "transformer_blocks.0.txt_mlp.w1.weight",
    "transformer_blocks.0.txt_mlp.w2.weight",
    "transformer_blocks.0.txt_mlp.w3.weight",
    "transformer_blocks.0.img_mod.1.weight",
    "transformer_blocks.0.txt_mod.1.weight",
    "norm_out.linear.weight",
    "proj_out.weight",
]


def fetch_header(repo_id: str, filename: str) -> dict:
    url = hf_hub_url(repo_id, filename)
    first = requests.get(url, headers={"Range": "bytes=0-7"}, allow_redirects=True, timeout=60)
    first.raise_for_status()
    header_len = struct.unpack("<Q", first.content)[0]
    header = requests.get(
        url,
        headers={"Range": f"bytes=8-{8 + header_len - 1}"},
        allow_redirects=True,
        timeout=60,
    )
    header.raise_for_status()
    return json.loads(header.content)


def main() -> int:
    for filename in FILES:
        meta = fetch_header("microsoft/Lens-Turbo", filename)
        print(f"FILE {filename} tensors={len(meta) - 1}")
        for key in KEYS:
            if key in meta:
                item = meta[key]
                print(f"  {key}: {item['dtype']} {item['shape']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
