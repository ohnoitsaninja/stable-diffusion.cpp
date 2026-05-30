#!/usr/bin/env python3
"""Dump source-backed Lens transformer oracle tensors.

This script intentionally loads only the Lens transformer component plus caller
provided prompt-feature/latent tensors. It does not load the GPT-OSS text
encoder or the VAE. Use it after creating precomputed Lens conditioning with the
Python split runner.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens-src", default="F:/Paralol/local/Lens", help="Path containing the local lens package")
    parser.add_argument("--model-dir", required=True, help="Lens-Turbo root containing transformer/")
    parser.add_argument("--latent-npy", required=True, help="Image-token latent/features as f32 [B,S_img,128]")
    parser.add_argument("--feature-npy", action="append", required=True, help="Repeat 4 times: f32 [B,S_txt,2880]")
    parser.add_argument("--mask-npy", required=True, help="bool or numeric [B,S_txt], nonzero means valid")
    parser.add_argument("--timestep", type=float, required=True, help="Lens transformer timestep in [0,1]")
    parser.add_argument("--img-shape", required=True, help="frame,height,width, e.g. 1,32,32 for S_img=1024")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def main() -> int:
    args = parse_args()
    if len(args.feature_npy) != 4:
        raise ValueError(f"expected exactly 4 --feature-npy inputs, got {len(args.feature_npy)}")

    lens_src = Path(args.lens_src).resolve()
    transformer_py = lens_src / "lens" / "transformer.py"
    spec = importlib.util.spec_from_file_location("lens_transformer_local", transformer_py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {transformer_py}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    LensTransformer2DModel = module.LensTransformer2DModel

    dtype = torch_dtype(args.dtype)
    device = torch.device(args.device)

    transformer = LensTransformer2DModel.from_pretrained(
        args.model_dir,
        subfolder="transformer",
        torch_dtype=dtype,
    )
    transformer.to(device)
    transformer.eval()

    hidden_states = torch.from_numpy(np.load(args.latent_npy)).to(device=device, dtype=dtype)
    features = [
        torch.from_numpy(np.load(path)).to(device=device, dtype=dtype)
        for path in args.feature_npy
    ]
    mask = torch.from_numpy(np.load(args.mask_npy)).to(device=device).bool()
    timestep = torch.full((hidden_states.shape[0],), args.timestep, device=device, dtype=dtype)
    img_shape = tuple(int(part) for part in args.img_shape.split(","))
    if len(img_shape) != 3:
        raise ValueError("--img-shape must be frame,height,width")

    activations: dict[str, np.ndarray] = {}
    handles = []

    def capture(name: str):
        def hook(_module, _inputs, output):
            value = output
            if isinstance(value, tuple):
                value = value[-1]
            activations[name] = value.detach().float().cpu().numpy()

        return hook

    handles.append(transformer.img_in.register_forward_hook(capture("img_in")))
    handles.append(transformer.txt_in.register_forward_hook(capture("txt_in")))
    if transformer.transformer_blocks:
        handles.append(transformer.transformer_blocks[0].register_forward_hook(capture("block_0_hidden")))
        mid = len(transformer.transformer_blocks) // 2
        handles.append(transformer.transformer_blocks[mid].register_forward_hook(capture(f"block_{mid}_hidden")))
        last = len(transformer.transformer_blocks) - 1
        handles.append(transformer.transformer_blocks[last].register_forward_hook(capture(f"block_{last}_hidden")))
    handles.append(transformer.norm_out.register_forward_hook(capture("norm_out")))

    with torch.no_grad():
        output = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=features,
            encoder_hidden_states_mask=mask,
            timestep=timestep,
            img_shapes=[img_shape],
        )

    for handle in handles:
        handle.remove()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_dir": str(Path(args.model_dir).resolve()),
        "timestep": args.timestep,
        "img_shape": img_shape,
        "dtype": args.dtype,
        "source": "F:/Paralol/local/Lens/lens/transformer.py",
    }
    np.savez_compressed(
        out_path,
        output=output.detach().float().cpu().numpy(),
        metadata=json.dumps(metadata),
        **activations,
    )
    print(f"wrote {out_path} output_shape={tuple(output.shape)} activations={sorted(activations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
