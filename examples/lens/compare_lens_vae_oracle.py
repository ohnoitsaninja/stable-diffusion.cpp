#!/usr/bin/env python3
"""Decode a Lens/Flux2 VAE latent with Diffusers and compare against native output.

The latent must be a NumPy f32 NCHW tensor with shape 1x32xHxW. The native
`sd-lens-vae-smoke` path accepts that same tensor, packs it to the Flux2 VAE
layout internally, and writes a PNG. This script is intentionally VAE-only so it
does not load the Lens text encoder or transformer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLFlux2
from PIL import Image


def pack_lens_latent(latent: torch.Tensor) -> torch.Tensor:
    """Pack 1x32xHxW Lens public latent to Flux2 BN layout 1x128xH/2xW/2."""
    if latent.ndim != 4 or latent.shape[0] != 1 or latent.shape[1] != 32:
        raise ValueError(f"expected 1x32xHxW latent, got {tuple(latent.shape)}")
    if latent.shape[2] % 2 != 0 or latent.shape[3] % 2 != 0:
        raise ValueError(f"latent H/W must be even, got {tuple(latent.shape)}")
    n, c, h, w = latent.shape
    packed = torch.empty((n, c * 4, h // 2, w // 2), dtype=latent.dtype, device=latent.device)
    for oy in range(2):
        for ox in range(2):
            packed[:, oy * 2 + ox :: 4, :, :] = latent[:, :, oy::2, ox::2]
    return packed


def unpack_flux2_latent(packed: torch.Tensor) -> torch.Tensor:
    """Unpack Flux2 VAE BN layout 1x128xH/2xW/2 to decoder layout 1x32xHxW."""
    if packed.ndim != 4 or packed.shape[0] != 1 or packed.shape[1] != 128:
        raise ValueError(f"expected 1x128xHxW packed latent, got {tuple(packed.shape)}")
    n, cp, h, w = packed.shape
    c = cp // 4
    latent = torch.empty((n, c, h * 2, w * 2), dtype=packed.dtype, device=packed.device)
    for oy in range(2):
        for ox in range(2):
            latent[:, :, oy::2, ox::2] = packed[:, oy * 2 + ox :: 4, :, :]
    return latent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-dir", required=True, help="Lens-Turbo root or local folder containing vae/")
    parser.add_argument("--latent-npy", required=True, help="f32 NCHW latent with shape 1x32xHxW")
    parser.add_argument("--native-png", required=True, help="PNG produced by sd-lens-vae-smoke")
    parser.add_argument("--python-png", required=True, help="Output PNG for Diffusers decode")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def to_u8_image(sample: torch.Tensor) -> np.ndarray:
    image = ((sample[0].permute(1, 2, 0) + 1.0) * 0.5).clamp(0, 1).mul(255).add(0.5)
    return image.to(torch.uint8).cpu().numpy()


def main() -> int:
    args = parse_args()
    latent = np.load(args.latent_npy)
    if latent.dtype != np.float32 or latent.shape[0] != 1 or latent.shape[1] != 32:
        raise ValueError(f"expected latent shape 1x32xHxW float32, got {latent.shape} {latent.dtype}")

    device = torch.device(args.device)
    vae = AutoencoderKLFlux2.from_pretrained(args.vae_dir, subfolder="vae", torch_dtype=torch.float32)
    vae.to(device)
    vae.eval()

    z = torch.from_numpy(latent).to(device)
    with torch.no_grad():
        packed = pack_lens_latent(z)
        bn_std = torch.sqrt(vae.bn.running_var.to(device) + vae.bn.eps).view(1, -1, 1, 1)
        bn_mean = vae.bn.running_mean.to(device).view(1, -1, 1, 1)
        decoder_latent = unpack_flux2_latent(packed * bn_std + bn_mean)
        decoded = vae.decode(decoder_latent).sample
    python_image = to_u8_image(decoded)

    python_png = Path(args.python_png)
    python_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(python_image).save(python_png)

    native_image = np.asarray(Image.open(args.native_png).convert("RGB"))
    if native_image.shape != python_image.shape:
        raise ValueError(f"image shape mismatch: native={native_image.shape}, python={python_image.shape}")

    delta = np.abs(native_image.astype(np.int16) - python_image.astype(np.int16))
    print(
        "Lens VAE oracle compare:",
        f"shape={python_image.shape}",
        f"max_abs_u8={int(delta.max())}",
        f"mean_abs_u8={float(delta.mean()):.6f}",
        f"nonzero={int((delta != 0).sum())}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
