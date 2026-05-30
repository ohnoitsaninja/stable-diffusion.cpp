#!/usr/bin/env python3
"""Export source-backed Lens first-image inputs for sd.cpp smoke replay.

This does not load the GPT-OSS text encoder, VAE, or the full Lens transformer.
It reads only the small timestep-embedding weights from transformer shards,
computes Lens RoPE from the local Lens source, and writes a LENSBLK1 fixture
that `sd-lens-transformer-smoke` can consume instead of synthetic placeholders.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens-src", default=r"F:\Paralol\local\Lens\lens")
    parser.add_argument("--transformer-dir", required=True)
    parser.add_argument("--cond", required=True, help="lens_cond_v1 safetensors bundle")
    parser.add_argument("--output", required=True, help="Output LENSBLK1 fixture path")
    parser.add_argument("--manifest", help="Optional JSON manifest path")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--vae-scale-factor", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timestep", type=float, default=1.0, help="Transformer timestep after /1000 scaling")
    parser.add_argument("--steps", type=int, default=4, help="Export per-step Lens timestep embeddings")
    return parser.parse_args()


def import_lens_transformer(lens_src: str):
    transformer_path = Path(lens_src) / "transformer.py"
    spec = importlib.util.spec_from_file_location("lens_transformer_direct", transformer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {transformer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_tensor(transformer_dir: Path, name: str) -> torch.Tensor:
    for shard in sorted(transformer_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            if name in handle.keys():
                return handle.get_tensor(name)
    raise KeyError(f"missing transformer tensor {name}")


def load_cond_mask(cond_path: Path) -> torch.Tensor:
    with safe_open(cond_path, framework="pt", device="cpu") as handle:
        if "feature_0" not in handle.keys() or "attention_mask" not in handle.keys():
            raise KeyError("conditioning bundle must contain feature_0 and attention_mask")
        feature_0 = handle.get_tensor("feature_0")
        mask = handle.get_tensor("attention_mask")
    if feature_0.ndim != 3 or feature_0.shape[0] != 1 or feature_0.shape[2] != 2880:
        raise ValueError(f"feature_0 must have shape [1,S,2880], got {tuple(feature_0.shape)}")
    if mask.shape != feature_0.shape[:2]:
        raise ValueError(f"attention_mask shape {tuple(mask.shape)} does not match feature_0 {tuple(feature_0.shape)}")
    return mask.bool()


def write_tensor(out, name: str, tensor: torch.Tensor) -> None:
    tensor = tensor.detach().cpu().contiguous().to(torch.float32)
    encoded = name.encode("utf-8")
    out.write(struct.pack("<II", len(encoded), tensor.ndim))
    out.write(encoded)
    for dim in tensor.shape:
        out.write(struct.pack("<q", int(dim)))
    out.write(tensor.numpy().tobytes(order="C"))


def calculate_shift(image_seq_len: int, steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * steps + b)


def flow_time_shift(mu: float, sigma: float) -> float:
    if sigma <= 0.0:
        return 0.0
    if sigma >= 1.0:
        return 1.0
    exp_mu = math.exp(mu)
    return exp_mu / (exp_mu + ((1.0 / sigma - 1.0) ** 1.0))


def lens_timesteps(model_root: Path, image_seq_len: int, steps: int) -> list[float]:
    if steps <= 0:
        raise ValueError("--steps must be positive")
    scheduler_path = model_root / "scheduler" / "scheduler_config.json"
    config = json.loads(scheduler_path.read_text(encoding="utf-8")) if scheduler_path.exists() else {}
    num_train_timesteps = int(config.get("num_train_timesteps", 1000))
    use_dynamic = bool(config.get("use_dynamic_shifting", True))
    shift = float(config.get("shift", 1.0))
    mu = calculate_shift(image_seq_len, steps)
    values = []
    for index in range(steps):
        sigma = 1.0
        if steps > 1:
            t = float(index) / float(steps - 1)
            sigma = 1.0 + (1.0 / float(steps) - 1.0) * t
        if use_dynamic:
            sigma = flow_time_shift(mu, sigma)
        else:
            sigma = shift * sigma / (1.0 + (shift - 1.0) * sigma)
        values.append(sigma * float(num_train_timesteps) / float(num_train_timesteps))
    return values


def main() -> int:
    args = parse_args()
    if args.height <= 0 or args.width <= 0:
        raise ValueError("height and width must be positive")
    if args.height % args.vae_scale_factor or args.width % args.vae_scale_factor:
        raise ValueError("height and width must be divisible by --vae-scale-factor")

    latent_h = args.height // args.vae_scale_factor
    latent_w = args.width // args.vae_scale_factor
    image_seq_len = latent_h * latent_w
    if image_seq_len <= 0:
        raise ValueError("computed image sequence length is zero")

    module = import_lens_transformer(args.lens_src)
    transformer_dir = Path(args.transformer_dir)
    cond_path = Path(args.cond)
    cond_mask = load_cond_mask(cond_path)
    text_seq_len = int(cond_mask.shape[1])

    timestep_embedder = module.LensTimestepProjEmbeddings(embedding_dim=1536).eval()
    timestep_embedder.load_state_dict(
        {
            "timestep_embedder.linear_1.weight": load_tensor(transformer_dir, "time_text_embed.timestep_embedder.linear_1.weight"),
            "timestep_embedder.linear_1.bias": load_tensor(transformer_dir, "time_text_embed.timestep_embedder.linear_1.bias"),
            "timestep_embedder.linear_2.weight": load_tensor(transformer_dir, "time_text_embed.timestep_embedder.linear_2.weight"),
            "timestep_embedder.linear_2.bias": load_tensor(transformer_dir, "time_text_embed.timestep_embedder.linear_2.bias"),
        },
        strict=True,
    )
    rope = module.LensEmbedRope(theta=10000, axes_dim=[8, 28, 28], scale_rope=True)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    packed_latents = torch.randn((1, image_seq_len, 128), generator=generator, dtype=torch.float32)
    step_timesteps = lens_timesteps(transformer_dir.parent, image_seq_len, args.steps)
    if args.timestep != 1.0:
        step_timesteps[0] = args.timestep
    with torch.no_grad():
        tembs = [
            timestep_embedder(
                torch.tensor([step_timestep], dtype=torch.float32),
                packed_latents.new_zeros((1, image_seq_len, 1536)),
            )
            for step_timestep in step_timesteps
        ]
        img_freqs, txt_freqs = rope([(1, latent_h, latent_w)], [text_seq_len], device=torch.device("cpu"))

    attention_mask = torch.zeros((1, image_seq_len + text_seq_len), dtype=torch.float32)
    attention_mask[:, image_seq_len:] = torch.where(
        cond_mask,
        torch.zeros_like(cond_mask, dtype=torch.float32),
        torch.full_like(cond_mask, float("-inf"), dtype=torch.float32),
    )

    tensors = {
        "full.input.hidden": packed_latents,
        "full.input.temb": tembs[0],
        "full.input.img_freqs": torch.view_as_real(img_freqs),
        "full.input.txt_freqs": torch.view_as_real(txt_freqs),
        "full.input.attention_mask": attention_mask,
    }
    for index, temb in enumerate(tembs):
        tensors[f"full.input.temb_{index}"] = temb

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as out:
        out.write(b"LENSBLK1")
        out.write(struct.pack("<I", len(tensors)))
        for name, tensor in tensors.items():
            write_tensor(out, name, tensor)

    manifest = {
        "schema": "lens_first_image_fixture_v1",
        "height": args.height,
        "width": args.width,
        "vae_scale_factor": args.vae_scale_factor,
        "latent_h": latent_h,
        "latent_w": latent_w,
        "image_seq_len": image_seq_len,
        "text_seq_len": text_seq_len,
        "seed": args.seed,
        "timesteps": step_timesteps,
        "steps": args.steps,
        "conditioning": str(cond_path.resolve()),
        "transformer_dir": str(transformer_dir.resolve()),
        "tensor_names": sorted(tensors),
    }
    manifest_path = Path(args.manifest) if args.manifest else output.with_suffix(output.suffix + ".json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        f"wrote {output} image_seq_len={image_seq_len} text_seq_len={text_seq_len} "
        f"height={args.height} width={args.width} seed={args.seed} manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
