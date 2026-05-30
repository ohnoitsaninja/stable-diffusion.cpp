#!/usr/bin/env python3
"""Export a real Lens-Turbo block-0 oracle fixture for native sd.cpp parity.

This intentionally loads only the Lens transformer and a precomputed
Lens conditioning bundle. It does not load GPT-OSS and must not be used as a
generation-path transformer trace; the exported expected tensors are parity
oracles for one native block forward.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import struct
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens-src", default=r"F:\Paralol\local\Lens\lens")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cond", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--full-transformer-oracle", action="store_true")
    parser.add_argument("--checkpoint-blocks", default="0,1,2,4,8,16,32,47")
    return parser.parse_args()


def cuda_mem(label: str) -> dict[str, float] | None:
    if not torch.cuda.is_available():
        print(f"[vram] {label}: cuda unavailable")
        return None
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    used = total - free
    snapshot = {
        "used_gib": used / 1024**3,
        "total_gib": total / 1024**3,
        "free_gib": free / 1024**3,
    }
    print(
        f"[vram] {label}: "
        f"{snapshot['used_gib']:.2f}/{snapshot['total_gib']:.2f} GiB "
        f"({snapshot['free_gib']:.2f} free)"
    )
    return snapshot


def import_lens_transformer(lens_src: str):
    transformer_path = Path(lens_src) / "transformer.py"
    spec = importlib.util.spec_from_file_location("lens_transformer_direct", transformer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {transformer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_cond(cond_path: Path, device: torch.device, dtype: torch.dtype) -> tuple[list[torch.Tensor], torch.Tensor]:
    features: list[torch.Tensor] = []
    with safe_open(cond_path, framework="pt", device="cpu") as handle:
        for index in range(4):
            name = f"feature_{index}"
            if name not in handle.keys():
                raise KeyError(f"conditioning bundle missing {name}")
            features.append(handle.get_tensor(name).to(device=device, dtype=dtype))
        if "attention_mask" not in handle.keys():
            raise KeyError("conditioning bundle missing attention_mask")
        mask = handle.get_tensor("attention_mask").to(device=device).bool()
    base_shape = tuple(features[0].shape)
    if len(base_shape) != 3 or base_shape[0] != 1 or base_shape[2] != 2880:
        raise ValueError(f"feature_0 must be [1,S,2880], got {base_shape}")
    for index, feature in enumerate(features[1:], start=1):
        if tuple(feature.shape) != base_shape:
            raise ValueError(f"feature_{index} shape {tuple(feature.shape)} does not match {base_shape}")
    if tuple(mask.shape) != base_shape[:2]:
        raise ValueError(f"attention_mask shape {tuple(mask.shape)} does not match features {base_shape[:2]}")
    return features, mask


def load_block_state(transformer_dir: Path, block_index: int) -> dict[str, torch.Tensor]:
    prefix = f"transformer_blocks.{block_index}."
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(transformer_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    state[key[len(prefix) :]] = handle.get_tensor(key).to(dtype=torch.float32)
    if not state:
        raise KeyError(f"missing transformer block tensors for {prefix}")
    return state


def load_transformer_tensor(transformer_dir: Path, name: str) -> torch.Tensor:
    for shard in sorted(transformer_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            if name in handle.keys():
                return handle.get_tensor(name).to(dtype=torch.float32)
    raise KeyError(f"missing transformer tensor {name}")


def final_ada_norm(hidden: torch.Tensor, temb: torch.Tensor, transformer_dir: Path) -> torch.Tensor:
    weight = load_transformer_tensor(transformer_dir, "norm_out.linear.weight")
    bias = load_transformer_tensor(transformer_dir, "norm_out.linear.bias")
    emb = F.linear(F.silu(temb), weight, bias)
    scale, shift = torch.chunk(emb, 2, dim=1)
    normed = F.layer_norm(hidden, (hidden.shape[-1],), eps=1e-6)
    return normed * (1.0 + scale[:, None, :]) + shift[:, None, :]


def final_projection(hidden: torch.Tensor, transformer_dir: Path) -> torch.Tensor:
    weight = load_transformer_tensor(transformer_dir, "proj_out.weight")
    bias = load_transformer_tensor(transformer_dir, "proj_out.bias")
    return F.linear(hidden, weight, bias)


def write_tensor(out, name: str, tensor: torch.Tensor) -> None:
    tensor = tensor.detach().cpu().contiguous().to(torch.float32)
    encoded = name.encode("utf-8")
    out.write(struct.pack("<II", len(encoded), tensor.ndim))
    out.write(encoded)
    for dim in tensor.shape:
        out.write(struct.pack("<q", int(dim)))
    out.write(tensor.numpy().tobytes(order="C"))


def compute_empirical_mu(image_seq_len: int, steps: int) -> float:
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


def lens_turbo_timesteps(model_dir: Path, image_seq_len: int, steps: int) -> list[float]:
    scheduler_path = model_dir / "scheduler" / "scheduler_config.json"
    config = json.loads(scheduler_path.read_text(encoding="utf-8")) if scheduler_path.exists() else {}
    num_train_timesteps = int(config.get("num_train_timesteps", 1000))
    use_dynamic = bool(config.get("use_dynamic_shifting", True))
    shift = float(config.get("shift", 1.0))
    mu = compute_empirical_mu(image_seq_len, steps)
    values: list[float] = []
    for index in range(steps):
        sigma = 1.0 if steps == 1 else 1.0 + (1.0 / float(steps) - 1.0) * (float(index) / float(steps - 1))
        if use_dynamic:
            sigma = flow_time_shift(mu, sigma)
        else:
            sigma = shift * sigma / (1.0 + (shift - 1.0) * sigma)
        values.append(sigma * float(num_train_timesteps) / float(num_train_timesteps))
    return values


def main() -> int:
    args = parse_args()
    if args.height != 256 or args.width != 256:
        raise ValueError("this block0 oracle checkpoint is intentionally limited to 256x256")
    if args.height % 16 or args.width % 16:
        raise ValueError("height and width must be divisible by 16")
    checkpoint_blocks = {int(x) for x in args.checkpoint_blocks.split(",") if x.strip()}
    if any(x < 0 or x > 47 for x in checkpoint_blocks):
        raise ValueError("--checkpoint-blocks entries must be in [0,47]")

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir)
    transformer_dir = model_dir / "transformer"
    cond_path = Path(args.cond)
    latent_h = args.height // 16
    latent_w = args.width // 16
    image_seq_len = latent_h * latent_w

    module = import_lens_transformer(args.lens_src)
    vram_log: dict[str, dict[str, float] | None] = {}
    vram_log["before_transformer_load"] = cuda_mem("before transformer load")
    start = time.perf_counter()
    transformer = module.LensTransformer2DModel.from_pretrained(
        model_dir,
        subfolder="transformer",
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)
    transformer.eval()
    vram_log["after_transformer_load"] = cuda_mem("after transformer load")

    features, mask = load_cond(cond_path, device, dtype)
    text_seq_len = int(mask.shape[1])
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    packed_latents = torch.randn((1, image_seq_len, 128), generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)
    timesteps = lens_turbo_timesteps(model_dir, image_seq_len, args.steps)
    timestep = torch.tensor([timesteps[0]], device=device, dtype=dtype)

    with torch.no_grad():
        hidden = transformer.img_in(packed_latents)
        normed = [transformer.txt_norm[i](features[i]) for i in range(4)]
        encoder = transformer.txt_in(torch.cat(normed, dim=-1))
        temb = transformer.time_text_embed(timestep, hidden)
        img_freqs, txt_freqs = transformer.pos_embed([(1, latent_h, latent_w)], [text_seq_len], device=device)
        attention_mask = transformer._build_joint_attention_mask(mask, image_seq_len)
        packed_latents_cpu = packed_latents.detach().cpu().float()
        hidden_cpu = hidden.detach().cpu().float()
        encoder_cpu = encoder.detach().cpu().float()
        temb_cpu = temb.detach().cpu().float()
        img_freqs_cpu = img_freqs.detach().cpu()
        txt_freqs_cpu = txt_freqs.detach().cpu()
        attention_mask_cpu = attention_mask.detach().cpu().float()
    del transformer, features, mask, packed_latents, hidden, encoder, temb, img_freqs, txt_freqs, attention_mask
    if device.type == "cuda":
        torch.cuda.empty_cache()
    vram_log["after_transformer_input_export"] = cuda_mem("after transformer input export")

    block = module.LensTransformerBlock(
        dim=1536,
        num_attention_heads=24,
        attention_head_dim=64,
        rms_norm=True,
        gate_mlp=True,
    ).eval()
    block.load_state_dict(load_block_state(transformer_dir, 0), strict=True)
    with torch.no_grad():
        expected_encoder, expected_hidden = block(
            hidden_states=hidden_cpu,
            encoder_hidden_states=encoder_cpu,
            temb=temb_cpu,
            image_rotary_emb=(img_freqs_cpu, txt_freqs_cpu),
            attention_mask=attention_mask_cpu,
        )
    full_tensors: dict[str, torch.Tensor] = {}
    if args.full_transformer_oracle:
        current_hidden = hidden_cpu
        current_encoder = encoder_cpu
        total_block_weight_bytes = 0
        for block_index in range(48):
            block = module.LensTransformerBlock(
                dim=1536,
                num_attention_heads=24,
                attention_head_dim=64,
                rms_norm=True,
                gate_mlp=True,
            ).eval()
            state = load_block_state(transformer_dir, block_index)
            total_block_weight_bytes += sum(t.numel() * t.element_size() for t in state.values())
            block.load_state_dict(state, strict=True)
            with torch.no_grad():
                current_encoder, current_hidden = block(
                    hidden_states=current_hidden,
                    encoder_hidden_states=current_encoder,
                    temb=temb_cpu,
                    image_rotary_emb=(img_freqs_cpu, txt_freqs_cpu),
                    attention_mask=attention_mask_cpu,
                )
            if block_index in checkpoint_blocks:
                full_tensors[f"checkpoint.block_{block_index}.encoder"] = current_encoder
                full_tensors[f"checkpoint.block_{block_index}.hidden"] = current_hidden
            del block, state
        with torch.no_grad():
            prediction = final_projection(final_ada_norm(current_hidden, temb_cpu, transformer_dir), transformer_dir)
        full_tensors["expected.prediction"] = prediction
    else:
        total_block_weight_bytes = 0
    elapsed = time.perf_counter() - start

    tensors = {
        "input.packed_latents": packed_latents_cpu,
        "input.hidden": hidden_cpu,
        "input.encoder": encoder_cpu,
        "input.temb": temb_cpu,
        "input.img_freqs": torch.view_as_real(img_freqs_cpu),
        "input.txt_freqs": torch.view_as_real(txt_freqs_cpu),
        "input.attention_mask": attention_mask_cpu.reshape(1, image_seq_len + text_seq_len),
        "expected.encoder": expected_encoder,
        "expected.hidden": expected_hidden,
    }
    tensors.update(full_tensors)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as out:
        out.write(b"LENSBLK1")
        out.write(struct.pack("<I", len(tensors)))
        for name, tensor in tensors.items():
            write_tensor(out, name, tensor)

    tensor_names = sorted(tensors)
    del tensors
    del full_tensors
    del packed_latents_cpu, hidden_cpu, encoder_cpu, temb_cpu, img_freqs_cpu, txt_freqs_cpu, attention_mask_cpu
    del expected_encoder, expected_hidden
    if device.type == "cuda":
        torch.cuda.empty_cache()
    vram_log["after_cleanup"] = cuda_mem("after cleanup")

    manifest = {
        "schema": "lens_full_oracle_v1" if args.full_transformer_oracle else "lens_block0_oracle_v1",
        "model_dir": str(model_dir.resolve()),
        "transformer_dir": str(transformer_dir.resolve()),
        "conditioning": str(cond_path.resolve()),
        "height": args.height,
        "width": args.width,
        "image_seq_len": image_seq_len,
        "text_seq_len": text_seq_len,
        "seed": args.seed,
        "dtype": args.dtype,
        "device": str(device),
        "steps": args.steps,
        "timesteps": timesteps,
        "scheduler": "lens_compute_empirical_mu_custom_turbo_sigmas",
        "runtime_seconds": elapsed,
        "tensor_names": tensor_names,
        "full_transformer_oracle": args.full_transformer_oracle,
        "checkpoint_blocks": sorted(checkpoint_blocks),
        "total_block_weight_bytes_for_oracle": total_block_weight_bytes,
        "vram": vram_log,
    }
    manifest_path = Path(args.manifest) if args.manifest else output.with_suffix(output.suffix + ".json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {output} image_seq_len={image_seq_len} text_seq_len={text_seq_len} "
        f"dtype={args.dtype} runtime_seconds={elapsed:.3f} manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
