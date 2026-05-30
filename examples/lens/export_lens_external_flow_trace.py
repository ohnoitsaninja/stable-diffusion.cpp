#!/usr/bin/env python3
"""Export Lens transformer model outputs for sd.cpp external-flow replay.

This script loads the Lens transformer but not the text encoder or VAE. It
consumes a precomputed lens_cond_v1 bundle, runs the real Lens transformer for
each Turbo step, updates latents with the Diffusers scheduler, and writes the
initial packed latents plus per-step model outputs in LENSBLK1 format.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens-src", default=r"F:\Paralol\local\Lens")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cond", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--work-dir", help="Directory for per-step trace shards; defaults to <output>.parts")
    parser.add_argument("--resume", action="store_true", help="Reuse completed per-step shards when present")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def cuda_note(label: str) -> None:
    if not torch.cuda.is_available():
        print(f"{label}: cuda unavailable")
        return
    free, total = torch.cuda.mem_get_info()
    print(f"{label}: cuda {(total - free) / 1024**3:.2f}/{total / 1024**3:.2f} GiB")


def import_transformer(lens_root: Path):
    transformer_path = lens_root / "lens" / "transformer.py"
    spec = importlib.util.spec_from_file_location("lens_transformer_direct", transformer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {transformer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.LensTransformer2DModel


def load_conditioning(path: Path, device: torch.device, dtype: torch.dtype) -> tuple[list[torch.Tensor], torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        features = [handle.get_tensor(f"feature_{i}").to(device=device, dtype=dtype) for i in range(4)]
        mask = handle.get_tensor("attention_mask").bool().to(device=device)
    return features, mask


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


def write_tensor(out, name: str, tensor: torch.Tensor) -> None:
    tensor = tensor.detach().cpu().contiguous().to(torch.float32)
    encoded = name.encode("utf-8")
    out.write(struct.pack("<II", len(encoded), tensor.ndim))
    out.write(encoded)
    for dim in tensor.shape:
        out.write(struct.pack("<q", int(dim)))
    out.write(tensor.numpy().tobytes(order="C"))


def save_npy(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, tensor.detach().cpu().contiguous().float().numpy())


def load_npy(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.load(path)).to(torch.float32)


def unpatchify_public_latent(tokens: torch.Tensor, latent_h: int, latent_w: int) -> torch.Tensor:
    bsz, seq, channels = tokens.shape
    if bsz != 1 or seq != latent_h * latent_w or channels != 128:
        raise ValueError(f"expected packed tokens [1,{latent_h * latent_w},128], got {tuple(tokens.shape)}")
    latents = tokens.reshape(1, latent_h, latent_w, 32, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(1, 32, latent_h * 2, latent_w * 2).contiguous()


def main() -> int:
    start_time = time.perf_counter()
    args = parse_args()
    if args.height % 16 or args.width % 16:
        raise ValueError("height and width must be divisible by 16")
    latent_h = args.height // 16
    latent_w = args.width // 16
    image_seq_len = latent_h * latent_w

    lens_root = Path(args.lens_src)
    sys.path.insert(0, str(lens_root))
    dtype = torch_dtype(args.dtype)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model_dir = Path(args.model_dir)
    output = Path(args.output)
    work_dir = Path(args.work_dir) if args.work_dir else output.with_suffix(output.suffix + ".parts")
    work_dir.mkdir(parents=True, exist_ok=True)

    Transformer = import_transformer(lens_root)
    features, mask = load_conditioning(Path(args.cond), device, dtype)

    cuda_note("before transformer load")
    transformer = Transformer.from_pretrained(model_dir, subfolder="transformer", torch_dtype=dtype, local_files_only=True)
    transformer.to(device)
    transformer.eval()
    cuda_note("after transformer load")

    initial_path = work_dir / "initial.npy"
    if args.resume and initial_path.exists():
        initial = load_npy(initial_path)
        latents = initial.to(device=device, dtype=dtype)
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        latents = torch.randn((1, image_seq_len, 128), generator=generator, device=device, dtype=dtype)
        initial = latents.detach().float().cpu()
        save_npy(initial_path, initial)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler", local_files_only=True)
    mu = compute_empirical_mu(image_seq_len, args.steps)
    sigmas = np.linspace(1.0, 1.0 / args.steps, args.steps)
    scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)

    outputs = []
    img_shapes = [(1, latent_h, latent_w)]
    with torch.no_grad():
        for step, timestep in enumerate(scheduler.timesteps):
            step_path = work_dir / f"model_output_step_{step:02d}.npy"
            sample_path = work_dir / f"sample_after_step_{step:02d}.npy"
            if args.resume and step_path.exists() and sample_path.exists():
                noise_cpu = load_npy(step_path)
                outputs.append(noise_cpu)
                latents = load_npy(sample_path).to(device=device, dtype=dtype)
                print(f"trace step={step} resumed=true shape={tuple(noise_cpu.shape)}")
                continue
            model_timestep = timestep.expand(1).to(dtype=latents.dtype) / 1000
            noise = transformer(
                hidden_states=latents,
                encoder_hidden_states=features,
                encoder_hidden_states_mask=mask,
                timestep=model_timestep,
                img_shapes=img_shapes,
            )
            noise_cpu = noise.detach().float().cpu()
            outputs.append(noise_cpu)
            save_npy(step_path, noise_cpu)
            latents = scheduler.step(noise, timestep, latents, return_dict=False)[0]
            save_npy(sample_path, latents.detach().float().cpu())
            max_abs = float(latents.detach().float().abs().max().item())
            mean_abs = float(latents.detach().float().abs().mean().item())
            print(
                f"trace step={step} timestep={float(timestep):.6f} sigma={float(scheduler.sigmas[step]):.6f} "
                f"max_sample={max_abs:.6g} mean_sample={mean_abs:.6g}"
            )
            del noise, noise_cpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    final_packed = latents.detach().float().cpu()
    save_npy(work_dir / "final_packed.npy", final_packed)
    public_latent = unpatchify_public_latent(final_packed, latent_h, latent_w)
    save_npy(work_dir / "final_public_latent.npy", public_latent)
    model_outputs = torch.stack(outputs, dim=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    sigmas_tensor = torch.tensor(scheduler.sigmas.detach().cpu().numpy(), dtype=torch.float32)
    timesteps_tensor = torch.tensor(scheduler.timesteps.detach().cpu().numpy(), dtype=torch.float32)
    tensors = {
        "external.initial": initial,
        "external.model_outputs": model_outputs,
        "external.final_packed_python": final_packed,
        "external.public_latent_python": public_latent,
        "external.sigmas": sigmas_tensor,
        "external.timesteps": timesteps_tensor,
        "external.height_width": torch.tensor([args.height, args.width], dtype=torch.float32),
        "external.schedule_kind": torch.tensor([1], dtype=torch.float32),
    }
    with output.open("wb") as out:
        out.write(b"LENSBLK1")
        out.write(struct.pack("<I", len(tensors)))
        for name, tensor in tensors.items():
            write_tensor(out, name, tensor)

    manifest = {
        "schema": "lens_external_flow_trace_v1",
        "height": args.height,
        "width": args.width,
        "latent_h": latent_h,
        "latent_w": latent_w,
        "image_seq_len": image_seq_len,
        "steps": args.steps,
        "seed": args.seed,
        "mu": mu,
        "sigmas": [float(x) for x in scheduler.sigmas.detach().cpu().numpy().tolist()],
        "timesteps": [float(x) for x in scheduler.timesteps.detach().cpu().numpy().tolist()],
        "conditioning": str(Path(args.cond).resolve()),
        "schedule_kind": "lens_pipeline_empirical_mu_custom_turbo_sigmas",
        "runtime_seconds": time.perf_counter() - start_time,
        "work_dir": str(work_dir.resolve()),
    }
    manifest_path = Path(args.manifest) if args.manifest else output.with_suffix(output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    del transformer, features, mask, latents, outputs, model_outputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    cuda_note("after transformer cleanup")
    print(
        f"wrote {output} model_outputs={tuple(tensors['external.model_outputs'].shape)} "
        f"final={tuple(final_packed.shape)} public_latent={tuple(public_latent.shape)} "
        f"runtime_seconds={time.perf_counter() - start_time:.3f} manifest={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
