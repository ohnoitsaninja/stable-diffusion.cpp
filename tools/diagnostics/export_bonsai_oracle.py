#!/usr/bin/env python3
"""Capture tensors from Prism's Bonsai GemLite/HQQ Python runtime.

This is an external diagnostic oracle only. stable-diffusion.cpp must not
depend on Python, Triton, GemLite, or HQQ at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def tensor_stats(t: Any) -> dict[str, Any]:
    import torch

    detached = t.detach()
    if detached.is_floating_point() or detached.is_complex():
        work = detached.float().cpu()
    else:
        work = detached.cpu()
    finite = torch.isfinite(work.float()) if work.numel() else torch.zeros_like(work, dtype=torch.bool)
    finite_values = work.float()[finite]
    stats: dict[str, Any] = {
        "shape": list(work.shape),
        "dtype": str(detached.dtype),
        "count": int(work.numel()),
        "finite": int(finite.sum().item()),
        "nan": int(torch.isnan(work.float()).sum().item()) if work.numel() else 0,
        "inf": int(torch.isinf(work.float()).sum().item()) if work.numel() else 0,
    }
    if finite_values.numel() > 0:
        stats.update(
            {
                "min": float(finite_values.min().item()),
                "max": float(finite_values.max().item()),
                "mean": float(finite_values.mean().item()),
                "std": float(finite_values.std(unbiased=False).item()),
            }
        )
    else:
        stats.update({"min": math.nan, "max": math.nan, "mean": math.nan, "std": math.nan})
    return stats


class Capture:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.meta: dict[str, Any] = {"tensors": {}}

    def save_tensor(self, name: str, t: Any) -> None:
        import numpy as np
        import torch

        if not torch.is_tensor(t):
            return
        safe = _safe_name(name)
        arr = t.detach().float().cpu().numpy() if t.detach().is_floating_point() else t.detach().cpu().numpy()
        npy = self.out_dir / f"{safe}.npy"
        np.save(npy, arr)
        self.meta["tensors"][name] = {"npy": str(npy), **tensor_stats(t)}

    def save_text(self, name: str, value: str) -> None:
        path = self.out_dir / f"{_safe_name(name)}.txt"
        path.write_text(value, encoding="utf-8")
        self.meta[name] = str(path)


def _find_subdir(root: Path, *hints: str) -> Path:
    matches = [p for p in root.iterdir() if p.is_dir() and any(h in p.name for h in hints)]
    if not matches:
        present = ", ".join(sorted(p.name for p in root.iterdir() if p.is_dir())) or "(empty)"
        raise FileNotFoundError(f"No subdir matching {hints!r} under {root}. Present: {present}")
    matches.sort(key=lambda p: len(p.name), reverse=True)
    return matches[0]


def _first_tensor(obj: Any) -> Any | None:
    import torch

    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (list, tuple)):
        for item in obj:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(obj, dict):
        for item in obj.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _top_level_tensors(obj: Any) -> list[Any]:
    import torch

    if torch.is_tensor(obj):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return [item for item in obj if torch.is_tensor(item)]
    if isinstance(obj, dict):
        return [item for item in obj.values() if torch.is_tensor(item)]
    return []


def _install_hooks(transformer: Any, capture: Capture) -> dict[str, Any]:
    """Capture first-call tensors around block 0 and one dominant single block."""
    import torch

    handles = []
    seen: set[str] = set()
    module_names = dict(transformer.named_modules())
    capture.meta["available_hook_name_samples"] = [
        name
        for name in module_names
        if name.startswith("transformer_blocks.0")
        or name.startswith("single_transformer_blocks.0")
        or name.startswith("single_transformer_blocks.12")
    ][:240]

    def save_once(name: str, value: Any) -> None:
        if name in seen:
            return
        tensor = _first_tensor(value)
        if tensor is None:
            return
        seen.add(name)
        capture.save_tensor(name, tensor)

    def pre_hook(label: str):
        def _hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            for idx, item in enumerate(args):
                save_once(f"{label}.input_arg{idx}", item)
            for key, item in kwargs.items():
                save_once(f"{label}.input_{key}", item)

        return _hook

    def post_hook(label: str):
        def _hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
            del args, kwargs
            save_once(f"{label}.output", output)
            tensors = _top_level_tensors(output)
            if len(tensors) > 1:
                for idx, tensor in enumerate(tensors):
                    save_once(f"{label}.output_{idx}", tensor)

        return _hook

    wanted = [
        "x_embedder",
        "context_embedder",
        "time_text_embed",
        "time_text_embed.timestep_embedder",
        "time_text_embed.guidance_embedder",
        "transformer_blocks.0",
        "transformer_blocks.0.norm1",
        "transformer_blocks.0.norm1_context",
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_k",
        "transformer_blocks.0.attn.to_v",
        "transformer_blocks.0.attn.add_q_proj",
        "transformer_blocks.0.attn.add_k_proj",
        "transformer_blocks.0.attn.add_v_proj",
        "transformer_blocks.0.ff.net.0.proj",
        "transformer_blocks.0.ff.net.2",
        "transformer_blocks.0.ff_context.net.0.proj",
        "transformer_blocks.0.ff_context.net.2",
        "single_transformer_blocks.0",
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj",
        "single_transformer_blocks.0.proj_out",
        "single_transformer_blocks.12.attn.to_qkv_mlp_proj",
        "single_transformer_blocks.12.proj_out",
        "norm_out",
        "proj_out",
    ]

    installed: list[str] = []
    for name in wanted:
        module = module_names.get(name)
        if module is None:
            continue
        installed.append(name)
        handles.append(module.register_forward_pre_hook(pre_hook(name), with_kwargs=True))
        handles.append(module.register_forward_hook(post_hook(name), with_kwargs=True))

    capture.meta["installed_hooks"] = installed
    capture.meta["missing_hooks"] = [name for name in wanted if name not in installed]
    capture.meta["hook_uses_with_kwargs"] = True
    return {"handles": handles, "seen": seen}


def run_capture(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    from PIL import Image

    demo_dir = Path(args.demo_dir).resolve()
    if str(demo_dir / "vendor" / "image-studio") not in sys.path:
        sys.path.insert(0, str(demo_dir / "vendor" / "image-studio"))
    if str(demo_dir / "vendor" / "image-studio" / "backend_gpu") not in sys.path:
        sys.path.insert(0, str(demo_dir / "vendor" / "image-studio" / "backend_gpu"))

    from backend_gpu.pipeline_gpu import GpuPipeline
    from backend_gpu import diffusion_klein
    from diffusers import Flux2Pipeline
    from diffusers.pipelines.flux2.pipeline_flux2 import retrieve_timesteps
    from gemlite.core import GemLiteLinearTriton

    model_root = Path(args.model_dir).resolve()
    transformer_dir = _find_subdir(model_root, "transformer")
    text_encoder_dir = _find_subdir(model_root, "text_encoder")
    vae_dir = _find_subdir(model_root, "vae")
    out_dir = Path(args.out_dir).resolve()
    capture = Capture(out_dir)
    capture.meta.update(
        {
            "status": "started",
            "model_root": str(model_root),
            "transformer_dir": str(transformer_dir),
            "text_encoder_dir": str(text_encoder_dir),
            "vae_dir": str(vae_dir),
            "prompt": args.prompt,
            "seed": args.seed,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "guidance": args.guidance,
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        }
    )
    if torch.cuda.is_available():
        capture.meta["cuda_device"] = torch.cuda.get_device_name(0)

    os.environ.setdefault("MFLUX_STUDIO_GPU_TERNARY_TRANSFORMER_PATH", str(transformer_dir))
    pipeline = GpuPipeline(
        backend="bonsai-binary-gemlite",
        binary_transformer_path=str(transformer_dir),
        ternary_transformer_path=os.environ["MFLUX_STUDIO_GPU_TERNARY_TRANSFORMER_PATH"],
        text_encoder_path=str(text_encoder_dir),
        vae_path=str(vae_dir),
        tokenizer_path=str(text_encoder_dir / "tokenizer"),
    )
    if args.gemlite_cache:
        cache = Path(args.gemlite_cache)
        if cache.exists():
            GemLiteLinearTriton.load_config(str(cache), print_error=False)

    t0 = time.perf_counter()
    pipeline.prewarm()
    capture.meta["prewarm_s"] = time.perf_counter() - t0
    hooks = _install_hooks(pipeline._transformer, capture)

    transformer = pipeline._transformer
    text_encoder = pipeline._text_encoder
    tokenizer = pipeline._tokenizer
    vae = pipeline._vae
    scheduler = pipeline._scheduler or diffusion_klein._build_default_scheduler()

    transformer_device = next(transformer.parameters()).device
    vae_device = next(vae.parameters()).device
    activation_dtype = getattr(transformer, "_inference_dtype", torch.float16)
    capture.meta["activation_dtype"] = str(activation_dtype)

    messages = [{"role": "user", "content": args.prompt}]
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    capture.save_text("rendered_prompt", rendered_prompt)
    inputs = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=args.max_sequence_length,
    )
    input_ids = inputs["input_ids"].to(text_encoder.device)
    attention_mask = inputs["attention_mask"].to(text_encoder.device)
    capture.save_tensor("conditioning.input_ids", input_ids)
    capture.save_tensor("conditioning.attention_mask", attention_mask)

    with torch.no_grad():
        text_output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        layer_stack = torch.stack(
            [text_output.hidden_states[k] for k in diffusion_klein.KLEIN_OUTPUT_LAYERS],
            dim=1,
        )
        prompt_embeds = (
            layer_stack.permute(0, 2, 1, 3)
            .reshape(layer_stack.shape[0], layer_stack.shape[2], layer_stack.shape[1] * layer_stack.shape[3])
        )
        capture.save_tensor("conditioning.hidden_layers_9_18_27_stacked", layer_stack)
        capture.save_tensor("conditioning.prompt_embeds_bsz_seq_7680", prompt_embeds)
        prompt_embeds_t = prompt_embeds.to(device=transformer_device, dtype=activation_dtype)
        capture.save_tensor("conditioning.prompt_embeds_transformer_dtype", prompt_embeds_t)
        text_ids = Flux2Pipeline._prepare_text_ids(prompt_embeds).to(transformer_device)
        capture.save_tensor("conditioning.text_ids", text_ids)

        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        h_lat = 2 * (int(args.height) // (vae_scale_factor * 2))
        w_lat = 2 * (int(args.width) // (vae_scale_factor * 2))
        in_channels_latents = transformer.config.in_channels // 4
        gen = torch.Generator(device="cpu").manual_seed(int(args.seed))
        noise_shape = (1, in_channels_latents * 4, h_lat // 2, w_lat // 2)
        latents_4d = torch.randn(noise_shape, generator=gen, dtype=torch.float32)
        capture.save_tensor("latents.initial_noise_cpu_f32_nchw", latents_4d)
        latents_4d = latents_4d.to(device=transformer_device, dtype=activation_dtype)
        capture.save_tensor("latents.initial_noise_device", latents_4d)
        latent_ids = Flux2Pipeline._prepare_latent_ids(latents_4d).to(transformer_device)
        capture.save_tensor("latents.latent_ids", latent_ids)
        latents = Flux2Pipeline._pack_latents(latents_4d)
        image_seq_len = latents.shape[1]
        capture.save_tensor("latents.initial_packed", latents)

        mu = diffusion_klein._mflux_empirical_mu(image_seq_len=image_seq_len, num_steps=args.steps)
        sigmas = np.linspace(1.0, 1.0 / args.steps, args.steps)
        if hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas:
            sigmas_arg = None
        else:
            sigmas_arg = sigmas
        timesteps, steps_eff = retrieve_timesteps(
            scheduler,
            args.steps,
            transformer_device,
            sigmas=sigmas_arg,
            mu=mu,
        )
        if hasattr(scheduler, "set_begin_index"):
            scheduler.set_begin_index(0)
        capture.meta["scheduler"] = {
            "mu": float(mu),
            "steps_eff": int(steps_eff),
            "input_sigmas": None if sigmas_arg is None else [float(x) for x in sigmas_arg],
            "timesteps": [float(x) for x in timesteps.detach().float().cpu().tolist()],
            "sigmas": [float(x) for x in getattr(scheduler, "sigmas", torch.empty(0)).detach().float().cpu().tolist()],
            "scheduler_class": scheduler.__class__.__name__,
            "scheduler_config": dict(scheduler.config),
        }
        capture.save_tensor("scheduler.timesteps", timesteps)
        if hasattr(scheduler, "sigmas"):
            capture.save_tensor("scheduler.sigmas", scheduler.sigmas)

        guidance_t = torch.full([1], args.guidance, device=transformer_device, dtype=torch.float32)
        guidance_t = guidance_t.expand(latents.shape[0])
        capture.save_tensor("conditioning.guidance", guidance_t)

        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0]).to(latents.dtype)
            step_prefix = f"step{i}"
            capture.save_tensor(f"{step_prefix}.timestep", t.reshape(1))
            capture.save_tensor(f"{step_prefix}.timestep_div1000_input", timestep / 1000)
            if hasattr(scheduler, "sigmas") and len(scheduler.sigmas) > i:
                capture.save_tensor(f"{step_prefix}.sigma", scheduler.sigmas[i].reshape(1))
            if hasattr(scheduler, "sigmas") and len(scheduler.sigmas) > i + 1:
                capture.save_tensor(f"{step_prefix}.sigma_next", scheduler.sigmas[i + 1].reshape(1))
            capture.save_tensor(f"{step_prefix}.latents_before_model", latents)
            noise_pred = transformer(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance_t,
                encoder_hidden_states=prompt_embeds_t,
                txt_ids=text_ids,
                img_ids=latent_ids,
                return_dict=False,
            )[0]
            capture.save_tensor(f"{step_prefix}.model_output", noise_pred)
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0].to(latents.dtype)
            capture.save_tensor(f"{step_prefix}.latents_after_scheduler", latents)
            if i == steps_eff - 1:
                capture.save_tensor("final.model_output", noise_pred)

        capture.save_tensor("final.latents_packed_before_decode", latents)
        latents_unpacked = Flux2Pipeline._unpack_latents_with_ids(latents, latent_ids).to(
            device=vae_device,
            dtype=torch.bfloat16,
        )
        capture.save_tensor("final.latents_unpacked_pre_bn", latents_unpacked)
        bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents_unpacked.device, latents_unpacked.dtype)
        bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
            latents_unpacked.device,
            latents_unpacked.dtype,
        )
        latents_denorm = latents_unpacked * bn_std + bn_mean
        capture.save_tensor("final.latents_unpacked_post_bn", latents_denorm)
        latents_unpatch = Flux2Pipeline._unpatchify_latents(latents_denorm)
        capture.save_tensor("final.latents_unpatchified_vae_input", latents_unpatch)
        image = vae.decode(latents_unpatch, return_dict=False)[0]
        capture.save_tensor("final.vae_output", image)

    for handle in hooks["handles"]:
        handle.remove()

    img = image[0].clamp(-1.0, 1.0).float()
    img = (img + 1.0) * 127.5
    img = img.clamp(0.0, 255.0).round().to(torch.uint8)
    img_arr = img.permute(1, 2, 0).cpu().numpy()
    image_path = out_dir / f"oracle_reference_{args.width}x{args.height}_seed{args.seed}.png"
    Image.fromarray(img_arr, mode="RGB").save(image_path)
    capture.meta["image"] = str(image_path)
    capture.meta["image_sha256"] = hashlib.sha256(image_path.read_bytes()).hexdigest().upper()
    capture.meta["image_stats"] = {
        "min": int(img_arr.min()),
        "max": int(img_arr.max()),
        "mean": float(img_arr.mean()),
        "std": float(img_arr.std()),
        "unique_rgb": int(len(np.unique(img_arr.reshape(-1, 3), axis=0))),
    }
    capture.meta["status"] = "ok"
    (out_dir / "oracle_status.json").write_text(json.dumps(capture.meta, indent=2), encoding="utf-8")
    print(json.dumps(capture.meta, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Bonsai Python reference tensors for sd.cpp parity.")
    parser.add_argument(
        "--demo-dir",
        default=r"F:\Paralol\local\bonsai-python-oracle\Bonsai-Image-Demo",
    )
    parser.add_argument(
        "--model-dir",
        default=r"F:\Paralol\local\bonsai-python-oracle\models\bonsai-image-binary-4B-gemlite-1bit",
    )
    parser.add_argument("--prompt", default="A bonsai tree in a quiet ceramic studio, soft morning light")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--out-dir", default=r"build-bonsai-int1\bonsai-oracle\python_capture_512")
    parser.add_argument(
        "--gemlite-cache",
        default=r"F:\Paralol\local\bonsai-python-oracle\Bonsai-Image-Demo\outputs\.gemlite_cache\autotune.json",
    )
    try:
        return run_capture(parser.parse_args())
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        out_dir = Path(parser.parse_args().out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        status = {"status": "blocked", "blocker": repr(exc), "python": sys.version}
        (out_dir / "oracle_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
