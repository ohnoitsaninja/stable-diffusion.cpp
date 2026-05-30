#!/usr/bin/env python
import argparse
import contextlib
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any


PROMPT = "a small glass robot standing on a wooden workbench, studio lighting, sharp focus"


def add_path(path: str) -> None:
    resolved = str(Path(path).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def sync(torch_module) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def cuda_mem(torch_module) -> dict[str, Any]:
    if not torch_module.cuda.is_available():
        return {"available": False}
    free, total = torch_module.cuda.mem_get_info()
    return {
        "available": True,
        "used_gib": (total - free) / 1024**3,
        "free_gib": free / 1024**3,
        "total_gib": total / 1024**3,
        "max_allocated_gib": torch_module.cuda.max_memory_allocated() / 1024**3,
        "max_reserved_gib": torch_module.cuda.max_memory_reserved() / 1024**3,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="B3.5 SDPA backend forcing for Lens denoise-only timing.")
    parser.add_argument("--lens-repo", default=r"F:\Paralol\local\Lens")
    parser.add_argument("--sdcpp-repo", default=r"F:\Paralol\local\stable-diffusion.cpp-speed")
    parser.add_argument("--model", default=r"F:\Paralol\local\models\microsoft\Lens-Turbo")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--txt-out", required=True)
    args = parser.parse_args()

    add_path(args.lens_repo)
    add_path(str(Path(args.sdcpp_repo) / "examples" / "lens"))

    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from benchmark_lens_python_official import encode_split
    from lens import LensPipeline

    torch.backends.cuda.matmul.allow_tf32 = True
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    class EncodeArgs:
        pass

    encode_args = EncodeArgs()
    encode_args.model = args.model
    encode_args.prompt = args.prompt
    encode_args.mxfp4 = "keep"
    encode_args.max_sequence_length = args.max_sequence_length

    result: dict[str, Any] = {
        "env": {
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "inputs": {
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "cfg": args.cfg,
            "seed": args.seed,
            "prompt": args.prompt,
            "max_sequence_length": args.max_sequence_length,
        },
        "memory": {"start": cuda_mem(torch)},
    }

    split = encode_split(encode_args, torch, torch.bfloat16)
    result["text_encoder_timings"] = split["timings"]
    result["memory"].update({f"text_encoder_{k}": v for k, v in split["memory"].items()})

    load_start = time.perf_counter()
    pipe = LensPipeline.from_pretrained(args.model, text_encoder=None, torch_dtype=torch.bfloat16, local_files_only=True)
    sync(torch)
    result["pipeline_load_seconds"] = time.perf_counter() - load_start
    to_cuda_start = time.perf_counter()
    pipe.to("cuda")
    sync(torch)
    result["pipeline_to_cuda_seconds"] = time.perf_counter() - to_cuda_start
    result["memory"]["after_pipeline_to_cuda"] = cuda_mem(torch)
    result["torch_compile_detected"] = hasattr(pipe.transformer, "_orig_mod")

    variants = [
        ("default", None),
        ("flash_only", SDPBackend.FLASH_ATTENTION),
        ("mem_efficient_only", SDPBackend.EFFICIENT_ATTENTION),
        ("math_only", SDPBackend.MATH),
    ]

    original_sdpa = F.scaled_dot_product_attention
    rows: list[dict[str, Any]] = []

    for name, backend in variants:
        captured: dict[str, Any] = {
            "sdpa_calls": 0,
            "sdpa_inputs": [],
            "warnings": [],
        }

        def wrapped_sdpa(q, k, v, *sdpa_args, **sdpa_kwargs):
            captured["sdpa_calls"] += 1
            if len(captured["sdpa_inputs"]) < 4:
                captured["sdpa_inputs"].append(
                    {
                        "q_shape": list(q.shape),
                        "k_shape": list(k.shape),
                        "v_shape": list(v.shape),
                        "q_dtype": str(q.dtype),
                        "k_dtype": str(k.dtype),
                        "v_dtype": str(v.dtype),
                        "q_device": str(q.device),
                        "attn_mask": "present" if sdpa_kwargs.get("attn_mask") is not None else "missing",
                        "kwargs": sorted(sdpa_kwargs.keys()),
                    }
                )
            return original_sdpa(q, k, v, *sdpa_args, **sdpa_kwargs)

        F.scaled_dot_product_attention = wrapped_sdpa
        context = contextlib.nullcontext() if backend is None else sdpa_kernel(backend)
        sync(torch)
        torch.cuda.empty_cache()
        row: dict[str, Any] = {
            "variant": name,
            "forced_backend": str(backend),
            "status": "not_run",
            "memory_before": cuda_mem(torch),
        }
        generator = torch.Generator(device=pipe._execution_device).manual_seed(args.seed)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                denoise_start = time.perf_counter()
                with context:
                    out = pipe(
                        prompt=None,
                        prompt_embeds=split["prompt_embeds"],
                        prompt_mask=split["prompt_mask"],
                        negative_prompt_embeds=split["negative_prompt_embeds"],
                        negative_prompt_mask=split["negative_prompt_mask"],
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.steps,
                        guidance_scale=args.cfg,
                        num_images_per_prompt=1,
                        generator=generator,
                        output_type="latent",
                    )
                sync(torch)
                row["denoise_seconds"] = time.perf_counter() - denoise_start
                row["latent_shape"] = list(out.images.shape)
                row["latent_dtype"] = str(out.images.dtype)
                row["latent_device"] = str(out.images.device)
                row["status"] = "ok"
                captured["warnings"] = [str(item.message) for item in caught]
        except Exception as exc:
            sync(torch)
            row["status"] = "failed"
            row["error"] = repr(exc)
        finally:
            F.scaled_dot_product_attention = original_sdpa

        row["sdpa_calls"] = captured["sdpa_calls"]
        row["sdpa_inputs"] = captured["sdpa_inputs"]
        row["warnings"] = captured["warnings"]
        row["memory_after"] = cuda_mem(torch)
        rows.append(row)

    result["variants"] = rows
    result["memory"]["end"] = cuda_mem(torch)

    out_json = Path(args.json_out)
    out_txt = Path(args.txt_out)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = ["B3.5 Python SDPA forcing", f"torch={torch.__version__} cuda={torch.version.cuda}", ""]
    for row in rows:
        lines.append(
            f"{row['variant']}: status={row['status']} seconds={row.get('denoise_seconds')} "
            f"sdpa_calls={row['sdpa_calls']} forced={row['forced_backend']}"
        )
        if row.get("error"):
            lines.append(f"  error={row['error']}")
        if row["warnings"]:
            lines.append(f"  warnings={row['warnings'][:3]}")
        if row["sdpa_inputs"]:
            lines.append(f"  first_sdpa_input={row['sdpa_inputs'][0]}")
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
