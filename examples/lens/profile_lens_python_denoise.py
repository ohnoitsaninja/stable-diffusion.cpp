from __future__ import annotations

import argparse
import json
import sys
import time
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


def summarize_profiler(prof) -> dict[str, Any]:
    rows = []
    for item in prof.key_averages():
        self_cuda_us = float(getattr(item, "self_cuda_time_total", 0.0) or 0.0)
        cuda_us = float(getattr(item, "cuda_time_total", 0.0) or 0.0)
        self_cpu_us = float(getattr(item, "self_cpu_time_total", 0.0) or 0.0)
        cpu_us = float(getattr(item, "cpu_time_total", 0.0) or 0.0)
        count = int(getattr(item, "count", 0) or 0)
        key = str(getattr(item, "key", ""))
        rows.append(
            {
                "name": key,
                "count": count,
                "self_cuda_ms": self_cuda_us / 1000.0,
                "cuda_total_ms": cuda_us / 1000.0,
                "self_cpu_ms": self_cpu_us / 1000.0,
                "cpu_total_ms": cpu_us / 1000.0,
            }
        )
    top_cuda = sorted(rows, key=lambda r: r["self_cuda_ms"], reverse=True)
    top_cpu = sorted(rows, key=lambda r: r["self_cpu_ms"], reverse=True)
    return {
        "top_cuda_by_self_ms": top_cuda[:25],
        "top_cpu_by_self_ms": top_cpu[:25],
        "raw_operator_count": len(rows),
    }


def infer_sdpa_backend(prof_rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = [row["name"] for row in prof_rows]
    lower = "\n".join(names).lower()
    if "flash" in lower or "_scaled_dot_product_flash_attention" in lower:
        backend = "flash_attention"
    elif "cudnn" in lower and "attention" in lower:
        backend = "cudnn_sdpa"
    elif "efficient_attention" in lower or "mem_efficient" in lower:
        backend = "memory_efficient_attention"
    elif "scaled_dot_product_attention" in lower or "bmm" in lower:
        backend = "math_or_composite"
    else:
        backend = "unknown"
    matching = [name for name in names if "attention" in name.lower() or "sdpa" in name.lower() or "flash" in name.lower()]
    return {"inferred_backend": backend, "matching_profiler_names": matching[:40]}


def make_component_bucket() -> dict[str, Any]:
    return {"calls": 0, "events": []}


def record_component_event(bucket: dict[str, Any], start_event, end_event) -> None:
    bucket["calls"] += 1
    bucket["events"].append((start_event, end_event))


def summarize_component_events(components: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, bucket in components.items():
        per_call_ms = [float(start.elapsed_time(end)) for start, end in bucket["events"]]
        summary[name] = {
            "calls": int(bucket["calls"]),
            "cuda_ms_total": float(sum(per_call_ms)),
            "cuda_ms_per_call_first8": per_call_ms[:8],
        }
    return dict(sorted(summary.items(), key=lambda item: item[1]["cuda_ms_total"], reverse=True))


def categorize_module_name(name: str) -> str:
    if name.endswith(".attn.img_qkv") or name.endswith(".attn.txt_qkv"):
        return "qkv_linear"
    if ".attn.to_out" in name or name.endswith(".attn.to_add_out"):
        return "attention_out_linear"
    if name.endswith(".img_mlp.w1") or name.endswith(".img_mlp.w2") or name.endswith(".img_mlp.w3"):
        return "img_mlp_linear"
    if name.endswith(".txt_mlp.w1") or name.endswith(".txt_mlp.w2") or name.endswith(".txt_mlp.w3"):
        return "txt_mlp_linear"
    if name.endswith(".img_mod.1") or name.endswith(".txt_mod.1"):
        return "modulation_linear"
    if "norm" in name:
        return "rmsnorm"
    if name.endswith(".attn"):
        return "attention_total"
    if name.endswith(".img_mlp") or name.endswith(".txt_mlp"):
        return "mlp_total"
    if name.endswith(".img_mod.0") or name.endswith(".txt_mod.0"):
        return "modulation_silu"
    return "other_module"


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Python Lens 256 denoise with torch.profiler.")
    parser.add_argument("--lens-repo", default=r"F:\Paralol\local\Lens")
    parser.add_argument("--sdcpp-repo", default=r"F:\Paralol\local\stable-diffusion.cpp-speed")
    parser.add_argument("--model", default=r"F:\Paralol\local\models\microsoft\Lens-Turbo")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
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
    from torch.profiler import ProfilerActivity, profile

    from benchmark_lens_python_official import encode_split
    from lens import LensPipeline
    import lens.transformer as lens_transformer_module

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = True

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

    start = time.perf_counter()
    pipe = LensPipeline.from_pretrained(args.model, text_encoder=None, torch_dtype=torch.bfloat16, local_files_only=True)
    sync(torch)
    result["pipeline_load_seconds"] = time.perf_counter() - start
    start = time.perf_counter()
    pipe.to("cuda")
    sync(torch)
    result["pipeline_to_cuda_seconds"] = time.perf_counter() - start
    result["memory"]["after_pipeline_to_cuda"] = cuda_mem(torch)
    result["torch_compile_detected"] = hasattr(pipe.transformer, "_orig_mod")

    captured: dict[str, Any] = {
        "sdpa_calls": 0,
        "sdpa_inputs": [],
        "transformer_forward_inputs": [],
    }
    components: dict[str, dict[str, Any]] = {
        "sdpa": make_component_bucket(),
        "rope": make_component_bucket(),
        "functional_silu": make_component_bucket(),
    }
    block_events: list[dict[str, Any]] = [{"calls": 0, "events": []} for _ in pipe.transformer.transformer_blocks]

    original_sdpa = F.scaled_dot_product_attention
    original_silu = F.silu
    original_rope = lens_transformer_module.apply_rotary_emb_lens

    def wrapped_sdpa(q, k, v, *sdpa_args, **sdpa_kwargs):
        captured["sdpa_calls"] += 1
        if len(captured["sdpa_inputs"]) < 8:
            captured["sdpa_inputs"].append(
                {
                    "q_dtype": str(q.dtype),
                    "k_dtype": str(k.dtype),
                    "v_dtype": str(v.dtype),
                    "q_device": str(q.device),
                    "k_device": str(k.device),
                    "v_device": str(v.device),
                    "q_shape": list(q.shape),
                    "k_shape": list(k.shape),
                    "v_shape": list(v.shape),
                    "kwargs": sorted(sdpa_kwargs.keys()),
                }
            )
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        out = original_sdpa(q, k, v, *sdpa_args, **sdpa_kwargs)
        end_event.record()
        record_component_event(components["sdpa"], start_event, end_event)
        return out

    def wrapped_silu(*silu_args, **silu_kwargs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        out = original_silu(*silu_args, **silu_kwargs)
        end_event.record()
        record_component_event(components["functional_silu"], start_event, end_event)
        return out

    def wrapped_rope(*rope_args, **rope_kwargs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        out = original_rope(*rope_args, **rope_kwargs)
        end_event.record()
        record_component_event(components["rope"], start_event, end_event)
        return out

    F.scaled_dot_product_attention = wrapped_sdpa
    F.silu = wrapped_silu
    lens_transformer_module.apply_rotary_emb_lens = wrapped_rope

    original_transformer_forward = pipe.transformer.forward

    def wrapped_transformer_forward(*tf_args, **tf_kwargs):
        hidden = tf_kwargs.get("hidden_states", tf_args[0] if tf_args else None)
        encoder = tf_kwargs.get("encoder_hidden_states", None)
        if len(captured["transformer_forward_inputs"]) < 4:
            encoder_dtypes = []
            encoder_devices = []
            if isinstance(encoder, (list, tuple)):
                encoder_dtypes = [str(t.dtype) for t in encoder]
                encoder_devices = [str(t.device) for t in encoder]
            elif encoder is not None:
                encoder_dtypes = [str(encoder.dtype)]
                encoder_devices = [str(encoder.device)]
            captured["transformer_forward_inputs"].append(
                {
                    "hidden_dtype": str(hidden.dtype) if hidden is not None else None,
                    "hidden_device": str(hidden.device) if hidden is not None else None,
                    "hidden_shape": list(hidden.shape) if hidden is not None else None,
                    "encoder_dtypes": encoder_dtypes,
                    "encoder_devices": encoder_devices,
                }
            )
        return original_transformer_forward(*tf_args, **tf_kwargs)

    pipe.transformer.forward = wrapped_transformer_forward

    for idx, block in enumerate(pipe.transformer.transformer_blocks):
        original = block.forward

        def make_wrapper(block_index, original_forward):
            def wrapped(*block_args, **block_kwargs):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                out = original_forward(*block_args, **block_kwargs)
                end_event.record()
                block_events[block_index]["events"].append((start_event, end_event))
                block_events[block_index]["calls"] += 1
                return out

            return wrapped

        block.forward = make_wrapper(idx, original)

    for name, module in pipe.transformer.named_modules():
        if not name.startswith("transformer_blocks."):
            continue
        cls_name = type(module).__name__
        if cls_name not in {"Linear", "RMSNorm", "SiLU", "GateMLP", "LensJointAttention"}:
            continue
        category = categorize_module_name(name)
        components.setdefault(category, make_component_bucket())
        components.setdefault(f"module:{name}", make_component_bucket())
        original_forward = module.forward

        def make_module_wrapper(component_category, module_key, original_module_forward):
            def wrapped_module(*module_args, **module_kwargs):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                out = original_module_forward(*module_args, **module_kwargs)
                end_event.record()
                record_component_event(components[component_category], start_event, end_event)
                record_component_event(components[module_key], start_event, end_event)
                return out

            return wrapped_module

        module.forward = make_module_wrapper(category, f"module:{name}", original_forward)

    generator = torch.Generator(device=pipe._execution_device).manual_seed(args.seed)
    sync(torch)
    denoise_wall_start = time.perf_counter()
    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            acc_events=True,
        ) as prof:
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
    finally:
        F.scaled_dot_product_attention = original_sdpa
        F.silu = original_silu
        lens_transformer_module.apply_rotary_emb_lens = original_rope
    denoise_wall = time.perf_counter() - denoise_wall_start

    block_timings = []
    for idx, item in enumerate(block_events):
        per_call_ms = [float(start.elapsed_time(end)) for start, end in item["events"]]
        block_timings.append(
            {
                "block": idx,
                "calls": item["calls"],
                "cuda_ms_total": float(sum(per_call_ms)),
                "cuda_ms_per_call": per_call_ms,
            }
        )

    profiler_summary = summarize_profiler(prof)
    component_summary = summarize_component_events(components)
    sdpa_info = infer_sdpa_backend(profiler_summary["top_cuda_by_self_ms"])
    result.update(
        {
            "denoise_wall_seconds": denoise_wall,
            "latent_dtype": str(out.images.dtype),
            "latent_device": str(out.images.device),
            "latent_shape": list(out.images.shape),
            "memory_after_denoise": cuda_mem(torch),
            "captured": captured,
            "block_timings": block_timings,
            "component_timings": component_summary,
            "profiler": profiler_summary,
            "sdpa_backend": sdpa_info,
        }
    )

    top_cuda = profiler_summary["top_cuda_by_self_ms"][:10]
    lines = [
        "B3.0 Python Lens denoise profile",
        f"denoise_wall_seconds={denoise_wall:.6f}",
        f"sdpa_calls={captured['sdpa_calls']}",
        f"sdpa_backend_inferred={sdpa_info['inferred_backend']}",
        f"latent={result['latent_shape']} {result['latent_dtype']} {result['latent_device']}",
        "",
        "Top CUDA ops/kernels by self time:",
    ]
    for row in top_cuda:
        lines.append(
            f"- {row['name']}: self_cuda_ms={row['self_cuda_ms']:.3f} "
            f"cuda_total_ms={row['cuda_total_ms']:.3f} count={row['count']}"
        )
    lines.append("")
    lines.append("CUDA-event component totals:")
    for name, row in list(component_summary.items())[:20]:
        if name.startswith("module:"):
            continue
        lines.append(f"- {name}: cuda_ms_total={row['cuda_ms_total']:.3f} calls={row['calls']}")
    lines.append("")
    lines.append("Top block CUDA totals:")
    for row in sorted(block_timings, key=lambda r: r["cuda_ms_total"], reverse=True)[:10]:
        lines.append(f"- block {row['block']}: cuda_ms_total={row['cuda_ms_total']:.3f} calls={row['calls']}")

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.txt_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    Path(args.txt_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
