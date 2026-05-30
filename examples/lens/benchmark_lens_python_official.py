from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence


PROMPT = "a small glass robot standing on a wooden workbench, studio lighting, sharp focus"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official Python Lens benchmark for native parity comparison.")
    parser.add_argument("--lens-repo", default=r"F:\Paralol\local\Lens")
    parser.add_argument("--model", default=r"F:\Paralol\local\models\microsoft\Lens-Turbo")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--variant", choices=["full-gpu", "full-offload", "split-gpu", "split-offload"], default="full-gpu")
    parser.add_argument("--mxfp4", choices=["keep", "dequant", "auto"], default="auto")
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json-out", default="")
    return parser.parse_args()


def add_lens_repo(path: str) -> None:
    repo = str(Path(path).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)


def torch_dtype(torch_module, name: str):
    return {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }[name]


def sync(torch_module) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def cuda_mem(torch_module) -> Dict[str, Any]:
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


def image_stats(path: Path) -> Dict[str, Any]:
    from PIL import Image
    import numpy as np

    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    return {
        "path": str(path),
        "sha256": digest,
        "width": int(arr.shape[1]),
        "height": int(arr.shape[0]),
        "channels": int(arr.shape[2]),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "nonzero_pixels": int(np.any(arr != 0, axis=2).sum()),
        "total_pixels": int(arr.shape[0] * arr.shape[1]),
    }


def make_mxfp4_config(mode: str) -> Dict[str, Any]:
    if mode == "auto":
        return {"status": "auto", "kwargs": {}}
    try:
        from transformers import Mxfp4Config
    except ImportError:
        return {"status": "Mxfp4Config unavailable", "kwargs": {}}
    dequantize = mode == "dequant"
    return {
        "status": f"Mxfp4Config(dequantize={dequantize})",
        "kwargs": {"quantization_config": Mxfp4Config(dequantize=dequantize)},
    }


def install_pipeline_timers(pipe, torch_module, timings: Dict[str, float], counts: Dict[str, int]) -> None:
    original_encode_prompt = pipe.encode_prompt

    def timed_encode_prompt(*args, **kwargs):
        sync(torch_module)
        start = time.perf_counter()
        out = original_encode_prompt(*args, **kwargs)
        sync(torch_module)
        timings["pipeline_encode_prompt"] = timings.get("pipeline_encode_prompt", 0.0) + time.perf_counter() - start
        counts["pipeline_encode_prompt"] = counts.get("pipeline_encode_prompt", 0) + 1
        return out

    pipe.encode_prompt = timed_encode_prompt

    original_transformer_forward = pipe.transformer.forward

    def timed_transformer_forward(*args, **kwargs):
        sync(torch_module)
        start = time.perf_counter()
        out = original_transformer_forward(*args, **kwargs)
        sync(torch_module)
        timings["transformer_forward"] = timings.get("transformer_forward", 0.0) + time.perf_counter() - start
        counts["transformer_forward"] = counts.get("transformer_forward", 0) + 1
        return out

    pipe.transformer.forward = timed_transformer_forward

    original_decode = pipe._decode

    def timed_decode(*args, **kwargs):
        sync(torch_module)
        start = time.perf_counter()
        out = original_decode(*args, **kwargs)
        sync(torch_module)
        timings["vae_decode"] = timings.get("vae_decode", 0.0) + time.perf_counter() - start
        counts["vae_decode"] = counts.get("vae_decode", 0) + 1
        return out

    pipe._decode = timed_decode


def render_prompts(tokenizer, prompts: Sequence[str]) -> List[str]:
    from lens.pipeline import _CHAT_ASSISTANT_THINKING, _CHAT_SYSTEM

    rendered: List[str] = []
    for prompt in prompts:
        conversation = [
            {"role": "system", "content": _CHAT_SYSTEM, "thinking": None},
            {"role": "user", "content": prompt, "thinking": None},
            {"role": "assistant", "thinking": _CHAT_ASSISTANT_THINKING, "content": ""},
        ]
        text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        rendered.append(text.split("<|return|>")[0])
    return rendered


def encode_split(args: argparse.Namespace, torch_module, dtype) -> Dict[str, Any]:
    from transformers import AutoTokenizer
    from lens import LensGptOssEncoder
    from lens.pipeline import DEFAULT_TXT_OFFSET

    model = Path(args.model)
    selected_layers = json.loads((model / "transformer" / "config.json").read_text())["selected_layer_index"]
    prompts = [args.prompt]
    timings: Dict[str, float] = {}
    mem: Dict[str, Any] = {}

    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, subfolder="tokenizer", local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    timings["tokenizer_load"] = time.perf_counter() - start

    kwargs = {
        "subfolder": "text_encoder",
        "dtype": dtype,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    mxfp4 = make_mxfp4_config(args.mxfp4)
    kwargs.update(mxfp4["kwargs"])

    mem["before_text_encoder_load"] = cuda_mem(torch_module)
    start = time.perf_counter()
    text_encoder = LensGptOssEncoder.from_pretrained(args.model, **kwargs)
    text_encoder.set_selected_layers(selected_layers)
    text_encoder.eval()
    timings["text_encoder_load"] = time.perf_counter() - start

    device = torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    start = time.perf_counter()
    text_encoder.to(device)
    sync(torch_module)
    timings["text_encoder_to_device"] = time.perf_counter() - start
    mem["after_text_encoder_to_device"] = cuda_mem(torch_module)

    encoded = tokenizer(
        render_prompts(tokenizer, prompts),
        padding=True,
        truncation=True,
        max_length=args.max_sequence_length,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attn_mask = encoded["attention_mask"].to(device)
    sync(torch_module)
    start = time.perf_counter()
    layer_outputs = text_encoder.encode_layers(input_ids, attn_mask)
    sync(torch_module)
    timings["text_encoder_encode"] = time.perf_counter() - start

    if input_ids.shape[1] > DEFAULT_TXT_OFFSET:
        prompt_embeds = [feat[:, DEFAULT_TXT_OFFSET:, :].contiguous().cpu() for feat in layer_outputs]
        prompt_mask = attn_mask[:, DEFAULT_TXT_OFFSET:].bool().cpu()
    else:
        zero_shape = (input_ids.shape[0], 0, layer_outputs[0].shape[-1])
        prompt_embeds = [layer_outputs[0].new_zeros(zero_shape).cpu() for _ in layer_outputs]
        prompt_mask = torch_module.zeros((input_ids.shape[0], 0), dtype=torch_module.bool)
    negative_prompt_embeds = [feat.new_zeros(feat.shape) for feat in prompt_embeds]
    negative_prompt_mask = torch_module.zeros_like(prompt_mask, dtype=torch_module.bool)

    del layer_outputs, input_ids, attn_mask, encoded, text_encoder
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
        torch_module.cuda.ipc_collect()
    mem["after_text_encoder_unload"] = cuda_mem(torch_module)

    return {
        "prompts": prompts,
        "prompt_embeds": prompt_embeds,
        "prompt_mask": prompt_mask,
        "negative_prompt_embeds": negative_prompt_embeds,
        "negative_prompt_mask": negative_prompt_mask,
        "timings": timings,
        "memory": mem,
        "mxfp4_status": mxfp4["status"],
    }


def run_full(args: argparse.Namespace, torch_module, dtype) -> Dict[str, Any]:
    from lens import LensGptOssEncoder, LensPipeline

    timings: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    mem: Dict[str, Any] = {"start": cuda_mem(torch_module)}

    kwargs = {
        "subfolder": "text_encoder",
        "dtype": dtype,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    mxfp4 = make_mxfp4_config(args.mxfp4)
    kwargs.update(mxfp4["kwargs"])

    start = time.perf_counter()
    text_encoder = LensGptOssEncoder.from_pretrained(args.model, **kwargs)
    timings["text_encoder_load"] = time.perf_counter() - start
    mem["after_text_encoder_load"] = cuda_mem(torch_module)

    start = time.perf_counter()
    pipe = LensPipeline.from_pretrained(args.model, text_encoder=text_encoder, torch_dtype=dtype, local_files_only=True)
    timings["pipeline_load"] = time.perf_counter() - start

    start = time.perf_counter()
    if args.variant == "full-offload":
        pipe.enable_model_cpu_offload()
        offload = True
    else:
        pipe.to("cuda")
        offload = False
    sync(torch_module)
    timings["pipeline_to_device_or_offload"] = time.perf_counter() - start
    mem["after_pipeline_ready"] = cuda_mem(torch_module)

    install_pipeline_timers(pipe, torch_module, timings, counts)

    generator = torch_module.Generator(device=pipe._execution_device).manual_seed(int(args.seed))
    sync(torch_module)
    start = time.perf_counter()
    out = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        num_images_per_prompt=1,
        generator=generator,
        output_type="pil",
    )
    sync(torch_module)
    timings["generation_wall"] = time.perf_counter() - start
    mem["after_generation"] = cuda_mem(torch_module)
    return {
        "images": list(out.images),
        "timings": timings,
        "counts": counts,
        "memory": mem,
        "mxfp4_status": mxfp4["status"],
        "offload": offload,
    }


def run_split(args: argparse.Namespace, torch_module, dtype) -> Dict[str, Any]:
    from lens import LensPipeline

    split = encode_split(args, torch_module, dtype)
    timings = dict(split["timings"])
    counts: Dict[str, int] = {}
    mem = dict(split["memory"])

    start = time.perf_counter()
    pipe = LensPipeline.from_pretrained(args.model, text_encoder=None, torch_dtype=dtype, local_files_only=True)
    timings["pipeline_load"] = time.perf_counter() - start

    start = time.perf_counter()
    if args.variant == "split-offload":
        pipe.enable_model_cpu_offload()
        offload = True
    else:
        pipe.to("cuda")
        offload = False
    sync(torch_module)
    timings["pipeline_to_device_or_offload"] = time.perf_counter() - start
    mem["after_pipeline_ready"] = cuda_mem(torch_module)

    install_pipeline_timers(pipe, torch_module, timings, counts)
    generator = torch_module.Generator(device=pipe._execution_device).manual_seed(int(args.seed))
    sync(torch_module)
    start = time.perf_counter()
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
        output_type="pil",
    )
    sync(torch_module)
    timings["generation_wall"] = time.perf_counter() - start
    mem["after_generation"] = cuda_mem(torch_module)
    return {
        "images": list(out.images),
        "timings": timings,
        "counts": counts,
        "memory": mem,
        "mxfp4_status": split["mxfp4_status"],
        "offload": offload,
    }


def main() -> None:
    args = parse_args()
    add_lens_repo(args.lens_repo)

    import torch

    dtype = torch_dtype(torch, args.dtype)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    cold_start = time.perf_counter()
    if args.variant.startswith("split"):
        result = run_split(args, torch, dtype)
    else:
        result = run_full(args, torch, dtype)
    cold_total = time.perf_counter() - cold_start

    if len(result["images"]) != 1:
        raise RuntimeError(f"expected exactly one image, got {len(result['images'])}")
    result["images"][0].save(output_path)
    stats = image_stats(output_path)
    timings = result["timings"]
    warm_generation = timings.get("generation_wall", 0.0)
    text_encoder_total = (
        timings.get("text_encoder_load", 0.0)
        + timings.get("text_encoder_to_device", 0.0)
        + timings.get("text_encoder_encode", 0.0)
        + timings.get("pipeline_encode_prompt", 0.0)
    )
    transformer_denoise = timings.get("transformer_forward", 0.0)
    vae_decode = timings.get("vae_decode", 0.0)
    transformer_load_context = timings.get("pipeline_load", 0.0) + timings.get("pipeline_to_device_or_offload", 0.0)
    summary = {
        "variant": args.variant,
        "mxfp4": args.mxfp4,
        "mxfp4_status": result["mxfp4_status"],
        "dtype": args.dtype,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "cfg": args.cfg,
        "seed": args.seed,
        "prompt": args.prompt,
        "offload": result["offload"],
        "timings": timings,
        "counts": result["counts"],
        "derived": {
            "text_encoder_total": text_encoder_total,
            "transformer_load_context": transformer_load_context,
            "transformer_denoise": transformer_denoise,
            "vae_decode": vae_decode,
            "warm_generation": warm_generation,
            "cold_total": cold_total,
        },
        "memory": result["memory"],
        "image": stats,
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("BENCH_JSON " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
