#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any


PROMPT = "a small glass robot standing on a wooden workbench, studio lighting, sharp focus"
MAGIC = b"LENSATTN1"


def add_path(path: str) -> None:
    resolved = str(Path(path).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def sync(torch_module) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


class StopAfterFixture(RuntimeError):
    pass


def tensor_info(t) -> dict[str, Any]:
    return {
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "contiguous": bool(t.is_contiguous()),
        "min": float(t.detach().float().amin().item()),
        "max": float(t.detach().float().amax().item()),
        "mean": float(t.detach().float().mean().item()),
    }


def as_float32_cpu(t):
    return t.detach().contiguous().to("cpu", dtype=None).float().numpy()


def write_f32(file, arr) -> None:
    arr.astype("<f4", copy=False).tofile(file)


def main() -> None:
    parser = argparse.ArgumentParser(description="B3.6 export one real Lens 512 SDPA call as a native fixture.")
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
    parser.add_argument("--bin-out", required=True)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()

    add_path(args.lens_repo)
    add_path(str(Path(args.sdcpp_repo) / "examples" / "lens"))

    import numpy as np
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

    split = encode_split(encode_args, torch, torch.bfloat16)
    pipe = LensPipeline.from_pretrained(args.model, text_encoder=None, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.to("cuda")
    sync(torch)

    original_sdpa = F.scaled_dot_product_attention
    captured: dict[str, Any] = {}

    def wrapped_sdpa(q, k, v, *sdpa_args, **sdpa_kwargs):
        out = original_sdpa(q, k, v, *sdpa_args, **sdpa_kwargs)
        sync(torch)
        attn_mask = sdpa_kwargs.get("attn_mask")
        captured.update(
            {
                "q": q.detach().contiguous().cpu(),
                "k": k.detach().contiguous().cpu(),
                "v": v.detach().contiguous().cpu(),
                "out": out.detach().contiguous().cpu(),
                "mask": None if attn_mask is None else attn_mask.detach().contiguous().cpu(),
                "q_info": tensor_info(q),
                "k_info": tensor_info(k),
                "v_info": tensor_info(v),
                "out_info": tensor_info(out),
                "mask_info": None if attn_mask is None else tensor_info(attn_mask),
                "kwargs": sorted(sdpa_kwargs.keys()),
            }
        )
        raise StopAfterFixture("captured first SDPA call")

    F.scaled_dot_product_attention = wrapped_sdpa
    generator = torch.Generator(device=pipe._execution_device).manual_seed(args.seed)
    status = "not_run"
    error = ""
    try:
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            pipe(
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
        status = "unexpected_full_run"
    except StopAfterFixture:
        status = "captured"
    except Exception as exc:
        status = "failed"
        error = repr(exc)
    finally:
        F.scaled_dot_product_attention = original_sdpa

    if status != "captured":
        raise RuntimeError(f"failed to capture SDPA fixture: {status} {error}")

    q = captured["q"]
    k = captured["k"]
    v = captured["v"]
    out = captured["out"]
    mask = captured["mask"]
    if q.ndim != 4:
        raise RuntimeError(f"expected [B,H,S,D] q, got {tuple(q.shape)}")
    b, h, s, d = [int(x) for x in q.shape]
    if tuple(k.shape) != (b, h, s, d) or tuple(v.shape) != (b, h, s, d) or tuple(out.shape) != (b, h, s, d):
        raise RuntimeError("q/k/v/out shapes do not match")

    # Native ggml_ext_attention_ext skip_reshape layout:
    # q/k: [D, S, B*H], v: [D, H, S, B], output: [D*H, S, B].
    q_np = as_float32_cpu(q).transpose(3, 2, 0, 1).reshape(d, s, b * h)
    k_np = as_float32_cpu(k).transpose(3, 2, 0, 1).reshape(d, s, b * h)
    v_np = as_float32_cpu(v).transpose(3, 1, 2, 0)
    out_np = as_float32_cpu(out).transpose(1, 3, 2, 0).reshape(h * d, s, b)
    if mask is None:
        mask_np = np.zeros((0,), dtype=np.float32)
        mask_shape = [0, 0, 0, 0]
        mask_nonzero = 0
    else:
        mask_f = as_float32_cpu(mask)
        mask_shape = list(mask_f.shape)
        mask_nonzero = int(np.count_nonzero(mask_f))
        mask_np = mask_f

    bin_path = Path(args.bin_out)
    json_path = Path(args.json_out)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with bin_path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<8i", b, h, s, d, *([int(x) for x in mask_shape] + [0, 0, 0, 0])[:4]))
        write_f32(f, q_np)
        write_f32(f, k_np)
        write_f32(f, v_np)
        write_f32(f, mask_np)
        write_f32(f, out_np)

    metadata = {
        "status": status,
        "backend": "torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "inputs": {
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "cfg": args.cfg,
            "seed": args.seed,
            "prompt": args.prompt,
            "max_sequence_length": args.max_sequence_length,
        },
        "shape": {"B": b, "H": h, "S": s, "D": d},
        "scale": 1.0 / math.sqrt(float(d)),
        "q": captured["q_info"],
        "k": captured["k_info"],
        "v": captured["v_info"],
        "out": captured["out_info"],
        "mask": captured["mask_info"],
        "mask_nonzero": mask_nonzero,
        "native_binary_layout": {
            "q": "[D,S,B*H] float32, converted from Python BF16",
            "k": "[D,S,B*H] float32, converted from Python BF16",
            "v": "[D,H,S,B] float32, converted from Python BF16",
            "mask": "raw Python mask float32 in original shape, if present",
            "out": "[D*H,S,B] float32, converted from Python BF16 SDPA output",
        },
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
