#!/usr/bin/env python3
"""Export real Lens prompt conditioning as lens_cond_v1.

This is the text-encoder-only half of the staged Lens workflow. It loads the
GPT-OSS encoder, writes selected layer features to CPU safetensors, and then
releases the encoder before any transformer/VAE work starts.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens-root", default=r"F:\Paralol\local\Lens")
    parser.add_argument("--model-dir", required=True, help="Local Lens-Turbo root")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--keep-mxfp4", action="store_true")
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def cuda_note(label: str) -> None:
    if not torch.cuda.is_available():
        print(f"{label}: cuda unavailable")
        return
    free, total = torch.cuda.mem_get_info()
    used = total - free
    print(f"{label}: cuda {used / 1024**3:.2f}/{total / 1024**3:.2f} GiB")


def main() -> int:
    args = parse_args()
    lens_root = Path(args.lens_root)
    sys.path.insert(0, str(lens_root))

    from lens import LensGptOssEncoder  # noqa: PLC0415
    from lens.pipeline import _CHAT_ASSISTANT_THINKING, _CHAT_SYSTEM, DEFAULT_TXT_OFFSET  # noqa: PLC0415

    model_dir = Path(args.model_dir)
    selected_layers = json.loads((model_dir / "transformer" / "config.json").read_text(encoding="utf-8"))[
        "selected_layer_index"
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_dir, subfolder="tokenizer", local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    conversation = [
        {"role": "system", "content": _CHAT_SYSTEM, "thinking": None},
        {"role": "user", "content": args.prompt, "thinking": None},
        {"role": "assistant", "thinking": _CHAT_ASSISTANT_THINKING, "content": ""},
    ]
    rendered = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    rendered = rendered.split("<|return|>")[0]

    kwargs = {
        "subfolder": "text_encoder",
        "dtype": torch_dtype(args.dtype),
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    try:
        from transformers import Mxfp4Config  # noqa: PLC0415

        kwargs["quantization_config"] = Mxfp4Config(dequantize=not args.keep_mxfp4)
    except ImportError:
        pass

    cuda_note("before text_encoder load")
    text_encoder = LensGptOssEncoder.from_pretrained(model_dir, **kwargs)
    text_encoder.set_selected_layers(selected_layers)
    text_encoder.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder.to(device)
    cuda_note("after text_encoder to device")

    encoded = tokenizer(
        [rendered],
        padding=True,
        truncation=True,
        max_length=args.max_sequence_length,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention = encoded["attention_mask"].to(device)

    with torch.no_grad():
        layer_outputs = text_encoder.encode_layers(input_ids, attention)

    if input_ids.shape[1] <= DEFAULT_TXT_OFFSET:
        raise RuntimeError(
            f"tokenized prompt length {input_ids.shape[1]} is not longer than Lens text offset {DEFAULT_TXT_OFFSET}; "
            "raise --max-sequence-length or use a longer prompt"
        )

    tensors = {}
    for idx, feature in enumerate(layer_outputs):
        tensors[f"feature_{idx}"] = feature[:, DEFAULT_TXT_OFFSET:, :].contiguous().to("cpu", torch.float32)
    tensors["attention_mask"] = attention[:, DEFAULT_TXT_OFFSET:].bool().to("cpu", torch.int32)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        output,
        metadata={
            "schema": "lens_cond_v1",
            "prompt": args.prompt,
            "selected_layer_index": json.dumps(selected_layers),
            "max_sequence_length": str(args.max_sequence_length),
        },
    )
    print(
        f"wrote {output} seq={tensors['attention_mask'].shape[1]} hidden={tensors['feature_0'].shape[2]} "
        f"valid_tokens={int(tensors['attention_mask'].sum().item())}"
    )

    del layer_outputs, input_ids, attention, encoded, text_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    cuda_note("after text_encoder unload")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    raise SystemExit(main())
