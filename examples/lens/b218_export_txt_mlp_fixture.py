#!/usr/bin/env python3
"""Export block0 txt_mlp intermediates for native BF16 linear layout tests."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default=r"build\diagnostics\lens_b212_block0_ref_stages.npz")
    parser.add_argument("--model", default=r"F:\Paralol\local\models\microsoft\Lens-Turbo")
    parser.add_argument("--output", default=r"build\diagnostics\lens_b218_txt_mlp_fixture.bin")
    parser.add_argument("--json-out", default=r"build\diagnostics\lens_b218_txt_mlp_export.json")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_tensor(out, name: str, tensor: torch.Tensor) -> None:
    arr = tensor.detach().cpu().contiguous().float().numpy()
    encoded = name.encode("utf-8")
    out.write(struct.pack("<II", len(encoded), arr.ndim))
    out.write(encoded)
    for dim in arr.shape:
        out.write(struct.pack("<q", int(dim)))
    out.write(arr.tobytes(order="C"))


def load_weight(model: str, name: str, device: torch.device) -> torch.Tensor:
    root = Path(model) / "transformer"
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            if name in handle.keys():
                return handle.get_tensor(name).to(device=device, dtype=torch.bfloat16)
    raise KeyError(name)


def meta(t: torch.Tensor, coords: list[tuple[int, ...]]) -> dict[str, object]:
    flat_bits = t.detach().cpu().contiguous().view(torch.int16).view(-1).numpy()
    values = {}
    tf = t.detach().float()
    for coord in coords:
        if len(coord) == t.ndim and all(coord[i] < t.shape[i] for i in range(t.ndim)):
            idx = np.ravel_multi_index(coord, tuple(t.shape))
            values[str(coord)] = {
                "value": float(tf[coord].item()),
                "bf16_bits_u16": int(np.uint16(flat_bits[idx]).item()),
            }
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "stride": list(t.stride()),
        "contiguous": bool(t.is_contiguous()),
        "min": float(tf.min().item()),
        "max": float(tf.max().item()),
        "mean_abs": float(tf.abs().mean().item()),
        "samples": values,
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    refs = np.load(args.npz)
    txt_modulated2 = torch.from_numpy(refs["ref.txt_modulated2"]).to(device=device, dtype=torch.bfloat16)
    prefix = "transformer_blocks.0.txt_mlp."
    w1 = load_weight(args.model, prefix + "w1.weight", device)
    w3 = load_weight(args.model, prefix + "w3.weight", device)
    w2 = load_weight(args.model, prefix + "w2.weight", device)
    with torch.no_grad():
        txt_w1 = F.linear(txt_modulated2, w1)
        txt_w3 = F.linear(txt_modulated2, w3)
        txt_silu_w1 = F.silu(txt_w1)
        txt_gated = txt_silu_w1 * txt_w3
        txt_mlp = F.linear(txt_gated, w2)

    tensors = {
        "input.txt_modulated2": txt_modulated2,
        "weight.txt_mlp.w1": w1,
        "weight.txt_mlp.w3": w3,
        "weight.txt_mlp.w2": w2,
        "expected.txt_w1": txt_w1,
        "expected.txt_w3": txt_w3,
        "expected.txt_silu_w1": txt_silu_w1,
        "expected.txt_gated": txt_gated,
        "expected.txt_mlp": txt_mlp,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as out:
        out.write(b"LENSBLK1")
        out.write(struct.pack("<I", len(tensors)))
        for name, tensor in tensors.items():
            write_tensor(out, name, tensor)

    coords3 = [(0, 0, 0), (0, 0, 1), (0, min(30, txt_modulated2.shape[1] - 1), 0), (0, min(30, txt_modulated2.shape[1] - 1), 1535)]
    coords2 = [(0, 0), (0, 1), (min(4095, w1.shape[0] - 1), 0), (min(4095, w1.shape[0] - 1), 1535)]
    report = {
        "fixture": args.output,
        "layout_note": "PyTorch Linear uses input [..., in] contiguous row-major, weight [out, in], computes x @ weight.T.",
        "tensors": {
            name: meta(t, coords3 if t.ndim == 3 else coords2)
            for name, t in tensors.items()
        },
    }
    Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"fixture": args.output, "json_out": args.json_out, "tensor_count": len(tensors)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
