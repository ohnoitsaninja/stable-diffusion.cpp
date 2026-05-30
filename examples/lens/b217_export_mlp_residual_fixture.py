#!/usr/bin/env python3
"""Export B2.17 MLP/residual microbench fixture from B2.12 NPZ."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default=r"build\diagnostics\lens_b212_block0_ref_stages.npz")
    parser.add_argument("--output", default=r"build\diagnostics\lens_b217_mlp_residual_fixture.bin")
    return parser.parse_args()


def write_tensor(out, name: str, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr.astype(np.float32))
    encoded = name.encode("utf-8")
    out.write(struct.pack("<II", len(encoded), arr.ndim))
    out.write(encoded)
    for dim in arr.shape:
        out.write(struct.pack("<q", int(dim)))
    out.write(arr.tobytes(order="C"))


def main() -> int:
    args = parse_args()
    source = np.load(args.npz)
    mapping = {
        "input.hidden_after_attn": "ref.hidden_after_attn",
        "input.encoder_after_attn": "ref.encoder_after_attn",
        "input.img_mod2": "ref.img_mod2",
        "input.txt_mod2": "ref.txt_mod2",
        "expected.img_norm2": "ref.img_norm2",
        "expected.txt_norm2": "ref.txt_norm2",
        "expected.img_modulated2": "ref.img_modulated2",
        "expected.txt_modulated2": "ref.txt_modulated2",
        "expected.img_mlp": "ref.img_mlp",
        "expected.txt_mlp": "ref.txt_mlp",
        "expected.final.hidden": "ref.final.hidden",
        "expected.final.encoder": "ref.final.encoder",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as out:
        out.write(b"LENSBLK1")
        out.write(struct.pack("<I", len(mapping)))
        for dst, src in mapping.items():
            if src not in source:
                raise KeyError(f"missing {src} in {args.npz}")
            write_tensor(out, dst, source[src])
    print(f"wrote {args.output} tensors={len(mapping)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
