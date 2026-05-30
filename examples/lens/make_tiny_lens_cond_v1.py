#!/usr/bin/env python3
"""Create a tiny lens_cond_v1 safetensors bundle for loader smoke tests."""

import argparse
import json
import struct
from pathlib import Path


def tensor_entry(dtype, shape, data, offset):
    size = len(data)
    return {
        "entry": {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        },
        "next_offset": offset + size,
    }


def pack_f32(values):
    return struct.pack("<" + "f" * len(values), *values)


def pack_i32(values):
    return struct.pack("<" + "i" * len(values), *values)


def write_safetensors(path, batch=1, seq_len=3, hidden_size=4):
    tensors = []
    offset = 0
    feature_elements = batch * seq_len * hidden_size
    for feature in range(4):
        values = [float(feature * 100 + i) / 1000.0 for i in range(feature_elements)]
        data = pack_f32(values)
        item = tensor_entry("F32", [batch, seq_len, hidden_size], data, offset)
        tensors.append((f"feature_{feature}", item["entry"], data))
        offset = item["next_offset"]

    mask_data = pack_i32([1] * (batch * seq_len))
    item = tensor_entry("I32", [batch, seq_len], mask_data, offset)
    tensors.append(("attention_mask", item["entry"], mask_data))

    header = {name: entry for name, entry, _ in tensors}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * padding

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for _, _, data in tensors:
            f.write(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch <= 0 or args.seq_len <= 0 or args.hidden_size <= 0:
        raise SystemExit("batch, seq-len, and hidden-size must be positive")
    write_safetensors(args.output, args.batch, args.seq_len, args.hidden_size)


if __name__ == "__main__":
    main()
