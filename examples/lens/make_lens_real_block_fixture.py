import argparse
import ctypes
import importlib.util
import struct
from pathlib import Path

import torch
from safetensors import safe_open


def load_lens_transformer_module(lens_src: str):
    transformer_path = Path(lens_src) / "transformer.py"
    spec = importlib.util.spec_from_file_location("lens_transformer_direct", transformer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_tensor(out, name: str, tensor: torch.Tensor):
    tensor = tensor.detach().cpu().contiguous().to(torch.float32)
    encoded = name.encode("utf-8")
    out.write(struct.pack("<II", len(encoded), tensor.ndim))
    out.write(encoded)
    for dim in tensor.shape:
        out.write(struct.pack("<q", int(dim)))
    out.write(tensor.numpy().tobytes(order="C"))


def load_weight(transformer_dir: Path, name: str) -> torch.Tensor:
    for shard in sorted(transformer_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            if name in handle.keys():
                return handle.get_tensor(name)
    raise KeyError(f"missing tensor {name}")


class MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def available_ram_gib() -> float:
    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return 0.0
    return status.ullAvailPhys / float(1024 ** 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens-src", default=r"F:\Paralol\local\Lens\lens")
    parser.add_argument("--transformer-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-large-real-tensor-load", action="store_true")
    parser.add_argument("--min-free-ram-gib", type=float, default=32.0)
    args = parser.parse_args()

    free_ram = available_ram_gib()
    if not args.allow_large_real_tensor_load:
        raise SystemExit(
            "Refusing to load real Lens transformer tensors by default. "
            "This path can allocate several GiB transiently and previously "
            "triggered a Windows MEMORY_MANAGEMENT crash on this machine. "
            "Pass --allow-large-real-tensor-load only after checking system "
            f"stability and RAM headroom. Current free RAM: {free_ram:.2f} GiB."
        )
    if free_ram < args.min_free_ram_gib:
        raise SystemExit(
            f"Refusing real Lens tensor load: free RAM {free_ram:.2f} GiB is "
            f"below --min-free-ram-gib {args.min_free_ram_gib:.2f} GiB."
        )

    module = load_lens_transformer_module(args.lens_src)
    block = module.LensTransformerBlock(
        dim=1536,
        num_attention_heads=24,
        attention_head_dim=64,
        rms_norm=True,
        gate_mlp=True,
    ).eval()

    state_names = list(block.state_dict().keys())
    state = {}
    transformer_dir = Path(args.transformer_dir)
    for name in state_names:
        state[name] = load_weight(transformer_dir, f"transformer_blocks.0.{name}")
    block.load_state_dict(state, strict=True)

    torch.manual_seed(5678)
    hidden = torch.randn(1, 4, 1536, dtype=torch.float32) * 0.01
    encoder = torch.randn(1, 3, 1536, dtype=torch.float32) * 0.01
    temb = torch.randn(1, 1536, dtype=torch.float32) * 0.01
    img_freqs = torch.polar(torch.ones(4, 32), torch.randn(4, 32) * 0.01)
    txt_freqs = torch.polar(torch.ones(3, 32), torch.randn(3, 32) * 0.01)
    attention_mask = torch.zeros(1, 1, 1, 7, dtype=torch.float32)
    attention_mask[:, :, :, -1] = float("-inf")

    with torch.no_grad():
        expected_encoder, expected_hidden = block(
            hidden_states=hidden,
            encoder_hidden_states=encoder,
            temb=temb,
            image_rotary_emb=(img_freqs, txt_freqs),
            attention_mask=attention_mask,
        )

    tensors = {
        "input.hidden": hidden,
        "input.encoder": encoder,
        "input.temb": temb,
        "input.img_freqs": torch.view_as_real(img_freqs),
        "input.txt_freqs": torch.view_as_real(txt_freqs),
        "input.attention_mask": attention_mask.reshape(1, 7),
        "expected.encoder": expected_encoder,
        "expected.hidden": expected_hidden,
    }
    tensors.update({f"state.{name}": value for name, value in block.state_dict().items()})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as out:
        out.write(b"LENSBLK1")
        out.write(struct.pack("<I", len(tensors)))
        for name, tensor in tensors.items():
            write_tensor(out, name, tensor)
    print(f"wrote {output} tensors={len(tensors)} real_block=0")


if __name__ == "__main__":
    main()
