#!/usr/bin/env python3
"""Compare the fork's Wan/Qwen VAE bridge output against ComfyUI.

This is a diagnostic harness for Anima/Qwen-image latent-format work. It
decodes a fork-exported f32 NCHW latent through ComfyUI's VAE in a few explicit
latent-format conventions and compares the result against a fork output image.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def image_to_u8_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def save_rgb_float(path: Path, image: np.ndarray) -> None:
    image_u8 = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(image_u8, mode="RGB").save(path)


def metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    diff = np.abs(af - bf)
    mse = float(np.mean((af - bf) ** 2))
    psnr = 99.0 if mse == 0.0 else float(20.0 * math.log10(255.0 / math.sqrt(mse)))
    return {
        "mean_abs": float(np.mean(diff)),
        "p95_abs": float(np.percentile(diff, 95.0)),
        "p99_abs": float(np.percentile(diff, 99.0)),
        "max_abs": float(np.max(diff)),
        "psnr": psnr,
    }


def make_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.float32) / 255.0
    # Make small differences visible without saturating everything.
    return np.clip(diff * 8.0, 0.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--latent", required=True)
    parser.add_argument("--fork-image", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    comfy_root = Path(args.comfy_root)
    sys.path.insert(0, str(comfy_root))
    os.chdir(comfy_root)

    import torch  # noqa: PLC0415
    import comfy.latent_formats  # noqa: PLC0415
    import comfy.sd  # noqa: PLC0415
    import comfy.utils  # noqa: PLC0415

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latent = np.load(args.latent).astype(np.float32)
    if latent.ndim != 4:
        raise ValueError(f"expected NCHW latent, got {latent.shape}")

    sd, metadata = comfy.utils.load_torch_file(args.vae, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    latent_nchw = torch.from_numpy(latent).to(device=device, dtype=torch.float32)
    latent_ncthw = latent_nchw.unsqueeze(2)
    wan = comfy.latent_formats.Wan21()

    variants: dict[str, torch.Tensor] = {
        "raw_ncthw": latent_ncthw,
        "wan21_process_out_ncthw": wan.process_out(latent_ncthw),
    }

    fork_rgb = image_to_u8_rgb(Path(args.fork_image))
    rows = []
    report = {
        "comfy_root": str(comfy_root),
        "vae_path": str(Path(args.vae)),
        "latent_path": str(Path(args.latent)),
        "fork_image": str(Path(args.fork_image)),
        "latent_shape_nchw": list(latent.shape),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "vae_dtype": str(getattr(vae, "vae_dtype", "")),
        "variants": {},
    }

    with torch.inference_mode():
        for name, tensor in variants.items():
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            decoded = vae.decode(tensor.detach().cpu())
            # Comfy VAE.decode returns NHWC float [0, 1].
            decoded_np = decoded.detach().cpu().numpy()
            decoded_shape = list(decoded_np.shape)
            if decoded_np.ndim == 5:
                # Comfy video VAEs return NHWTC/NTHWC depending on the VAE wrapper.
                # Wan image latents decode to a single-frame NHWTC tensor here.
                if decoded_np.shape[1] == 1 and decoded_np.shape[-1] in (3, 4):
                    decoded_np = decoded_np[:, 0]
                elif decoded_np.shape[3] == 1 and decoded_np.shape[-1] in (3, 4):
                    decoded_np = decoded_np[:, :, :, 0, :]
                else:
                    raise ValueError(f"unhandled decoded 5D image shape {decoded_shape}")
            if decoded_np.ndim != 4 or decoded_np.shape[-1] < 3:
                raise ValueError(f"unhandled decoded image shape {decoded_shape}")
            decoded_rgb = decoded_np[0, :, :, :3]
            image_path = out_dir / f"comfy_{name}.png"
            save_rgb_float(image_path, decoded_rgb)
            comfy_u8 = image_to_u8_rgb(image_path)
            if comfy_u8.shape != fork_rgb.shape:
                report["variants"][name] = {
                    "image_path": str(image_path),
                    "shape": list(comfy_u8.shape),
                    "fork_shape": list(fork_rgb.shape),
                    "shape_mismatch": True,
                }
                continue
            m = metrics(comfy_u8, fork_rgb)
            diff_path = out_dir / f"diff_{name}_vs_fork.png"
            save_rgb_float(diff_path, make_diff(comfy_u8, fork_rgb))
            contact = np.concatenate([fork_rgb, comfy_u8, image_to_u8_rgb(diff_path)], axis=1)
            contact_path = out_dir / f"contact_{name}_fork_comfy_diff.png"
            Image.fromarray(contact, mode="RGB").save(contact_path)
            report["variants"][name] = {
                "image_path": str(image_path),
                "decoded_shape": decoded_shape,
                "diff_path": str(diff_path),
                "contact_path": str(contact_path),
                **m,
            }
            rows.append((name, m["mean_abs"], m["p99_abs"], m["psnr"], str(contact_path)))

    report_path = out_dir / "anima_vae_bridge_parity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path = out_dir / "anima_vae_bridge_parity_report.md"
    lines = [
        "# Anima Wan/Qwen VAE Bridge Parity",
        "",
        f"- VAE dtype: `{report['vae_dtype']}`",
        f"- Latent shape: `{report['latent_shape_nchw']}`",
        "",
        "| variant | mean abs | p99 abs | PSNR | contact |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name, mean_abs, p99, psnr, contact_path in rows:
        lines.append(f"| `{name}` | {mean_abs:.4f} | {p99:.4f} | {psnr:.2f} | `{contact_path}` |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    print(f"wrote {md_path}")
    for row in rows:
        print(f"variant={row[0]} mean_abs={row[1]:.4f} p99={row[2]:.4f} psnr={row[3]:.2f} contact={row[4]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
