#!/usr/bin/env python3
"""Compare sdcpp COMFY_NORMAL VAE output with ComfyUI normal VAE.

The harness runs full-frame normal VAE only. It does not enable TAESD or tiled
VAE. The sdcpp side uses sd-latent-smoke so the public latent/image handles stay
inside the DLL and the normal API reports can be parsed from stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


REPORT_RE = re.compile(r"(\w+)=((?:\"[^\"]*\")|\S+)")


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(image, 0.0, 1.0)
    Image.fromarray(np.round(image * 255.0).astype(np.uint8), "RGB").save(path)


def image_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    mse = float(np.mean((a - b) ** 2))
    return {
        "mean_abs_diff": float(np.mean(diff)),
        "p95_abs_diff": float(np.quantile(diff, 0.95)),
        "p99_abs_diff": float(np.quantile(diff, 0.99)),
        "p999_abs_diff": float(np.quantile(diff, 0.999)),
        "max_abs_diff": float(np.max(diff)),
        "psnr": float("inf") if mse == 0.0 else 10.0 * math.log10(1.0 / mse),
    }


def parse_value(value: str) -> Any:
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def parse_report_line(text: str) -> dict[str, Any]:
    return {key: parse_value(value) for key, value in REPORT_RE.findall(text)}


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_commit(repo_root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-c", f"safe.directory={repo_root.as_posix()}", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        )
        return proc.stdout.strip()
    except Exception:
        return None


def run_sdcpp(args: argparse.Namespace, out_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    output = out_dir / "sdcpp_comfy_normal_roundtrip.png"
    stdout_path = out_dir / "sdcpp.stdout.log"
    stderr_path = out_dir / "sdcpp.stderr.log"
    cmd = [
        str(args.sdcpp_smoke),
        "--model",
        str(args.model),
        "--image",
        str(args.image),
        "--image-channels",
        "3",
        "--type-f16",
        "--split-decode-context",
    ]
    if args.mode == "encode":
        cmd.append("--no-decode")
    else:
        cmd.extend(["--output", str(output)])

    env = os.environ.copy()
    env["SDCPP_VAE_STRICT_COMFY_NORMAL"] = "1"
    env.pop("SDCPP_DISABLE_COMFY_NORMAL_VAE", None)
    env.pop("SDCPP_DISABLE_DEFAULT_VAE_CONV_DIRECT", None)

    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, env=env, check=False, timeout=args.timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(exc.stdout or "", encoding="utf-8")
        stderr_path.write_text(exc.stderr or "", encoding="utf-8")
        raise RuntimeError(f"sd-latent-smoke timed out after {args.timeout_seconds}s") from exc

    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"sd-latent-smoke failed with exit code {proc.returncode}; see {stdout_path} and {stderr_path}")

    reports: dict[str, Any] = {}
    combined = proc.stdout + "\n" + proc.stderr
    for name in ("encode_report", "decode_report", "estimate_encode_report", "estimate_decode_report"):
        for line in combined.splitlines():
            marker = f"{name} "
            if marker not in line:
                continue
            segment = line[line.index(name):]
            reports[name] = parse_report_line(segment)
            reports[name]["raw"] = segment
            break
    return (None if args.mode == "encode" else output), reports


def load_comfy(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    comfy_root = Path(args.comfy_root).resolve()
    if not comfy_root.exists():
        raise RuntimeError(f"ComfyUI root not found: {comfy_root}")
    sys.path.insert(0, str(comfy_root))
    os.chdir(comfy_root)

    import torch  # type: ignore
    import comfy.sd  # type: ignore
    import comfy.model_management  # type: ignore

    device = comfy.model_management.get_torch_device()
    model, clip, vae, clipvision = comfy.sd.load_checkpoint_guess_config(
        str(args.model),
        output_vae=True,
        output_clip=False,
        output_clipvision=False,
        embedding_directory=None,
    )
    del model, clip, clipvision
    metadata = {
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(device) if torch.cuda.is_available() else str(device),
        "comfy_vae_dtype": str(getattr(vae, "vae_dtype", "unknown")),
    }
    return torch, vae, metadata


def run_comfy_roundtrip(args: argparse.Namespace, out_dir: Path) -> tuple[Path, dict[str, Any]]:
    torch, vae, metadata = load_comfy(args)
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else next(iter(vae.first_stage_model.parameters())).device
    image = read_rgb(args.image)
    pixels = torch.from_numpy(image)[None, ...].to(device=device, dtype=torch.float32)

    with torch.inference_mode():
        warm_latent = vae.encode(pixels)
        warm_decoded = vae.decode(warm_latent)
        del warm_latent, warm_decoded
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        latent = vae.encode(pixels)
        decoded = vae.decode(latent).detach().float().cpu().numpy()[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    peak_allocated = float(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else 0.0
    peak_reserved = float(torch.cuda.max_memory_reserved(device)) if torch.cuda.is_available() else 0.0

    output = out_dir / "comfy_normal_roundtrip.png"
    write_rgb(output, decoded)
    metadata.update(
        {
            "comfy_peak_allocated_mb": peak_allocated / 1024.0 / 1024.0,
            "comfy_peak_reserved_mb": peak_reserved / 1024.0 / 1024.0,
        }
    )
    return output, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--sdcpp-smoke", type=Path, required=True)
    parser.add_argument("--sdcpp-dll", type=Path)
    parser.add_argument("--checkpoint", "--model", dest="model", type=Path, required=True)
    parser.add_argument("--latent", type=Path, help="Reserved for future raw latent parity inputs; current smoke path uses --image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["decode", "encode", "roundtrip", "all"], default="all")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()

    if args.latent and args.mode == "decode":
        raise SystemExit("--latent raw decode parity is not implemented by sd-latent-smoke yet; use --mode roundtrip or --mode all.")

    repo_root = Path(__file__).resolve().parents[1]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dll_path = args.sdcpp_dll or (Path(args.sdcpp_smoke).resolve().parent / "stable-diffusion.dll")

    sdcpp_image, sdcpp_reports = run_sdcpp(args, args.out_dir)
    comfy_image = None
    comfy_report: dict[str, Any] = {}
    diff_report: dict[str, Any] = {}
    diff_image = None

    if args.mode in ("decode", "roundtrip", "all"):
        comfy_image, comfy_report = run_comfy_roundtrip(args, args.out_dir)
        if sdcpp_image is not None:
            sdcpp_pixels = read_rgb(sdcpp_image)
            comfy_pixels = read_rgb(comfy_image)
            diff_report = image_metrics(sdcpp_pixels, comfy_pixels)
            diff_image = args.out_dir / "vae_absdiff_x8.png"
            write_rgb(diff_image, np.abs(sdcpp_pixels - comfy_pixels) * 8.0)

    report = {
        "comfy_root": str(Path(args.comfy_root).resolve()),
        "mode": args.mode,
        "model_path": str(Path(args.model).resolve()),
        "latent_path": str(Path(args.latent).resolve()) if args.latent else None,
        "image_path": str(Path(args.image).resolve()),
        "sdcpp_smoke": str(Path(args.sdcpp_smoke).resolve()),
        "sdcpp_dll_path": str(Path(dll_path).resolve()),
        "sdcpp_dll_sha256": sha256_file(Path(dll_path)),
        "sdcpp_commit_hash": repo_commit(repo_root),
        "sdcpp_image": str(sdcpp_image) if sdcpp_image else None,
        "comfy_image": str(comfy_image) if comfy_image else None,
        "diff_image": str(diff_image) if diff_image else None,
        "sdcpp_reports": sdcpp_reports,
        "comfy_report": comfy_report,
        "metrics": diff_report,
    }
    report_path = args.out_dir / "vae_comfy_parity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
