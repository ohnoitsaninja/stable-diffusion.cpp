#!/usr/bin/env python3
"""Compare native Lens-Turbo schedule traces against Diffusers.

This is a scheduler-only oracle. It loads the local stable-diffusion.dll with
ctypes and compares sd_lens_turbo_build_schedule(...) with Diffusers'
FlowMatchEulerDiscreteScheduler using the Lens-Turbo scheduler_config.json.
"""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path

import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler


SD_VAE_API_VERSION = 1
DLL_DIRECTORIES = []


class LensScheduleOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("steps", ctypes.c_int),
        ("image_seq_len", ctypes.c_int),
        ("num_train_timesteps", ctypes.c_int),
        ("base_image_seq_len", ctypes.c_int),
        ("max_image_seq_len", ctypes.c_int),
        ("base_shift", ctypes.c_float),
        ("max_shift", ctypes.c_float),
        ("shift", ctypes.c_float),
        ("mu", ctypes.c_float),
        ("has_mu", ctypes.c_bool),
        ("use_dynamic_shifting", ctypes.c_bool),
        ("reserved", ctypes.c_uint32 * 8),
    ]


class LensScheduleDesc(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("steps", ctypes.c_int),
        ("sigma_count", ctypes.c_int),
        ("timestep_count", ctypes.c_int),
        ("image_seq_len", ctypes.c_int),
        ("mu", ctypes.c_float),
        ("use_dynamic_shifting", ctypes.c_bool),
        ("reserved", ctypes.c_uint32 * 8),
    ]


def parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dll", required=True, help="Path to stable-diffusion.dll")
    parser.add_argument("--scheduler-dir", required=True, help="Lens-Turbo root containing scheduler/")
    parser.add_argument("--steps", default="1,2,4,8")
    parser.add_argument("--image-seq-lens", default="256,1024,4096")
    parser.add_argument("--atol", type=float, default=1.0e-4)
    return parser.parse_args()


def calculate_shift(image_seq_len: int, steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * steps + b)


def load_dll(path: Path) -> ctypes.CDLL:
    if os.name == "nt":
        DLL_DIRECTORIES.append(os.add_dll_directory(str(path.parent)))
        for search_dir in os.environ.get("PATH", "").split(os.pathsep):
            if not search_dir:
                continue
            candidate = Path(search_dir) / "cublas64_13.dll"
            if candidate.exists():
                DLL_DIRECTORIES.append(os.add_dll_directory(str(candidate.parent)))
        os.environ["PATH"] = str(path.parent) + os.pathsep + os.environ.get("PATH", "")
    dll = ctypes.CDLL(str(path))
    dll.sd_lens_schedule_options_init.argtypes = [ctypes.POINTER(LensScheduleOptions)]
    dll.sd_lens_schedule_options_init.restype = None
    dll.sd_lens_schedule_desc_init.argtypes = [ctypes.POINTER(LensScheduleDesc)]
    dll.sd_lens_schedule_desc_init.restype = None
    dll.sd_lens_turbo_build_schedule.argtypes = [
        ctypes.POINTER(LensScheduleOptions),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.POINTER(LensScheduleDesc),
    ]
    dll.sd_lens_turbo_build_schedule.restype = ctypes.c_bool
    return dll


def native_trace(
    dll: ctypes.CDLL,
    scheduler: FlowMatchEulerDiscreteScheduler,
    steps: int,
    image_seq_len: int,
) -> tuple[np.ndarray, np.ndarray, LensScheduleDesc]:
    options = LensScheduleOptions()
    desc = LensScheduleDesc()
    dll.sd_lens_schedule_options_init(ctypes.byref(options))
    dll.sd_lens_schedule_desc_init(ctypes.byref(desc))

    config = scheduler.config
    options.steps = steps
    options.image_seq_len = image_seq_len
    options.num_train_timesteps = int(config.num_train_timesteps)
    options.base_image_seq_len = int(config.base_image_seq_len)
    options.max_image_seq_len = int(config.max_image_seq_len)
    options.base_shift = float(config.base_shift)
    options.max_shift = float(config.max_shift)
    options.shift = float(config.shift)
    options.has_mu = False
    options.use_dynamic_shifting = bool(config.use_dynamic_shifting)

    sigmas = (ctypes.c_float * (steps + 1))()
    timesteps = (ctypes.c_float * steps)()
    ok = dll.sd_lens_turbo_build_schedule(
        ctypes.byref(options),
        sigmas,
        steps + 1,
        timesteps,
        steps,
        ctypes.byref(desc),
    )
    if not ok:
        raise RuntimeError(f"native scheduler failed for steps={steps} image_seq_len={image_seq_len}")
    return (
        np.fromiter(sigmas, dtype=np.float32),
        np.fromiter(timesteps, dtype=np.float32),
        desc,
    )


def diffusers_trace(
    scheduler: FlowMatchEulerDiscreteScheduler,
    steps: int,
    image_seq_len: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    mu = calculate_shift(image_seq_len, steps)
    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    scheduler.set_timesteps(sigmas=sigmas, device="cpu", mu=mu)
    return (
        scheduler.sigmas.detach().cpu().numpy().astype(np.float32),
        scheduler.timesteps.detach().cpu().numpy().astype(np.float32),
        mu,
    )


def main() -> int:
    args = parse_args()
    dll = load_dll(Path(args.dll).resolve())
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.scheduler_dir, subfolder="scheduler")

    failures: list[str] = []
    for steps in parse_csv_ints(args.steps):
        for image_seq_len in parse_csv_ints(args.image_seq_lens):
            native_sigmas, native_timesteps, desc = native_trace(dll, scheduler, steps, image_seq_len)
            oracle_sigmas, oracle_timesteps, mu = diffusers_trace(scheduler, steps, image_seq_len)
            sigma_delta = np.abs(native_sigmas - oracle_sigmas)
            timestep_delta = np.abs(native_timesteps - oracle_timesteps)
            desc_ok = (
                desc.steps == steps
                and desc.sigma_count == steps + 1
                and desc.timestep_count == steps
                and desc.image_seq_len == image_seq_len
                and abs(float(desc.mu) - mu) <= args.atol
                and bool(desc.use_dynamic_shifting) == bool(scheduler.config.use_dynamic_shifting)
            )
            ok = desc_ok and sigma_delta.max(initial=0.0) <= args.atol and timestep_delta.max(initial=0.0) <= args.atol
            print(
                "Lens scheduler oracle:",
                f"steps={steps}",
                f"image_seq_len={image_seq_len}",
                f"mu={mu:.8f}",
                f"max_sigma_abs={float(sigma_delta.max(initial=0.0)):.8g}",
                f"max_timestep_abs={float(timestep_delta.max(initial=0.0)):.8g}",
                f"desc_ok={desc_ok}",
            )
            if not ok:
                failures.append(f"steps={steps} image_seq_len={image_seq_len}")

    if failures:
        raise RuntimeError("scheduler oracle mismatch: " + ", ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
