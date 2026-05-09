#!/usr/bin/env python3
"""Smoke-check COMFY_NORMAL VAE C API exports from stable-diffusion.dll."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path


SD_VAE_API_VERSION = 1


class SdVaeCapabilities(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("supports_comfy_normal", ctypes.c_bool),
        ("supports_device_resident_stages", ctypes.c_bool),
        ("supports_bf16_storage", ctypes.c_bool),
        ("supports_f16_storage", ctypes.c_bool),
        ("supports_normal_encode", ctypes.c_bool),
        ("supports_normal_decode", ctypes.c_bool),
        ("supports_memory_report", ctypes.c_bool),
        ("supports_no_im2col_guard", ctypes.c_bool),
        ("reserved", ctypes.c_uint32 * 8),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--dependency-dir", type=Path, action="append", default=[])
    args = parser.parse_args()

    dll_path = args.dll.resolve()
    if not dll_path.exists():
        raise SystemExit(f"DLL not found: {dll_path}")

    os.environ["PATH"] = str(dll_path.parent) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(dll_path.parent))
        for dependency_dir in args.dependency_dir:
            os.add_dll_directory(str(dependency_dir.resolve()))
    for dependency_dir in args.dependency_dir:
        os.environ["PATH"] = str(dependency_dir.resolve()) + os.pathsep + os.environ.get("PATH", "")

    dll = ctypes.CDLL(str(dll_path))
    required_exports = [
        "sd_decode_latent_normal",
        "sd_encode_image_normal",
        "sd_estimate_vae_normal_memory",
        "sd_get_vae_capabilities",
    ]
    missing = [name for name in required_exports if not hasattr(dll, name)]
    if missing:
        raise SystemExit(f"missing exports: {', '.join(missing)}")

    dll.sd_get_vae_capabilities.argtypes = [ctypes.c_void_p, ctypes.POINTER(SdVaeCapabilities)]
    dll.sd_get_vae_capabilities.restype = ctypes.c_bool

    caps = SdVaeCapabilities()
    caps.struct_size = ctypes.sizeof(SdVaeCapabilities)
    ok = bool(dll.sd_get_vae_capabilities(None, ctypes.byref(caps)))
    result = {
        "dll": str(dll_path),
        "exports": required_exports,
        "sd_get_vae_capabilities_ok": ok,
        "struct_size": int(caps.struct_size),
        "version": int(caps.version),
        "supports_comfy_normal": bool(caps.supports_comfy_normal),
        "supports_device_resident_stages": bool(caps.supports_device_resident_stages),
        "supports_normal_encode": bool(caps.supports_normal_encode),
        "supports_normal_decode": bool(caps.supports_normal_decode),
        "supports_no_im2col_guard": bool(caps.supports_no_im2col_guard),
        "supports_memory_report": bool(caps.supports_memory_report),
        "supports_bf16_storage": bool(caps.supports_bf16_storage),
        "supports_f16_storage": bool(caps.supports_f16_storage),
    }
    print(json.dumps(result, indent=2))

    checks = [
        ok,
        caps.version == SD_VAE_API_VERSION,
        caps.supports_comfy_normal,
        caps.supports_normal_encode,
        caps.supports_normal_decode,
        caps.supports_no_im2col_guard,
        caps.supports_memory_report,
    ]
    if not all(checks):
        raise SystemExit("capability smoke failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
