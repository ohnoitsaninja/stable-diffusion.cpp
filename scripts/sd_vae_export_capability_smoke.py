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


class SdGpuCapabilities(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("supports_gpu_handles", ctypes.c_bool),
        ("supports_cuda_gpu_handles", ctypes.c_bool),
        ("supports_gpu_latent_output", ctypes.c_bool),
        ("supports_gpu_latent_input", ctypes.c_bool),
        ("supports_sampler_gpu_latent_output", ctypes.c_bool),
        ("supports_vae_gpu_latent_input", ctypes.c_bool),
        ("supports_vae_encode_gpu_latent_output", ctypes.c_bool),
        ("supports_vae_encode_gpu_latent_bridge_output", ctypes.c_bool),
        ("supports_gpu_image_output", ctypes.c_bool),
        ("supports_gpu_image_to_rgba8", ctypes.c_bool),
        ("supports_gpu_download", ctypes.c_bool),
        ("supports_gpu_latent_download", ctypes.c_bool),
        ("supports_gpu_latent_upload", ctypes.c_bool),
        ("supports_dlpack_export", ctypes.c_bool),
        ("supports_cuda_pointer_borrow", ctypes.c_bool),
        ("supports_cuda_ipc_export", ctypes.c_bool),
        ("supports_external_memory_interop", ctypes.c_bool),
        ("reserved", ctypes.c_uint32 * 9),
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
        "sd_get_gpu_capabilities",
        "sd_sample_latent_gpu",
        "sd_gpu_latent_download",
        "sd_cpu_latent_upload",
        "sd_decode_latent_normal_gpu",
        "sd_decode_gpu_latent_normal_gpu",
        "sd_gpu_image_download",
        "sd_gpu_image_download_to_buffer",
        "sd_free_downloaded_image",
        "sd_gpu_handle_get_desc",
        "sd_gpu_handle_retain",
        "sd_gpu_handle_release",
        "sd_gpu_handle_borrow_cuda_ptr",
    ]
    missing = [name for name in required_exports if not hasattr(dll, name)]
    if missing:
        raise SystemExit(f"missing exports: {', '.join(missing)}")

    dll.sd_get_vae_capabilities.argtypes = [ctypes.c_void_p, ctypes.POINTER(SdVaeCapabilities)]
    dll.sd_get_vae_capabilities.restype = ctypes.c_bool
    dll.sd_get_gpu_capabilities.argtypes = [ctypes.c_void_p, ctypes.POINTER(SdGpuCapabilities)]
    dll.sd_get_gpu_capabilities.restype = ctypes.c_bool

    caps = SdVaeCapabilities()
    caps.struct_size = ctypes.sizeof(SdVaeCapabilities)
    ok = bool(dll.sd_get_vae_capabilities(None, ctypes.byref(caps)))
    gpu_caps = SdGpuCapabilities()
    gpu_caps.struct_size = ctypes.sizeof(SdGpuCapabilities)
    gpu_ok = bool(dll.sd_get_gpu_capabilities(None, ctypes.byref(gpu_caps)))
    result = {
        "dll": str(dll_path),
        "exports": required_exports,
        "sd_get_vae_capabilities_ok": ok,
        "sd_get_gpu_capabilities_ok": gpu_ok,
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
        "gpu": {
            "struct_size": int(gpu_caps.struct_size),
            "version": int(gpu_caps.version),
            "supports_gpu_handles": bool(gpu_caps.supports_gpu_handles),
            "supports_cuda_gpu_handles": bool(gpu_caps.supports_cuda_gpu_handles),
            "supports_gpu_latent_output": bool(gpu_caps.supports_gpu_latent_output),
            "supports_gpu_latent_input": bool(gpu_caps.supports_gpu_latent_input),
            "supports_sampler_gpu_latent_output": bool(gpu_caps.supports_sampler_gpu_latent_output),
            "supports_vae_gpu_latent_input": bool(gpu_caps.supports_vae_gpu_latent_input),
            "supports_vae_encode_gpu_latent_output": bool(gpu_caps.supports_vae_encode_gpu_latent_output),
            "supports_vae_encode_gpu_latent_bridge_output": bool(gpu_caps.supports_vae_encode_gpu_latent_bridge_output),
            "supports_gpu_image_output": bool(gpu_caps.supports_gpu_image_output),
            "supports_gpu_download": bool(gpu_caps.supports_gpu_download),
            "supports_gpu_latent_download": bool(gpu_caps.supports_gpu_latent_download),
            "supports_gpu_latent_upload": bool(gpu_caps.supports_gpu_latent_upload),
            "supports_cuda_pointer_borrow": bool(gpu_caps.supports_cuda_pointer_borrow),
        },
    }
    print(json.dumps(result, indent=2))

    checks = [
        ok,
        gpu_ok,
        caps.version == SD_VAE_API_VERSION,
        gpu_caps.version == SD_VAE_API_VERSION,
        caps.supports_comfy_normal,
        caps.supports_normal_encode,
        caps.supports_normal_decode,
        caps.supports_no_im2col_guard,
        caps.supports_memory_report,
        gpu_caps.supports_gpu_handles,
        gpu_caps.supports_cuda_gpu_handles,
        gpu_caps.supports_gpu_latent_output,
        gpu_caps.supports_gpu_latent_input,
        gpu_caps.supports_vae_gpu_latent_input,
        not gpu_caps.supports_vae_encode_gpu_latent_output,
        not gpu_caps.supports_vae_encode_gpu_latent_bridge_output,
        gpu_caps.supports_gpu_image_output,
        gpu_caps.supports_gpu_download,
        gpu_caps.supports_gpu_latent_download,
        gpu_caps.supports_gpu_latent_upload,
        gpu_caps.supports_cuda_pointer_borrow,
    ]
    if not all(checks):
        raise SystemExit("capability smoke failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
