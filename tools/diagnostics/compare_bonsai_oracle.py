#!/usr/bin/env python3
"""Compare Bonsai Python oracle tensors against sd.cpp dumps.

This is diagnostic-only and intentionally supports the narrow dump formats
used by the Bonsai GemLite INT1 spike.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_sdcpp_json(path: Path) -> np.ndarray:
    meta = json.loads(path.read_text(encoding="utf-8"))
    if "npy" in meta:
        return np.load(meta["npy"])
    if "binary" not in meta:
        raise ValueError(f"{path} has neither npy nor binary field")
    dtype = {"f16": np.float16, "f32": np.float32}[meta["dtype"]]
    return np.fromfile(meta["binary"], dtype=dtype).reshape(meta["shape"])


def compare(label: str, a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    diff = np.abs(a - b)
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    cosine = float(np.dot(a, b) / (an * bn)) if an > 0 and bn > 0 else float("nan")
    corr = float(np.corrcoef(a, b)[0, 1]) if n > 1 else float("nan")
    return {
        "label": label,
        "count": int(n),
        "mean_abs": float(diff.mean()),
        "max_abs": float(diff.max()),
        "cosine": cosine,
        "corr": corr,
        "a_min": float(np.nanmin(a)),
        "a_max": float(np.nanmax(a)),
        "a_mean": float(np.nanmean(a)),
        "a_std": float(np.nanstd(a)),
        "b_min": float(np.nanmin(b)),
        "b_max": float(np.nanmax(b)),
        "b_mean": float(np.nanmean(b)),
        "b_std": float(np.nanstd(b)),
    }


def compare_if_exists(results: list[dict[str, Any]], label: str, a_path: Path, b_path: Path) -> None:
    if not a_path.exists() or not b_path.exists():
        results.append(
            {
                "label": label,
                "missing": True,
                "python_exists": a_path.exists(),
                "sdcpp_exists": b_path.exists(),
            }
        )
        return
    results.append(compare(label, np.load(a_path), load_sdcpp_json(b_path)))


def split_qkv(arr: np.ndarray, tokens: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    work = arr.reshape(tokens, 9216)
    return work[:, 0:3072], work[:, 3072:6144], work[:, 6144:9216]


def compare_qkv_chunks(
    results: list[dict[str, Any]],
    label: str,
    py_q: np.ndarray,
    py_k: np.ndarray,
    py_v: np.ndarray,
    sd_qkv: np.ndarray,
    tokens: int,
) -> None:
    sd_q, sd_k, sd_v = split_qkv(sd_qkv, tokens)
    py_items = {"q": py_q, "k": py_k, "v": py_v}
    sd_items = {"q": sd_q, "k": sd_k, "v": sd_v}
    for name in ("q", "k", "v"):
        results.append(compare(f"{label}.{name}_output expected_chunk", py_items[name], sd_items[name]))
        best = min(
            ((chunk, compare(f"{label}.{name}_output vs sd_{chunk}", py_items[name], value)) for chunk, value in sd_items.items()),
            key=lambda item: item[1]["mean_abs"],
        )
        best_result = best[1]
        best_result["best_sd_chunk"] = best[0]
        results.append(best_result)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Bonsai oracle and sd.cpp tensors.")
    parser.add_argument("--python-dir", required=True)
    parser.add_argument("--sdcpp-dump-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    py = Path(args.python_dir)
    sd = Path(args.sdcpp_dump_dir)
    results: list[dict[str, Any]] = []

    # Python stores prompt embeds as [1,512,7680]. The sd.cpp debug NPY's
    # metadata says [7680,512], but the exported flat data compares correctly
    # as [512,7680] for this diagnostic boundary.
    py_cond = np.load(py / "conditioning.prompt_embeds_bsz_seq_7680.npy")[0]
    sd_cond = np.load(sd / "0000_generate_image_cond_crossattn.npy").reshape(512, 7680)
    results.append(compare("conditioning.prompt_embeds [512,7680]", py_cond, sd_cond))

    py_txt = np.load(py / "transformer_blocks.0.input_encoder_hidden_states.npy")[0]
    sd_txt = load_sdcpp_json(sd / "dbg_0000_block00_double_txt_stream_input.json")
    results.append(compare("block00.text_stream_input [512,3072]", py_txt, sd_txt))

    py_qin = np.load(py / "transformer_blocks.0.attn.add_q_proj.input_arg0.npy")[0]
    sd_qin = load_sdcpp_json(sd / "dbg_0004_block00_double_txt_qkv_input_after_modulate.json")
    results.append(compare("block00.text_qkv_input_after_modulate [512,3072]", py_qin, sd_qin))

    py_noise = np.load(py / "latents.initial_packed.npy")[0]
    sd_noise_raw = np.load(sd / "0001_cpu_sampler_step1_noised_input.npy")
    sd_noise_legacy = sd_noise_raw.reshape(32, 32, 128, 1)[:, :, :, 0].reshape(1024, 128)
    results.append(compare("initial_packed_latent cpu_dump_legacy_view [1024,128]", py_noise, sd_noise_legacy))
    img_in_input = next(sd.glob("*flux2_img_in_input.json"), None)
    if img_in_input is not None:
        results.append(compare("x_embedder.input / flux2.img_in.input [1024,128]", np.load(py / "x_embedder.input_arg0.npy")[0], load_sdcpp_json(img_in_input)))
    img_in_output = next(sd.glob("*flux2_img_in_output.json"), None)
    if img_in_output is not None:
        results.append(compare("x_embedder.output / flux2.img_in.output [1024,3072]", np.load(py / "x_embedder.output.npy")[0], load_sdcpp_json(img_in_output)))

    py_out = np.load(py / "step0.model_output.npy")[0]
    sd_out = np.load(sd / "0002_cpu_sampler_step1_cond_model_output.npy").reshape(32, 32, 128, 1)[:, :, :, 0].reshape(1024, 128)
    results.append(compare("step0.model_output [1024,128]", py_out, sd_out))

    py_q = np.load(py / "transformer_blocks.0.attn.add_q_proj.output.npy")[0]
    sd_qkv = load_sdcpp_json(sd / "0001_model_diffusion_model_double_blocks_0_txt_attn_qkv_weight_q_output.json")
    sd_q = sd_qkv.reshape(512, 9216)[:, :3072]
    results.append(compare("block00.text_q_output [512,3072]", py_q, sd_q))

    compare_if_exists(
        results,
        "block00.image_stream_input [1024,3072]",
        py / "transformer_blocks.0.input_hidden_states.npy",
        sd / "dbg_0005_block00_double_img_stream_input.json",
    )
    compare_if_exists(
        results,
        "block00.image_norm1_output [1024,3072]",
        py / "transformer_blocks.0.norm1.output.npy",
        sd / "dbg_0006_block00_double_img_norm1_output.json",
    )
    compare_if_exists(
        results,
        "block00.image_qkv_input_after_modulate [1024,3072]",
        py / "transformer_blocks.0.attn.to_q.input_arg0.npy",
        sd / "dbg_0009_block00_double_img_qkv_input_after_modulate.json",
    )
    py_img_q = np.load(py / "transformer_blocks.0.attn.to_q.output.npy")[0]
    py_img_k = np.load(py / "transformer_blocks.0.attn.to_k.output.npy")[0]
    py_img_v = np.load(py / "transformer_blocks.0.attn.to_v.output.npy")[0]
    sd_img_qkv = load_sdcpp_json(sd / "0003_model_diffusion_model_double_blocks_0_img_attn_qkv_weight_q_output.json")
    compare_qkv_chunks(results, "block00.image", py_img_q, py_img_k, py_img_v, sd_img_qkv, 1024)

    py_txt_q = np.load(py / "transformer_blocks.0.attn.add_q_proj.output.npy")[0]
    py_txt_k = np.load(py / "transformer_blocks.0.attn.add_k_proj.output.npy")[0]
    py_txt_v = np.load(py / "transformer_blocks.0.attn.add_v_proj.output.npy")[0]
    compare_qkv_chunks(results, "block00.text", py_txt_q, py_txt_k, py_txt_v, sd_qkv, 512)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"comparisons": results}, indent=2), encoding="utf-8")
    for item in results:
        if item.get("missing"):
            print(
                f"{item['label']}: missing python_exists={item['python_exists']} "
                f"sdcpp_exists={item['sdcpp_exists']}"
            )
            continue
        print(
            f"{item['label']}: mean_abs={item['mean_abs']:.6g} "
            f"max_abs={item['max_abs']:.6g} corr={item['corr']:.6g} cosine={item['cosine']:.6g}"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
