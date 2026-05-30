import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def tensor_stats(t):
    tf = t.detach().float()
    finite = torch.isfinite(tf)
    finite_vals = tf[finite]
    if finite_vals.numel() == 0:
        return {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "finite": int(finite.sum().item()),
            "nan": int(torch.isnan(tf).sum().item()),
            "inf": int(torch.isinf(tf).sum().item()),
        }
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "finite": int(finite.sum().item()),
        "nan": int(torch.isnan(tf).sum().item()),
        "inf": int(torch.isinf(tf).sum().item()),
        "min": float(finite_vals.min().item()),
        "max": float(finite_vals.max().item()),
        "mean": float(finite_vals.mean().item()),
        "std": float(finite_vals.std(unbiased=False).item()),
    }


def percentile(values, p):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[idx]


def main():
    parser = argparse.ArgumentParser(description="GemLite one-shape Bonsai INT1 oracle.")
    parser.add_argument("--pack", required=True)
    parser.add_argument("--linear", default="single_transformer_blocks.0.attn.to_qkv_mlp_proj")
    parser.add_argument("--m", type=int, default=1536)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--manual-matmul", default="", help="Optional GemLite matmul type override, for example GEMM.")
    args = parser.parse_args()

    from gemlite.core import (
        GEMLITE_MATMUL_TYPES_MAPPING,
        GEMLITE_TRITON_CONFIG_CACHE,
        GemLiteLinearTriton,
        get_matmul_type,
    )

    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GemLite timing")

    pack = Path(args.pack)
    state = torch.load(pack, map_location="cpu")
    prefix = args.linear
    required = ["W_q", "scales", "zeros", "metadata", "orig_shape"]
    missing = [suffix for suffix in required if f"{prefix}.{suffix}" not in state]
    if missing:
        raise RuntimeError(f"missing GemLite tensors for {prefix}: {missing}")

    metadata = state[f"{prefix}.metadata"].tolist()
    orig_shape = state[f"{prefix}.orig_shape"].tolist()
    n, k = int(orig_shape[0]), int(orig_shape[1])
    w_nbits = int(metadata[1])
    group_size = int(metadata[2])
    elements_per_sample = int(metadata[4])
    input_dtype_value = int(metadata[5])
    type_id = input_dtype_value * 100 + w_nbits
    matmul_type = args.manual_matmul if args.manual_matmul else get_matmul_type(args.m, w_nbits, False)

    module = GemLiteLinearTriton()
    module.load_state_dict({
        "W_q": state[f"{prefix}.W_q"].cuda(non_blocking=False),
        "scales": state[f"{prefix}.scales"].cuda(non_blocking=False),
        "zeros": state[f"{prefix}.zeros"].cuda(non_blocking=False),
        "metadata": state[f"{prefix}.metadata"].cuda(non_blocking=False),
        "orig_shape": state[f"{prefix}.orig_shape"].cuda(non_blocking=False),
    })
    x = torch.randn((args.m, k), device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()

    def run_once():
        if args.manual_matmul:
            return module.forward_manual(x, args.manual_matmul)
        return module(x)

    compile_start = time.perf_counter()
    y = run_once()
    torch.cuda.synchronize()
    compile_plus_first_ms = (time.perf_counter() - compile_start) * 1000.0

    for _ in range(args.warmup):
        y = run_once()
    torch.cuda.synchronize()

    times = []
    for _ in range(args.runs):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        y = run_once()
        stop.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(stop)))

    cache_dump = Path(args.out).with_name("gemlite_autotune_cache_after_run.json")
    try:
        GemLiteLinearTriton.cache_config(str(cache_dump))
    except Exception as exc:
        cache_dump.write_text(json.dumps({"error": str(exc)}, indent=2), encoding="utf-8")

    cache_entry = None
    cache_source = "GEMLITE_TRITON_CONFIG_CACHE"
    cache_key = str((args.m, n, k, group_size, elements_per_sample, type_id))
    if matmul_type in GEMLITE_TRITON_CONFIG_CACHE:
        cache_entry = GEMLITE_TRITON_CONFIG_CACHE[matmul_type].get(cache_key)
    if cache_entry is None and cache_dump.exists():
        try:
            dumped = json.loads(cache_dump.read_text(encoding="utf-8"))
            cache_entry = dumped.get(matmul_type, {}).get(cache_key)
            if cache_entry is not None:
                cache_source = str(cache_dump)
        except Exception:
            pass

    output = {
        "linear": prefix,
        "pack": str(pack),
        "device": torch.cuda.get_device_name(0),
        "M": args.m,
        "K": k,
        "N": n,
        "W_q_shape": list(state[f"{prefix}.W_q"].shape),
        "scales_shape": list(state[f"{prefix}.scales"].shape),
        "zeros_shape": list(state[f"{prefix}.zeros"].shape),
        "metadata": metadata,
        "W_nbits": w_nbits,
        "group_size": group_size,
        "packing_bitwidth": int(metadata[4] * w_nbits),
        "elements_per_sample": elements_per_sample,
        "W_group_mode": int(metadata[10]),
        "type_id": type_id,
        "matmul_type": matmul_type,
        "cache_key": cache_key,
        "cache_entry": cache_entry,
        "cache_source": cache_source,
        "cache_dump": str(cache_dump),
        "compile_plus_first_ms": compile_plus_first_ms,
        "warmup": args.warmup,
        "runs": args.runs,
        "times_ms": times,
        "median_ms": statistics.median(times),
        "mean_ms": statistics.mean(times),
        "p90_ms": percentile(times, 0.9),
        "output_stats": tensor_stats(y),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "linear": prefix,
        "shape": [args.m, k, n],
        "matmul_type": matmul_type,
        "median_ms": output["median_ms"],
        "mean_ms": output["mean_ms"],
        "cache_key": cache_key,
        "cache_entry": cache_entry,
        "cache_source": cache_source,
    }, indent=2))


if __name__ == "__main__":
    main()
