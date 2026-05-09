#!/usr/bin/env python3
"""Validate COMFY_NORMAL VAE parity report against production guardrails."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def report_field(report: dict[str, Any], name: str, field: str, default: Any = None) -> Any:
    return report.get("sdcpp_reports", {}).get(name, {}).get(field, default)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-decode-workspace-mb", type=float, default=3200.0)
    parser.add_argument("--max-encode-workspace-mb", type=float, default=2048.0)
    parser.add_argument("--max-mean-abs-diff", type=float, default=0.01)
    parser.add_argument("--max-p99-abs-diff", type=float, default=0.05)
    parser.add_argument("--min-psnr", type=float, default=40.0)
    args = parser.parse_args()

    data = json.loads(args.report.read_text(encoding="utf-8"))
    failures: list[str] = []

    for name in ("encode_report", "decode_report"):
        require(report_field(data, name, "used_im2col") is False, f"{name} used IM2COL", failures)
        require(report_field(data, name, "used_tiling") is False, f"{name} used tiled VAE", failures)
        require(report_field(data, name, "used_taesd") is False, f"{name} used TAESD", failures)
        require(int(report_field(data, name, "host_copies", 0)) == 0, f"{name} had host stage copies", failures)

    decode_workspace = float(report_field(data, "decode_report", "planned_mb", 0.0))
    encode_workspace = float(report_field(data, "encode_report", "planned_mb", 0.0))
    require(decode_workspace <= args.max_decode_workspace_mb, f"decode workspace {decode_workspace} MB exceeded limit", failures)
    require(encode_workspace <= args.max_encode_workspace_mb, f"encode workspace {encode_workspace} MB exceeded limit", failures)

    metrics = data.get("metrics", {})
    if metrics:
        require(float(metrics.get("mean_abs_diff", 999.0)) <= args.max_mean_abs_diff, "mean abs diff exceeded limit", failures)
        require(float(metrics.get("p99_abs_diff", 999.0)) <= args.max_p99_abs_diff, "p99 abs diff exceeded limit", failures)
        require(float(metrics.get("psnr", 0.0)) >= args.min_psnr, "PSNR below limit", failures)

    if failures:
        print(json.dumps({"ok": False, "failures": failures}, indent=2))
        return 1

    print(json.dumps({
        "ok": True,
        "encode_workspace_mb": encode_workspace,
        "decode_workspace_mb": decode_workspace,
        "metrics": metrics,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
