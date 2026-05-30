#!/usr/bin/env python3
"""Subgraph-level Lens block0 parity recipes.

This is intentionally Python-only. It consumes the trusted B2.12 block0
checkpoints and evaluates stable subgraph recipes before any C++ port.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import b215_fast_block0_harness as b215


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default=r"build\diagnostics\lens_b212_block0_ref_stages.npz")
    parser.add_argument("--audit-json", default=r"build\diagnostics\lens_b212_block0_audit.json")
    parser.add_argument("--model", default=r"F:\Paralol\local\models\microsoft\Lens-Turbo")
    parser.add_argument("--json-out", default=r"build\diagnostics\lens_b216_subgraph_parity.json")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def score_named(outputs: dict[str, torch.Tensor], refs: dict[str, torch.Tensor]) -> dict[str, object]:
    stats = {name: b215.diff_stats(value, refs[name]) for name, value in outputs.items()}
    mean = sum(float(s["mean_diff"]) for s in stats.values()) / max(1, len(stats))
    max_diff = max((float(s["max_diff"]) for s in stats.values()), default=0.0)
    return {"mean_diff": mean, "max_diff": max_diff, "stages": stats}


def final_stats(final: dict[str, torch.Tensor], refs: dict[str, torch.Tensor]) -> dict[str, object]:
    return {
        "hidden": b215.diff_stats(final["final.hidden"], refs["final.hidden"]),
        "encoder": b215.diff_stats(final["final.encoder"], refs["final.encoder"]),
    }


def make_ref_pre(refs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "img_mod1": refs["img_mod1"],
        "img_mod2": refs["img_mod2"],
        "txt_mod1": refs["txt_mod1"],
        "txt_mod2": refs["txt_mod2"],
        "img_norm1": refs["img_norm1"],
        "txt_norm1": refs["txt_norm1"],
        "img_modulated": refs["img_modulated"],
        "txt_modulated": refs["txt_modulated"],
        "img_gate1": refs["img_mod1"].chunk(3, dim=-1)[2].unsqueeze(1),
        "txt_gate1": refs["txt_mod1"].chunk(3, dim=-1)[2].unsqueeze(1),
    }


def pre_variant(refs, state, name: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if name == "python_reference":
        pre = make_ref_pre(refs)
        outputs = {
            "img_mod1": pre["img_mod1"],
            "txt_mod1": pre["txt_mod1"],
            "img_norm1": pre["img_norm1"],
            "txt_norm1": pre["txt_norm1"],
            "img_modulated": pre["img_modulated"],
            "txt_modulated": pre["txt_modulated"],
        }
        return outputs, pre

    modulation_fn = {
        "current_native": b215._modulation_current,
        "bf16_exact_mod_current_norm_mod": b215._modulation_bf16,
        "bf16_exact_mod_python_style_norm_mod": b215._modulation_bf16,
        "bf16_exact_mod_bf16_step_pre": b215._modulation_bf16,
    }[name]
    _, mod = modulation_fn(refs["input.temb"], state)

    norm_fn = {
        "current_native": b215.rms_current,
        "bf16_exact_mod_current_norm_mod": b215.rms_current,
        "bf16_exact_mod_python_style_norm_mod": b215.rms_f32_reduce_bf16_out,
        "bf16_exact_mod_bf16_step_pre": b215.rms_bf16_step,
    }[name]
    _, norm = b215._norm1(refs["input.hidden"], refs["input.encoder"], state, norm_fn)

    modulate_fn = {
        "current_native": b215.modulate_current,
        "bf16_exact_mod_current_norm_mod": b215.modulate_current,
        "bf16_exact_mod_python_style_norm_mod": b215.modulate_f32_fused_bf16_out,
        "bf16_exact_mod_bf16_step_pre": b215.modulate_bf16_step,
    }[name]
    outputs, modulated = b215._modulate1(norm["img_norm1"], norm["txt_norm1"], mod["img_mod1"], mod["txt_mod1"], modulate_fn)
    pre = {
        **mod,
        **norm,
        **modulated,
    }
    compare = {
        "img_mod1": mod["img_mod1"],
        "txt_mod1": mod["txt_mod1"],
        "img_norm1": norm["img_norm1"],
        "txt_norm1": norm["txt_norm1"],
        **outputs,
    }
    return compare, pre


def attention_variant(refs, state, pre: dict[str, torch.Tensor], name: str):
    bsz, img_seq, _ = refs["input.hidden"].shape
    txt_seq = refs["input.encoder"].shape[1]
    heads = 24
    head_dim = 64
    if name == "python_reference":
        outputs = {
            "img_qkv": refs["img_qkv"],
            "txt_qkv": refs["txt_qkv"],
            "img_q_norm": refs["img_q_norm"],
            "txt_q_norm": refs["txt_q_norm"],
            "img_q_rope": refs["img_q_rope"],
            "txt_q_rope": refs["txt_q_rope"],
            "sdpa_joint": refs["sdpa_joint"],
            "img_attn": refs["img_attn"],
            "txt_attn": refs["txt_attn"],
        }
        return outputs, {"img_attn": refs["img_attn"], "txt_attn": refs["txt_attn"]}

    lin = {
        "current_native": b215.linear_current,
        "bf16_qkv_proj_current_attention": b215.linear_bf16,
        "bf16_qkv_proj_bf16_sdpa": b215.linear_bf16,
    }[name]
    _, qkv = b215._qkv(pre["img_modulated"], pre["txt_modulated"], state, lin, bsz, img_seq, txt_seq, heads, head_dim)
    norm_fn = b215.rms_bf16_step if name != "current_native" else b215.rms_current
    _, qk = b215._qk_norm(qkv, state, norm_fn)
    _, rope = b215._rope(qk, refs["input.img_freqs"], refs["input.txt_freqs"], b215.apply_rope_current)
    sdpa_fn = b215.sdpa_bf16 if name == "bf16_qkv_proj_bf16_sdpa" else b215.sdpa_current_f32
    _, attn_pre = b215._attention(rope, qkv, refs["input.attention_mask"], sdpa_fn, img_seq)
    _, proj = b215._out_proj(attn_pre, state, lin)
    outputs = {
        "img_qkv": qkv["img_qkv"],
        "txt_qkv": qkv["txt_qkv"],
        "img_q_norm": qk["img_q_norm"],
        "txt_q_norm": qk["txt_q_norm"],
        "img_q_rope": rope["img_q_rope"],
        "txt_q_rope": rope["txt_q_rope"],
        "sdpa_joint": attn_pre["sdpa_joint"],
        "img_attn": proj["img_attn"],
        "txt_attn": proj["txt_attn"],
    }
    return outputs, proj


def residual1_variant(refs, pre, attn, name: str):
    if name == "python_reference":
        final = {"hidden_after_attn": refs["hidden_after_attn"], "encoder_after_attn": refs["encoder_after_attn"]}
        return final, final
    outputs, updates = b215._residual1(
        refs["input.hidden"],
        refs["input.encoder"],
        pre["img_gate1"],
        pre["txt_gate1"],
        attn["img_attn"],
        attn["txt_attn"],
        name == "bf16_stepwise",
    )
    return outputs, updates


def mlp_residual_variant(refs, state, hidden1, encoder1, img_mod2, txt_mod2, name: str):
    if name == "python_reference":
        outputs = {
            "img_norm2": refs["img_norm2"],
            "txt_norm2": refs["txt_norm2"],
            "img_modulated2": refs["img_modulated2"],
            "txt_modulated2": refs["txt_modulated2"],
            "img_mlp": refs["img_mlp"],
            "txt_mlp": refs["txt_mlp"],
            "final.hidden": refs["final.hidden"],
            "final.encoder": refs["final.encoder"],
        }
        return outputs, {"final.hidden": refs["final.hidden"], "final.encoder": refs["final.encoder"]}

    if name == "current_native":
        norm_fn = b215.rms_current
        modulate_fn = b215.modulate_current
        lin = b215.linear_current
        step = False
    elif name == "python_style_bf16_step":
        norm_fn = b215.rms_bf16_step
        modulate_fn = b215.modulate_bf16_step
        lin = b215.linear_bf16
        step = True
    elif name == "f32_intermediate_bf16_outputs":
        norm_fn = b215.rms_f32_reduce_bf16_out
        modulate_fn = b215.modulate_f32_fused_bf16_out
        lin = b215.linear_bf16_f32acc
        step = False
    else:
        raise ValueError(name)

    _, norm = b215._norm2(hidden1, encoder1, state, norm_fn)
    _, mod = b215._modulate2(norm["img_norm2"], norm["txt_norm2"], img_mod2, txt_mod2, modulate_fn)
    _, mlp = b215._mlp(mod["img_modulated2"], mod["txt_modulated2"], state, lin, step)
    _, final = b215._residual2(hidden1, encoder1, mod["img_gate2"], mod["txt_gate2"], mlp["img_mlp"], mlp["txt_mlp"], step)
    outputs = {
        **norm,
        "img_modulated2": mod["img_modulated2"],
        "txt_modulated2": mod["txt_modulated2"],
        **mlp,
        **final,
    }
    return outputs, final


def independent_subgraph_rows(refs, state):
    rows = []
    for name in ["current_native", "bf16_exact_mod_current_norm_mod", "bf16_exact_mod_python_style_norm_mod", "bf16_exact_mod_bf16_step_pre", "python_reference"]:
        outputs, _ = pre_variant(refs, state, name)
        rows.append({"subgraph": "pre_attention", "variant": name, **score_named(outputs, refs)})

    ref_pre = make_ref_pre(refs)
    for name in ["current_native", "bf16_qkv_proj_current_attention", "bf16_qkv_proj_bf16_sdpa", "python_reference"]:
        outputs, _ = attention_variant(refs, state, ref_pre, name)
        rows.append({"subgraph": "attention", "variant": name, **score_named(outputs, refs)})

    ref_attn = {"img_attn": refs["img_attn"], "txt_attn": refs["txt_attn"]}
    for name in ["current_native", "bf16_stepwise", "python_reference"]:
        outputs, _ = residual1_variant(refs, ref_pre, ref_attn, name)
        rows.append({"subgraph": "residual1", "variant": name, **score_named(outputs, refs)})

    for name in ["current_native", "f32_intermediate_bf16_outputs", "python_style_bf16_step", "python_reference"]:
        outputs, _ = mlp_residual_variant(refs, state, refs["hidden_after_attn"], refs["encoder_after_attn"], refs["img_mod2"], refs["txt_mod2"], name)
        rows.append({"subgraph": "mlp_residual", "variant": name, **score_named(outputs, refs)})
    return rows


def mlp_residual_decomposition(refs, state):
    rows = []
    for name, norm_fn in [
        ("current_native", b215.rms_current),
        ("f32_reduce_bf16_output", b215.rms_f32_reduce_bf16_out),
        ("bf16_stepwise_approx", b215.rms_bf16_step),
    ]:
        outputs, _ = b215._norm2(refs["hidden_after_attn"], refs["encoder_after_attn"], state, norm_fn)
        rows.append({"stage_group": "norm2_only", "variant": name, **score_named(outputs, refs)})

    for name, modulate_fn in [
        ("current_native", b215.modulate_current),
        ("f32_fused_bf16_output", b215.modulate_f32_fused_bf16_out),
        ("bf16_stepwise", b215.modulate_bf16_step),
    ]:
        outputs, _ = b215._modulate2(refs["img_norm2"], refs["txt_norm2"], refs["img_mod2"], refs["txt_mod2"], modulate_fn)
        rows.append({"stage_group": "modulate2_only", "variant": name, **score_named(outputs, refs)})

    for name, lin, step in [
        ("current_native", b215.linear_current, False),
        ("bf16_cublaslt_linears_bf16_swiglu", b215.linear_bf16, True),
        ("bf16_weight_f32acc_linears", b215.linear_bf16_f32acc, False),
    ]:
        outputs, _ = b215._mlp(refs["img_modulated2"], refs["txt_modulated2"], state, lin, step)
        rows.append({"stage_group": "mlp_only_ref_input", "variant": name, **score_named(outputs, refs)})

    img_gate2 = refs["img_mod2"].chunk(3, dim=-1)[2].unsqueeze(1)
    txt_gate2 = refs["txt_mod2"].chunk(3, dim=-1)[2].unsqueeze(1)
    for name, step in [("current_native", False), ("bf16_stepwise", True)]:
        outputs, _ = b215._residual2(
            refs["hidden_after_attn"],
            refs["encoder_after_attn"],
            img_gate2,
            txt_gate2,
            refs["img_mlp"],
            refs["txt_mlp"],
            step,
        )
        rows.append({"stage_group": "residual2_only_ref_mlp", "variant": name, **score_named(outputs, refs)})
    return rows


def run_whole_recipe(refs, state, recipe: dict[str, str]):
    _, pre = pre_variant(refs, state, recipe["pre_attention"])
    _, attn = attention_variant(refs, state, pre, recipe["attention"])
    _, res1 = residual1_variant(refs, pre, attn, recipe["residual1"])
    outputs, final = mlp_residual_variant(
        refs,
        state,
        res1["hidden_after_attn"],
        res1["encoder_after_attn"],
        pre["img_mod2"],
        pre["txt_mod2"],
        recipe["mlp_residual"],
    )
    return {
        "recipe": recipe,
        "final": final_stats(final, refs),
        "mlp_residual_score": score_named(outputs, refs),
    }


def classify(rows, recipes, mlp_rows):
    first_unstable = None
    for row in rows:
        if row["variant"] == "python_reference":
            continue
        if float(row["mean_diff"]) > 0.1:
            first_unstable = {
                "subgraph": row["subgraph"],
                "variant": row["variant"],
                "max_diff": row["max_diff"],
                "mean_diff": row["mean_diff"],
            }
            break
    if first_unstable is None:
        first_unstable = {"subgraph": "mlp_residual", "reason": "largest remaining downstream amplification"}

    comparable_recipes = [r for r in recipes if r["recipe"]["name"] != "oracle_all_subgraphs"]
    best = min(comparable_recipes, key=lambda r: (r["final"]["hidden"]["mean_diff"] + r["final"]["encoder"]["mean_diff"], r["final"]["hidden"]["max_diff"] + r["final"]["encoder"]["max_diff"]))

    cause = "not_classified"
    mlp_only = [r for r in mlp_rows if r["stage_group"] == "mlp_only_ref_input" and r["variant"] == "current_native"][0]
    residual_only = [r for r in mlp_rows if r["stage_group"] == "residual2_only_ref_mlp" and r["variant"] == "current_native"][0]
    if float(mlp_only["mean_diff"]) > 1.0:
        cause = "mlp_linear_cast_behavior"
    elif float(residual_only["mean_diff"]) > 1.0:
        cause = "gate_multiply_or_residual_add_behavior"
    else:
        cause = "bad_input_from_previous_subgraph"
    return first_unstable, best, cause


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    refs = b215.load_refs(args.npz, device)
    state = b215.load_block0_state(args.model, device)
    with torch.no_grad():
        rows = independent_subgraph_rows(refs, state)
        mlp_rows = mlp_residual_decomposition(refs, state)
        recipes = [
            run_whole_recipe(refs, state, {
                "name": "candidate1_python_pre_current_rest",
                "pre_attention": "python_reference",
                "attention": "current_native",
                "residual1": "current_native",
                "mlp_residual": "current_native",
            }),
            run_whole_recipe(refs, state, {
                "name": "candidate2_bf16_pre_current_rest",
                "pre_attention": "bf16_exact_mod_bf16_step_pre",
                "attention": "current_native",
                "residual1": "current_native",
                "mlp_residual": "current_native",
            }),
            run_whole_recipe(refs, state, {
                "name": "candidate3_bf16_pre_bf16_attention_current_mlp",
                "pre_attention": "bf16_exact_mod_bf16_step_pre",
                "attention": "bf16_qkv_proj_bf16_sdpa",
                "residual1": "current_native",
                "mlp_residual": "current_native",
            }),
            run_whole_recipe(refs, state, {
                "name": "candidate4_bf16_pre_attention_mlp",
                "pre_attention": "bf16_exact_mod_bf16_step_pre",
                "attention": "bf16_qkv_proj_bf16_sdpa",
                "residual1": "bf16_stepwise",
                "mlp_residual": "python_style_bf16_step",
            }),
            run_whole_recipe(refs, state, {
                "name": "candidate5_python_pre_current_attention_bf16_res_mlp",
                "pre_attention": "python_reference",
                "attention": "current_native",
                "residual1": "bf16_stepwise",
                "mlp_residual": "python_style_bf16_step",
            }),
            run_whole_recipe(refs, state, {
                "name": "candidate6_python_pre_bf16_attention_bf16_res_mlp",
                "pre_attention": "python_reference",
                "attention": "bf16_qkv_proj_bf16_sdpa",
                "residual1": "bf16_stepwise",
                "mlp_residual": "python_style_bf16_step",
            }),
            run_whole_recipe(refs, state, {
                "name": "candidate7_python_pre_attention_current_res_bf16_mlp",
                "pre_attention": "python_reference",
                "attention": "python_reference",
                "residual1": "current_native",
                "mlp_residual": "python_style_bf16_step",
            }),
            run_whole_recipe(refs, state, {
                "name": "oracle_all_subgraphs",
                "pre_attention": "python_reference",
                "attention": "python_reference",
                "residual1": "python_reference",
                "mlp_residual": "python_reference",
            }),
        ]
    first_unstable, best, cause = classify(rows, recipes, mlp_rows)
    report = {
        "schema": "lens_b216_subgraph_parity_v1",
        "npz": args.npz,
        "audit_json": args.audit_json,
        "model": args.model,
        "device": str(device),
        "independent_subgraph_rows": rows,
        "mlp_residual_decomposition": mlp_rows,
        "whole_subgraph_recipes": recipes,
        "first_unstable_subgraph": first_unstable,
        "encoder_explosion_cause": cause,
        "best_stable_recipe": best,
        "cxx_patch_applied": False,
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "json_out": args.json_out,
        "first_unstable_subgraph": first_unstable,
        "encoder_explosion_cause": cause,
        "best_stable_recipe": best["recipe"]["name"],
        "best_final": best["final"],
        "cxx_patch_applied": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
