#!/usr/bin/env python3
"""Aggressive one-block Lens transformer parity harness.

This consumes the trusted B2.12 block0 Python checkpoints and evaluates a small,
bounded set of native implementation recipes. It does not run full generation
and does not depend on the native C++ runner.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


BF16 = torch.bfloat16
F32 = torch.float32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default=r"build\diagnostics\lens_b212_block0_ref_stages.npz")
    parser.add_argument("--audit-json", default=r"build\diagnostics\lens_b212_block0_audit.json")
    parser.add_argument("--model", default=r"F:\Paralol\local\models\microsoft\Lens-Turbo")
    parser.add_argument("--json-out", default=r"build\diagnostics\lens_b215_fast_block0_harness.json")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_block0_state(model_dir: str, device: torch.device) -> dict[str, torch.Tensor]:
    root = Path(model_dir) / "transformer"
    prefix = "transformer_blocks.0."
    out: dict[str, torch.Tensor] = {}
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    out[key[len(prefix) :]] = handle.get_tensor(key).to(device=device, dtype=F32)
    if not out:
        raise RuntimeError(f"missing block0 weights under {root}")
    return out


def load_refs(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    data = np.load(path)
    refs: dict[str, torch.Tensor] = {}
    for key in data.files:
        refs[key.removeprefix("ref.")] = torch.from_numpy(data[key]).to(device=device, dtype=F32)
    return refs


def diff_stats(actual: torch.Tensor, ref: torch.Tensor) -> dict[str, object]:
    a = actual.detach().float()
    r = ref.detach().float()
    d = (a - r).abs()
    idx = int(d.argmax().item()) if d.numel() else 0
    return {
        "max_diff": float(d.max().item()) if d.numel() else 0.0,
        "mean_diff": float(d.mean().item()) if d.numel() else 0.0,
        "worst_flat_index": idx,
        "shape": list(a.shape),
    }


def score_outputs(outputs: dict[str, torch.Tensor], refs: dict[str, torch.Tensor]) -> tuple[float, float]:
    means = []
    maxes = []
    for name, value in outputs.items():
        stats = diff_stats(value, refs[name])
        means.append(float(stats["mean_diff"]))
        maxes.append(float(stats["max_diff"]))
    return (float(sum(means) / max(1, len(means))), float(max(maxes) if maxes else 0.0))


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(BF16)


def linear_current(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
    return F.linear(x.float(), w.float(), None if b is None else b.float())


def linear_bf16(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
    return F.linear(bf16(x), bf16(w), None if b is None else bf16(b))


def linear_bf16_f32acc(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
    return F.linear(x.float(), bf16(w).float(), None if b is None else bf16(b).float())


def rms_current(x: torch.Tensor, weight: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    y = x.float()
    var = y.pow(2).mean(dim=-1, keepdim=True)
    out = y * torch.rsqrt(var + eps) * weight.float()
    return out.to(x.dtype)


def rms_f32_reduce_bf16_out(x: torch.Tensor, weight: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    y = x.float()
    var = y.pow(2).mean(dim=-1, keepdim=True)
    return bf16(y * torch.rsqrt(var + eps) * bf16(weight).float())


def rms_bf16_step(x: torch.Tensor, weight: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    # Approximation only: reductions stay in F32, but normalize/multiply are
    # rounded through BF16 to mimic a stepwise BF16 path.
    y = bf16(x)
    var = y.float().pow(2).mean(dim=-1, keepdim=True)
    inv = bf16(torch.rsqrt(var + eps))
    return bf16(bf16(y * inv) * bf16(weight))


def modulate_current(x: torch.Tensor, mod: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shift, scale, gate = mod.chunk(3, dim=-1)
    return x.float() * (1.0 + scale.float().unsqueeze(1)) + shift.float().unsqueeze(1), gate.unsqueeze(1)


def modulate_f32_fused_bf16_out(x: torch.Tensor, mod: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    y, gate = modulate_current(x, mod)
    return bf16(y), bf16(gate)


def modulate_bf16_step(x: torch.Tensor, mod: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shift, scale, gate = mod.chunk(3, dim=-1)
    one_plus = bf16(1.0 + bf16(scale).unsqueeze(1))
    prod = bf16(bf16(x) * one_plus)
    out = bf16(prod + bf16(shift).unsqueeze(1))
    return out, bf16(gate).unsqueeze(1)


def apply_rope_current(x: torch.Tensor, freqs_real: torch.Tensor) -> torch.Tensor:
    freqs = torch.view_as_complex(freqs_real.float().contiguous())
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * freqs.unsqueeze(1)).flatten(3).type_as(x)


def apply_rope_bf16_out(x: torch.Tensor, freqs_real: torch.Tensor) -> torch.Tensor:
    return bf16(apply_rope_current(x, freqs_real))


def sdpa_current_f32(
    img_q: torch.Tensor,
    img_k: torch.Tensor,
    img_v: torch.Tensor,
    txt_q: torch.Tensor,
    txt_k: torch.Tensor,
    txt_v: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    q = torch.cat([img_q, txt_q], dim=1).transpose(1, 2).float()
    k = torch.cat([img_k, txt_k], dim=1).transpose(1, 2).float()
    v = torch.cat([img_v, txt_v], dim=1).transpose(1, 2).float()
    scale = 1.0 / math.sqrt(float(q.shape[-1]))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    scores = scores + attention_mask.float()
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


def sdpa_bf16(
    img_q: torch.Tensor,
    img_k: torch.Tensor,
    img_v: torch.Tensor,
    txt_q: torch.Tensor,
    txt_k: torch.Tensor,
    txt_v: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    q = torch.cat([img_q, txt_q], dim=1).transpose(1, 2).to(BF16)
    k = torch.cat([img_k, txt_k], dim=1).transpose(1, 2).to(BF16)
    v = torch.cat([img_v, txt_v], dim=1).transpose(1, 2).to(BF16)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask.to(BF16))
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


class Harness:
    def __init__(self, refs: dict[str, torch.Tensor], state: dict[str, torch.Tensor]):
        self.refs = refs
        self.state = state
        self.rows: list[dict[str, object]] = []
        self.selected: dict[str, str] = {}
        self.first_unavoidable: dict[str, object] | None = None

    def choose(
        self,
        component: str,
        candidates: dict[str, Callable[[], tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]],
    ) -> dict[str, torch.Tensor]:
        best_name = ""
        best_outputs: dict[str, torch.Tensor] = {}
        best_updates: dict[str, torch.Tensor] = {}
        best_score: tuple[float, float] | None = None
        candidate_rows = []
        for name, fn in candidates.items():
            outputs, updates = fn()
            mean, maxd = score_outputs(outputs, self.refs)
            candidate_rows.append({"variant": name, "mean_diff": mean, "max_diff": maxd})
            if best_score is None or (mean, maxd) < best_score:
                best_name = name
                best_score = (mean, maxd)
                best_outputs = outputs
                best_updates = updates
        self.selected[component] = best_name
        best_stats = {name: diff_stats(value, self.refs[name]) for name, value in best_outputs.items()}
        self.rows.append({
            "component": component,
            "selected": best_name,
            "candidates": candidate_rows,
            "selected_stats": best_stats,
        })
        if self.first_unavoidable is None:
            for stage, stats in best_stats.items():
                if float(stats["max_diff"]) > 0.01:
                    self.first_unavoidable = {
                        "component": component,
                        "stage": stage,
                        **stats,
                    }
                    break
        return best_updates


def run_recipe(refs: dict[str, torch.Tensor], state: dict[str, torch.Tensor], recipe: dict[str, str]) -> dict[str, torch.Tensor]:
    h = Harness(refs, state)
    return run_greedy(refs, state, h, fixed_recipe=recipe)


def run_greedy(
    refs: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    harness: Harness | None = None,
    fixed_recipe: dict[str, str] | None = None,
) -> dict[str, torch.Tensor]:
    selected = {}

    def pick(component: str, candidates):
        if fixed_recipe is not None:
            outputs, updates = candidates[fixed_recipe[component]]()
            selected[component] = fixed_recipe[component]
            return updates
        assert harness is not None
        updates = harness.choose(component, candidates)
        selected[component] = harness.selected[component]
        return updates

    hidden0 = refs["input.hidden"]
    encoder0 = refs["input.encoder"]
    temb = refs["input.temb"]
    img_freqs = refs["input.img_freqs"]
    txt_freqs = refs["input.txt_freqs"]
    attention_mask = refs["input.attention_mask"]
    bsz, img_seq, hidden = hidden0.shape
    txt_seq = encoder0.shape[1]
    heads = 24
    head_dim = 64

    modulation = pick("modulation", {
        "current_f32_silu_f32_linear": lambda: _modulation_current(temb, state),
        "bf16_silu_f32acc_linear": lambda: _modulation_bf16_f32acc(temb, state),
        "bf16_silu_bf16_cublaslt_linear": lambda: _modulation_bf16(temb, state),
    })
    img_mod1 = modulation["img_mod1"]
    img_mod2 = modulation["img_mod2"]
    txt_mod1 = modulation["txt_mod1"]
    txt_mod2 = modulation["txt_mod2"]

    norm1 = pick("rmsnorm1", {
        "current_native": lambda: _norm1(hidden0, encoder0, state, rms_current),
        "f32_reduce_bf16_output": lambda: _norm1(hidden0, encoder0, state, rms_f32_reduce_bf16_out),
        "bf16_stepwise_approx": lambda: _norm1(hidden0, encoder0, state, rms_bf16_step),
    })
    img_norm1 = norm1["img_norm1"]
    txt_norm1 = norm1["txt_norm1"]

    modulated1 = pick("modulate1", {
        "current_native": lambda: _modulate1(img_norm1, txt_norm1, img_mod1, txt_mod1, modulate_current),
        "f32_fused_bf16_output": lambda: _modulate1(img_norm1, txt_norm1, img_mod1, txt_mod1, modulate_f32_fused_bf16_out),
        "bf16_stepwise": lambda: _modulate1(img_norm1, txt_norm1, img_mod1, txt_mod1, modulate_bf16_step),
    })
    img_modulated = modulated1["img_modulated"]
    txt_modulated = modulated1["txt_modulated"]
    img_gate1 = modulated1["img_gate1"]
    txt_gate1 = modulated1["txt_gate1"]

    qkv = pick("qkv_linear", {
        "current_native": lambda: _qkv(img_modulated, txt_modulated, state, linear_current, bsz, img_seq, txt_seq, heads, head_dim),
        "bf16_cublaslt_fused_bias": lambda: _qkv(img_modulated, txt_modulated, state, linear_bf16, bsz, img_seq, txt_seq, heads, head_dim),
        "bf16_weight_f32acc": lambda: _qkv(img_modulated, txt_modulated, state, linear_bf16_f32acc, bsz, img_seq, txt_seq, heads, head_dim),
    })

    qk_norm = pick("qk_rmsnorm", {
        "current_native": lambda: _qk_norm(qkv, state, rms_current),
        "f32_reduce_bf16_output": lambda: _qk_norm(qkv, state, rms_f32_reduce_bf16_out),
        "bf16_stepwise_approx": lambda: _qk_norm(qkv, state, rms_bf16_step),
    })

    rope = pick("rope", {
        "current_native": lambda: _rope(qk_norm, img_freqs, txt_freqs, apply_rope_current),
        "bf16_output": lambda: _rope(qk_norm, img_freqs, txt_freqs, apply_rope_bf16_out),
    })

    attn = pick("attention", {
        "current_regular_f32_manual": lambda: _attention(rope, qkv, attention_mask, sdpa_current_f32, img_seq),
        "bf16_sdpa": lambda: _attention(rope, qkv, attention_mask, sdpa_bf16, img_seq),
    })

    proj = pick("output_projection", {
        "current_native": lambda: _out_proj(attn, state, linear_current),
        "bf16_cublaslt_fused_bias": lambda: _out_proj(attn, state, linear_bf16),
        "bf16_weight_f32acc": lambda: _out_proj(attn, state, linear_bf16_f32acc),
    })
    img_attn = proj["img_attn"]
    txt_attn = proj["txt_attn"]

    residual1 = pick("residual1", {
        "current_native": lambda: _residual1(hidden0, encoder0, img_gate1, txt_gate1, img_attn, txt_attn, False),
        "bf16_stepwise": lambda: _residual1(hidden0, encoder0, img_gate1, txt_gate1, img_attn, txt_attn, True),
    })
    hidden1 = residual1["hidden_after_attn"]
    encoder1 = residual1["encoder_after_attn"]

    norm2 = pick("rmsnorm2", {
        "current_native": lambda: _norm2(hidden1, encoder1, state, rms_current),
        "f32_reduce_bf16_output": lambda: _norm2(hidden1, encoder1, state, rms_f32_reduce_bf16_out),
        "bf16_stepwise_approx": lambda: _norm2(hidden1, encoder1, state, rms_bf16_step),
    })

    modulated2 = pick("modulate2", {
        "current_native": lambda: _modulate2(norm2["img_norm2"], norm2["txt_norm2"], img_mod2, txt_mod2, modulate_current),
        "f32_fused_bf16_output": lambda: _modulate2(norm2["img_norm2"], norm2["txt_norm2"], img_mod2, txt_mod2, modulate_f32_fused_bf16_out),
        "bf16_stepwise": lambda: _modulate2(norm2["img_norm2"], norm2["txt_norm2"], img_mod2, txt_mod2, modulate_bf16_step),
    })

    mlp = pick("mlp", {
        "current_native": lambda: _mlp(modulated2["img_modulated2"], modulated2["txt_modulated2"], state, linear_current, False),
        "bf16_cublaslt_linears_bf16_swiglu": lambda: _mlp(modulated2["img_modulated2"], modulated2["txt_modulated2"], state, linear_bf16, True),
        "bf16_weight_f32acc_linears": lambda: _mlp(modulated2["img_modulated2"], modulated2["txt_modulated2"], state, linear_bf16_f32acc, False),
    })

    final = pick("residual2", {
        "current_native": lambda: _residual2(hidden1, encoder1, modulated2["img_gate2"], modulated2["txt_gate2"], mlp["img_mlp"], mlp["txt_mlp"], False),
        "bf16_stepwise": lambda: _residual2(hidden1, encoder1, modulated2["img_gate2"], modulated2["txt_gate2"], mlp["img_mlp"], mlp["txt_mlp"], True),
    })
    final["selected_recipe"] = selected  # type: ignore[index]
    return final


def _modulation_current(temb, state):
    silu = F.silu(temb.float())
    img = linear_current(silu, state["img_mod.1.weight"], state["img_mod.1.bias"])
    txt = linear_current(silu, state["txt_mod.1.weight"], state["txt_mod.1.bias"])
    return _modulation_outputs(silu, img, txt)


def _modulation_bf16_f32acc(temb, state):
    silu = bf16(F.silu(bf16(temb)))
    img = linear_bf16_f32acc(silu, state["img_mod.1.weight"], state["img_mod.1.bias"])
    txt = linear_bf16_f32acc(silu, state["txt_mod.1.weight"], state["txt_mod.1.bias"])
    return _modulation_outputs(silu, img, txt)


def _modulation_bf16(temb, state):
    silu = F.silu(bf16(temb))
    img = linear_bf16(silu, state["img_mod.1.weight"], state["img_mod.1.bias"])
    txt = linear_bf16(silu, state["txt_mod.1.weight"], state["txt_mod.1.bias"])
    return _modulation_outputs(silu, img, txt)


def _modulation_outputs(silu, img, txt):
    img1, img2 = img.chunk(2, dim=-1)
    txt1, txt2 = txt.chunk(2, dim=-1)
    outputs = {
        "silu_temb": silu,
        "img_mod_full": img,
        "txt_mod_full": txt,
        "img_mod1": img1,
        "img_mod2": img2,
        "txt_mod1": txt1,
        "txt_mod2": txt2,
    }
    return outputs, {"img_mod1": img1, "img_mod2": img2, "txt_mod1": txt1, "txt_mod2": txt2}


def _norm1(hidden, encoder, state, norm_fn):
    img = norm_fn(hidden, state["img_norm1.weight"])
    txt = norm_fn(encoder, state["txt_norm1.weight"])
    return {"img_norm1": img, "txt_norm1": txt}, {"img_norm1": img, "txt_norm1": txt}


def _norm2(hidden, encoder, state, norm_fn):
    img = norm_fn(hidden, state["img_norm2.weight"])
    txt = norm_fn(encoder, state["txt_norm2.weight"])
    return {"img_norm2": img, "txt_norm2": txt}, {"img_norm2": img, "txt_norm2": txt}


def _modulate1(img_norm, txt_norm, img_mod, txt_mod, fn):
    img, img_gate = fn(img_norm, img_mod)
    txt, txt_gate = fn(txt_norm, txt_mod)
    return {"img_modulated": img, "txt_modulated": txt}, {
        "img_modulated": img,
        "txt_modulated": txt,
        "img_gate1": img_gate,
        "txt_gate1": txt_gate,
    }


def _modulate2(img_norm, txt_norm, img_mod, txt_mod, fn):
    img, img_gate = fn(img_norm, img_mod)
    txt, txt_gate = fn(txt_norm, txt_mod)
    return {"img_modulated2": img, "txt_modulated2": txt}, {
        "img_modulated2": img,
        "txt_modulated2": txt,
        "img_gate2": img_gate,
        "txt_gate2": txt_gate,
    }


def _qkv(img, txt, state, lin, bsz, img_seq, txt_seq, heads, head_dim):
    img_qkv = lin(img, state["attn.img_qkv.weight"], state["attn.img_qkv.bias"]).view(bsz, img_seq, 3, heads, head_dim)
    txt_qkv = lin(txt, state["attn.txt_qkv.weight"], state["attn.txt_qkv.bias"]).view(bsz, txt_seq, 3, heads, head_dim)
    img_q, img_k, img_v = img_qkv.unbind(dim=2)
    txt_q, txt_k, txt_v = txt_qkv.unbind(dim=2)
    outputs = {
        "img_qkv": img_qkv,
        "txt_qkv": txt_qkv,
        "img_q": img_q,
        "img_k": img_k,
        "img_v": img_v,
        "txt_q": txt_q,
        "txt_k": txt_k,
        "txt_v": txt_v,
    }
    return outputs, outputs.copy()


def _qk_norm(qkv, state, norm_fn):
    out = {
        "img_q_norm": norm_fn(qkv["img_q"], state["attn.norm_q.weight"]),
        "img_k_norm": norm_fn(qkv["img_k"], state["attn.norm_k.weight"]),
        "txt_q_norm": norm_fn(qkv["txt_q"], state["attn.norm_added_q.weight"]),
        "txt_k_norm": norm_fn(qkv["txt_k"], state["attn.norm_added_k.weight"]),
    }
    return out, out.copy()


def _rope(qk, img_freqs, txt_freqs, rope_fn):
    out = {
        "img_q_rope": rope_fn(qk["img_q_norm"], img_freqs),
        "img_k_rope": rope_fn(qk["img_k_norm"], img_freqs),
        "txt_q_rope": rope_fn(qk["txt_q_norm"], txt_freqs),
        "txt_k_rope": rope_fn(qk["txt_k_norm"], txt_freqs),
    }
    return out, out.copy()


def _attention(rope, qkv, mask, fn, img_seq):
    joint = fn(rope["img_q_rope"], rope["img_k_rope"], qkv["img_v"], rope["txt_q_rope"], rope["txt_k_rope"], qkv["txt_v"], mask)
    img = joint[:, :img_seq, :]
    txt = joint[:, img_seq:, :]
    return {"sdpa_joint": joint, "img_attn_pre_out": img, "txt_attn_pre_out": txt}, {
        "sdpa_joint": joint,
        "img_attn_pre_out": img,
        "txt_attn_pre_out": txt,
    }


def _out_proj(attn, state, lin):
    img = lin(attn["img_attn_pre_out"], state["attn.to_out.0.weight"], state["attn.to_out.0.bias"])
    txt = lin(attn["txt_attn_pre_out"], state["attn.to_add_out.weight"], state["attn.to_add_out.bias"])
    return {"img_attn": img, "txt_attn": txt}, {"img_attn": img, "txt_attn": txt}


def _residual1(hidden, encoder, img_gate, txt_gate, img_attn, txt_attn, step_bf16: bool):
    if step_bf16:
        img_update = bf16(bf16(img_gate) * bf16(img_attn))
        txt_update = bf16(bf16(txt_gate) * bf16(txt_attn))
        out_h = bf16(bf16(hidden) + img_update)
        out_e = bf16(bf16(encoder) + txt_update)
    else:
        out_h = hidden.float() + img_gate.float() * img_attn.float()
        out_e = encoder.float() + txt_gate.float() * txt_attn.float()
    return {"hidden_after_attn": out_h, "encoder_after_attn": out_e}, {"hidden_after_attn": out_h, "encoder_after_attn": out_e}


def _mlp_one(x, state, prefix, lin, step_bf16):
    w1 = lin(x, state[f"{prefix}.w1.weight"], None)
    w3 = lin(x, state[f"{prefix}.w3.weight"], None)
    if step_bf16:
        gated = bf16(F.silu(bf16(w1)) * bf16(w3))
    else:
        gated = F.silu(w1.float()) * w3.float()
    return lin(gated, state[f"{prefix}.w2.weight"], None)


def _mlp(img, txt, state, lin, step_bf16):
    img_out = _mlp_one(img, state, "img_mlp", lin, step_bf16)
    txt_out = _mlp_one(txt, state, "txt_mlp", lin, step_bf16)
    return {"img_mlp": img_out, "txt_mlp": txt_out}, {"img_mlp": img_out, "txt_mlp": txt_out}


def _residual2(hidden, encoder, img_gate, txt_gate, img_mlp, txt_mlp, step_bf16: bool):
    if step_bf16:
        img_update = bf16(bf16(img_gate) * bf16(img_mlp))
        txt_update = bf16(bf16(txt_gate) * bf16(txt_mlp))
        out_h = bf16(bf16(hidden) + img_update)
        out_e = bf16(bf16(encoder) + txt_update)
    else:
        out_h = hidden.float() + img_gate.float() * img_mlp.float()
        out_e = encoder.float() + txt_gate.float() * txt_mlp.float()
    return {"final.hidden": out_h, "final.encoder": out_e}, {"final.hidden": out_h, "final.encoder": out_e}


def final_pair_stats(values: dict[str, torch.Tensor], refs: dict[str, torch.Tensor]) -> dict[str, dict[str, object]]:
    return {
        "hidden": diff_stats(values["final.hidden"], refs["final.hidden"]),
        "encoder": diff_stats(values["final.encoder"], refs["final.encoder"]),
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    refs = load_refs(args.npz, device)
    state = load_block0_state(args.model, device)
    harness = Harness(refs, state)

    with torch.no_grad():
        selected_final = run_greedy(refs, state, harness)
        old_f32_recipe = {
            "modulation": "current_f32_silu_f32_linear",
            "rmsnorm1": "current_native",
            "modulate1": "current_native",
            "qkv_linear": "current_native",
            "qk_rmsnorm": "current_native",
            "rope": "current_native",
            "attention": "current_regular_f32_manual",
            "output_projection": "current_native",
            "residual1": "current_native",
            "rmsnorm2": "current_native",
            "modulate2": "current_native",
            "mlp": "current_native",
            "residual2": "current_native",
        }
        gpu_full_bf16_like_recipe = {
            "modulation": "bf16_silu_bf16_cublaslt_linear",
            "rmsnorm1": "current_native",
            "modulate1": "current_native",
            "qkv_linear": "bf16_cublaslt_fused_bias",
            "qk_rmsnorm": "current_native",
            "rope": "current_native",
            "attention": "current_regular_f32_manual",
            "output_projection": "bf16_cublaslt_fused_bias",
            "residual1": "current_native",
            "rmsnorm2": "current_native",
            "modulate2": "current_native",
            "mlp": "bf16_cublaslt_linears_bf16_swiglu",
            "residual2": "current_native",
        }
        old_f32 = run_recipe(refs, state, old_f32_recipe)
        gpu_full_bf16_like = run_recipe(refs, state, gpu_full_bf16_like_recipe)

    report = {
        "schema": "lens_b215_fast_block0_harness_v1",
        "npz": str(args.npz),
        "audit_json": str(args.audit_json),
        "model": str(args.model),
        "device": str(device),
        "component_rows": harness.rows,
        "greedy_selected_recipe": harness.selected,
        "first_unavoidable_bad_boundary": harness.first_unavoidable,
        "final_block0_diff": {
            "old_native_f32_recipe": final_pair_stats(old_f32, refs),
            "gpu_full_bf16_like_recipe": final_pair_stats(gpu_full_bf16_like, refs),
            "selected_aggressive_recipe": final_pair_stats(selected_final, refs),
        },
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "json_out": args.json_out,
        "first_unavoidable_bad_boundary": harness.first_unavoidable,
        "greedy_selected_recipe": harness.selected,
        "final_block0_diff": report["final_block0_diff"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
