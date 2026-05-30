#!/usr/bin/env python3
"""Export a LensGptOssEncoder oracle bundle for native sd.cpp bring-up."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.gpt_oss.modeling_gpt_oss import (
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)


PROMPT = "a small glass robot standing on a wooden workbench, studio lighting, sharp focus"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens-root", default=r"F:\Paralol\local\Lens")
    parser.add_argument("--model-dir", default=r"F:\Paralol\local\models\microsoft\Lens-Turbo")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-sequence-length", type=int, default=128)
    parser.add_argument("--keep-mxfp4", action="store_true")
    parser.add_argument("--compare-cond", default="")
    parser.add_argument(
        "--bf16-reduced-precision-reduction",
        choices=["default", "true", "false"],
        default="default",
        help="Override torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction for golden/debug exports",
    )
    parser.add_argument(
        "--oracle-mode",
        choices=["golden", "debug-projection", "full"],
        default="golden",
        help="golden writes lens_cond_v1 only; debug-projection writes layer-0 projection/debug tensors only; full writes both",
    )
    parser.add_argument("--moe-debug-token", type=int, default=16)
    parser.add_argument("--moe-debug-channel", type=int, default=1167)
    parser.add_argument(
        "--max-oracle-layer",
        type=int,
        default=5,
        help="Highest decoder layer to export per-layer debug checkpoints for",
    )
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def cuda_note(label: str) -> None:
    if not torch.cuda.is_available():
        print(f"{label}: cuda unavailable")
        return
    free, total = torch.cuda.mem_get_info()
    used = total - free
    print(f"{label}: cuda {used / 1024**3:.2f}/{total / 1024**3:.2f} GiB")


def write_npy(path: Path, tensor: torch.Tensor | np.ndarray) -> None:
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    np.save(path, np.ascontiguousarray(tensor))


def bf16_bits_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(torch.bfloat16).cpu().view(torch.int16).to(torch.int32)


def render_prompt(tokenizer, prompt: str, chat_system: str, chat_thinking: str) -> str:
    conversation = [
        {"role": "system", "content": chat_system, "thinking": None},
        {"role": "user", "content": prompt, "thinking": None},
        {"role": "assistant", "thinking": chat_thinking, "content": ""},
    ]
    rendered = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    return rendered.split("<|return|>")[0]


def export_small_mxfp4_reference(text_encoder_dir: Path, output_dir: Path) -> dict[str, object]:
    """Export one tiny dequantized slice used by the native MXFP4 proof.

    The reference is gate_up_proj[expert=0, hidden=0, out=0:32], decoded using
    the Transformers MXFP4 routine from the raw *_blocks/_scales tensors.
    """
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors  # noqa: PLC0415

    shard = text_encoder_dir / "model-00001-of-00003.safetensors"
    tensors = load_file(str(shard), device="cpu")
    blocks = tensors["model.layers.0.mlp.experts.gate_up_proj_blocks"]
    scales = tensors["model.layers.0.mlp.experts.gate_up_proj_scales"]
    decoded = convert_moe_packed_tensors(blocks[:1, :32, :1, :], scales[:1, :32, :1], dtype=torch.float32)
    ref = decoded[0, 0, :32].contiguous().cpu().numpy().astype(np.float32)
    path = output_dir / "mxfp4_gate_up_e0_hidden0_out0_31_f32.npy"
    np.save(path, ref)
    return {
        "tensor": "model.layers.0.mlp.experts.gate_up_proj_blocks",
        "scale_tensor": "model.layers.0.mlp.experts.gate_up_proj_scales",
        "reference": path.name,
        "slice": "dequantized gate_up_proj[0, 0, 0:32]",
        "raw_blocks_slice": "blocks[0, 0:32, 0, 0]",
        "raw_scales_slice": "scales[0, 0:32, 0]",
    }


def scalar_meta(value: torch.Tensor | float) -> dict[str, object]:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        scalar = float(tensor.float().item())
        bits = int(tensor.reshape(()).to(torch.bfloat16).cpu().view(torch.int16).item() & 0xFFFF)
        dtype = str(tensor.dtype)
    else:
        scalar = float(value)
        t = torch.tensor(scalar, dtype=torch.bfloat16)
        bits = int(t.view(torch.int16).item() & 0xFFFF)
        dtype = "float"
    return {"value": scalar, "bf16_bits": bits, "dtype": dtype}


@torch.no_grad()
def export_layer0_moe_debug(
    layer0,
    post_attn_norm: torch.Tensor,
    router_scores: torch.Tensor,
    router_indices: torch.Tensor,
    output_dir: Path,
    token: int,
    channel: int,
) -> dict[str, object]:
    experts = layer0.mlp.experts
    hidden = int(post_attn_norm.shape[-1])
    if token < 0 or token >= int(post_attn_norm.shape[1]):
        raise ValueError(f"moe debug token {token} out of range")
    if channel < 0 or channel >= hidden:
        raise ValueError(f"moe debug channel {channel} out of range")
    if not all(hasattr(experts, name) for name in ("gate_up_proj", "gate_up_proj_bias", "down_proj", "down_proj_bias")):
        raise RuntimeError("MoE debug requires dequantized GptOssExperts tensors")

    flat_scores = router_scores.reshape(post_attn_norm.shape[1], -1)
    flat_indices = router_indices.reshape(post_attn_norm.shape[1], -1).to(torch.long)
    selected = flat_indices[token].detach().cpu()
    selected_scores = flat_scores[token].detach()
    current_state = post_attn_norm.reshape(-1, hidden)[token]
    write_npy(output_dir / f"layer0_moe_token{token}_topk_indices_i64.npy", selected.to(torch.int64))
    write_npy(output_dir / f"layer0_moe_token{token}_topk_weights_f32.npy", selected_scores.float())
    write_npy(output_dir / f"layer0_moe_token{token}_current_state_f32.npy", current_state.float())
    write_npy(output_dir / f"layer0_moe_token{token}_current_state_bf16_bits_i32.npy", bf16_bits_tensor(current_state))

    accumulated = torch.zeros((hidden,), dtype=current_state.dtype, device=current_state.device)
    metadata: dict[str, object] = {
        "token": token,
        "channel": channel,
        "current_state_dtype": str(current_state.dtype),
        "router_scores_dtype": str(router_scores.dtype),
        "router_indices": [int(x) for x in selected.tolist()],
        "router_weights": [float(x) for x in selected_scores.float().cpu().tolist()],
        "expert_order": [],
        "experts": {},
    }

    for expert_id in sorted({int(x) for x in selected.tolist()}):
        positions = [int(i) for i, x in enumerate(selected.tolist()) if int(x) == expert_id]
        expert_meta: dict[str, object] = {
            "top_k_positions": positions,
            "gate_up_proj_dtype": str(experts.gate_up_proj.dtype),
            "gate_up_proj_bias_dtype": str(experts.gate_up_proj_bias.dtype),
            "down_proj_dtype": str(experts.down_proj.dtype),
            "down_proj_bias_dtype": str(experts.down_proj_bias.dtype),
        }
        gate_w = experts.gate_up_proj[expert_id]
        gate_b = experts.gate_up_proj_bias[expert_id]
        down_w = experts.down_proj[expert_id]
        down_b = experts.down_proj_bias[expert_id]
        gate_up_pre_bias = current_state @ gate_w
        gate_up_post_bias = gate_up_pre_bias + gate_b
        gate = gate_up_post_bias[..., ::2]
        up = gate_up_post_bias[..., 1::2]
        gate_clamped = gate.clamp(min=None, max=experts.limit)
        up_clamped = up.clamp(min=-experts.limit, max=experts.limit)
        gate_alpha = gate_clamped * experts.alpha
        sigmoid = torch.sigmoid(gate_alpha)
        glu = gate_clamped * sigmoid
        gated_output = (up_clamped + 1) * glu
        down_pre_bias = gated_output @ down_w
        down_post_bias = down_pre_bias + down_b

        prefix = output_dir / f"layer0_moe_token{token}_expert{expert_id}"
        write_npy(prefix.with_name(prefix.name + "_gate_up_proj_f32.npy"), gate_w.float())
        write_npy(prefix.with_name(prefix.name + "_gate_up_proj_bf16_bits_i32.npy"), bf16_bits_tensor(gate_w))
        write_npy(prefix.with_name(prefix.name + "_gate_up_proj_bias_f32.npy"), gate_b.float())
        write_npy(prefix.with_name(prefix.name + "_gate_up_pre_bias_f32.npy"), gate_up_pre_bias.float())
        write_npy(prefix.with_name(prefix.name + "_gate_up_post_bias_f32.npy"), gate_up_post_bias.float())
        write_npy(prefix.with_name(prefix.name + "_gate_up_post_bias_bf16_bits_i32.npy"), bf16_bits_tensor(gate_up_post_bias))
        write_npy(prefix.with_name(prefix.name + "_gate_slice_f32.npy"), gate.float())
        write_npy(prefix.with_name(prefix.name + "_up_slice_f32.npy"), up.float())
        write_npy(prefix.with_name(prefix.name + "_gate_clamped_f32.npy"), gate_clamped.float())
        write_npy(prefix.with_name(prefix.name + "_up_clamped_f32.npy"), up_clamped.float())
        write_npy(prefix.with_name(prefix.name + "_gate_alpha_f32.npy"), gate_alpha.float())
        write_npy(prefix.with_name(prefix.name + "_sigmoid_f32.npy"), sigmoid.float())
        write_npy(prefix.with_name(prefix.name + "_glu_f32.npy"), glu.float())
        write_npy(prefix.with_name(prefix.name + "_gated_output_f32.npy"), gated_output.float())
        write_npy(prefix.with_name(prefix.name + "_gated_output_bf16_bits_i32.npy"), bf16_bits_tensor(gated_output))
        write_npy(prefix.with_name(prefix.name + "_down_proj_f32.npy"), down_w.float())
        write_npy(prefix.with_name(prefix.name + "_down_proj_bf16_bits_i32.npy"), bf16_bits_tensor(down_w))
        write_npy(prefix.with_name(prefix.name + "_down_proj_bias_f32.npy"), down_b.float())
        write_npy(prefix.with_name(prefix.name + f"_down_proj_channel{channel}_f32.npy"), down_w[:, channel].float())
        write_npy(prefix.with_name(prefix.name + "_down_pre_bias_f32.npy"), down_pre_bias.float())
        write_npy(prefix.with_name(prefix.name + "_down_post_bias_f32.npy"), down_post_bias.float())
        write_npy(prefix.with_name(prefix.name + "_down_post_bias_bf16_bits_i32.npy"), bf16_bits_tensor(down_post_bias))

        for top_k_pos in positions:
            route = selected_scores[top_k_pos]
            weighted = down_post_bias * route
            weighted_hidden = weighted.to(current_state.dtype)
            accumulated = accumulated + weighted_hidden
            write_npy(prefix.with_name(prefix.name + f"_topk{top_k_pos}_weighted_output_f32.npy"), weighted.float())
            write_npy(prefix.with_name(prefix.name + f"_topk{top_k_pos}_weighted_output_hidden_dtype_f32.npy"), weighted_hidden.float())
            write_npy(prefix.with_name(prefix.name + f"_topk{top_k_pos}_accumulated_after_f32.npy"), accumulated.float())
            expert_meta[f"topk_{top_k_pos}"] = {
                "routing_weight": scalar_meta(route),
                "down_pre_bias_channel": scalar_meta(down_pre_bias[channel]),
                "down_post_bias_channel": scalar_meta(down_post_bias[channel]),
                "weighted_channel": scalar_meta(weighted[channel]),
                "weighted_hidden_dtype_channel": scalar_meta(weighted_hidden[channel]),
                "accumulated_channel_after": scalar_meta(accumulated[channel]),
            }
            metadata["expert_order"].append({"expert_id": expert_id, "top_k_pos": top_k_pos})

        expert_meta["gate_up_pre_bias_channel_pair"] = [
            scalar_meta(gate_up_pre_bias[channel * 2]),
            scalar_meta(gate_up_pre_bias[channel * 2 + 1]),
        ] if channel * 2 + 1 < gate_up_pre_bias.numel() else []
        expert_meta["down_channel"] = {
            "pre_bias": scalar_meta(down_pre_bias[channel]),
            "post_bias": scalar_meta(down_post_bias[channel]),
        }
        metadata["experts"][str(expert_id)] = expert_meta

    write_npy(output_dir / f"layer0_moe_token{token}_manual_accumulated_f32.npy", accumulated.float())
    write_npy(output_dir / f"layer0_moe_token{token}_manual_accumulated_bf16_bits_i32.npy", bf16_bits_tensor(accumulated))
    metadata["final_channel"] = scalar_meta(accumulated[channel])
    return metadata


@torch.no_grad()
def export_layer0_manual_checkpoints(
    text_encoder,
    embedding: torch.Tensor,
    input_ids: torch.Tensor,
    attention: torch.Tensor,
    output_dir: Path,
    moe_debug_token: int = 16,
    moe_debug_channel: int = 1167,
) -> dict[str, object]:
    model = text_encoder.model
    layer0 = model.layers[0]

    def tensor_layout_meta(tensor: torch.Tensor) -> dict[str, object]:
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "stride": list(tensor.stride()),
            "is_contiguous": bool(tensor.is_contiguous()),
            "device": str(tensor.device),
        }

    def diff_meta(actual: torch.Tensor, ref: torch.Tensor) -> dict[str, object]:
        diff = (actual.float() - ref.float()).abs()
        return {
            "max_diff": float(diff.max().item()) if diff.numel() else 0.0,
            "mean_diff": float(diff.mean().item()) if diff.numel() else 0.0,
            "max_index": int(diff.reshape(-1).argmax().item()) if diff.numel() else 0,
        }

    def projection_formula_meta(name: str, module: torch.nn.Linear, input_tensor: torch.Tensor, ref: torch.Tensor) -> dict[str, object]:
        weight = module.weight
        bias = module.bias
        original_reduced = bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
        reduced_precision_runs: dict[str, object] = {}
        module_outputs: dict[bool, torch.Tensor] = {}
        try:
            for allow_reduced in (True, False):
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = allow_reduced
                variants: dict[str, torch.Tensor] = {
                    "module_forward": module(input_tensor),
                    "f_linear": torch.nn.functional.linear(input_tensor, weight, bias),
                    "input_float_weight_float_bias_float_to_bf16": (
                        input_tensor.float() @ weight.float().transpose(0, 1) + bias.float()
                    ).to(torch.bfloat16),
                    "input_bf16_weight_bf16_add_bias_bf16": (
                        input_tensor.to(torch.bfloat16) @ weight.to(torch.bfloat16).transpose(0, 1) + bias.to(torch.bfloat16)
                    ),
                }
                matmul_bf16 = input_tensor.to(torch.bfloat16) @ weight.to(torch.bfloat16).transpose(0, 1)
                variants["matmul_bf16_to_bf16_then_add_bias_bf16"] = matmul_bf16.to(torch.bfloat16) + bias.to(torch.bfloat16)
                variants["matmul_bf16_to_float_add_bias_float_to_bf16"] = (
                    matmul_bf16.float() + bias.float()
                ).to(torch.bfloat16)
                module_outputs[allow_reduced] = variants["module_forward"].detach()
                reduced_precision_runs[str(allow_reduced).lower()] = {
                    "formula_diffs": {key: diff_meta(value, ref) for key, value in variants.items()},
                    "module_forward_vs_f_linear": diff_meta(variants["module_forward"], variants["f_linear"]),
                    "module_forward_file": f"layer0_projection_{name}_module_allow_bf16_reduction_{str(allow_reduced).lower()}_f32.npy",
                    "f_linear_file": f"layer0_projection_{name}_f_linear_allow_bf16_reduction_{str(allow_reduced).lower()}_f32.npy",
                }
                write_npy(
                    output_dir / f"layer0_projection_{name}_module_allow_bf16_reduction_{str(allow_reduced).lower()}_f32.npy",
                    variants["module_forward"].float(),
                )
                write_npy(
                    output_dir / f"layer0_projection_{name}_f_linear_allow_bf16_reduction_{str(allow_reduced).lower()}_f32.npy",
                    variants["f_linear"].float(),
                )
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original_reduced
        return {
            "input": tensor_layout_meta(input_tensor),
            "weight": tensor_layout_meta(weight),
            "bias": tensor_layout_meta(bias),
            "output": tensor_layout_meta(ref),
            "allow_bf16_reduced_precision_reduction_original": original_reduced,
            "reduced_precision_runs": reduced_precision_runs,
            "module_forward_true_vs_false": (
                diff_meta(module_outputs[True], module_outputs[False])
                if True in module_outputs and False in module_outputs
                else {}
            ),
        }

    def rmsnorm_oracle_meta(module, input_tensor: torch.Tensor, ref: torch.Tensor) -> dict[str, object]:
        input_dtype = input_tensor.dtype
        hidden_fp32 = input_tensor.to(torch.float32)
        variance = hidden_fp32.pow(2).mean(-1, keepdim=True)
        rsqrt = torch.rsqrt(variance + module.variance_epsilon)
        normalized = hidden_fp32 * rsqrt
        weighted = module.weight * normalized
        output = weighted.to(input_dtype)
        worst_flat = int((ref.float() - output.float()).abs().reshape(-1).argmax().item())
        hidden = int(input_tensor.shape[-1])
        selected = [(0, 0), (worst_flat // hidden, worst_flat % hidden), (18, 92)]
        selected = sorted({(int(t), int(c)) for t, c in selected if t < input_tensor.shape[1] and c < hidden})

        write_npy(output_dir / "layer0_rmsnorm_input_f32.npy", input_tensor.float())
        write_npy(output_dir / "layer0_rmsnorm_weight_f32.npy", module.weight.float())
        write_npy(output_dir / "layer0_rmsnorm_weight_bf16_bits_i32.npy", bf16_bits_tensor(module.weight))
        write_npy(output_dir / "layer0_rmsnorm_variance_f32.npy", variance.float())
        write_npy(output_dir / "layer0_rmsnorm_rsqrt_f32.npy", rsqrt.float())
        write_npy(output_dir / "layer0_rmsnorm_normalized_f32.npy", normalized.float())
        write_npy(output_dir / "layer0_rmsnorm_weighted_f32.npy", weighted.float())
        write_npy(output_dir / "layer0_rmsnorm_output_f32.npy", output.float())
        write_npy(output_dir / "layer0_rmsnorm_output_bf16_bits_i32.npy", bf16_bits_tensor(output))

        scalar = []
        for token, channel in selected:
            scalar.append(
                {
                    "token": token,
                    "channel": channel,
                    "input_f32": float(input_tensor[0, token, channel].float().item()),
                    "input_bf16_bits": int(bf16_bits_tensor(input_tensor[0, token, channel]).item()) & 0xFFFF,
                    "weight_f32": float(module.weight[channel].float().item()),
                    "weight_dtype": str(module.weight.dtype),
                    "weight_bf16_bits": int(bf16_bits_tensor(module.weight[channel]).item()) & 0xFFFF,
                    "variance_f32": float(variance[0, token, 0].float().item()),
                    "rsqrt_f32": float(rsqrt[0, token, 0].float().item()),
                    "normalized_f32": float(normalized[0, token, channel].float().item()),
                    "weighted_f32": float(weighted[0, token, channel].float().item()),
                    "output_f32": float(output[0, token, channel].float().item()),
                    "output_bf16_bits": int(bf16_bits_tensor(output[0, token, channel]).item()) & 0xFFFF,
                }
            )
        return {
            "input": tensor_layout_meta(input_tensor),
            "weight": tensor_layout_meta(module.weight),
            "eps": float(module.variance_epsilon),
            "output": tensor_layout_meta(output),
            "formula": "input.to(float32); variance=mean(x^2); x=x*rsqrt(variance+eps); output=(weight*x).to(input_dtype)",
            "output_vs_module": diff_meta(output, ref),
            "selected_scalars": scalar,
            "files": {
                "input": "layer0_rmsnorm_input_f32.npy",
                "weight": "layer0_rmsnorm_weight_f32.npy",
                "weight_bf16_bits": "layer0_rmsnorm_weight_bf16_bits_i32.npy",
                "variance": "layer0_rmsnorm_variance_f32.npy",
                "rsqrt": "layer0_rmsnorm_rsqrt_f32.npy",
                "normalized": "layer0_rmsnorm_normalized_f32.npy",
                "weighted": "layer0_rmsnorm_weighted_f32.npy",
                "output": "layer0_rmsnorm_output_f32.npy",
                "output_bf16_bits": "layer0_rmsnorm_output_bf16_bits_i32.npy",
            },
        }

    position_ids = torch.arange(embedding.shape[1], device=embedding.device).unsqueeze(0).expand_as(input_ids)
    mask_kwargs = {
        "config": model.config,
        "inputs_embeds": embedding,
        "attention_mask": attention,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    causal_mask_mapping = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
    }
    position_embeddings = model.rotary_emb(embedding, position_ids)
    inv_freq_expanded = model.rotary_emb.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(embedding.device)
    position_ids_expanded = position_ids[:, None, :].float()
    device_type = embedding.device.type if isinstance(embedding.device.type, str) and embedding.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        rope_freqs_fp32 = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        rope_cos_fp32 = rope_freqs_fp32.cos() * model.rotary_emb.attention_scaling
        rope_sin_fp32 = rope_freqs_fp32.sin() * model.rotary_emb.attention_scaling
    rope_cos_cast = rope_cos_fp32.to(embedding.dtype)
    rope_sin_cast = rope_sin_fp32.to(embedding.dtype)

    residual0 = embedding
    norm0 = layer0.input_layernorm(embedding)
    q_proj = layer0.self_attn.q_proj(norm0)
    k_proj = layer0.self_attn.k_proj(norm0)
    v_proj = layer0.self_attn.v_proj(norm0)

    input_shape = norm0.shape[:-1]
    hidden_shape = (*input_shape, -1, layer0.self_attn.head_dim)
    q_states = q_proj.view(hidden_shape).transpose(1, 2)
    k_states = k_proj.view(hidden_shape).transpose(1, 2)
    v_states = v_proj.view(hidden_shape).transpose(1, 2)
    q_rope, k_rope = apply_rotary_pos_emb(q_states, k_states, *position_embeddings)

    k_repeat = repeat_kv(k_rope, layer0.self_attn.num_key_value_groups)
    v_repeat = repeat_kv(v_states, layer0.self_attn.num_key_value_groups)
    attn_mask = causal_mask_mapping[model.config.layer_types[0]]
    attn_logits_raw = torch.matmul(q_rope, k_repeat.transpose(2, 3)) * layer0.self_attn.scaling
    attn_logits_masked = attn_logits_raw + attn_mask
    sinks = layer0.self_attn.sinks.reshape(1, -1, 1, 1).expand(q_rope.shape[0], -1, q_rope.shape[-2], -1)
    combined_logits = torch.cat([attn_logits_masked, sinks], dim=-1)
    row_max = combined_logits.max(dim=-1, keepdim=True).values
    combined_logits_centered = combined_logits - row_max
    combined_probs = torch.nn.functional.softmax(combined_logits_centered, dim=-1, dtype=combined_logits_centered.dtype)
    token_scores = combined_probs[..., :-1]
    token_scores_value_dtype = token_scores.to(v_repeat.dtype)
    attn_pre_o_exact = torch.matmul(token_scores_value_dtype, v_repeat)

    attn_pre_o, _ = eager_attention_forward(
        layer0.self_attn,
        q_rope,
        k_rope,
        v_states,
        causal_mask_mapping[model.config.layer_types[0]],
        dropout=0.0,
        scaling=layer0.self_attn.scaling,
        sliding_window=layer0.self_attn.sliding_window,
        s_aux=layer0.self_attn.sinks,
    )
    attn_pre_o_flat = attn_pre_o.reshape(*input_shape, -1).contiguous()
    attn_pre_o_exact_flat = attn_pre_o_exact.transpose(1, 2).contiguous().reshape(*input_shape, -1).contiguous()
    attn_o = layer0.self_attn.o_proj(attn_pre_o_flat)
    post_attn_residual = residual0 + attn_o
    post_attn_norm = layer0.post_attention_layernorm(post_attn_residual)
    router_logits, router_scores, router_indices = layer0.mlp.router(
        post_attn_norm.reshape(-1, post_attn_norm.shape[-1])
    )
    mlp_out, _ = layer0.mlp(post_attn_norm)
    final_hidden = post_attn_residual + mlp_out

    tensors = {
        "layer0_manual_position_ids_i64.npy": position_ids.to(torch.int64),
        "layer0_manual_rope_inv_freq_f32.npy": model.rotary_emb.inv_freq.float(),
        "layer0_manual_rope_attention_scaling_f32.npy": torch.tensor([float(model.rotary_emb.attention_scaling)], device=embedding.device, dtype=torch.float32),
        "layer0_manual_rope_freqs_fp32.npy": rope_freqs_fp32.float(),
        "layer0_manual_rope_cos_fp32_precise.npy": rope_cos_fp32.float(),
        "layer0_manual_rope_sin_fp32_precise.npy": rope_sin_fp32.float(),
        "layer0_manual_rope_cos_f32.npy": position_embeddings[0].float(),
        "layer0_manual_rope_sin_f32.npy": position_embeddings[1].float(),
        "layer0_manual_rope_cos_bf16_bits_i32.npy": bf16_bits_tensor(rope_cos_cast),
        "layer0_manual_rope_sin_bf16_bits_i32.npy": bf16_bits_tensor(rope_sin_cast),
        "layer0_manual_input_norm_output_f32.npy": norm0.float(),
        "layer0_manual_q_proj_f32.npy": q_proj.float(),
        "layer0_manual_k_proj_f32.npy": k_proj.float(),
        "layer0_manual_v_proj_f32.npy": v_proj.float(),
        "layer0_manual_q_proj_weight_f32.npy": layer0.self_attn.q_proj.weight.float(),
        "layer0_manual_q_proj_bias_f32.npy": layer0.self_attn.q_proj.bias.float(),
        "layer0_manual_k_proj_weight_f32.npy": layer0.self_attn.k_proj.weight.float(),
        "layer0_manual_k_proj_bias_f32.npy": layer0.self_attn.k_proj.bias.float(),
        "layer0_manual_v_proj_weight_f32.npy": layer0.self_attn.v_proj.weight.float(),
        "layer0_manual_v_proj_bias_f32.npy": layer0.self_attn.v_proj.bias.float(),
        "layer0_manual_q_states_f32.npy": q_states.float(),
        "layer0_manual_k_states_f32.npy": k_states.float(),
        "layer0_manual_v_states_pre_repeat_f32.npy": v_states.float(),
        "layer0_manual_q_rope_f32.npy": q_rope.float(),
        "layer0_manual_k_rope_f32.npy": k_rope.float(),
        "layer0_manual_v_states_f32.npy": v_states.float(),
        "layer0_manual_k_repeat_f32.npy": k_repeat.float(),
        "layer0_manual_v_repeat_f32.npy": v_repeat.float(),
        "layer0_manual_attention_mask_f32.npy": attn_mask.float(),
        "layer0_manual_attention_logits_raw_f32.npy": attn_logits_raw.float(),
        "layer0_manual_attention_logits_masked_f32.npy": attn_logits_masked.float(),
        "layer0_manual_attention_sinks_f32.npy": layer0.self_attn.sinks.float(),
        "layer0_manual_attention_sinks_expanded_f32.npy": sinks.float(),
        "layer0_manual_attention_combined_logits_f32.npy": combined_logits.float(),
        "layer0_manual_attention_row_max_f32.npy": row_max.float(),
        "layer0_manual_attention_combined_logits_centered_f32.npy": combined_logits_centered.float(),
        "layer0_manual_attention_probs_with_sink_f32.npy": combined_probs.float(),
        "layer0_manual_attention_token_scores_f32.npy": token_scores.float(),
        "layer0_manual_attention_token_scores_value_dtype_f32.npy": token_scores_value_dtype.float(),
        "layer0_manual_attention_pre_o_exact_f32.npy": attn_pre_o_exact_flat.float(),
        "layer0_manual_attention_pre_o_f32.npy": attn_pre_o_flat.float(),
        "layer0_manual_attention_o_f32.npy": attn_o.float(),
        "layer0_manual_o_proj_weight_f32.npy": layer0.self_attn.o_proj.weight.float(),
        "layer0_manual_o_proj_bias_f32.npy": layer0.self_attn.o_proj.bias.float(),
        "layer0_manual_post_attention_residual_f32.npy": post_attn_residual.float(),
        "layer0_manual_post_attention_norm_f32.npy": post_attn_norm.float(),
        "layer0_manual_router_logits_f32.npy": router_logits.reshape(1, embedding.shape[1], -1).float(),
        "layer0_manual_router_scores_f32.npy": router_scores.reshape(1, embedding.shape[1], -1).float(),
        "layer0_manual_router_indices_i64.npy": router_indices.reshape(1, embedding.shape[1], -1).to(torch.int64),
        "layer0_manual_mlp_output_f32.npy": mlp_out.float(),
        "layer0_manual_final_hidden_f32.npy": final_hidden.float(),
        "layer0_input_norm_output_f32.npy": norm0.float(),
        "layer0_attention_output_f32.npy": attn_o.float(),
        "layer0_post_attention_norm_output_f32.npy": post_attn_norm.float(),
        "layer0_router_logits_f32.npy": router_logits.reshape(1, embedding.shape[1], -1).float(),
        "layer0_router_scores_f32.npy": router_scores.reshape(1, embedding.shape[1], -1).float(),
        "layer0_router_indices_i64.npy": router_indices.reshape(1, embedding.shape[1], -1).to(torch.int64),
        "layer0_mlp_output_f32.npy": mlp_out.float(),
        "layer0_final_hidden_f32.npy": final_hidden.float(),
    }
    for name, tensor in tensors.items():
        write_npy(output_dir / name, tensor)
    active = sorted({int(v) for v in router_indices.detach().cpu().reshape(-1).tolist()})
    counts = torch.bincount(router_indices.detach().cpu().reshape(-1), minlength=model.config.num_local_experts)
    moe_debug_meta = export_layer0_moe_debug(
        layer0,
        post_attn_norm,
        router_scores,
        router_indices,
        output_dir,
        moe_debug_token,
        moe_debug_channel,
    )
    try:
        autocast_cuda_enabled = bool(torch.is_autocast_enabled("cuda"))
    except TypeError:
        autocast_cuda_enabled = bool(torch.is_autocast_cuda_enabled()) if hasattr(torch, "is_autocast_cuda_enabled") else False
    projection_meta = {
        "q": projection_formula_meta("q", layer0.self_attn.q_proj, norm0, q_proj),
        "k": projection_formula_meta("k", layer0.self_attn.k_proj, norm0, k_proj),
        "v": projection_formula_meta("v", layer0.self_attn.v_proj, norm0, v_proj),
        "o": projection_formula_meta("o", layer0.self_attn.o_proj, attn_pre_o_flat, attn_o),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(embedding.device) if embedding.is_cuda else "cpu",
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cuda_matmul_allow_bf16_reduced_precision_reduction": bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "autocast_enabled": bool(torch.is_autocast_enabled()),
        "autocast_cuda_enabled": autocast_cuda_enabled,
    }
    return {
        "layer_type": model.config.layer_types[0],
        "q_proj_shape": list(q_proj.shape),
        "k_proj_shape": list(k_proj.shape),
        "v_proj_shape": list(v_proj.shape),
        "q_rope_shape": list(q_rope.shape),
        "k_repeat_shape": list(k_repeat.shape),
        "v_repeat_shape": list(v_repeat.shape),
        "attention_mask_shape": list(attn_mask.shape),
        "attention_mask_dtype": str(attn_mask.dtype),
        "attention_mask_min": float(attn_mask.float().min().item()),
        "attention_mask_max": float(attn_mask.float().max().item()),
        "attention_layer_type": model.config.layer_types[0],
        "attention_sliding_window": int(layer0.self_attn.sliding_window or 0),
        "attention_sinks_shape": list(layer0.self_attn.sinks.shape),
        "attention_sinks_expanded_shape": list(sinks.shape),
        "attention_sink_sample": [float(x) for x in layer0.self_attn.sinks.detach().float().cpu()[:8].tolist()],
        "num_query_heads": int(layer0.self_attn.config.num_attention_heads),
        "num_kv_heads": int(layer0.self_attn.config.num_key_value_heads),
        "num_kv_repeats": int(layer0.self_attn.num_key_value_groups),
        "head_dim": int(layer0.self_attn.head_dim),
        "attention_scale": float(layer0.self_attn.scaling),
        "attention_path": "manual HuggingFace GPT-OSS eager_attention_forward with sink logits",
        "attention_pre_o_exact_vs_eager_max_diff": float((attn_pre_o_exact_flat.float() - attn_pre_o_flat.float()).abs().max().item()),
        "attention_pre_o_exact_vs_eager_mean_diff": float((attn_pre_o_exact_flat.float() - attn_pre_o_flat.float()).abs().mean().item()),
        "attention_pre_o_shape": list(attn_pre_o_flat.shape),
        "projection_oracle": projection_meta,
        "rmsnorm_oracle": rmsnorm_oracle_meta(layer0.input_layernorm, embedding, norm0),
        "rope_oracle": {
            "position_ids": tensor_layout_meta(position_ids),
            "inv_freq": tensor_layout_meta(model.rotary_emb.inv_freq),
            "attention_scaling": float(model.rotary_emb.attention_scaling),
            "freqs_fp32": tensor_layout_meta(rope_freqs_fp32),
            "cos_fp32_precise": tensor_layout_meta(rope_cos_fp32),
            "sin_fp32_precise": tensor_layout_meta(rope_sin_fp32),
            "cos_after_cast": tensor_layout_meta(rope_cos_cast),
            "sin_after_cast": tensor_layout_meta(rope_sin_cast),
            "q_before_rope": tensor_layout_meta(q_states),
            "k_before_rope": tensor_layout_meta(k_states),
            "q_after_rope": tensor_layout_meta(q_rope),
            "k_after_rope": tensor_layout_meta(k_rope),
            "formula": "split-half rotation; cos/sin computed FP32 with autocast disabled, cast to input dtype; BF16 elementwise products are rounded before add/sub",
            "selected_coordinates": {
                "q_previous_worst": [0, 13, 11, 2],
                "k_previous_worst": [0, 0, 8, 11],
                "stable": [0, 0, 0, 0],
            },
            "files": {
                "position_ids": "layer0_manual_position_ids_i64.npy",
                "inv_freq": "layer0_manual_rope_inv_freq_f32.npy",
                "attention_scaling": "layer0_manual_rope_attention_scaling_f32.npy",
                "freqs_fp32": "layer0_manual_rope_freqs_fp32.npy",
                "cos_fp32_precise": "layer0_manual_rope_cos_fp32_precise.npy",
                "sin_fp32_precise": "layer0_manual_rope_sin_fp32_precise.npy",
                "cos_after_cast": "layer0_manual_rope_cos_f32.npy",
                "sin_after_cast": "layer0_manual_rope_sin_f32.npy",
                "cos_bf16_bits": "layer0_manual_rope_cos_bf16_bits_i32.npy",
                "sin_bf16_bits": "layer0_manual_rope_sin_bf16_bits_i32.npy",
                "q_before_rope": "layer0_manual_q_states_f32.npy",
                "k_before_rope": "layer0_manual_k_states_f32.npy",
                "q_after_rope": "layer0_manual_q_rope_f32.npy",
                "k_after_rope": "layer0_manual_k_rope_f32.npy",
            },
        },
        "router_logits_shape": [1, int(embedding.shape[1]), int(model.config.num_local_experts)],
        "router_topk_shape": [1, int(embedding.shape[1]), int(model.config.num_experts_per_tok)],
        "active_experts": active,
        "tokens_per_expert_topk_hits": [int(x) for x in counts.tolist()],
        "moe_debug": moe_debug_meta,
    }


@torch.no_grad()
def export_layer_manual_checkpoints(
    text_encoder,
    layer_idx: int,
    layer_input: torch.Tensor,
    input_ids: torch.Tensor,
    attention: torch.Tensor,
    output_dir: Path,
) -> dict[str, object]:
    model = text_encoder.model
    layer = model.layers[layer_idx]
    prefix = f"layer{layer_idx}"
    layer_input = layer_input.to(device=input_ids.device, dtype=model.embed_tokens.weight.dtype).contiguous()

    def layout(tensor: torch.Tensor) -> dict[str, object]:
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "stride": list(tensor.stride()),
            "is_contiguous": bool(tensor.is_contiguous()),
            "device": str(tensor.device),
        }

    position_ids = torch.arange(layer_input.shape[1], device=layer_input.device).unsqueeze(0).expand_as(input_ids)
    mask_kwargs = {
        "config": model.config,
        "inputs_embeds": layer_input,
        "attention_mask": attention,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    masks = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
    }
    position_embeddings = model.rotary_emb(layer_input, position_ids)
    residual = layer_input
    norm0 = layer.input_layernorm(layer_input)
    q_proj = layer.self_attn.q_proj(norm0)
    k_proj = layer.self_attn.k_proj(norm0)
    v_proj = layer.self_attn.v_proj(norm0)

    input_shape = norm0.shape[:-1]
    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
    q_states = q_proj.view(hidden_shape).transpose(1, 2)
    k_states = k_proj.view(hidden_shape).transpose(1, 2)
    v_states = v_proj.view(hidden_shape).transpose(1, 2)
    q_rope, k_rope = apply_rotary_pos_emb(q_states, k_states, *position_embeddings)
    k_repeat = repeat_kv(k_rope, layer.self_attn.num_key_value_groups)
    v_repeat = repeat_kv(v_states, layer.self_attn.num_key_value_groups)
    attn_mask = masks[model.config.layer_types[layer_idx]]
    attn_logits_raw = torch.matmul(q_rope, k_repeat.transpose(2, 3)) * layer.self_attn.scaling
    attn_logits_masked = attn_logits_raw + attn_mask
    sinks = layer.self_attn.sinks.reshape(1, -1, 1, 1).expand(q_rope.shape[0], -1, q_rope.shape[-2], -1)
    combined_logits = torch.cat([attn_logits_masked, sinks], dim=-1)
    row_max = combined_logits.max(dim=-1, keepdim=True).values
    combined_logits_centered = combined_logits - row_max
    combined_probs = torch.nn.functional.softmax(combined_logits_centered, dim=-1, dtype=combined_logits_centered.dtype)
    token_scores = combined_probs[..., :-1]
    token_scores_value_dtype = token_scores.to(v_repeat.dtype)
    attn_pre_o_exact = torch.matmul(token_scores_value_dtype, v_repeat)
    attn_pre_o, _ = eager_attention_forward(
        layer.self_attn,
        q_rope,
        k_rope,
        v_states,
        attn_mask,
        dropout=0.0,
        scaling=layer.self_attn.scaling,
        sliding_window=layer.self_attn.sliding_window,
        s_aux=layer.self_attn.sinks,
    )
    attn_pre_o_flat = attn_pre_o.reshape(*input_shape, -1).contiguous()
    attn_pre_o_exact_flat = attn_pre_o_exact.transpose(1, 2).contiguous().reshape(*input_shape, -1).contiguous()
    attn_o = layer.self_attn.o_proj(attn_pre_o_flat)
    post_attn_residual = residual + attn_o
    post_attn_norm = layer.post_attention_layernorm(post_attn_residual)
    router_logits, router_scores, router_indices = layer.mlp.router(
        post_attn_norm.reshape(-1, post_attn_norm.shape[-1])
    )
    mlp_out, _ = layer.mlp(post_attn_norm)
    final_hidden = post_attn_residual + mlp_out

    tensors = {
        f"{prefix}_manual_input_norm_output_f32.npy": norm0.float(),
        f"{prefix}_manual_q_proj_f32.npy": q_proj.float(),
        f"{prefix}_manual_k_proj_f32.npy": k_proj.float(),
        f"{prefix}_manual_v_proj_f32.npy": v_proj.float(),
        f"{prefix}_manual_q_states_f32.npy": q_states.float(),
        f"{prefix}_manual_k_states_f32.npy": k_states.float(),
        f"{prefix}_manual_v_states_pre_repeat_f32.npy": v_states.float(),
        f"{prefix}_manual_q_rope_f32.npy": q_rope.float(),
        f"{prefix}_manual_k_rope_f32.npy": k_rope.float(),
        f"{prefix}_manual_v_states_f32.npy": v_states.float(),
        f"{prefix}_manual_k_repeat_f32.npy": k_repeat.float(),
        f"{prefix}_manual_v_repeat_f32.npy": v_repeat.float(),
        f"{prefix}_manual_attention_mask_f32.npy": attn_mask.float(),
        f"{prefix}_manual_attention_logits_raw_f32.npy": attn_logits_raw.float(),
        f"{prefix}_manual_attention_logits_masked_f32.npy": attn_logits_masked.float(),
        f"{prefix}_manual_attention_sinks_f32.npy": layer.self_attn.sinks.float(),
        f"{prefix}_manual_attention_sinks_expanded_f32.npy": sinks.float(),
        f"{prefix}_manual_attention_combined_logits_f32.npy": combined_logits.float(),
        f"{prefix}_manual_attention_row_max_f32.npy": row_max.float(),
        f"{prefix}_manual_attention_combined_logits_centered_f32.npy": combined_logits_centered.float(),
        f"{prefix}_manual_attention_probs_with_sink_f32.npy": combined_probs.float(),
        f"{prefix}_manual_attention_token_scores_f32.npy": token_scores.float(),
        f"{prefix}_manual_attention_token_scores_value_dtype_f32.npy": token_scores_value_dtype.float(),
        f"{prefix}_manual_attention_pre_o_exact_f32.npy": attn_pre_o_exact_flat.float(),
        f"{prefix}_manual_attention_pre_o_f32.npy": attn_pre_o_flat.float(),
        f"{prefix}_manual_attention_o_f32.npy": attn_o.float(),
        f"{prefix}_manual_post_attention_residual_f32.npy": post_attn_residual.float(),
        f"{prefix}_manual_post_attention_norm_f32.npy": post_attn_norm.float(),
        f"{prefix}_manual_router_logits_f32.npy": router_logits.reshape(1, layer_input.shape[1], -1).float(),
        f"{prefix}_manual_router_scores_f32.npy": router_scores.reshape(1, layer_input.shape[1], -1).float(),
        f"{prefix}_manual_router_indices_i64.npy": router_indices.reshape(1, layer_input.shape[1], -1).to(torch.int64),
        f"{prefix}_manual_mlp_output_f32.npy": mlp_out.float(),
        f"{prefix}_manual_final_hidden_f32.npy": final_hidden.float(),
    }
    for name, tensor in tensors.items():
        write_npy(output_dir / name, tensor)

    return {
        "layer_type": model.config.layer_types[layer_idx],
        "q_proj_shape": list(q_proj.shape),
        "k_proj_shape": list(k_proj.shape),
        "v_proj_shape": list(v_proj.shape),
        "q_rope_shape": list(q_rope.shape),
        "k_repeat_shape": list(k_repeat.shape),
        "v_repeat_shape": list(v_repeat.shape),
        "attention_mask_shape": list(attn_mask.shape),
        "attention_mask_dtype": str(attn_mask.dtype),
        "attention_mask_min": float(attn_mask.float().min().item()),
        "attention_mask_max": float(attn_mask.float().max().item()),
        "attention_sinks_shape": list(layer.self_attn.sinks.shape),
        "attention_sinks_expanded_shape": list(sinks.shape),
        "attention_sink_sample": [float(x) for x in layer.self_attn.sinks.detach().float().cpu()[:8].tolist()],
        "num_query_heads": int(layer.self_attn.config.num_attention_heads),
        "num_kv_heads": int(layer.self_attn.config.num_key_value_heads),
        "num_kv_repeats": int(layer.self_attn.num_key_value_groups),
        "head_dim": int(layer.self_attn.head_dim),
        "attention_scale": float(layer.self_attn.scaling),
        "attention_path": "manual HuggingFace GPT-OSS eager_attention_forward with sink logits",
        "attention_pre_o_exact_vs_eager_max_diff": float((attn_pre_o_exact_flat.float() - attn_pre_o_flat.float()).abs().max().item()),
        "attention_pre_o_exact_vs_eager_mean_diff": float((attn_pre_o_exact_flat.float() - attn_pre_o_flat.float()).abs().mean().item()),
        "input": layout(layer_input),
        "final_hidden": layout(final_hidden),
    }


def main() -> int:
    args = parse_args()
    if args.bf16_reduced_precision_reduction != "default":
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = args.bf16_reduced_precision_reduction == "true"
    lens_root = Path(args.lens_root)
    sys.path.insert(0, str(lens_root))

    from lens import LensGptOssEncoder  # noqa: PLC0415
    from lens.pipeline import _CHAT_ASSISTANT_THINKING, _CHAT_SYSTEM, DEFAULT_TXT_OFFSET  # noqa: PLC0415

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    enable_conditioning = args.oracle_mode in {"golden", "full"}
    enable_layer_debug = args.oracle_mode in {"debug-projection", "full"}
    enable_layer_hooks = args.oracle_mode == "full"

    transformer_config = json.loads((model_dir / "transformer" / "config.json").read_text(encoding="utf-8"))
    selected_layers = [int(x) for x in transformer_config["selected_layer_index"]]

    tokenizer = AutoTokenizer.from_pretrained(model_dir, subfolder="tokenizer", local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    rendered = render_prompt(tokenizer, args.prompt, _CHAT_SYSTEM, _CHAT_ASSISTANT_THINKING)
    (output_dir / "rendered_prompt.txt").write_text(rendered, encoding="utf-8")

    encoded = tokenizer(
        [rendered],
        padding=True,
        truncation=True,
        max_length=args.max_sequence_length,
        return_tensors="pt",
        add_special_tokens=True,
    )
    write_npy(output_dir / "input_ids_i64.npy", encoded["input_ids"].to(torch.int64))
    write_npy(output_dir / "attention_mask_i64.npy", encoded["attention_mask"].to(torch.int64))

    kwargs = {
        "subfolder": "text_encoder",
        "dtype": torch_dtype(args.dtype),
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    try:
        from transformers import Mxfp4Config  # noqa: PLC0415

        kwargs["quantization_config"] = Mxfp4Config(dequantize=not args.keep_mxfp4)
    except ImportError:
        pass

    cuda_note("before text_encoder load")
    text_encoder = LensGptOssEncoder.from_pretrained(model_dir, **kwargs)
    text_encoder.set_selected_layers(selected_layers)
    text_encoder.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder.to(device)
    cuda_note("after text_encoder to device")

    captures: dict[str, torch.Tensor | tuple[torch.Tensor, ...]] = {}

    def capture_hook(name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                captures[name] = tuple(
                    item.detach().float().cpu() for item in output if isinstance(item, torch.Tensor)
                )
            elif isinstance(output, torch.Tensor):
                captures[name] = output.detach().float().cpu()
        return hook

    def capture_layer_input_hook(name: str):
        def hook(_module, inputs):
            if inputs and isinstance(inputs[0], torch.Tensor):
                captures[name] = inputs[0].detach().float().cpu()
        return hook

    hooks = []
    layer_capture_meta: dict[str, object] = {}
    if args.max_oracle_layer < 0:
        raise ValueError("--max-oracle-layer must be non-negative")
    max_oracle_layer = min(args.max_oracle_layer, len(text_encoder.model.layers) - 1)
    for layer_idx in range(max_oracle_layer + 1):
        prefix = f"layer{layer_idx}"
        layer_capture_meta[str(layer_idx)] = {
            "attention_mode": text_encoder.model.config.layer_types[layer_idx],
            "sliding_window": int(text_encoder.model.config.sliding_window),
            "capture_output_file": f"{prefix}_final_hidden_f32.npy",
            "router_logits_file": f"{prefix}_router_logits_f32.npy",
            "router_scores_file": f"{prefix}_router_scores_f32.npy",
            "router_indices_file": f"{prefix}_router_indices_i64.npy",
        }
        if enable_layer_hooks:
            layer = text_encoder.model.layers[layer_idx]
            hooks.extend(
                [
                    layer.register_forward_pre_hook(capture_layer_input_hook(f"{prefix}_input")),
                    layer.input_layernorm.register_forward_hook(capture_hook(f"{prefix}_input_norm_output")),
                    layer.self_attn.register_forward_hook(capture_hook(f"{prefix}_attention_output")),
                    layer.post_attention_layernorm.register_forward_hook(capture_hook(f"{prefix}_post_attention_norm_output")),
                    layer.mlp.router.register_forward_hook(capture_hook(f"{prefix}_router_output")),
                    layer.mlp.register_forward_hook(capture_hook(f"{prefix}_mlp_output")),
                    layer.register_forward_hook(capture_hook(f"{prefix}_final_hidden")),
                ]
            )

    input_ids = encoded["input_ids"].to(device)
    attention = encoded["attention_mask"].to(device)
    layer0_manual_meta: dict[str, object] = {}
    layer_manual_meta: dict[str, object] = {}
    layer_outputs = None
    with torch.no_grad():
        embedding = text_encoder.model.embed_tokens(input_ids)
        write_npy(output_dir / "token_embedding_f32.npy", embedding.float())
        write_npy(output_dir / "layer0_input_f32.npy", embedding.float())
        if enable_layer_debug:
            layer0_manual_meta = export_layer0_manual_checkpoints(
                text_encoder,
                embedding,
                input_ids,
                attention,
                output_dir,
                args.moe_debug_token,
                args.moe_debug_channel,
            )
        if enable_conditioning or enable_layer_hooks:
            layer_outputs = text_encoder.encode_layers(input_ids, attention)

    for hook in hooks:
        hook.remove()

    if input_ids.shape[1] > DEFAULT_TXT_OFFSET:
        mask = attention[:, DEFAULT_TXT_OFFSET:].bool().cpu()
    else:
        mask = torch.zeros((input_ids.shape[0], 0), dtype=torch.bool)
    write_npy(output_dir / "attention_mask_trimmed_i64.npy", mask.to(torch.int64))

    features: list[torch.Tensor] = []
    if enable_conditioning:
        if layer_outputs is None:
            raise RuntimeError("conditioning mode requires layer outputs")
        if input_ids.shape[1] > DEFAULT_TXT_OFFSET:
            features = [feat[:, DEFAULT_TXT_OFFSET:, :].contiguous().float().cpu() for feat in layer_outputs]
        else:
            zero_shape = (input_ids.shape[0], 0, layer_outputs[0].shape[-1])
            features = [layer_outputs[0].new_zeros(zero_shape).float().cpu() for _ in layer_outputs]
        for idx, feature in enumerate(features):
            write_npy(output_dir / f"feature_{idx}_layer{selected_layers[idx]}_trimmed_f32.npy", feature)
            write_npy(output_dir / f"feature_{idx}_layer{selected_layers[idx]}_untrimmed_f32.npy", layer_outputs[idx].float())

        cond_tensors = {f"feature_{idx}": feature for idx, feature in enumerate(features)}
        cond_tensors["attention_mask"] = mask.to(torch.int32)
        save_file(
            cond_tensors,
            output_dir / "lens_cond_v1.safetensors",
            metadata={
                "schema": "lens_cond_v1",
                "prompt": args.prompt,
                "selected_layer_index": json.dumps(selected_layers),
                "max_sequence_length": str(args.max_sequence_length),
                "txt_offset": str(DEFAULT_TXT_OFFSET),
            },
        )

    if enable_layer_hooks:
        for layer_idx in range(max_oracle_layer + 1):
            prefix = f"layer{layer_idx}"
            if isinstance(captures.get(f"{prefix}_input"), torch.Tensor):
                write_npy(output_dir / f"{prefix}_input_f32.npy", captures[f"{prefix}_input"])
            if isinstance(captures.get(f"{prefix}_input_norm_output"), torch.Tensor):
                write_npy(output_dir / f"{prefix}_input_norm_output_f32.npy", captures[f"{prefix}_input_norm_output"])
            attn = captures.get(f"{prefix}_attention_output")
            if isinstance(attn, tuple) and attn:
                write_npy(output_dir / f"{prefix}_attention_output_f32.npy", attn[0])
            post = captures.get(f"{prefix}_post_attention_norm_output")
            if isinstance(post, torch.Tensor):
                write_npy(output_dir / f"{prefix}_post_attention_norm_output_f32.npy", post)
            router = captures.get(f"{prefix}_router_output")
            if isinstance(router, tuple) and len(router) >= 3:
                write_npy(output_dir / f"{prefix}_router_logits_f32.npy", router[0])
                write_npy(output_dir / f"{prefix}_router_scores_f32.npy", router[1])
                write_npy(output_dir / f"{prefix}_router_indices_i64.npy", router[2].to(torch.int64))
            mlp = captures.get(f"{prefix}_mlp_output")
            if isinstance(mlp, tuple) and mlp:
                write_npy(output_dir / f"{prefix}_mlp_output_f32.npy", mlp[0])
            final = captures.get(f"{prefix}_final_hidden")
            if isinstance(final, torch.Tensor):
                write_npy(output_dir / f"{prefix}_final_hidden_f32.npy", final)
        for manual_layer_idx in (1,):
            captured_input = captures.get(f"layer{manual_layer_idx}_input")
            if isinstance(captured_input, torch.Tensor):
                layer_manual_meta[str(manual_layer_idx)] = export_layer_manual_checkpoints(
                    text_encoder,
                    manual_layer_idx,
                    captured_input,
                    input_ids,
                    attention,
                    output_dir,
                )

    mxfp4_meta = export_small_mxfp4_reference(model_dir / "text_encoder", output_dir)

    compare = {}
    if args.compare_cond and enable_conditioning:
        existing = load_file(args.compare_cond, device="cpu")
        for idx, feature in enumerate(features):
            key = f"feature_{idx}"
            if key in existing:
                diff = (feature - existing[key].float()).abs()
                compare[key] = {
                    "shape": list(feature.shape),
                    "max_diff": float(diff.max().item()) if diff.numel() else 0.0,
                    "mean_diff": float(diff.mean().item()) if diff.numel() else 0.0,
                }
        if "attention_mask" in existing:
            compare["attention_mask_equal"] = bool(torch.equal(mask.to(torch.int32), existing["attention_mask"].to(torch.int32)))
    elif args.compare_cond:
        compare["skipped"] = f"compare-cond is disabled in oracle-mode={args.oracle_mode}"

    metadata = {
        "schema": "lens_text_encoder_oracle_v1",
        "oracle_mode": args.oracle_mode,
        "prompt": args.prompt,
        "rendered_prompt_file": "rendered_prompt.txt",
        "model_dir": str(model_dir),
        "text_encoder_dir": str(model_dir / "text_encoder"),
        "tokenizer_dir": str(model_dir / "tokenizer"),
        "dtype": args.dtype,
        "max_sequence_length": args.max_sequence_length,
        "torch_version": torch.__version__,
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cuda_matmul_allow_bf16_reduced_precision_reduction": bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "autocast_enabled": bool(torch.is_autocast_enabled()),
        "txt_offset": int(DEFAULT_TXT_OFFSET),
        "captured_layer_indices": selected_layers,
        "input_ids_shape": list(encoded["input_ids"].shape),
        "attention_mask_shape": list(encoded["attention_mask"].shape),
        "trimmed_attention_mask_shape": list(mask.shape),
        "feature_shapes": [list(feature.shape) for feature in features],
        "feature_dtype": "float32",
        "token_embedding_shape": list(embedding.shape),
        "tokenizer": {
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "padding_side": tokenizer.padding_side,
        },
        "mxfp4_reference": mxfp4_meta,
        "layer0_manual_checkpoints": layer0_manual_meta,
        "layer_manual_checkpoints": layer_manual_meta,
        "layer_checkpoints": layer_capture_meta,
        "feature_captures": [
            {
                "feature_index": idx,
                "selected_layer": int(layer_idx),
                "capture_semantics": "hidden state returned immediately after decoder layer forward",
                "untrimmed_file": f"feature_{idx}_layer{layer_idx}_untrimmed_f32.npy",
                "trimmed_file": f"feature_{idx}_layer{layer_idx}_trimmed_f32.npy",
                "txt_offset": int(DEFAULT_TXT_OFFSET),
            }
            for idx, layer_idx in enumerate(selected_layers)
        ],
        "compare_existing_lens_cond": compare,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    hidden_dim = features[0].shape[-1] if features else int(embedding.shape[-1])
    print(
        f"wrote {output_dir} raw_seq={encoded['input_ids'].shape[1]} "
        f"trimmed_seq={mask.shape[1]} hidden={hidden_dim}"
    )
    if compare:
        print("compare_existing_lens_cond=" + json.dumps(compare, sort_keys=True))

    del layer_outputs, input_ids, attention, encoded, text_encoder, embedding
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    cuda_note("after text_encoder unload")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    raise SystemExit(main())
