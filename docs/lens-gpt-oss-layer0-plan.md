# Lens GPT-OSS Layer-0 Parity Plan

This note is the next checkpoint after the B1 text-encoder bootstrap. It is
intentionally limited to layer 0 and does not start the full 24-layer forward.

## Confirmed Inputs

- Source weights: `F:\Paralol\local\models\microsoft\Lens-Turbo\text_encoder`
- Tokenizer source: `F:\Paralol\local\models\microsoft\Lens-Turbo\tokenizer`
- Oracle bundle: `build\diagnostics\lens_text_encoder_oracle_robot_128`
- Prompt sequence: `[1,128]`
- Lens trim offset: `97`
- Captured layer taps: `5, 11, 17, 23`
- Hidden size: `2880`
- Attention heads: `64`
- KV heads: `8`
- Head dim: `64`
- Experts: `32`
- Experts per token: `4`
- Layer 0 attention type: `sliding_attention`

## Layer-0 Operation Order

Match `LensGptOssEncoder.forward(...)` and `GptOssDecoderLayer.forward(...)`:

1. Token embedding lookup: `model.embed_tokens.weight[input_ids]`.
2. Position IDs: `arange(seq_len)` expanded to `[1, seq_len]`.
3. Build both causal masks once:
   - `create_causal_mask(...)`
   - `create_sliding_window_causal_mask(...)`
4. Build YaRN RoPE cos/sin from `model.rotary_emb(...)`.
5. Layer 0 input RMSNorm:
   - tensor: `model.layers.0.input_layernorm.weight`
   - eps: `1e-5`
   - compute variance in f32, then cast to the working dtype boundary.
6. Self-attention:
   - q: `model.layers.0.self_attn.q_proj.{weight,bias}`
   - k: `model.layers.0.self_attn.k_proj.{weight,bias}`
   - v: `model.layers.0.self_attn.v_proj.{weight,bias}`
   - apply GPT-OSS RoPE to q/k.
   - repeat KV heads by `num_attention_heads / num_key_value_heads = 8`.
   - use the layer-0 sliding attention mask.
   - append attention sinks from `model.layers.0.self_attn.sinks`.
   - subtract max before softmax, matching the eager implementation.
   - drop the sink probability column before multiplying values.
   - project with `model.layers.0.self_attn.o_proj.{weight,bias}`.
7. First residual add.
8. Post-attention RMSNorm:
   - tensor: `model.layers.0.post_attention_layernorm.weight`
9. Router:
   - logits: `linear(hidden, model.layers.0.mlp.router.weight, bias)`
   - select top 4 experts per token.
   - softmax only over selected top-k logits.
10. MoE experts:
    - dequantize only selected expert rows as needed from MXFP4 blocks/scales.
    - `gate_up_proj` layout after decode: `[expert, hidden, 2 * intermediate]`.
    - split even/odd columns into gate/up.
    - clamp gate to max `7.0`; clamp up to `[-7.0, 7.0]`.
    - `glu = gate * sigmoid(gate * 1.702)`.
    - `gated_output = (up + 1) * glu`.
    - `down_proj` layout after decode: `[expert, intermediate, hidden]`.
    - add F32 expert biases.
    - multiply each expert output by its top-k routing weight.
    - accumulate by token index.
11. Second residual add.

## Required Native Checkpoints

The current oracle exporter already writes the first useful checkpoints:

- `token_embedding_f32.npy`
- `layer0_input_f32.npy`
- `layer0_input_norm_output_f32.npy`
- `layer0_attention_output_f32.npy`
- `layer0_post_attention_norm_output_f32.npy`
- `layer0_router_logits_f32.npy`
- `layer0_router_scores_f32.npy`
- `layer0_router_indices_i64.npy`
- `layer0_mlp_output_f32.npy`
- `layer0_final_hidden_f32.npy`

The first layer-0 implementation should compare in this order:

1. RMSNorm output.
2. q/k/v projection outputs.
3. RoPE-applied q/k.
4. attention logits before sink append.
5. softmax probabilities after sink drop.
6. attention output before o-proj.
7. attention output after o-proj.
8. router logits / top-k indices / top-k scores.
9. selected expert gate/up/dequant slices.
10. MLP output.
11. final layer-0 hidden.

## Memory/Staging Policy

Layer-0 parity may stream CPU-side expert MXFP4 blocks and dequantize only
selected experts. That is acceptable for the GPT-OSS text-encoder stage because
the encoder is sparse MoE and must be fully released before loading the dense
Lens transformer. Do not apply this policy to the dense Lens DiT.
