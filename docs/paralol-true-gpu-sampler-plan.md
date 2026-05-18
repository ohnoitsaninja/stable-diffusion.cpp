# Paralol True GPU Sampler Plan

This note explains which KSampler GPU APIs are true device-resident paths and
which paths remain compatibility bridges.

## Current blocker

The sampler loop still uses host-side `sd::Tensor<float>` values as its public
and internal currency:

- `src/denoiser.hpp`: `denoise_cb_t` is
  `std::function<sd::Tensor<float>(const sd::Tensor<float>&, float, int)>`.
- `src/denoiser.hpp`: k-diffusion sampler methods update `x`, `denoised`,
  Brownian noise, CFG deltas, and final latents as `sd::Tensor<float>`.
- `src/stable-diffusion.cpp`: `StableDiffusionGGML::sample(...)` returns
  `sd::Tensor<float>` and performs denoise/CFG math on host tensors.
- `src/diffusion_model.hpp`: `DiffusionModel::compute(...)` returns
  `sd::Tensor<float>`.
- `src/ggml_extend.hpp`: `GGMLRunner::compute<T>(...)` materializes graph
  output with `sd::make_sd_tensor_from_ggml<T>(...)`.

Because of those contracts, unsupported samplers and model families still run
the existing sampler, receive a CPU latent, then upload it into an owned CUDA
`SD_GPU_RESOURCE_LATENT`. The supported exception is the env-gated SD1/SDXL
Euler backend path described below.

Without `SDCPP_EXPERIMENTAL_GPU_SAMPLER_EULER=1`, the capability API remains
intentionally honest:

- `supports_sampler_gpu_latent_output=false`
- `supports_sampler_gpu_latent_bridge_output=true`
- `supports_sampler_gpu_init_latent_input=false`
- `supports_sampler_gpu_init_latent_bridge_input=true`

With `SDCPP_EXPERIMENTAL_GPU_SAMPLER_EULER=1` on SD1/SDXL CUDA contexts, the
true Euler path reports:

- `supports_sampler_gpu_latent_output=true`
- `supports_sampler_gpu_latent_bridge_output=true`
- `supports_sampler_gpu_init_latent_input=true`
- `supports_sampler_gpu_init_latent_bridge_input=true`

Strict mode refuses bridge fallbacks and allows only the true path.

## Implemented status

The fork now has the first backend-resident sampler seam for SDXL/SD1 Euler.
It remains env-gated and narrow, but it is no longer only a proof API.

Two env-gated paths exist:

- `SDCPP_EXPERIMENTAL_TRUE_GPU_SAMPLER=1` enables
  `sd_sample_latent_gpu_true_euler_spike(...)`. This is a strict-mode proof
  path: latent state, UNet input/output, CFG-disabled denoised reconstruction,
  and Euler updates stay as backend tensors, and the returned latent handle has
  no CPU bridge flags. It deliberately uses deterministic device-procedural
  Gaussian noise instead of production Philox, so it proves residency, not
  image parity.
- `SDCPP_EXPERIMENTAL_GPU_SAMPLER_EULER=1` changes the SDXL/SD1 Euler GPU
  sampler APIs to use a backend tensor sampler loop. T2I creates Philox
  Gaussian noise on CUDA, scales it on the backend, and keeps per-step latent
  state and sampler math on the backend. I2I validates the CUDA init latent,
  copies it device-to-device into the sampler backend, creates matching CUDA
  noise, applies strength on the backend, then samples without downloading the
  init latent. The returned handle has no CPU bridge flags on success.

The practical backend path has been checked against the existing CPU sampler
with `sd-latent-smoke --compare-gpu-sampler-backend-euler` on SDXL 1024,
Euler, 8 steps, CFG 1.2:

- descriptor: CUDA latent, f32, `SD_LAYOUT_WHCN_GGML`, `1x4x128x128`,
  `262144` bytes
- latent comparison after CUDA Philox: shape/count match, no NaN/Inf, mean abs
  `0.00979606`, p99 abs `0.161348`, max abs `1.86009`
- decoded image comparison against the CPU sampler output: mean pixel abs
  `0.54797/255`, p99 `6/255`, PSNR `41.65 dB`
- strict mode allows the experimental path when the descriptor has flags `4`
  (`SAMPLER_OUTPUT`) and no CPU bridge flags

This proves the sampler-loop/backend-tensor refactor is feasible and usable for
the narrow SDXL/SD1 Euler lane. It is not a global sampler implementation: the
path is env-gated and limited to Euler, batch 1, no masks, no ControlNet, no
reference/edit/image-CFG paths, and non-flow denoisers.

## CFG and UNet conv update

The first CFG performance pass found two separate issues:

- batch-2 CFG exposed a real flash-attention shape bug. `ggml_ext_attention_ext`
  now preserves the batch dimension when flash attention returns
  `[d_head, heads * batch, tokens]`, so SDXL batch-2 UNet graphs no longer
  abort during the attention output reshape.
- plain cond/uncond batching is not a default win on this fork yet. With legacy
  diffusion conv lowering, batch-2 UNet doubles the large IM2COL workspaces. The
  experimental batch mode is therefore gated behind
  `SDCPP_EXPERIMENTAL_GPU_EULER_BATCHED_CFG=1`.

For the env-gated GPU Euler sampler, the default SDXL/SD1 UNet path now enables
diffusion direct conv and routes CUDA `CONV_2D` through the imported
implicit-GEMM kernel. This is scoped to the GPU Euler backend path and can be
disabled with:

```text
SDCPP_DISABLE_GPU_EULER_DIFFUSION_IMPLICIT_GEMM_CONV=1
```

In a bounded SDXL 1024, 8-step, CFG 1.2 real-prompt smoke, this lowered the
per-UNet compute buffer from about `491.99 MB` to `270.86 MB`. The same smoke
reported:

- direct implicit-GEMM CFG-separate path: `denoise_ms=2480`
- legacy diffusion conv escape hatch: `denoise_ms=2839`
- explicit batch CFG plus implicit-GEMM conv: `denoise_ms=2576`

Those numbers vary with cache/interleaving, but the result is clear enough for
the fork contract: implicit-GEMM diffusion conv is the useful default for the
narrow GPU Euler backend; batch CFG remains opt-in until the backend has better
batch-2 UNet graph efficiency.

## Required refactor

1. Add a backend tensor value type.

   Introduce an owned `sd_backend_tensor`/resource wrapper that can represent a
   CUDA tensor with dtype, layout, shape, stream/event readiness, and lifetime.
   This should be able to wrap ggml graph outputs without downloading them and
   should reuse the existing `sd_gpu_handle_t` ownership rules at API boundaries.

2. Add diffusion model GPU compute.

   Add `DiffusionModel::compute_backend(...)` or equivalent for SDXL first. It
   must return a backend-owned tensor for the model prediction instead of
   `sd::Tensor<float>`. `GGMLRunner` needs a mode that keeps graph outputs on
   the selected backend and pins/copies them into owned buffers when needed.

3. Port CFG and denoiser math to backend tensors.

   Move these operations off host `sd::Tensor<float>`:

   - `cond_out`, `uncond_out`, and CFG blend
   - `c_out`, `c_skip`, sigma scaling
   - sampler step updates such as Euler, Euler ancestral, DPM++ SDE, and ER-SDE
   - latent noise add/mul/sub chains

   The first production target should be the smallest sampler set needed by
   Paralol acceptance: Euler for Flux/Z/Anima and Euler/DPM++ SDE for SDXL.

4. Add backend RNG/noise.

   T2I needs device-resident initial noise. I2I needs device-resident noise
   blending and strength application. Either generate noise on CUDA with a
   deterministic seed-compatible path or make any CPU-generated noise upload a
   clearly reported bridge until the device RNG lands.

5. Add true public APIs after SDXL passes.

   Once the SDXL Euler path is device-resident, add or promote APIs so:

   - `sd_sample_latent_gpu(...)` returns a handle without
     `CPU_BRIDGE_UPLOAD`.
   - `sd_sample_latent_gpu_with_init_gpu(...)` consumes the init handle without
     `CPU_BRIDGE_DOWNLOAD`.
   - strict mode allows those true paths.
   - bridge paths remain available under separate capability flags.

6. Expand model families.

   After SDXL, validate Flux.1, Flux2/Klein, Z-Image, and Anima separately. The
   latent channel count, VAE scale, guidance behavior, reference image/edit
   conditioning, and model-specific latent transforms differ enough that success
   for SDXL should not imply success for the other families.

## Acceptance criteria

For SDXL 1024:

- sampled and init latents report `SD_GPU_RESOURCE_LATENT`, CUDA, f32,
  `SD_LAYOUT_WHCN_GGML`, `1x4x128x128`, `262144` bytes.
- true sampler output handle has no `CPU_BRIDGE_UPLOAD` or
  `CPU_BRIDGE_DOWNLOAD` flags.
- true init-latent sampler input does not call `sd_gpu_latent_download(...)`.
- `SDCPP_STRICT_GPU_RESIDENT=1` allows true T2I and I2I sampler paths.
- VAE Decode can consume the returned latent through
  `sd_decode_gpu_latent_normal_gpu(...)`.
- output parity against the current CPU sampler is within the existing image
  thresholds for a fixed seed.
- bridge paths still work and remain visibly marked as bridges.

Those conditions are now met for the env-gated SDXL/SD1 Euler T2I path and the
env-gated SDXL/SD1 Euler I2I path where the init latent is already an
`SD_GPU_RESOURCE_LATENT`. Paralol should continue treating unsupported
samplers, model families, masks, ControlNet, reference/edit/image-CFG paths,
and requests without `SDCPP_EXPERIMENTAL_GPU_SAMPLER_EULER=1` as bridge paths
only.
