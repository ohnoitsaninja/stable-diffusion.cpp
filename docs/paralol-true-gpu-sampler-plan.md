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
backend sampler path described below.

Without `SDCPP_EXPERIMENTAL_GPU_SAMPLER_BACKEND=1`, the capability API remains
intentionally honest:

- `supports_sampler_gpu_latent_output=false`
- `supports_sampler_gpu_latent_bridge_output=true`
- `supports_sampler_gpu_init_latent_input=false`
- `supports_sampler_gpu_init_latent_bridge_input=true`

With `SDCPP_EXPERIMENTAL_GPU_SAMPLER_BACKEND=1` on SD1/SDXL CUDA contexts, the
true backend sampler path reports:

- `supports_sampler_gpu_latent_output=true`
- `supports_sampler_gpu_latent_bridge_output=true`
- `supports_sampler_gpu_init_latent_input=true`
- `supports_sampler_gpu_init_latent_bridge_input=true`

Strict mode refuses bridge fallbacks and allows only the true path.

## Implemented status

The fork now has the first backend-resident sampler seam for several SDXL/SD1
k-diffusion samplers, including ancestral and DPM++ SDE variants. It remains
env-gated and narrow, but it is no longer only a proof API.

Two env-gated paths exist:

- `SDCPP_EXPERIMENTAL_TRUE_GPU_SAMPLER=1` enables
  `sd_sample_latent_gpu_true_euler_spike(...)`. This is a strict-mode proof
  path: latent state, UNet input/output, CFG-disabled denoised reconstruction,
  and Euler updates stay as backend tensors, and the returned latent handle has
  no CPU bridge flags. It deliberately uses deterministic device-procedural
  Gaussian noise instead of production Philox, so it proves residency, not
  image parity.
- `SDCPP_EXPERIMENTAL_GPU_SAMPLER_BACKEND=1` changes the SDXL/SD1 GPU sampler
  APIs to use a backend tensor sampler loop for supported k-diffusion methods.
  T2I creates Philox
  Gaussian noise on CUDA, scales it on the backend, and keeps per-step latent
  state and sampler math on the backend. I2I validates the CUDA init latent,
  copies it device-to-device into the sampler backend, creates matching CUDA
  noise, applies strength on the backend, then samples without downloading the
  init latent. The returned handle has no CPU bridge flags on success.

  `SDCPP_EXPERIMENTAL_GPU_SAMPLER_EULER=1` remains accepted as a compatibility
  alias for the original Euler rollout, but new validation should use
  `SDCPP_EXPERIMENTAL_GPU_SAMPLER_BACKEND=1`.

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

The env-gated backend sampler path now covers every public non-flow sampler in
the SDXL/SD1 sampler enum. Bounded SDXL 512, 2-step, strict GPU-resident smokes
with conditioning handles completed through
`sd_sample_latent_gpu_with_conditioning(...)` using:

- `--sampling-method euler`
- `--sampling-method euler_a`
- `--sampling-method heun`
- `--sampling-method dpm2`
- `--sampling-method dpm++2s_a`
- `--sampling-method dpm++2m`
- `--sampling-method dpm++2mv2`
- `--sampling-method ipndm`
- `--sampling-method ipndm_v`
- `--sampling-method lcm`
- `--sampling-method ddim_trailing`
- `--sampling-method tcd`
- `--sampling-method res_multistep`
- `--sampling-method res_2s`
- `--sampling-method er_sde`
- `--sampling-method dpmpp_sde`
- `--sampling-method dpmpp_sde_gpu`
- `--sampling-method dpmpp_2m_sde`
- `--sampling-method dpmpp_2m_sde_gpu`
- `--sampling-method dpmpp_2m_sde_heun`
- `--sampling-method dpmpp_2m_sde_heun_gpu`
- `--sampling-method dpmpp_3m_sde`
- `--sampling-method dpmpp_3m_sde_gpu`

For each smoke:

- sampler math stayed `gpu_backend_tensor`
- output handle was a CUDA latent with `SAMPLER_OUTPUT` and no CPU bridge flags
- CFG used the same batch-2 UNet path as Euler
- stochastic/ancestral paths generated step noise as CUDA Philox tensors
- no VAE decode was required for the smoke

This proves the sampler-loop/backend-tensor refactor is feasible beyond Euler.
It is not a global sampler implementation: the path is env-gated and currently
limited to SDXL/SD1, batch 1, no masks, no ControlNet, no
reference/edit/image-CFG paths, and non-flow denoisers.

## CFG and UNet conv update

The first CFG performance pass found two separate issues:

- batch-2 CFG exposed a real flash-attention shape bug. `ggml_ext_attention_ext`
  now preserves the batch dimension when flash attention returns
  `[d_head, heads * batch, tokens]`, so SDXL batch-2 UNet graphs no longer
  abort during the attention output reshape.
- plain cond/uncond batching is now the default for eligible SDXL/SD1 Euler
  CFG in the backend sampler path because same-noise Comfy parity improves
  measurably. It can still be disabled with
  `SDCPP_DISABLE_GPU_EULER_BATCHED_CFG=1` when comparing speed or isolating
  CFG behavior.

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

- direct implicit-GEMM CFG-separate path: `denoise_ms=2384`
- legacy diffusion conv escape hatch: `denoise_ms=2839`
- default batch CFG plus implicit-GEMM conv: `denoise_ms=2647`

Those numbers vary with cache/interleaving, but the result is clear enough for
the fork contract: implicit-GEMM diffusion conv is the useful default for the
narrow GPU Euler backend, and batch CFG is the parity default. The separate
CFG path remains available through `SDCPP_DISABLE_GPU_EULER_BATCHED_CFG=1`
because the batch-2 graph is currently a little slower on the reference SDXL
1024 run.

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

   The first production target was the SDXL/SD1 public sampler set. Flow-model
   Euler for Flux/Z/Anima is still separate because those denoisers use
   different latent channels, guidance behavior, and flow timestep semantics.

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

Those conditions are now met for the env-gated SDXL/SD1 Euler T2I path, the
env-gated SDXL/SD1 Euler I2I path where the init latent is already an
`SD_GPU_RESOURCE_LATENT`, and first strict T2I backend smokes for the remaining
SDXL/SD1 public sampler methods: Euler A, Heun, DPM2, DPM++ 2S A, DPM++ 2M,
modified DPM++ 2M, iPNDM, iPNDM_v, LCM, DDIM trailing, TCD, Res Multistep,
Res 2S, ER-SDE, and the DPM++ SDE family. Paralol should continue treating
unsupported model families, masks, ControlNet, reference/edit/image-CFG paths,
and requests without
`SDCPP_EXPERIMENTAL_GPU_SAMPLER_BACKEND=1` as bridge paths only. The newly added
samplers should be promoted through the same full T2I/I2I parity matrix as
Euler before Paralol exposes them as equally production-ready.
