# Paralol True GPU Sampler Plan

This note explains why the current KSampler GPU APIs are compatibility bridges
and what must change before `SDCPP_STRICT_GPU_RESIDENT=1` can allow sampler
input/output paths.

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

Because of those contracts, `sd_sample_latent_gpu(...)` currently runs the
existing sampler, receives a CPU latent, then uploads it into an owned CUDA
`SD_GPU_RESOURCE_LATENT`. `sd_sample_latent_gpu_with_init_gpu(...)` validates
its CUDA init latent, downloads it into the CPU sampler, then returns the same
bridge-uploaded output.

The capability API is intentionally honest:

- `supports_sampler_gpu_latent_output=false`
- `supports_sampler_gpu_latent_bridge_output=true`
- `supports_sampler_gpu_init_latent_input=false`
- `supports_sampler_gpu_init_latent_bridge_input=true`

Strict mode must keep refusing both bridge APIs until the contracts above are
replaced.

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

Until those conditions are met, Paralol should treat sampler GPU output and
sampler GPU init-latent input as bridge paths only.
