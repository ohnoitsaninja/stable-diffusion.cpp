# Paralol GPU Latent Flow

This note covers the current stable-diffusion.cpp latent path used by Paralol.

## Current sampler path

`sd_sample_latent` is still a host-side sampler API:

- `StableDiffusionGGML::sample(...)` returns `sd::Tensor<float>`.
- `denoise_cb_t` is `std::function<sd::Tensor<float>(const sd::Tensor<float>&, float, int)>`.
- The k-diffusion sampler implementations in `denoiser.hpp` update `x`, `denoised`, and noise as `sd::Tensor<float>` values.
- Each diffusion model callback returns `sd::Tensor<float>` from `DiffusionModel::compute(...)`.
- `GGMLRunner::compute<T>(...)` materializes the final ggml graph tensor with `sd::make_sd_tensor_from_ggml<T>(...)`.

That means the current sampler does not have a final device tensor that can be wrapped directly as a true GPU-resident KSampler output. A narrow `sd_sample_latent_gpu` wrapper can upload the final CPU latent into an `sd_gpu_handle_t`, but strict GPU-resident mode rejects that bridge because it is not an all-device sampler path.

## Current materialization points

Sampler output materializes here:

- `src/stable-diffusion.cpp`: `sd_sample_latent(...)`
- `src/stable-diffusion.cpp`: `sd::Tensor<float> final_latent = sd_ctx->sd->sample(...)`
- `src/stable-diffusion.cpp`: `return make_sd_latent(std::move(final_latent))`

The pre-existing CPU VAE decode path uploaded this latent by converting the `sd_latent_t` back into `sd::Tensor<float>` and passing it into a VAE graph input.

## GPU latent handle support

The fork now supports `SD_GPU_RESOURCE_LATENT` handles and explicit latent download/upload helpers:

- `sd_sample_latent_gpu(...)`
- `sd_gpu_latent_download(...)`
- `sd_cpu_latent_upload(...)`
- `sd_decode_gpu_latent_normal_gpu(...)`

`sd_decode_gpu_latent_normal_gpu(...)` consumes a CUDA latent handle directly. For SDXL/SD1-style scalar latent transforms it keeps the diffusion-latent to VAE-latent conversion on the selected backend, then runs COMFY_NORMAL VAE decode and returns a GPU image handle.

`sd_sample_latent_gpu(...)` is intentionally marked as a compatibility bridge today. It calls the existing host-side sampler and uploads the final latent to CUDA. It sets `SD_GPU_RESOURCE_FLAG_SAMPLER_OUTPUT | SD_GPU_RESOURCE_FLAG_CPU_BRIDGE_UPLOAD` on the returned handle. With `SDCPP_STRICT_GPU_RESIDENT=1`, it fails instead of hiding the CPU materialization.

## SDXL 1024 latent descriptor

For a 1024x1024 SDXL image:

- Shape: `N=1, C=4, H=128, W=128`
- GGML descriptor order: `w=128, h=128, c=4, n=1`
- Layout: `SD_LAYOUT_WHCN_GGML`
- Dtype: `SD_DTYPE_F32`
- Byte size: `128 * 128 * 4 * 1 * sizeof(float) = 262144`
- Backend for GPU handles: `SD_BACKEND_CUDA`

## Paralol handoff

For the next Paralol worker integration:

- Treat latent values as `CPU latent OR GPU latent handle`.
- Treat image values as `CPU image OR GPU image handle`.
- If KSampler calls `sd_sample_latent_gpu`, inspect handle flags. `CPU_BRIDGE_UPLOAD` means the sampler itself still materialized on CPU.
- Latent Decode should prefer `sd_decode_gpu_latent_normal_gpu` when it receives a CUDA latent handle.
- Save/export/debug consumers should explicitly call `sd_gpu_latent_download` or `sd_gpu_image_download`.

The real all-GPU KSampler project is a separate sampler backend refactor: the denoiser callback and sampler math need backend tensor/resource variants instead of `sd::Tensor<float>` host values.
