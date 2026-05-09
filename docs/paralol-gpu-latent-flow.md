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

CPU sampled latents can also be uploaded explicitly with `sd_cpu_latent_upload(...)`.
That path is intended as a compatibility bridge for existing CPU latent values.
The uploaded handle is an owned CUDA buffer with `SD_GPU_RESOURCE_FLAG_CPU_BRIDGE_UPLOAD`;
if the source latent came from `sd_sample_latent(...)`, it also carries
`SD_GPU_RESOURCE_FLAG_SAMPLER_OUTPUT`.

## VAE-encoded latent limitation

VAE Encode GPU latent handoff is capability-gated off in this build:

- `supports_vae_encode_gpu_latent_output=false`
- `supports_vae_encode_gpu_latent_bridge_output=false`

The disabled path is deliberate. During the GPU-handoff smoke work, an
encoded-image latent followed by GPU VAE Decode tripped a CUDA illegal memory
access inside the implicit-GEMM decode convolution. The T2I sampled-latent path
and explicit CPU sampled-latent upload path both decode correctly, but the
encoded-latent reconstruction path is not safe enough to expose as a Paralol
contract yet.

For safety:

- `sd_encode_image_normal_gpu(...)` returns a clear refusal instead of returning
  a half-working GPU latent handle.
- CPU latents produced by `sd_encode_image(...)` / `sd_encode_image_normal(...)`
  are tagged internally as `vae_encode`.
- `sd_decode_latent_normal_gpu(...)` refuses internally tagged VAE-encoded CPU
  latents before uploading them.
- `sd_cpu_latent_upload(...)` refuses internally tagged VAE-encoded CPU latents.
- `sd_decode_gpu_latent_normal_gpu(...)` refuses any handle carrying
  `SD_GPU_RESOURCE_FLAG_VAE_ENCODE_OUTPUT`.

This keeps I2I encoded latent handoff honest: Paralol should treat VAE Encode
GPU output as unsupported until this specific path has a separate fix and smoke
coverage.

## SDXL 1024 latent descriptor

For a 1024x1024 SDXL image:

- Shape: `N=1, C=4, H=128, W=128`
- GGML descriptor order: `w=128, h=128, c=4, n=1`
- Layout: `SD_LAYOUT_WHCN_GGML`
- Dtype: `SD_DTYPE_F32`
- Byte size: `128 * 128 * 4 * 1 * sizeof(float) = 262144`
- Backend for GPU handles: `SD_BACKEND_CUDA`

## Ownership and lifetime

Exported GPU handles are opaque ids owned by the `sd_ctx_t` that created them.
Consumers must not persist borrowed CUDA pointers after releasing the handle.

Current exported latent/image handles use owned backend buffers, not transient
ggml graph tensors:

- Producer returns a handle with `refcount=1`.
- `sd_gpu_handle_retain(...)` increments ownership for stored graph values.
- `sd_gpu_handle_release(...)` releases ownership and destroys the resource at
  the final release.
- `sd_gpu_handle_get_desc(...)` reports kind, backend, dtype, layout, shape,
  strides, byte size, flags, and refcount.
- `sd_gpu_handle_borrow_cuda_ptr(...)` is same-process only and does not
  transfer ownership.

Do not pass raw CUDA pointers across process boundaries. Future cross-process
or graphics interop should use a separate IPC/external-memory API.

## Paralol handoff

For the next Paralol worker integration:

- Treat latent values as `CPU latent OR GPU latent handle`.
- Treat image values as `CPU image OR GPU image handle`.
- If KSampler calls `sd_sample_latent_gpu`, inspect handle flags. `CPU_BRIDGE_UPLOAD` means the sampler itself still materialized on CPU.
- Latent Decode should prefer `sd_decode_gpu_latent_normal_gpu` when it receives a CUDA latent handle.
- Save/export/debug consumers should explicitly call `sd_gpu_latent_download` or `sd_gpu_image_download`.
- VAE Encode should stay on the CPU latent path for now. Do not request GPU
  encoded-latent handles until `supports_vae_encode_gpu_latent_output=true`.

## Supported handoff paths

Supported now:

- Non-strict T2I bridge:
  `sd_sample_latent_gpu -> sd_decode_gpu_latent_normal_gpu -> sd_gpu_image_download`
- CPU sampled latent bridge:
  `sd_sample_latent -> sd_cpu_latent_upload -> sd_decode_gpu_latent_normal_gpu`
- GPU image output:
  `sd_decode_gpu_latent_normal_gpu -> SD_GPU_RESOURCE_IMAGE -> sd_gpu_image_download`

Refused now:

- Strict sampler GPU output:
  `SDCPP_STRICT_GPU_RESIDENT=1 sd_sample_latent_gpu`
- VAE Encode GPU latent output:
  `sd_encode_image_normal_gpu`
- VAE-encoded latent upload/decode handoff:
  `sd_cpu_latent_upload` or `sd_decode_latent_normal_gpu` with an internally
  tagged VAE-encoded latent

The real all-GPU KSampler project is a separate sampler backend refactor: the denoiser callback and sampler math need backend tensor/resource variants instead of `sd::Tensor<float>` host values.
