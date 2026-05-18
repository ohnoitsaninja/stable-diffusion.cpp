# Paralol Conditioning Handles

This fork exposes a first-pass conditioning handle API so Paralol can make
`sd.native.clip_text_encode` a real worker-side encode step instead of a prompt
descriptor that KSampler re-encodes every run.

## API Surface

- `sd_get_conditioning_capabilities`
- `sd_conditioning_encode_options_init`
- `sd_conditioning_encode_text`
- `sd_conditioning_retain`
- `sd_conditioning_release`
- `sd_conditioning_get_desc`
- `sd_conditioning_debug_name`
- `sd_sample_latent_gpu_with_conditioning`
- `sd_sample_latent_gpu_with_init_gpu_and_conditioning`

The handle type is `sd_conditioning_handle_t`. Handles are scoped to the
`sd_ctx_t` that created them and must not be used with another context.

## Supported Scope

The sampler handle path is intentionally narrow:

- SD1.x and SDXL text-to-image.
- SD1.x and SDXL image-to-image when the init latent is already encoded and
  passed as an `SD_GPU_RESOURCE_LATENT` handle.
- Batch size 1.
- No init image, masks, ControlNet, reference image, edit mode, LoRA list, or
  image CFG.
- KSampler output is still the existing sampled-latent bridge upload unless the
  separate true GPU sampler path is used.

Unsupported requests fail clearly instead of falling back to prompt strings.

## Residency Status

Conditioning handles are resident in the `sd_ctx_t`, but the current
`SDCondition` implementation is host-backed `sd::Tensor<float>` storage.
Capabilities therefore report:

- `supports_conditioning_handles=true`
- `supports_sampler_conditioning_handle_input=true` for SD1.x/SDXL contexts
- `supports_conditioning_cpu_resident=true`
- `supports_conditioning_gpu_resident=false`

The handle-based sampler entrypoint bypasses CLIP prompt encoding and reports
`prompt_encode_ms=0`, but it does not yet make conditioning tensors true CUDA
resources.

For I2I, `sd_sample_latent_gpu_with_init_gpu_and_conditioning` combines the
resident conditioning handles with the current non-strict GPU init-latent
bridge. The function validates the GPU latent descriptor, downloads the init
latent to the existing CPU sampler path, samples with cached conditioning, then
uploads the sampled latent back to a GPU handle. It reports
`init_bridge_download=true`, `output_bridge_upload=true`, and
`prompt_encode_ms=0`.

`SDCPP_STRICT_GPU_RESIDENT=1` still refuses this I2I path because sampler math
is not yet fully backend-resident. This is expected and keeps the capability
contract honest.

## Lifetime

`sd_conditioning_encode_text` returns a handle with refcount 1. Paralol should
retain a handle while a graph value or cache entry owns it and release it when
that value is evicted.

Handle reuse is only advertised when `free_params_immediately=false`. If a
context frees model params after a sample, the conditioning handles still exist
but the same context is not suitable for repeated KSampler calls without
reloading params.

The standalone encode API deliberately does not free CLIP params after each
handle. Callers that want to unload CLIP after creating all needed handles should
use the existing explicit release API.

## SDXL Negative Prompt

For SDXL empty negative prompts, encode the negative conditioning with
`force_zero_uncond=true`. This matches the existing folded KSampler path.

## Paralol Dataflow

Recommended T2I integration:

1. `sd.native.clip_text_encode` calls `sd_conditioning_encode_text`.
2. The node outputs an opaque conditioning handle value plus descriptor.
3. `sd.native.ksampler` receives positive and negative handles.
4. KSampler calls `sd_sample_latent_gpu_with_conditioning`.
5. Latent decode continues through `sd_decode_gpu_latent_normal_gpu`.

This removes prompt encoding from KSampler timing for supported SD1.x/SDXL T2I
workflows while keeping capability flags honest about the remaining CPU-backed
conditioning storage.

Recommended I2I integration:

1. `sd.native.clip_text_encode` calls `sd_conditioning_encode_text`.
2. VAE Encode produces a GPU latent handle.
3. `sd.native.ksampler` receives positive/negative handles plus the init latent
   handle.
4. KSampler calls `sd_sample_latent_gpu_with_init_gpu_and_conditioning`.
5. Latent Decode consumes the sampled GPU latent through
   `sd_decode_gpu_latent_normal_gpu`.

This gives I2I the same CLIP-cache behavior as T2I while retaining the existing
bridge labels for sampler init-latent and sampled-latent movement.

## Smoke

The fork smoke target supports:

```powershell
build\codex\bin\sd-latent-smoke.exe `
  --model <sdxl-checkpoint> `
  --image <any-rgba-image> `
  --prompt "a clean studio product shot of a blue glass bottle on a white background" `
  --negative-prompt "blurry, low quality, noisy" `
  --width 1024 --height 1024 `
  --steps 8 --cfg-scale 1.2 `
  --sampling-method euler `
  --type-f16 `
  --condition-handles-reuse `
  --gpu-latent-decode-input `
  --gpu-decode-output `
  --download-gpu-output-buffer `
  --dump-gpu-handle-desc
```

Expected markers:

- `conditioning_capabilities ... sampler_input=true reuse=true`
- `sd_conditioning_encode_text ... storage=host_tensor device_resident=false`
- `sd_sample_latent_with_conditioning ... prompt_encode_ms=0`
- `conditioning_handle_reuse=true`
- GPU latent descriptor `1x4x128x128`
- GPU image descriptor `1x3x1024x1024`

For the I2I combined path, add `--gpu-encode-output --gpu-init-sample-input`
and use `--condition-handles` instead of `--condition-handles-reuse`:

```powershell
build\codex\bin\sd-latent-smoke.exe `
  --model <sdxl-checkpoint> `
  --image <rgba-image> `
  --prompt "a clean studio product shot of a blue glass bottle on a white background" `
  --negative-prompt "blurry, low quality, noisy" `
  --width 1024 --height 1024 `
  --steps 8 --cfg-scale 1.2 `
  --sampling-method euler `
  --type-f16 `
  --gpu-encode-output `
  --gpu-init-sample-input `
  --condition-handles `
  --gpu-latent-decode-input `
  --gpu-decode-output `
  --download-gpu-output-buffer `
  --dump-gpu-handle-desc
```

Expected I2I markers:

- `gpu_encoded_latent_desc ... kind=3 ... shape_nchw=1x4x128x128`
- `calling sd_sample_latent_gpu_with_init_gpu_and_conditioning`
- `sd_sample_latent_with_conditioning ... prompt_encode_ms=0 ... init_latent=true`
- `sd_sample_latent_gpu_with_init_gpu_and_conditioning ... init_bridge_download=true output_bridge_upload=true`
- GPU sampled latent descriptor `1x4x128x128`
- GPU image descriptor `1x3x1024x1024`
