# Paralol TAESD Sampling Previews

This fork exposes TAESD/TAE previews during sampling so Paralol can publish
intermediate images while KSampler is still running.

The verified targets are:

- SDXL with TAESDXL.
- Flux.1 with TAEF1.
- Flux.2 / Flux2 Klein with TAEF2.

The context must be created with `taesd_path` pointing at a compatible tiny
autoencoder checkpoint and `tae_preview_only=true` if the tiny autoencoder
should be used for previews without replacing the final model VAE.

## API

Legacy callers can continue using:

```c
sd_set_preview_callback(cb, PREVIEW_TAE, 4, true, false, data);
```

That now means "emit denoised TAESD previews every 4 denoise steps, plus the
final step by default."

New callers should prefer:

```c
sd_preview_options_t options;
sd_preview_options_init(&options);
options.mode = PREVIEW_TAE;
options.schedule_mode = SD_PREVIEW_SCHEDULE_PERCENT_INTERVAL;
options.percent_interval = 0.25f;
options.denoised = true;
options.noisy = false;

sd_set_preview_callback_v2(cb, &options, data);
```

Supported schedule modes:

- `SD_PREVIEW_SCHEDULE_EVERY_N_STEPS`: emit at `step_interval`, e.g. 4, 8, 12.
- `SD_PREVIEW_SCHEDULE_PERCENT_INTERVAL`: emit at ratio increments, e.g. 0.25
  gives 25%, 50%, 75%, and 100%.
- `SD_PREVIEW_SCHEDULE_EXPLICIT_PERCENTS`: emit at the configured
  `percent_points[]`, useful for exact 0.25/0.50/1.0 style schedules.

`include_first_step` and `include_final_step` can force step 1 or the final
step. The default includes the final step.

The callback receives CPU `sd_image_t` preview frames and must copy any pixels it
wants to keep before returning. The frame memory remains DLL-owned and valid only
for the callback duration, matching the upstream preview callback contract.

## Paralol Notes

For KSampler:

- Configure `taesd_path` from the workflow/local settings.
- Set `tae_preview_only=true` so final Latent Decode still uses the model VAE.
- Use `sd_set_preview_callback_v2` immediately around a single sample run and
  clear it after the run.
- Publish preview images as transient progress events, not final graph values.

The preview callback is process-global in stable-diffusion.cpp. The current
Paralol SD native worker runs one SD sample at a time inside the worker process,
so this is acceptable. If the worker later allows concurrent in-process samples,
the callback must become per-context or protected by a run token.

For Flux.2 / Flux2 Klein, use:

- `taesd_path = %models%/VAE/taef2.safetensors`
- the normal Flux2 VAE path for final decode, e.g. `flux2-vae.safetensors`
- the Flux2/Klein LLM text encoder already required by the model

The fork's tiny autoencoder implementation automatically switches to the TAEF2
shape when the loaded model version is Flux.2:

- latent channels: 32 inside TAEF2
- diffusion latent shape at 1024: `1x128x64x64`
- patch/unpatch scale: 2

Keep TAeF2 as preview-only until final image quality is explicitly accepted for
the workflow.

For Flux.1, use:

- `taesd_path = %models%/VAE/taef1.safetensors`
- the normal Flux.1 AE path for final decode, e.g. `ae.safetensors`
- the CLIP-L and T5XXL text encoders already required by Flux.1

The fork's tiny autoencoder implementation uses the standard Flux/DiT TAE
shape for Flux.1:

- latent channels: 16
- diffusion latent shape at 512: `1x16x64x64`
- no Flux.2 32-channel patch/unpatch path

Keep TAEF1 as preview-only until final image quality is explicitly accepted for
the workflow.

## Verification

Local SDXL smoke:

```powershell
.\build\codex\bin\sd-latent-smoke.exe `
  --model "F:\automatic1111\Stability\Models\StableDiffusion\creapromptLightning_creapromtHypersdxlV1.2.safetensors" `
  --taesd "F:\Paralol\build\runtime\models\taesd\taesdxl-diffusion_pytorch_model.safetensors" `
  --image "F:\Paralol\examples\orc.png" `
  --image-channels 3 `
  --prompt "cat" --negative-prompt "bad" `
  --steps 8 --cfg-scale 1.2 --width 1024 --height 1024 `
  --type-f16 --sample-without-init --gpu-sample-output `
  --preview-tae --preview-every 4 `
  --preview-prefix "C:\tmp\stable-diffusion.cpp-paralol\build\preview-smoke\every4" `
  --skip-estimate --no-decode
```

Expected previews:

- `every4_step4_denoised_frame0.png`
- `every4_step8_denoised_frame0.png`

Percentage schedule smoke:

```powershell
.\build\codex\bin\sd-latent-smoke.exe ... `
  --steps 4 `
  --preview-tae --preview-percent-interval 0.5 `
  --preview-prefix "C:\tmp\stable-diffusion.cpp-paralol\build\preview-smoke\pct50"
```

Expected previews:

- `pct50_step2_denoised_frame0.png`
- `pct50_step4_denoised_frame0.png`

Flux2 Klein + TAEF2 smoke:

```powershell
.\build\codex\bin\sd-latent-smoke.exe `
  --diffusion-model "F:\automatic1111\Stability\Models\DiffusionModels\flux-2-klein-4b-fp8.safetensors" `
  --vae "F:\automatic1111\Stability\Models\VAE\flux2-vae.safetensors" `
  --llm "F:\automatic1111\Stability\Models\TextEncoders\Qwen3-4B-Q5_K_M.gguf" `
  --taesd "F:\automatic1111\Stability\Models\VAE\taef2.safetensors" `
  --image "F:\Paralol\examples\orc.png" `
  --image-channels 3 `
  --prompt "a lovely cat" `
  --negative-prompt "" `
  --steps 4 --cfg-scale 1.0 --width 1024 --height 1024 `
  --sample-without-init --gpu-sample-output `
  --preview-tae --preview-every 2 `
  --preview-prefix "C:\tmp\stable-diffusion.cpp-paralol\build\taef2-smoke\flux2_1024" `
  --skip-estimate --no-decode
```

Expected previews:

- `flux2_1024_step2_denoised_frame0.png`
- `flux2_1024_step4_denoised_frame0.png`

Flux.1 + TAEF1 smoke:

```powershell
.\build\codex\bin\sd-latent-smoke.exe `
  --diffusion-model "F:\automatic1111\Stability\Models\DiffusionModels\flux1-kontext-dev-Q5_K_M.gguf" `
  --vae "F:\automatic1111\Stability\Models\VAE\ae.safetensors" `
  --clip-l "F:\automatic1111\Stability\Models\TextEncoders\clip_l.safetensors" `
  --t5xxl "F:\automatic1111\Stability\Models\TextEncoders\t5-v1_1-xxl-encoder-Q3_K_L.gguf" `
  --taesd "F:\automatic1111\Stability\Models\VAE\taef1.safetensors" `
  --image "F:\Paralol\examples\orc.png" `
  --image-channels 3 `
  --prompt "a lovely cat, detailed, sharp, high quality" `
  --negative-prompt "" `
  --steps 4 --cfg-scale 1.0 --width 512 --height 512 `
  --sample-without-init --gpu-sample-output `
  --preview-tae --preview-every 4 `
  --preview-prefix "C:\tmp\stable-diffusion.cpp-paralol\build\taef1-smoke\flux1_4step" `
  --skip-estimate --no-decode
```

Expected preview:

- `flux1_4step_step4_denoised_frame0.png`

This is a preview decoder, not the final Flux AE. The verified 4-step preview is
coherent but has visible tiny-autoencoder artifacts; final image quality should
still be judged through the full Flux AE decode.
