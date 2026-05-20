# Paralol SDXL Low-Step Speed Experiments

This branch keeps normal stable-diffusion.cpp behavior unchanged and adds one
opt-in sampler experiment for SDXL/SD1 low-step Turbo/Lightning-style models.

## GPU Euler Denoised Cache

Set:

```powershell
$env:SDCPP_EXPERIMENTAL_GPU_EULER_DENOISED_CACHE = "1"
```

The GPU backend Euler sampler will compute the denoised prediction for the first
step, then reuse or linearly forecast cached denoised tensors on selected later
steps. This skips the UNet on those steps and keeps the update math in ggml
backend tensors.

Tuning knobs:

- `SDCPP_GPU_EULER_DENOISED_CACHE_WARMUP`: computed warmup steps before skips,
  default `1`.
- `SDCPP_GPU_EULER_DENOISED_CACHE_REFRESH_STRIDE`: compute steps divisible by
  this stride and skip other eligible steps, default `2`.
- `SDCPP_GPU_EULER_DENOISED_CACHE_REUSE_ONLY=1`: disable linear forecasting and
  reuse the last denoised tensor.
- `SDCPP_GPU_EULER_DENOISED_CACHE_ALLOW_FINAL=1`: allow skipping the final
  denoise step. This is riskier and is off by default.

The path is intentionally limited to Euler, non-flow SDXL/SD1 GPU backend
sampling, and it disables itself when UNet boundary parity tracing is active.
Logs report skipped UNet calls and forecast count.

## Distilled Model Defaults

`sd_get_model_pipeline_capabilities` now treats SDXL/SD1 checkpoint paths that
contain `turbo`, `lightning`, `hyper-sd`, `hypersd`, `hyper_sd`, `lcm`, or `tcd`
as low-step distilled hints:

- default CFG becomes `1.0`
- default steps become `4`
- `lcm` model names default to the LCM sampler/scheduler

Callers can still override all of these values explicitly.

## First Benchmark Shape

For a 4-step CFG-free SDXL Turbo checkpoint, compare:

1. baseline GPU backend Euler, CFG `1.0`, four steps
2. `SDCPP_EXPERIMENTAL_GPU_EULER_DENOISED_CACHE=1`
3. optional `SDCPP_GPU_EULER_DENOISED_CACHE_ALLOW_FINAL=1`

The third run should be considered quality-risky. The useful signal is whether
skipping one middle UNet call gives a visible wall-clock gain without obvious
image collapse.

## Sampler Parity Notes

The supported SDXL/SD1 GPU backend sampler lane now has two different parity
noise mechanisms:

- initial latent noise can use Comfy/PyTorch CPU noise with
  `SDCPP_NOISE_BACKEND=comfy_cpu_torch` or
  `SDCPP_EXPERIMENTAL_COMFY_CPU_NOISE=1`.
- default per-step ancestral noise uses the CUDA Philox randn-call offset that
  matches Comfy's `default_noise_sampler` behavior. This covers samplers such as
  Euler A and DPM++ 2S ancestral.

SDE samplers are different. Comfy uses `torchsde.BrownianTree` through
`BrownianTreeNoiseSampler`, not independent `randn` calls per step. Exact
same-seed parity for `dpmpp_sde`, `dpmpp_2m_sde`, `dpmpp_3m_sde`, and related
GPU variants needs a BrownianTree-compatible implementation or imported
Brownian increments.

For that diagnostic path the DLL exports:

```c
sd_sample_latent_gpu_with_init_gpu_and_conditioning_and_noise_schedule_gpu(...)
sd_sampler_uses_step_noise(...)
sd_sampler_uses_brownian_step_noise(...)
sd_sampler_step_noise_count(...)
```

`sd_get_gpu_capabilities(...)` also reports:

- `supports_sampler_imported_initial_noise`
- `supports_sampler_imported_step_noise_schedule`
- `supports_sampler_brownian_step_noise_import`
- `supports_sampler_step_noise_count_query`

and `sd-latent-smoke` accepts repeated:

```powershell
--import-step-noise-npy <path>
```

Each imported step-noise file is uploaded as an explicit GPU latent handle and
consumed by index whenever the sampler asks for step noise. The path is
debug/parity-only and fail-closed: if a supplied schedule is missing a requested
noise tensor, has the wrong tensor count, or targets a sampler that does not
consume step noise, sampling fails rather than falling back to generated noise.

Validation used the SDXL 1024 glass-bottle 4-step CFG=1 parity case against
fresh Comfy reference images. Generating torchsde `BrownianTreeNoiseSampler`
increments externally and importing them through the new API materially closes
or narrows the SDE gap:

| sampler | native fork PSNR vs Comfy | imported Brownian PSNR | imported mean abs | imported p99 |
| --- | ---: | ---: | ---: | ---: |
| `dpmpp_sde` | 24.67 dB | 45.34 dB | 0.56 | 4 |
| `dpmpp_2m_sde` | 19.71 dB | 33.80 dB | 1.50 | 19 |
| `dpmpp_3m_sde` | 12.88 dB | 29.69 dB | 4.36 | 32 |
| `dpmpp_sde_gpu` | 21.63 dB | 40.51 dB | 0.70 | 6 |

Artifacts:

- `F:\Paralol\build\diagnostics\sdcpp-sampler-brownian-20260520\metrics_imported_brownian_all.json`
- `F:\Paralol\build\diagnostics\sdcpp-sampler-brownian-20260520\contact_dpmpp_sde_imported_brownian.png`
- `F:\Paralol\build\diagnostics\sdcpp-sampler-brownian-20260520\contact_dpmpp_sde_gpu_imported_brownian.png`
- `F:\Paralol\build\diagnostics\sdcpp-sampler-brownian-family-20260520\sde_family_brownian_metrics.json`
- `F:\Paralol\build\diagnostics\sdcpp-sampler-brownian-family-20260520\sde_family_brownian_overview.png`

The `dpmpp_sde` result proves that sampler's remaining same-seed drift is
BrownianTree noise generation, not conditioning, VAE, sigma scheduling, or the
main update equations. The multistep SDE samplers also improve sharply when fed
Comfy Brownian increments, but they do not collapse to the same near-exact level.
That leaves a second, sampler-specific drift boundary in the DPM++ 2M/3M SDE
multistep coefficient/update path. The production choices are therefore:

1. implement a native BrownianTree-compatible backend for exact same-seed SDE
   noise generation;
2. keep imported Brownian schedules for explicit parity/debug runs;
3. separately audit `dpmpp_2m_sde` and `dpmpp_3m_sde` step-update math against
   Comfy if exact multistep SDE image parity is required.
