# VAE BF16 Feasibility Baseline

Generated from bounded trace runs under `C:\tmp\stable-diffusion.cpp-paralol\build\vae-bf16-feasibility`.

The previous stalled wrapper did not produce usable allocation data. It left empty stdout/stderr files and an `encode_nvml.csv` that sampled idle VRAM for about 899 seconds, so NVML peak is not used.

Useful trace files:

- `encode_trace.out.log`
- `decode_trace.out.log`
- `encode_trace.err.log` was empty
- `decode_trace.err.log` was empty

Baseline:

- encode planned workspace: 1536 MB
- decode planned workspace: 2816 MB
- encode/decode stage count: 6
- device-resident stage boundaries: yes
- stage boundary host copies: 0
- IM2COL present: false
- direct conv present: true
- resolved storage dtype: f32

The detailed stage/op tables are in `docs/vae-bf16-feasibility.md`.

