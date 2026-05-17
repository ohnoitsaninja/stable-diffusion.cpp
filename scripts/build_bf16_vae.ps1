param(
    [string]$BuildDir = "build\codex",
    [string]$Model,
    [string]$Image,
    [string]$ComfyRoot,
    [switch]$SkipSmoke,
    [switch]$SkipComfyParity
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BuildPath = Join-Path $RepoRoot $BuildDir

& (Join-Path $PSScriptRoot "apply_ggml_bf16_vae_patches.ps1") -RepoRoot $RepoRoot -Force

if (-not (Test-Path (Join-Path $BuildPath "CMakeCache.txt"))) {
    Write-Host "Configuring Release CUDA build in $BuildPath"
    cmake -S $RepoRoot -B $BuildPath -DSD_CUDA=ON -DSD_BUILD_SHARED_LIBS=ON
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configure failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Building stable-diffusion, sd-latent-smoke, and sd-vae-op-bench"
cmake --build $BuildPath --config Release --target stable-diffusion sd-latent-smoke sd-vae-op-bench
if ($LASTEXITCODE -ne 0) {
    throw "CMake build failed with exit code $LASTEXITCODE"
}

if ($SkipSmoke) {
    return
}

$SmokeExe = Join-Path $BuildPath "bin\sd-latent-smoke.exe"
if (-not (Test-Path $SmokeExe)) {
    throw "Smoke executable not found: $SmokeExe"
}

if ([string]::IsNullOrWhiteSpace($Model) -or [string]::IsNullOrWhiteSpace($Image)) {
    Write-Host "Build complete. Pass -Model and -Image to run BF16 VAE smoke."
    return
}

$env:SDCPP_EXPERIMENTAL_VAE_BF16 = "1"
$env:SDCPP_VAE_STRICT_COMFY_NORMAL = "1"

Write-Host "Running BF16 VAE smoke"
$SmokeOutDir = Join-Path $BuildPath "bf16-vae-smoke"
New-Item -ItemType Directory -Path $SmokeOutDir -Force | Out-Null
& $SmokeExe --model $Model --image $Image --image-channels 3 --type-f16 --split-decode-context --output (Join-Path $SmokeOutDir "bf16-roundtrip.png")
if ($LASTEXITCODE -ne 0) {
    throw "BF16 VAE smoke failed with exit code $LASTEXITCODE"
}

if ($SkipComfyParity) {
    return
}

if (-not [string]::IsNullOrWhiteSpace($ComfyRoot)) {
    $ParityScript = Join-Path $RepoRoot "scripts\vae_comfy_parity.py"
    $PythonExe = "python"
    $ComfyPython = Join-Path $ComfyRoot "venv\Scripts\python.exe"
    if (Test-Path $ComfyPython) {
        $PythonExe = $ComfyPython
    }
    $DllPath = Join-Path $BuildPath "bin\stable-diffusion.dll"
    Write-Host "Running Comfy parity harness"
    & $PythonExe $ParityScript --comfy-root $ComfyRoot --sdcpp-smoke $SmokeExe --sdcpp-dll $DllPath --model $Model --image $Image --out-dir (Join-Path $BuildPath "bf16-vae-parity") --mode all
    if ($LASTEXITCODE -ne 0) {
        throw "Comfy parity failed with exit code $LASTEXITCODE"
    }
}
