# CUDA Tile C++ Smoke

Standalone CUDA 13.3 diagnostic for checking whether CUDA Tile C++ can compile a tiny half GEMM and emit Tensor Core instructions on this machine.

This is not integrated with the Bonsai runtime.

Build from the fork root with a Visual Studio developer shell active:

```powershell
& 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe' -std=c++20 -arch=sm_89 -enable-tile -c tools\diagnostics\cuda_tile_cpp_smoke\tile_gemm_smoke.cu -o build-bonsai-int1\bonsai-cuda133-eval\tile_cpp_smoke.obj
```
