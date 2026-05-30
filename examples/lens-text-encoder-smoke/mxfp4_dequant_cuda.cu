#include <cuda_runtime.h>

#include <cstdint>
#include <cstddef>

static __device__ __forceinline__ uint16_t fp32_to_bf16_rne(float s) {
    union {
        float f;
        uint32_t i;
    } u;
    u.f = s;
    if ((u.i & 0x7fffffffU) > 0x7f800000U) {
        return static_cast<uint16_t>((u.i >> 16) | 64U);
    }
    return static_cast<uint16_t>((u.i + (0x7fffU + ((u.i >> 16) & 1U))) >> 16);
}

static __device__ __forceinline__ float fp4_lut(uint8_t nibble) {
    switch (nibble & 0x0f) {
        case 0x0: return  0.0f;
        case 0x1: return  0.5f;
        case 0x2: return  1.0f;
        case 0x3: return  1.5f;
        case 0x4: return  2.0f;
        case 0x5: return  3.0f;
        case 0x6: return  4.0f;
        case 0x7: return  6.0f;
        case 0x8: return -0.0f;
        case 0x9: return -0.5f;
        case 0xa: return -1.0f;
        case 0xb: return -1.5f;
        case 0xc: return -2.0f;
        case 0xd: return -3.0f;
        case 0xe: return -4.0f;
        default:  return -6.0f;
    }
}

__global__ void lens_mxfp4_dequant_bf16_kernel(const uint8_t* __restrict__ blocks,
                                               const uint8_t* __restrict__ scales,
                                               uint16_t* __restrict__ dst,
                                               int total,
                                               int out_dim,
                                               int groups,
                                               int bytes_per_group,
                                               int in_dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int in = idx % in_dim;
    const int out = (idx / in_dim) % out_dim;
    const int expert = idx / (in_dim * out_dim);
    const int group = in / 32;
    const int byte = (in % 32) / 2;
    const bool high = (in % 2) != 0;
    const int block_idx = ((expert * out_dim + out) * groups + group) * bytes_per_group + byte;
    const int scale_idx = (expert * out_dim + out) * groups + group;
    const uint8_t packed = blocks[block_idx];
    const uint8_t nibble = high ? static_cast<uint8_t>(packed >> 4) : static_cast<uint8_t>(packed & 0x0f);
    const int exp = static_cast<int>(scales[scale_idx]) - 127;
    const float v = ldexpf(fp4_lut(nibble), exp);
    dst[idx] = fp32_to_bf16_rne(v);
}

extern "C" int lens_mxfp4_dequant_bf16_to_device(const uint8_t* blocks_host,
                                                  size_t blocks_bytes,
                                                  const uint8_t* scales_host,
                                                  size_t scales_bytes,
                                                  int experts,
                                                  int out_dim,
                                                  int groups,
                                                  int bytes_per_group,
                                                  int in_dim,
                                                  void* dst_device,
                                                  cudaStream_t stream,
                                                  float* upload_ms,
                                                  float* kernel_ms) {
    uint8_t* d_blocks = nullptr;
    uint8_t* d_scales = nullptr;
    cudaEvent_t upload_start = nullptr;
    cudaEvent_t upload_stop = nullptr;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_stop = nullptr;
    if (upload_ms != nullptr) *upload_ms = 0.0f;
    if (kernel_ms != nullptr) *kernel_ms = 0.0f;

    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_blocks), blocks_bytes);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(reinterpret_cast<void**>(&d_scales), scales_bytes);
    if (err != cudaSuccess) goto cleanup;
    err = cudaEventCreate(&upload_start);
    if (err != cudaSuccess) goto cleanup;
    err = cudaEventCreate(&upload_stop);
    if (err != cudaSuccess) goto cleanup;
    err = cudaEventCreate(&kernel_start);
    if (err != cudaSuccess) goto cleanup;
    err = cudaEventCreate(&kernel_stop);
    if (err != cudaSuccess) goto cleanup;

    err = cudaEventRecord(upload_start, stream);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpyAsync(d_blocks, blocks_host, blocks_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpyAsync(d_scales, scales_host, scales_bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) goto cleanup;
    err = cudaEventRecord(upload_stop, stream);
    if (err != cudaSuccess) goto cleanup;

    err = cudaEventRecord(kernel_start, stream);
    if (err != cudaSuccess) goto cleanup;
    {
        const int total = experts * out_dim * in_dim;
        const int block = 256;
        const int grid = (total + block - 1) / block;
        lens_mxfp4_dequant_bf16_kernel<<<grid, block, 0, stream>>>(d_blocks,
                                                                   d_scales,
                                                                   reinterpret_cast<uint16_t*>(dst_device),
                                                                   total,
                                                                   out_dim,
                                                                   groups,
                                                                   bytes_per_group,
                                                                   in_dim);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) goto cleanup;
    err = cudaEventRecord(kernel_stop, stream);
    if (err != cudaSuccess) goto cleanup;
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) goto cleanup;

    if (upload_ms != nullptr) {
        cudaEventElapsedTime(upload_ms, upload_start, upload_stop);
    }
    if (kernel_ms != nullptr) {
        cudaEventElapsedTime(kernel_ms, kernel_start, kernel_stop);
    }

cleanup:
    if (kernel_stop != nullptr) cudaEventDestroy(kernel_stop);
    if (kernel_start != nullptr) cudaEventDestroy(kernel_start);
    if (upload_stop != nullptr) cudaEventDestroy(upload_stop);
    if (upload_start != nullptr) cudaEventDestroy(upload_start);
    if (d_scales != nullptr) cudaFree(d_scales);
    if (d_blocks != nullptr) cudaFree(d_blocks);
    return static_cast<int>(err);
}
