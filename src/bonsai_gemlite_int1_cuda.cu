#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "bonsai_gemlite_int1.hpp"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace {

extern "C" cudaStream_t ggml_cuda_get_current_custom_op_stream(void);

namespace wmma = nvcuda::wmma;

__device__ __forceinline__ unsigned bonsai_smem_addr(const void* ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void bonsai_cp_async_cg_16(void* dst_smem, const void* src_gmem) {
#if __CUDA_ARCH__ >= 800
    const unsigned dst = bonsai_smem_addr(dst_smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src_gmem));
#else
    *reinterpret_cast<int4*>(dst_smem) = *reinterpret_cast<const int4*>(src_gmem);
#endif
}

__device__ __forceinline__ void bonsai_cp_async_commit_group() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void bonsai_cp_async_wait_group_2() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 2;\n" ::);
#endif
}

__device__ __forceinline__ void bonsai_cp_async_wait_group_4() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 4;\n" ::);
#endif
}

__device__ __forceinline__ void bonsai_cp_async_wait_all() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_all;\n" ::);
#endif
}

__device__ __forceinline__ uint32_t bonsai_triton_shared2_b_f16_offset_bytes(int k, int n) {
    const int phase = (k >> 1) & 3;
    const int n_group = n >> 3;
    const int n_in_vec = n & 7;
    const int phys_n = ((n_group ^ phase) << 3) | n_in_vec;
    return static_cast<uint32_t>(((k << 5) + phys_n) << 1);
}

__device__ __forceinline__ void bonsai_ldmatrix_x4_b16(uint32_t& x0,
                                                       uint32_t& x1,
                                                       uint32_t& x2,
                                                       uint32_t& x3,
                                                       uint32_t smem_addr) {
#if __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
                 : "r"(smem_addr));
#else
    (void)smem_addr;
    x0 = x1 = x2 = x3 = 0;
#endif
}

__device__ __forceinline__ void bonsai_ldmatrix_x4_trans_b16(uint32_t& x0,
                                                             uint32_t& x1,
                                                             uint32_t& x2,
                                                             uint32_t& x3,
                                                             uint32_t smem_addr) {
#if __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3)
                 : "r"(smem_addr));
#else
    (void)smem_addr;
    x0 = x1 = x2 = x3 = 0;
#endif
}

__device__ __forceinline__ void bonsai_mma_m16n8k16_f32(uint32_t a0,
                                                        uint32_t a1,
                                                        uint32_t a2,
                                                        uint32_t a3,
                                                        uint32_t b0,
                                                        uint32_t b1,
                                                        float& d0,
                                                        float& d1,
                                                        float& d2,
                                                        float& d3) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, "
                 "{%4, %5, %6, %7}, "
                 "{%8, %9}, "
                 "{%0, %1, %2, %3};\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
    d0 = d1 = d2 = d3 = 0.0f;
#endif
}

__global__ void bonsai_int1_linear_kernel(const half* __restrict__ a,
                                          const uint8_t* __restrict__ wq,
                                          const float* __restrict__ scales,
                                          const float* __restrict__ zeros,
                                          half* __restrict__ c,
                                          int m,
                                          int k,
                                          int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) {
        return;
    }
    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        const int packed_row = kk >> 3;
        const int bit_offset = kk & 7;
        const uint8_t byte = wq[packed_row * n + col];
        const int bit = (byte >> bit_offset) & 1;
        const int group = kk >> 7;
        const float weight = fmaf((float)bit, scales[group * n + col], zeros[group * n + col]);
        acc = fmaf(__half2float(a[row * k + kk]), weight, acc);
    }
    c[row * n + col] = __float2half(acc);
}

__global__ void bonsai_int1_linear_strided_out_kernel(const half* __restrict__ a,
                                                      const uint8_t* __restrict__ wq,
                                                      const float* __restrict__ scales,
                                                      const float* __restrict__ zeros,
                                                      half* __restrict__ c,
                                                      int m,
                                                      int k,
                                                      int n,
                                                      int c_stride,
                                                      int c_col_offset) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) {
        return;
    }
    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        const int packed_row = kk >> 3;
        const int bit_offset = kk & 7;
        const uint8_t byte = wq[packed_row * n + col];
        const int bit = (byte >> bit_offset) & 1;
        const int group = kk >> 7;
        const float weight = fmaf((float)bit, scales[group * n + col], zeros[group * n + col]);
        acc = fmaf(__half2float(a[row * k + kk]), weight, acc);
    }
    c[row * c_stride + c_col_offset + col] = __float2half(acc);
}

template<int BM, int BN, int BK>
__global__ void bonsai_int1_linear_tiled_kernel(const half* __restrict__ a,
                                                const uint8_t* __restrict__ wq,
                                                const float* __restrict__ scales,
                                                const float* __restrict__ zeros,
                                                half* __restrict__ c,
                                                int m,
                                                int k,
                                                int n) {
    __shared__ half a_tile[BM][BK];

    const int local_col = threadIdx.x;
    const int local_row = threadIdx.y;
    const int col       = blockIdx.x * BN + local_col;
    const int row       = blockIdx.y * BM + local_row;
    float acc           = 0.0f;

    for (int kb = 0; kb < k; kb += BK) {
        for (int idx = local_row * BN + local_col; idx < BM * BK; idx += BM * BN) {
            const int tile_row = idx / BK;
            const int tile_k   = idx - tile_row * BK;
            const int src_row  = blockIdx.y * BM + tile_row;
            const int src_k    = kb + tile_k;
            a_tile[tile_row][tile_k] = (src_row < m && src_k < k) ? a[src_row * k + src_k] : __float2half(0.0f);
        }
        __syncthreads();

        if (row < m && col < n) {
            const int group = kb >> 7;
            const float scale = scales[group * n + col];
            const float zero  = zeros[group * n + col];
            #pragma unroll 4
            for (int kk = 0; kk < BK && kb + kk < k; ++kk) {
                const int packed_row = (kb + kk) >> 3;
                const int bit_offset = (kb + kk) & 7;
                const uint8_t byte = wq[packed_row * n + col];
                const int bit = (byte >> bit_offset) & 1;
                const float weight = fmaf(static_cast<float>(bit), scale, zero);
                acc = fmaf(__half2float(a_tile[local_row][kk]), weight, acc);
            }
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = __float2half(acc);
    }
}

template<int BM, int BN, int BK>
__global__ void bonsai_int1_linear_tiled_strided_out_kernel(const half* __restrict__ a,
                                                            const uint8_t* __restrict__ wq,
                                                            const float* __restrict__ scales,
                                                            const float* __restrict__ zeros,
                                                            half* __restrict__ c,
                                                            int m,
                                                            int k,
                                                            int n,
                                                            int c_stride,
                                                            int c_col_offset) {
    __shared__ half a_tile[BM][BK];

    const int local_col = threadIdx.x;
    const int local_row = threadIdx.y;
    const int col       = blockIdx.x * BN + local_col;
    const int row       = blockIdx.y * BM + local_row;
    float acc           = 0.0f;

    for (int kb = 0; kb < k; kb += BK) {
        for (int idx = local_row * BN + local_col; idx < BM * BK; idx += BM * BN) {
            const int tile_row = idx / BK;
            const int tile_k   = idx - tile_row * BK;
            const int src_row  = blockIdx.y * BM + tile_row;
            const int src_k    = kb + tile_k;
            a_tile[tile_row][tile_k] = (src_row < m && src_k < k) ? a[src_row * k + src_k] : __float2half(0.0f);
        }
        __syncthreads();

        if (row < m && col < n) {
            const int group = kb >> 7;
            const float scale = scales[group * n + col];
            const float zero  = zeros[group * n + col];
            #pragma unroll 4
            for (int kk = 0; kk < BK && kb + kk < k; ++kk) {
                const int packed_row = (kb + kk) >> 3;
                const int bit_offset = (kb + kk) & 7;
                const uint8_t byte = wq[packed_row * n + col];
                const int bit = (byte >> bit_offset) & 1;
                const float weight = fmaf(static_cast<float>(bit), scale, zero);
                acc = fmaf(__half2float(a_tile[local_row][kk]), weight, acc);
            }
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * c_stride + c_col_offset + col] = __float2half(acc);
    }
}

__global__ void bonsai_int1_linear_gemlite_shape_kernel(const half* __restrict__ a,
                                                        const uint8_t* __restrict__ wq,
                                                        const float* __restrict__ scales,
                                                        const float* __restrict__ zeros,
                                                        half* __restrict__ c,
                                                        int m,
                                                        int k,
                                                        int n) {
    __shared__ half a_group[8][128];

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) {
        return;
    }

    float acc = 0.0f;
    const int local_row = threadIdx.y;
    const int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    constexpr int group_size = 128;
    constexpr int bytes_per_group = group_size / 8;
    const int groups = k / group_size;
    for (int group = 0; group < groups; ++group) {
        const int k_base = group * group_size;
        for (int idx = linear_tid; idx < blockDim.y * group_size; idx += blockDim.x * blockDim.y) {
            const int tile_row = idx / group_size;
            const int tile_k = idx - tile_row * group_size;
            const int src_row = blockIdx.y * blockDim.y + tile_row;
            a_group[tile_row][tile_k] = (src_row < m) ? a[static_cast<size_t>(src_row) * static_cast<size_t>(k) + k_base + tile_k] : __float2half(0.0f);
        }
        __syncthreads();

        float sum_all = 0.0f;
        float sum_bits = 0.0f;
        const uint8_t* group_wq = wq + static_cast<size_t>(group * bytes_per_group) * static_cast<size_t>(n) + col;

        #pragma unroll 4
        for (int byte_idx = 0; byte_idx < bytes_per_group; ++byte_idx) {
            const uint8_t bits = group_wq[static_cast<size_t>(byte_idx) * static_cast<size_t>(n)];
            const int kk = byte_idx * 8;
            const float a0 = __half2float(a_group[local_row][kk + 0]);
            const float a1 = __half2float(a_group[local_row][kk + 1]);
            const float a2 = __half2float(a_group[local_row][kk + 2]);
            const float a3 = __half2float(a_group[local_row][kk + 3]);
            const float a4 = __half2float(a_group[local_row][kk + 4]);
            const float a5 = __half2float(a_group[local_row][kk + 5]);
            const float a6 = __half2float(a_group[local_row][kk + 6]);
            const float a7 = __half2float(a_group[local_row][kk + 7]);
            sum_all += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
            sum_bits += ((bits & 0x01) ? a0 : 0.0f)
                      + ((bits & 0x02) ? a1 : 0.0f)
                      + ((bits & 0x04) ? a2 : 0.0f)
                      + ((bits & 0x08) ? a3 : 0.0f)
                      + ((bits & 0x10) ? a4 : 0.0f)
                      + ((bits & 0x20) ? a5 : 0.0f)
                      + ((bits & 0x40) ? a6 : 0.0f)
                      + ((bits & 0x80) ? a7 : 0.0f);
        }
        const float scale = scales[static_cast<size_t>(group) * static_cast<size_t>(n) + col];
        const float zero = zeros[static_cast<size_t>(group) * static_cast<size_t>(n) + col];
        acc = fmaf(scale, sum_bits, acc);
        acc = fmaf(zero, sum_all, acc);
        __syncthreads();
    }

    c[static_cast<size_t>(row) * static_cast<size_t>(n) + col] = __float2half(acc);
}

template<int BM, int BN, int COLS_PER_THREAD>
__global__ void bonsai_int1_linear_gemlite_gemm_v2_kernel(const half* __restrict__ a,
                                                          const uint8_t* __restrict__ wq,
                                                          const float* __restrict__ scales,
                                                          const float* __restrict__ zeros,
                                                          half* __restrict__ c,
                                                          int m,
                                                          int k,
                                                          int n) {
    constexpr int GROUP_SIZE = 128;
    constexpr int BYTES_PER_GROUP = GROUP_SIZE / 8;
    constexpr int COL_THREADS = BN / COLS_PER_THREAD;
    __shared__ half a_group[BM][GROUP_SIZE];

    const int local_col_thread = threadIdx.x;
    const int local_row = threadIdx.y;
    const int row = blockIdx.y * BM + local_row;
    const int col_base = blockIdx.x * BN + local_col_thread * COLS_PER_THREAD;
    const int linear_tid = local_row * COL_THREADS + local_col_thread;
    constexpr int THREADS_PER_BLOCK = BM * COL_THREADS;

    float acc[COLS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < COLS_PER_THREAD; ++i) {
        acc[i] = 0.0f;
    }

    const int groups = k / GROUP_SIZE;
    for (int group = 0; group < groups; ++group) {
        const int k_base = group * GROUP_SIZE;
        for (int idx = linear_tid; idx < BM * GROUP_SIZE; idx += THREADS_PER_BLOCK) {
            const int tile_row = idx / GROUP_SIZE;
            const int tile_k = idx - tile_row * GROUP_SIZE;
            const int src_row = blockIdx.y * BM + tile_row;
            a_group[tile_row][tile_k] = (src_row < m) ? a[static_cast<size_t>(src_row) * static_cast<size_t>(k) + k_base + tile_k] : __float2half(0.0f);
        }
        __syncthreads();

        if (row < m) {
            float sum_all = 0.0f;
            float sum_bits[COLS_PER_THREAD];
            #pragma unroll
            for (int i = 0; i < COLS_PER_THREAD; ++i) {
                sum_bits[i] = 0.0f;
            }

            #pragma unroll
            for (int byte_idx = 0; byte_idx < BYTES_PER_GROUP; ++byte_idx) {
                const int kk = byte_idx * 8;
                const float a0 = __half2float(a_group[local_row][kk + 0]);
                const float a1 = __half2float(a_group[local_row][kk + 1]);
                const float a2 = __half2float(a_group[local_row][kk + 2]);
                const float a3 = __half2float(a_group[local_row][kk + 3]);
                const float a4 = __half2float(a_group[local_row][kk + 4]);
                const float a5 = __half2float(a_group[local_row][kk + 5]);
                const float a6 = __half2float(a_group[local_row][kk + 6]);
                const float a7 = __half2float(a_group[local_row][kk + 7]);
                sum_all += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;

                #pragma unroll
                for (int ci = 0; ci < COLS_PER_THREAD; ++ci) {
                    const int col = col_base + ci;
                    if (col < n) {
                        const uint8_t bits = wq[(static_cast<size_t>(group * BYTES_PER_GROUP + byte_idx) * static_cast<size_t>(n)) + col];
                        sum_bits[ci] += ((bits & 0x01) ? a0 : 0.0f)
                                      + ((bits & 0x02) ? a1 : 0.0f)
                                      + ((bits & 0x04) ? a2 : 0.0f)
                                      + ((bits & 0x08) ? a3 : 0.0f)
                                      + ((bits & 0x10) ? a4 : 0.0f)
                                      + ((bits & 0x20) ? a5 : 0.0f)
                                      + ((bits & 0x40) ? a6 : 0.0f)
                                      + ((bits & 0x80) ? a7 : 0.0f);
                    }
                }
            }

            #pragma unroll
            for (int ci = 0; ci < COLS_PER_THREAD; ++ci) {
                const int col = col_base + ci;
                if (col < n) {
                    const size_t meta_index = static_cast<size_t>(group) * static_cast<size_t>(n) + static_cast<size_t>(col);
                    acc[ci] = fmaf(scales[meta_index], sum_bits[ci], acc[ci]);
                    acc[ci] = fmaf(zeros[meta_index], sum_all, acc[ci]);
                }
            }
        }
        __syncthreads();
    }

    if (row < m) {
        #pragma unroll
        for (int ci = 0; ci < COLS_PER_THREAD; ++ci) {
            const int col = col_base + ci;
            if (col < n) {
                c[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] = __float2half(acc[ci]);
            }
        }
    }
}

template<int BM, int BN, int COLS_PER_THREAD>
__global__ void bonsai_int1_linear_gemlite_scale_only_kernel(const half* __restrict__ a,
                                                             const uint8_t* __restrict__ wq,
                                                             const float* __restrict__ scales,
                                                             half* __restrict__ c,
                                                             int m,
                                                             int k,
                                                             int n) {
    constexpr int GROUP_SIZE = 128;
    constexpr int BYTES_PER_GROUP = GROUP_SIZE / 8;
    constexpr int COL_THREADS = BN / COLS_PER_THREAD;
    __shared__ half a_group[BM][GROUP_SIZE];

    const int local_col_thread = threadIdx.x;
    const int local_row = threadIdx.y;
    const int row = blockIdx.y * BM + local_row;
    const int col_base = blockIdx.x * BN + local_col_thread * COLS_PER_THREAD;
    const int linear_tid = local_row * COL_THREADS + local_col_thread;
    constexpr int THREADS_PER_BLOCK = BM * COL_THREADS;

    float acc[COLS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < COLS_PER_THREAD; ++i) {
        acc[i] = 0.0f;
    }

    const int groups = k / GROUP_SIZE;
    for (int group = 0; group < groups; ++group) {
        const int k_base = group * GROUP_SIZE;
        for (int idx = linear_tid; idx < BM * GROUP_SIZE; idx += THREADS_PER_BLOCK) {
            const int tile_row = idx / GROUP_SIZE;
            const int tile_k = idx - tile_row * GROUP_SIZE;
            const int src_row = blockIdx.y * BM + tile_row;
            a_group[tile_row][tile_k] = (src_row < m) ? a[static_cast<size_t>(src_row) * static_cast<size_t>(k) + k_base + tile_k] : __float2half(0.0f);
        }
        __syncthreads();

        if (row < m) {
            float sum_all = 0.0f;
            float sum_bits[COLS_PER_THREAD];
            #pragma unroll
            for (int i = 0; i < COLS_PER_THREAD; ++i) {
                sum_bits[i] = 0.0f;
            }

            #pragma unroll
            for (int byte_idx = 0; byte_idx < BYTES_PER_GROUP; ++byte_idx) {
                const int kk = byte_idx * 8;
                const float a0 = __half2float(a_group[local_row][kk + 0]);
                const float a1 = __half2float(a_group[local_row][kk + 1]);
                const float a2 = __half2float(a_group[local_row][kk + 2]);
                const float a3 = __half2float(a_group[local_row][kk + 3]);
                const float a4 = __half2float(a_group[local_row][kk + 4]);
                const float a5 = __half2float(a_group[local_row][kk + 5]);
                const float a6 = __half2float(a_group[local_row][kk + 6]);
                const float a7 = __half2float(a_group[local_row][kk + 7]);
                sum_all += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;

                #pragma unroll
                for (int ci = 0; ci < COLS_PER_THREAD; ++ci) {
                    const int col = col_base + ci;
                    if (col < n) {
                        const uint8_t bits = wq[(static_cast<size_t>(group * BYTES_PER_GROUP + byte_idx) * static_cast<size_t>(n)) + col];
                        sum_bits[ci] += ((bits & 0x01) ? a0 : 0.0f)
                                      + ((bits & 0x02) ? a1 : 0.0f)
                                      + ((bits & 0x04) ? a2 : 0.0f)
                                      + ((bits & 0x08) ? a3 : 0.0f)
                                      + ((bits & 0x10) ? a4 : 0.0f)
                                      + ((bits & 0x20) ? a5 : 0.0f)
                                      + ((bits & 0x40) ? a6 : 0.0f)
                                      + ((bits & 0x80) ? a7 : 0.0f);
                    }
                }
            }

            #pragma unroll
            for (int ci = 0; ci < COLS_PER_THREAD; ++ci) {
                const int col = col_base + ci;
                if (col < n) {
                    const float scale = scales[static_cast<size_t>(group) * static_cast<size_t>(n) + static_cast<size_t>(col)];
                    acc[ci] = fmaf(scale, sum_bits[ci] - 0.5f * sum_all, acc[ci]);
                }
            }
        }
        __syncthreads();
    }

    if (row < m) {
        #pragma unroll
        for (int ci = 0; ci < COLS_PER_THREAD; ++ci) {
            const int col = col_base + ci;
            if (col < n) {
                c[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] = __float2half(acc[ci]);
            }
        }
    }
}

template<int BM, int BN, int COLS_PER_THREAD, int SPLITS>
__global__ void bonsai_int1_linear_gemlite_splitk_partial_kernel(const half* __restrict__ a,
                                                                 const uint8_t* __restrict__ wq,
                                                                 const float* __restrict__ scales,
                                                                 float* __restrict__ partial,
                                                                 int m,
                                                                 int k,
                                                                 int n) {
    constexpr int GROUP_SIZE = 128;
    constexpr int BYTES_PER_GROUP = GROUP_SIZE / 8;
    constexpr int COL_THREADS = BN / COLS_PER_THREAD;
    __shared__ half a_group[BM][GROUP_SIZE];

    const int split = blockIdx.z;
    const int local_col_thread = threadIdx.x;
    const int local_row = threadIdx.y;
    const int row = blockIdx.y * BM + local_row;
    const int col_base = blockIdx.x * BN + local_col_thread * COLS_PER_THREAD;
    const int linear_tid = local_row * COL_THREADS + local_col_thread;
    constexpr int THREADS_PER_BLOCK = BM * COL_THREADS;

    float acc[COLS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < COLS_PER_THREAD; ++i) {
        acc[i] = 0.0f;
    }

    const int groups = k / GROUP_SIZE;
    const int groups_per_split = groups / SPLITS;
    const int group_begin = split * groups_per_split;
    const int group_end = (split == SPLITS - 1) ? groups : group_begin + groups_per_split;
    for (int group = group_begin; group < group_end; ++group) {
        const int k_base = group * GROUP_SIZE;
        for (int idx = linear_tid; idx < BM * GROUP_SIZE; idx += THREADS_PER_BLOCK) {
            const int tile_row = idx / GROUP_SIZE;
            const int tile_k = idx - tile_row * GROUP_SIZE;
            const int src_row = blockIdx.y * BM + tile_row;
            a_group[tile_row][tile_k] = (src_row < m) ? a[static_cast<size_t>(src_row) * static_cast<size_t>(k) + k_base + tile_k] : __float2half(0.0f);
        }
        __syncthreads();

        if (row < m) {
            float sum_all = 0.0f;
            float sum_bits[COLS_PER_THREAD];
            #pragma unroll
            for (int i = 0; i < COLS_PER_THREAD; ++i) {
                sum_bits[i] = 0.0f;
            }

            #pragma unroll
            for (int byte_idx = 0; byte_idx < BYTES_PER_GROUP; ++byte_idx) {
                const int kk = byte_idx * 8;
                const float a0 = __half2float(a_group[local_row][kk + 0]);
                const float a1 = __half2float(a_group[local_row][kk + 1]);
                const float a2 = __half2float(a_group[local_row][kk + 2]);
                const float a3 = __half2float(a_group[local_row][kk + 3]);
                const float a4 = __half2float(a_group[local_row][kk + 4]);
                const float a5 = __half2float(a_group[local_row][kk + 5]);
                const float a6 = __half2float(a_group[local_row][kk + 6]);
                const float a7 = __half2float(a_group[local_row][kk + 7]);
                sum_all += a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;

                #pragma unroll
                for (int ci = 0; ci < COLS_PER_THREAD; ++ci) {
                    const int col = col_base + ci;
                    if (col < n) {
                        const uint8_t bits = wq[(static_cast<size_t>(group * BYTES_PER_GROUP + byte_idx) * static_cast<size_t>(n)) + col];
                        sum_bits[ci] += ((bits & 0x01) ? a0 : 0.0f)
                                      + ((bits & 0x02) ? a1 : 0.0f)
                                      + ((bits & 0x04) ? a2 : 0.0f)
                                      + ((bits & 0x08) ? a3 : 0.0f)
                                      + ((bits & 0x10) ? a4 : 0.0f)
                                      + ((bits & 0x20) ? a5 : 0.0f)
                                      + ((bits & 0x40) ? a6 : 0.0f)
                                      + ((bits & 0x80) ? a7 : 0.0f);
                    }
                }
            }

            #pragma unroll
            for (int ci = 0; ci < COLS_PER_THREAD; ++ci) {
                const int col = col_base + ci;
                if (col < n) {
                    const float scale = scales[static_cast<size_t>(group) * static_cast<size_t>(n) + static_cast<size_t>(col)];
                    acc[ci] = fmaf(scale, sum_bits[ci] - 0.5f * sum_all, acc[ci]);
                }
            }
        }
        __syncthreads();
    }

    if (row < m) {
        #pragma unroll
        for (int ci = 0; ci < COLS_PER_THREAD; ++ci) {
            const int col = col_base + ci;
            if (col < n) {
                partial[(static_cast<size_t>(split) * static_cast<size_t>(m) + static_cast<size_t>(row)) * static_cast<size_t>(n) + static_cast<size_t>(col)] = acc[ci];
            }
        }
    }
}

template<int SPLITS>
__global__ void bonsai_int1_linear_gemlite_splitk_reduce_kernel(const float* __restrict__ partial,
                                                                half* __restrict__ c,
                                                                int m,
                                                                int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = m * n;
    if (idx >= total) {
        return;
    }
    float acc = 0.0f;
    #pragma unroll
    for (int split = 0; split < SPLITS; ++split) {
        acc += partial[static_cast<size_t>(split) * static_cast<size_t>(total) + static_cast<size_t>(idx)];
    }
    c[idx] = __float2half(acc);
}

__global__ void bonsai_int1_linear_gemlite_tc_gemm_kernel(const half* __restrict__ a,
                                                          const uint8_t* __restrict__ wq,
                                                          const float* __restrict__ scales,
                                                          const float* __restrict__ zeros,
                                                          half* __restrict__ c,
                                                          int m,
                                                          int k,
                                                          int n) {
    __shared__ half a_smem[64][128];
    __shared__ half b_smem[128][128];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int row_base = blockIdx.y * 64;
    const int col_base = blockIdx.x * 128;

    wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc[8];
    #pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
        wmma::fill_fragment(acc[tile], __float2half(0.0f));
    }

    constexpr int bytes_per_group = 16;
    const int groups = k / 128;
    for (int group = 0; group < groups; ++group) {
        const int k_base = group * 128;

        for (int idx = tid; idx < 64 * 128; idx += 128) {
            const int r = idx / 128;
            const int kk = idx - r * 128;
            a_smem[r][kk] = a[static_cast<size_t>(row_base + r) * static_cast<size_t>(k) + static_cast<size_t>(k_base + kk)];
        }

        for (int idx = tid; idx < 128 * 128; idx += 128) {
            const int kk = idx / 128;
            const int cn = idx - kk * 128;
            const int col = col_base + cn;
            const int byte_idx = kk >> 3;
            const int bit_offset = kk & 7;
            const uint8_t bits = wq[(static_cast<size_t>(group * bytes_per_group + byte_idx) * static_cast<size_t>(n)) + static_cast<size_t>(col)];
            const int bit = (bits >> bit_offset) & 1;
            const size_t meta_index = static_cast<size_t>(group) * static_cast<size_t>(n) + static_cast<size_t>(col);
            const float weight = fmaf(static_cast<float>(bit), scales[meta_index], zeros[meta_index]);
            b_smem[kk][cn] = __float2half(weight);
        }
        __syncthreads();

        #pragma unroll
        for (int tile = 0; tile < 8; ++tile) {
            const int linear_tile = warp_id + tile * 4;
            const int frag_m = linear_tile >> 3;
            const int frag_n = linear_tile & 7;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
            #pragma unroll
            for (int kk = 0; kk < 128; kk += 16) {
                wmma::load_matrix_sync(a_frag, &a_smem[frag_m * 16][kk], 128);
                wmma::load_matrix_sync(b_frag, &b_smem[kk][frag_n * 16], 128);
                wmma::mma_sync(acc[tile], a_frag, b_frag, acc[tile]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
        const int linear_tile = warp_id + tile * 4;
        const int frag_m = linear_tile >> 3;
        const int frag_n = linear_tile & 7;
        wmma::store_matrix_sync(&c[(static_cast<size_t>(row_base + frag_m * 16) * static_cast<size_t>(n)) + static_cast<size_t>(col_base + frag_n * 16)],
                                acc[tile],
                                n,
                                wmma::mem_row_major);
    }
}

__device__ __forceinline__ uint32_t bonsai_pack_half_pair(half lo, half hi) {
    return static_cast<uint32_t>(__half_as_ushort(lo)) |
           (static_cast<uint32_t>(__half_as_ushort(hi)) << 16);
}

__global__ void bonsai_int1_linear_gemlite_tc_linear1_largesmem_kernel(const half* __restrict__ a,
                                                                       const uint8_t* __restrict__ wq,
                                                                       const float* __restrict__ scales,
                                                                       const float* __restrict__ zeros,
                                                                       half* __restrict__ c,
                                                                       int m,
                                                                       int k,
                                                                       int n) {
    extern __shared__ __align__(16) unsigned char smem[];
    half* a_smem = reinterpret_cast<half*>(smem);
    half* b_smem = reinterpret_cast<half*>(smem + 64 * 128 * sizeof(half));

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int row_base = blockIdx.y * 64;
    const int col_base = blockIdx.x * 128;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
        wmma::fill_fragment(acc[tile], 0.0f);
    }

    constexpr int bytes_per_group = 16;
    const int groups = k / 128;
    for (int group = 0; group < groups; ++group) {
        const int k_base = group * 128;

        for (int vec = tid; vec < (64 * 128 / 8); vec += 128) {
            const int r = vec >> 4;
            const int kk = (vec & 15) << 3;
            const uint4 packed_a = *reinterpret_cast<const uint4*>(
                &a[static_cast<size_t>(row_base + r) * static_cast<size_t>(k) + static_cast<size_t>(k_base + kk)]);
            *reinterpret_cast<uint4*>(&a_smem[r * 128 + kk]) = packed_a;
        }

        for (int idx = tid; idx < 128 * 16; idx += 128) {
            const int kk = idx >> 4;
            const int cn = (idx & 15) << 3;
            const int col = col_base + cn;
            const int byte_idx = kk >> 3;
            const int bit_offset = kk & 7;
            const uint64_t bits8 = *reinterpret_cast<const uint64_t*>(
                &wq[(static_cast<size_t>(group * bytes_per_group + byte_idx) * static_cast<size_t>(n)) +
                    static_cast<size_t>(col)]);
            uint4 packed_b;
            #pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const int c0 = pair * 2;
                const int c1 = c0 + 1;
                const uint8_t byte0 = static_cast<uint8_t>((bits8 >> (8 * c0)) & 0xffu);
                const uint8_t byte1 = static_cast<uint8_t>((bits8 >> (8 * c1)) & 0xffu);
                const int bit0 = (byte0 >> bit_offset) & 1;
                const int bit1 = (byte1 >> bit_offset) & 1;
                const size_t meta0 = static_cast<size_t>(group) * static_cast<size_t>(n) + static_cast<size_t>(col + c0);
                const size_t meta1 = static_cast<size_t>(group) * static_cast<size_t>(n) + static_cast<size_t>(col + c1);
                const half h0 = __float2half(fmaf(static_cast<float>(bit0), scales[meta0], zeros[meta0]));
                const half h1 = __float2half(fmaf(static_cast<float>(bit1), scales[meta1], zeros[meta1]));
                const uint32_t word = bonsai_pack_half_pair(h0, h1);
                if (pair == 0) packed_b.x = word;
                if (pair == 1) packed_b.y = word;
                if (pair == 2) packed_b.z = word;
                if (pair == 3) packed_b.w = word;
            }
            *reinterpret_cast<uint4*>(&b_smem[kk * 128 + cn]) = packed_b;
        }
        __syncthreads();

        #pragma unroll
        for (int tile = 0; tile < 8; ++tile) {
            const int linear_tile = warp_id + tile * 4;
            const int frag_m = linear_tile >> 3;
            const int frag_n = linear_tile & 7;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
            #pragma unroll
            for (int kk = 0; kk < 128; kk += 16) {
                wmma::load_matrix_sync(a_frag, &a_smem[frag_m * 16 * 128 + kk], 128);
                wmma::load_matrix_sync(b_frag, &b_smem[kk * 128 + frag_n * 16], 128);
                wmma::mma_sync(acc[tile], a_frag, b_frag, acc[tile]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
        const int linear_tile = warp_id + tile * 4;
        const int frag_m = linear_tile >> 3;
        const int frag_n = linear_tile & 7;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> out_frag;
        #pragma unroll
        for (int i = 0; i < out_frag.num_elements; ++i) {
            out_frag.x[i] = __float2half(acc[tile].x[i]);
        }
        wmma::store_matrix_sync(&c[(static_cast<size_t>(row_base + frag_m * 16) * static_cast<size_t>(n)) +
                                   static_cast<size_t>(col_base + frag_n * 16)],
                                out_frag,
                                n,
                                wmma::mem_row_major);
    }
}

__global__ void bonsai_int1_linear_gemlite_tc_gemm_v2_kernel(const half* __restrict__ a,
                                                             const uint8_t* __restrict__ wq,
                                                             const float* __restrict__ scales,
                                                             const float* __restrict__ zeros,
                                                             half* __restrict__ c,
                                                             int m,
                                                             int k,
                                                             int n) {
    __shared__ half a_smem[64][16];
    __shared__ half b_smem[16][128];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int row_base = blockIdx.y * 64;
    const int col_base = blockIdx.x * 128;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
        wmma::fill_fragment(acc[tile], 0.0f);
    }

    constexpr int bytes_per_group = 16;
    const int groups = k / 128;
    for (int group = 0; group < groups; ++group) {
        const int k_group_base = group * 128;
        #pragma unroll
        for (int inner = 0; inner < 128; inner += 16) {
            for (int idx = tid; idx < 64 * 16; idx += 128) {
                const int r = idx / 16;
                const int kk = idx - r * 16;
                a_smem[r][kk] = a[static_cast<size_t>(row_base + r) * static_cast<size_t>(k) + static_cast<size_t>(k_group_base + inner + kk)];
            }

            for (int idx = tid; idx < 16 * 128; idx += 128) {
                const int kk = idx / 128;
                const int cn = idx - kk * 128;
                const int logical_k = inner + kk;
                const int col = col_base + cn;
                const int byte_idx = logical_k >> 3;
                const int bit_offset = logical_k & 7;
                const uint8_t bits = wq[(static_cast<size_t>(group * bytes_per_group + byte_idx) * static_cast<size_t>(n)) + static_cast<size_t>(col)];
                const int bit = (bits >> bit_offset) & 1;
                const size_t meta_index = static_cast<size_t>(group) * static_cast<size_t>(n) + static_cast<size_t>(col);
                const float weight = fmaf(static_cast<float>(bit), scales[meta_index], zeros[meta_index]);
                b_smem[kk][cn] = __float2half(weight);
            }
            __syncthreads();

            #pragma unroll
            for (int tile = 0; tile < 8; ++tile) {
                const int linear_tile = warp_id + tile * 4;
                const int frag_m = linear_tile >> 3;
                const int frag_n = linear_tile & 7;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(a_frag, &a_smem[frag_m * 16][0], 16);
                wmma::load_matrix_sync(b_frag, &b_smem[0][frag_n * 16], 128);
                wmma::mma_sync(acc[tile], a_frag, b_frag, acc[tile]);
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
        const int linear_tile = warp_id + tile * 4;
        const int frag_m = linear_tile >> 3;
        const int frag_n = linear_tile & 7;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> out_frag;
        #pragma unroll
        for (int i = 0; i < out_frag.num_elements; ++i) {
            out_frag.x[i] = __float2half(acc[tile].x[i]);
        }
        wmma::store_matrix_sync(&c[(static_cast<size_t>(row_base + frag_m * 16) * static_cast<size_t>(n)) + static_cast<size_t>(col_base + frag_n * 16)],
                                out_frag,
                                n,
                                wmma::mem_row_major);
    }
}

__global__ void bonsai_int1_linear_gemlite_tc_gemm_v2_strided_out_kernel(const half* __restrict__ a,
                                                                         const uint8_t* __restrict__ wq,
                                                                         const float* __restrict__ scales,
                                                                         const float* __restrict__ zeros,
                                                                         half* __restrict__ c,
                                                                         int m,
                                                                         int k,
                                                                         int n,
                                                                         int c_stride,
                                                                         int c_col_offset) {
    __shared__ half a_smem[64][16];
    __shared__ half b_smem[16][128];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int row_base = blockIdx.y * 64;
    const int col_base = blockIdx.x * 128;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
        wmma::fill_fragment(acc[tile], 0.0f);
    }

    constexpr int bytes_per_group = 16;
    const int groups = k / 128;
    for (int group = 0; group < groups; ++group) {
        const int k_group_base = group * 128;
        #pragma unroll
        for (int inner = 0; inner < 128; inner += 16) {
            for (int idx = tid; idx < 64 * 16; idx += 128) {
                const int r = idx / 16;
                const int kk = idx - r * 16;
                a_smem[r][kk] = a[static_cast<size_t>(row_base + r) * static_cast<size_t>(k) + static_cast<size_t>(k_group_base + inner + kk)];
            }

            for (int idx = tid; idx < 16 * 128; idx += 128) {
                const int kk = idx / 128;
                const int cn = idx - kk * 128;
                const int logical_k = inner + kk;
                const int col = col_base + cn;
                const int byte_idx = logical_k >> 3;
                const int bit_offset = logical_k & 7;
                const uint8_t bits = wq[(static_cast<size_t>(group * bytes_per_group + byte_idx) * static_cast<size_t>(n)) + static_cast<size_t>(col)];
                const int bit = (bits >> bit_offset) & 1;
                const size_t meta_index = static_cast<size_t>(group) * static_cast<size_t>(n) + static_cast<size_t>(col);
                const float weight = fmaf(static_cast<float>(bit), scales[meta_index], zeros[meta_index]);
                b_smem[kk][cn] = __float2half(weight);
            }
            __syncthreads();

            #pragma unroll
            for (int tile = 0; tile < 8; ++tile) {
                const int linear_tile = warp_id + tile * 4;
                const int frag_m = linear_tile >> 3;
                const int frag_n = linear_tile & 7;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(a_frag, &a_smem[frag_m * 16][0], 16);
                wmma::load_matrix_sync(b_frag, &b_smem[0][frag_n * 16], 128);
                wmma::mma_sync(acc[tile], a_frag, b_frag, acc[tile]);
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
        const int linear_tile = warp_id + tile * 4;
        const int frag_m = linear_tile >> 3;
        const int frag_n = linear_tile & 7;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> out_frag;
        #pragma unroll
        for (int i = 0; i < out_frag.num_elements; ++i) {
            out_frag.x[i] = __float2half(acc[tile].x[i]);
        }
        wmma::store_matrix_sync(&c[(static_cast<size_t>(row_base + frag_m * 16) * static_cast<size_t>(c_stride)) +
                                   static_cast<size_t>(c_col_offset + col_base + frag_n * 16)],
                                out_frag,
                                c_stride,
                                wmma::mem_row_major);
    }
}

__global__ void bonsai_int1_linear_gemlite_tc_attn_proj_txt_kernel(const half* __restrict__ a,
                                                                   const uint8_t* __restrict__ wq,
                                                                   const float* __restrict__ scales,
                                                                   const float* __restrict__ zeros,
                                                                   half* __restrict__ c,
                                                                   int m,
                                                                   int k,
                                                                   int n) {
    __shared__ half a_smem[64][16];
    __shared__ half b_smem[16][64];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int row_base = blockIdx.y * 64;
    const int col_base = blockIdx.x * 64;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    #pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
        wmma::fill_fragment(acc[tile], 0.0f);
    }

    for (int k_base = 0; k_base < k; k_base += 64) {
        #pragma unroll
        for (int inner = 0; inner < 64; inner += 16) {
            for (int idx = tid; idx < 64 * 16; idx += 128) {
                const int r = idx / 16;
                const int kk = idx - r * 16;
                a_smem[r][kk] = a[static_cast<size_t>(row_base + r) * static_cast<size_t>(k) +
                                   static_cast<size_t>(k_base + inner + kk)];
            }

            for (int idx = tid; idx < 16 * 64; idx += 128) {
                const int kk = idx / 64;
                const int cn = idx - kk * 64;
                const int logical_k = k_base + inner + kk;
                const int col = col_base + cn;
                const int byte_idx = logical_k >> 3;
                const int bit_offset = logical_k & 7;
                const int group = logical_k >> 7;
                const uint8_t bits = wq[static_cast<size_t>(byte_idx) * static_cast<size_t>(n) +
                                        static_cast<size_t>(col)];
                const int bit = (bits >> bit_offset) & 1;
                const size_t meta_index = static_cast<size_t>(group) * static_cast<size_t>(n) + static_cast<size_t>(col);
                const float weight = fmaf(static_cast<float>(bit), scales[meta_index], zeros[meta_index]);
                b_smem[kk][cn] = __float2half(weight);
            }
            __syncthreads();

            #pragma unroll
            for (int tile = 0; tile < 4; ++tile) {
                const int linear_tile = warp_id + tile * 4;
                const int frag_m = linear_tile >> 2;
                const int frag_n = linear_tile & 3;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(a_frag, &a_smem[frag_m * 16][0], 16);
                wmma::load_matrix_sync(b_frag, &b_smem[0][frag_n * 16], 64);
                wmma::mma_sync(acc[tile], a_frag, b_frag, acc[tile]);
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
        const int linear_tile = warp_id + tile * 4;
        const int frag_m = linear_tile >> 2;
        const int frag_n = linear_tile & 3;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> out_frag;
        #pragma unroll
        for (int i = 0; i < out_frag.num_elements; ++i) {
            out_frag.x[i] = __float2half(acc[tile].x[i]);
        }
        wmma::store_matrix_sync(&c[(static_cast<size_t>(row_base + frag_m * 16) * static_cast<size_t>(n)) +
                                   static_cast<size_t>(col_base + frag_n * 16)],
                                out_frag,
                                n,
                                wmma::mem_row_major);
    }
}

__global__ void bonsai_int1_linear_gemlite_tc_gemm_v2_prepack_kernel(const half* __restrict__ a,
                                                                     const uint8_t* __restrict__ wq_tile,
                                                                     const float* __restrict__ scales_tile,
                                                                     const float* __restrict__ zeros_tile,
                                                                     half* __restrict__ c,
                                                                     int m,
                                                                     int k,
                                                                     int n,
                                                                     int n_tiles) {
    __shared__ half a_smem[64][16];
    __shared__ half b_smem[16][128];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int row_base = blockIdx.y * 64;
    const int tile_n = blockIdx.x;
    const int col_base = tile_n * 128;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
        wmma::fill_fragment(acc[tile], 0.0f);
    }

    constexpr int bytes_per_group = 16;
    const int groups = k / 128;
    for (int group = 0; group < groups; ++group) {
        const int k_group_base = group * 128;
        const size_t packed_tile_base = (static_cast<size_t>(group) * static_cast<size_t>(n_tiles) + static_cast<size_t>(tile_n)) * bytes_per_group * 128ull;
        const size_t meta_tile_base = (static_cast<size_t>(group) * static_cast<size_t>(n_tiles) + static_cast<size_t>(tile_n)) * 128ull;
        #pragma unroll
        for (int inner = 0; inner < 128; inner += 16) {
            for (int idx = tid; idx < 64 * 16; idx += 128) {
                const int r = idx / 16;
                const int kk = idx - r * 16;
                a_smem[r][kk] = a[static_cast<size_t>(row_base + r) * static_cast<size_t>(k) + static_cast<size_t>(k_group_base + inner + kk)];
            }

            for (int idx = tid; idx < 16 * 128; idx += 128) {
                const int kk = idx / 128;
                const int cn = idx - kk * 128;
                const int logical_k = inner + kk;
                const int byte_idx = logical_k >> 3;
                const int bit_offset = logical_k & 7;
                const uint8_t bits = wq_tile[packed_tile_base + static_cast<size_t>(byte_idx) * 128ull + static_cast<size_t>(cn)];
                const int bit = (bits >> bit_offset) & 1;
                const size_t meta_index = meta_tile_base + static_cast<size_t>(cn);
                const float weight = fmaf(static_cast<float>(bit), scales_tile[meta_index], zeros_tile[meta_index]);
                b_smem[kk][cn] = __float2half(weight);
            }
            __syncthreads();

            #pragma unroll
            for (int tile = 0; tile < 8; ++tile) {
                const int linear_tile = warp_id + tile * 4;
                const int frag_m = linear_tile >> 3;
                const int frag_n = linear_tile & 7;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(a_frag, &a_smem[frag_m * 16][0], 16);
                wmma::load_matrix_sync(b_frag, &b_smem[0][frag_n * 16], 128);
                wmma::mma_sync(acc[tile], a_frag, b_frag, acc[tile]);
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
        const int linear_tile = warp_id + tile * 4;
        const int frag_m = linear_tile >> 3;
        const int frag_n = linear_tile & 7;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> out_frag;
        #pragma unroll
        for (int i = 0; i < out_frag.num_elements; ++i) {
            out_frag.x[i] = __float2half(acc[tile].x[i]);
        }
        wmma::store_matrix_sync(&c[(static_cast<size_t>(row_base + frag_m * 16) * static_cast<size_t>(n)) + static_cast<size_t>(col_base + frag_n * 16)],
                                out_frag,
                                n,
                                wmma::mem_row_major);
    }
}

__global__ void bonsai_int1_linear_gemlite_tc_gemm_v2_prepack_wq_only_kernel(const half* __restrict__ a,
                                                                             const uint8_t* __restrict__ wq_tile,
                                                                             const float* __restrict__ scales,
                                                                             const float* __restrict__ zeros,
                                                                             half* __restrict__ c,
                                                                             int m,
                                                                             int k,
                                                                             int n,
                                                                             int n_tiles) {
    __shared__ half a_smem[64][16];
    __shared__ half b_smem[16][128];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int row_base = blockIdx.y * 64;
    const int tile_n = blockIdx.x;
    const int col_base = tile_n * 128;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
        wmma::fill_fragment(acc[tile], 0.0f);
    }

    constexpr int bytes_per_group = 16;
    const int groups = k / 128;
    for (int group = 0; group < groups; ++group) {
        const int k_group_base = group * 128;
        const size_t packed_tile_base = (static_cast<size_t>(group) * static_cast<size_t>(n_tiles) + static_cast<size_t>(tile_n)) * bytes_per_group * 128ull;
        #pragma unroll
        for (int inner = 0; inner < 128; inner += 16) {
            for (int idx = tid; idx < 64 * 16; idx += 128) {
                const int r = idx / 16;
                const int kk = idx - r * 16;
                a_smem[r][kk] = a[static_cast<size_t>(row_base + r) * static_cast<size_t>(k) + static_cast<size_t>(k_group_base + inner + kk)];
            }

            for (int idx = tid; idx < 16 * 128; idx += 128) {
                const int kk = idx / 128;
                const int cn = idx - kk * 128;
                const int logical_k = inner + kk;
                const int byte_idx = logical_k >> 3;
                const int bit_offset = logical_k & 7;
                const uint8_t bits = wq_tile[packed_tile_base + static_cast<size_t>(byte_idx) * 128ull + static_cast<size_t>(cn)];
                const int bit = (bits >> bit_offset) & 1;
                const size_t meta_index = static_cast<size_t>(group) * static_cast<size_t>(n) + static_cast<size_t>(col_base + cn);
                const float weight = fmaf(static_cast<float>(bit), scales[meta_index], zeros[meta_index]);
                b_smem[kk][cn] = __float2half(weight);
            }
            __syncthreads();

            #pragma unroll
            for (int tile = 0; tile < 8; ++tile) {
                const int linear_tile = warp_id + tile * 4;
                const int frag_m = linear_tile >> 3;
                const int frag_n = linear_tile & 7;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(a_frag, &a_smem[frag_m * 16][0], 16);
                wmma::load_matrix_sync(b_frag, &b_smem[0][frag_n * 16], 128);
                wmma::mma_sync(acc[tile], a_frag, b_frag, acc[tile]);
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
        const int linear_tile = warp_id + tile * 4;
        const int frag_m = linear_tile >> 3;
        const int frag_n = linear_tile & 7;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> out_frag;
        #pragma unroll
        for (int i = 0; i < out_frag.num_elements; ++i) {
            out_frag.x[i] = __float2half(acc[tile].x[i]);
        }
        wmma::store_matrix_sync(&c[(static_cast<size_t>(row_base + frag_m * 16) * static_cast<size_t>(n)) + static_cast<size_t>(col_base + frag_n * 16)],
                                out_frag,
                                n,
                                wmma::mem_row_major);
    }
}

__global__ void bonsai_int1_linear_gemlite_tc_gemm_v2_prepack_padded_kernel(const half* __restrict__ a,
                                                                            const uint8_t* __restrict__ wq_tile,
                                                                            const float* __restrict__ scales_tile,
                                                                            const float* __restrict__ zeros_tile,
                                                                            half* __restrict__ c,
                                                                            int m,
                                                                            int k,
                                                                            int n,
                                                                            int n_tiles) {
    __shared__ half a_smem[64][16];
    __shared__ half b_smem[16][136];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int row_base = blockIdx.y * 64;
    const int tile_n = blockIdx.x;
    const int col_base = tile_n * 128;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
        wmma::fill_fragment(acc[tile], 0.0f);
    }

    constexpr int bytes_per_group = 16;
    const int groups = k / 128;
    for (int group = 0; group < groups; ++group) {
        const int k_group_base = group * 128;
        const size_t packed_tile_base = (static_cast<size_t>(group) * static_cast<size_t>(n_tiles) + static_cast<size_t>(tile_n)) * bytes_per_group * 128ull;
        const size_t meta_tile_base = (static_cast<size_t>(group) * static_cast<size_t>(n_tiles) + static_cast<size_t>(tile_n)) * 128ull;
        #pragma unroll
        for (int inner = 0; inner < 128; inner += 16) {
            for (int idx = tid; idx < 64 * 16; idx += 128) {
                const int r = idx / 16;
                const int kk = idx - r * 16;
                a_smem[r][kk] = a[static_cast<size_t>(row_base + r) * static_cast<size_t>(k) + static_cast<size_t>(k_group_base + inner + kk)];
            }

            for (int idx = tid; idx < 16 * 128; idx += 128) {
                const int kk = idx / 128;
                const int cn = idx - kk * 128;
                const int logical_k = inner + kk;
                const int byte_idx = logical_k >> 3;
                const int bit_offset = logical_k & 7;
                const uint8_t bits = wq_tile[packed_tile_base + static_cast<size_t>(byte_idx) * 128ull + static_cast<size_t>(cn)];
                const int bit = (bits >> bit_offset) & 1;
                const size_t meta_index = meta_tile_base + static_cast<size_t>(cn);
                const float weight = fmaf(static_cast<float>(bit), scales_tile[meta_index], zeros_tile[meta_index]);
                b_smem[kk][cn] = __float2half(weight);
            }
            __syncthreads();

            #pragma unroll
            for (int tile = 0; tile < 8; ++tile) {
                const int linear_tile = warp_id + tile * 4;
                const int frag_m = linear_tile >> 3;
                const int frag_n = linear_tile & 7;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(a_frag, &a_smem[frag_m * 16][0], 16);
                wmma::load_matrix_sync(b_frag, &b_smem[0][frag_n * 16], 136);
                wmma::mma_sync(acc[tile], a_frag, b_frag, acc[tile]);
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
        const int linear_tile = warp_id + tile * 4;
        const int frag_m = linear_tile >> 3;
        const int frag_n = linear_tile & 7;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> out_frag;
        #pragma unroll
        for (int i = 0; i < out_frag.num_elements; ++i) {
            out_frag.x[i] = __float2half(acc[tile].x[i]);
        }
        wmma::store_matrix_sync(&c[(static_cast<size_t>(row_base + frag_m * 16) * static_cast<size_t>(n)) + static_cast<size_t>(col_base + frag_n * 16)],
                                out_frag,
                                n,
                                wmma::mem_row_major);
    }
}

__global__ void bonsai_int1_linear_gemlite_tc_gemm_v2_prepack_bankfix_kernel(const half* __restrict__ a,
                                                                             const uint8_t* __restrict__ wq_tile,
                                                                             const float* __restrict__ scales_tile,
                                                                             const float* __restrict__ zeros_tile,
                                                                             half* __restrict__ c,
                                                                             int m,
                                                                             int k,
                                                                             int n,
                                                                             int n_tiles) {
    __shared__ half a_smem[64][24];
    __shared__ half b_smem[16][136];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int row_base = blockIdx.y * 64;
    const int tile_n = blockIdx.x;
    const int col_base = tile_n * 128;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
        wmma::fill_fragment(acc[tile], 0.0f);
    }

    constexpr int bytes_per_group = 16;
    const int groups = k / 128;
    for (int group = 0; group < groups; ++group) {
        const int k_group_base = group * 128;
        const size_t packed_tile_base = (static_cast<size_t>(group) * static_cast<size_t>(n_tiles) + static_cast<size_t>(tile_n)) * bytes_per_group * 128ull;
        const size_t meta_tile_base = (static_cast<size_t>(group) * static_cast<size_t>(n_tiles) + static_cast<size_t>(tile_n)) * 128ull;
        #pragma unroll
        for (int inner = 0; inner < 128; inner += 16) {
            for (int idx = tid; idx < 64 * 16; idx += 128) {
                const int r = idx / 16;
                const int kk = idx - r * 16;
                a_smem[r][kk] = a[static_cast<size_t>(row_base + r) * static_cast<size_t>(k) + static_cast<size_t>(k_group_base + inner + kk)];
            }

            for (int idx = tid; idx < 16 * 128; idx += 128) {
                const int kk = idx / 128;
                const int cn = idx - kk * 128;
                const int logical_k = inner + kk;
                const int byte_idx = logical_k >> 3;
                const int bit_offset = logical_k & 7;
                const uint8_t bits = wq_tile[packed_tile_base + static_cast<size_t>(byte_idx) * 128ull + static_cast<size_t>(cn)];
                const int bit = (bits >> bit_offset) & 1;
                const size_t meta_index = meta_tile_base + static_cast<size_t>(cn);
                const float weight = fmaf(static_cast<float>(bit), scales_tile[meta_index], zeros_tile[meta_index]);
                b_smem[kk][cn] = __float2half(weight);
            }
            __syncthreads();

            #pragma unroll
            for (int tile = 0; tile < 8; ++tile) {
                const int linear_tile = warp_id + tile * 4;
                const int frag_m = linear_tile >> 3;
                const int frag_n = linear_tile & 7;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(a_frag, &a_smem[frag_m * 16][0], 24);
                wmma::load_matrix_sync(b_frag, &b_smem[0][frag_n * 16], 136);
                wmma::mma_sync(acc[tile], a_frag, b_frag, acc[tile]);
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
        const int linear_tile = warp_id + tile * 4;
        const int frag_m = linear_tile >> 3;
        const int frag_n = linear_tile & 7;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> out_frag;
        #pragma unroll
        for (int i = 0; i < out_frag.num_elements; ++i) {
            out_frag.x[i] = __float2half(acc[tile].x[i]);
        }
        wmma::store_matrix_sync(&c[(static_cast<size_t>(row_base + frag_m * 16) * static_cast<size_t>(n)) + static_cast<size_t>(col_base + frag_n * 16)],
                                out_frag,
                                n,
                                wmma::mem_row_major);
    }
}

__global__ void bonsai_int1_linear_gemlite_tc_gemm_v2_dequant_only_kernel(const half* __restrict__ a,
                                                                         const uint8_t* __restrict__ wq,
                                                                         const float* __restrict__ scales,
                                                                         const float* __restrict__ zeros,
                                                                         half* __restrict__ c,
                                                                         int m,
                                                                         int k,
                                                                         int n) {
    __shared__ half a_smem[64][16];
    __shared__ half b_smem[16][128];
    const int tid = threadIdx.x;
    const int row_base = blockIdx.y * 64;
    const int col_base = blockIdx.x * 128;
    constexpr int bytes_per_group = 16;
    const int groups = k / 128;
    float checksum = 0.0f;
    for (int group = 0; group < groups; ++group) {
        const int k_group_base = group * 128;
        #pragma unroll
        for (int inner = 0; inner < 128; inner += 16) {
            for (int idx = tid; idx < 64 * 16; idx += 128) {
                const int r = idx / 16;
                const int kk = idx - r * 16;
                const half av = a[static_cast<size_t>(row_base + r) * static_cast<size_t>(k) + static_cast<size_t>(k_group_base + inner + kk)];
                a_smem[r][kk] = av;
                checksum += __half2float(av) * 0.000001f;
            }
            for (int idx = tid; idx < 16 * 128; idx += 128) {
                const int kk = idx / 128;
                const int cn = idx - kk * 128;
                const int logical_k = inner + kk;
                const int col = col_base + cn;
                const int byte_idx = logical_k >> 3;
                const int bit_offset = logical_k & 7;
                const uint8_t bits = wq[(static_cast<size_t>(group * bytes_per_group + byte_idx) * static_cast<size_t>(n)) + static_cast<size_t>(col)];
                const int bit = (bits >> bit_offset) & 1;
                const size_t meta_index = static_cast<size_t>(group) * static_cast<size_t>(n) + static_cast<size_t>(col);
                const float weight = fmaf(static_cast<float>(bit), scales[meta_index], zeros[meta_index]);
                b_smem[kk][cn] = __float2half(weight);
                checksum += weight * 0.000001f;
            }
            __syncthreads();
        }
    }
    if (tid == 0) {
        c[static_cast<size_t>(row_base) * static_cast<size_t>(n) + static_cast<size_t>(col_base)] = __float2half(checksum);
    }
}

__global__ void bonsai_int1_linear_gemlite_tc_gemm_v2_mma_only_kernel(const half* __restrict__ a,
                                                                     half* __restrict__ c,
                                                                     int m,
                                                                     int k,
                                                                     int n) {
    __shared__ half a_smem[64][16];
    __shared__ half b_smem[16][128];
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int row_base = blockIdx.y * 64;
    const int col_base = blockIdx.x * 128;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
    #pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
        wmma::fill_fragment(acc[tile], 0.0f);
    }
    const int groups = k / 128;
    for (int group = 0; group < groups; ++group) {
        const int k_group_base = group * 128;
        #pragma unroll
        for (int inner = 0; inner < 128; inner += 16) {
            for (int idx = tid; idx < 64 * 16; idx += 128) {
                const int r = idx / 16;
                const int kk = idx - r * 16;
                a_smem[r][kk] = a[static_cast<size_t>(row_base + r) * static_cast<size_t>(k) + static_cast<size_t>(k_group_base + inner + kk)];
            }
            for (int idx = tid; idx < 16 * 128; idx += 128) {
                const int kk = idx / 128;
                const int cn = idx - kk * 128;
                b_smem[kk][cn] = __float2half(((kk + cn) & 1) ? 0.5f : -0.5f);
            }
            __syncthreads();
            #pragma unroll
            for (int tile = 0; tile < 8; ++tile) {
                const int linear_tile = warp_id + tile * 4;
                const int frag_m = linear_tile >> 3;
                const int frag_n = linear_tile & 7;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(a_frag, &a_smem[frag_m * 16][0], 16);
                wmma::load_matrix_sync(b_frag, &b_smem[0][frag_n * 16], 128);
                wmma::mma_sync(acc[tile], a_frag, b_frag, acc[tile]);
            }
            __syncthreads();
        }
    }
    #pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
        const int linear_tile = warp_id + tile * 4;
        const int frag_m = linear_tile >> 3;
        const int frag_n = linear_tile & 7;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> out_frag;
        #pragma unroll
        for (int i = 0; i < out_frag.num_elements; ++i) {
            out_frag.x[i] = __float2half(acc[tile].x[i]);
        }
        wmma::store_matrix_sync(&c[(static_cast<size_t>(row_base + frag_m * 16) * static_cast<size_t>(n)) + static_cast<size_t>(col_base + frag_n * 16)],
                                out_frag,
                                n,
                                wmma::mem_row_major);
    }
}

__global__ void bonsai_int1_linear_gemlite_tc_linear2_kernel(const half* __restrict__ a,
                                                             const uint8_t* __restrict__ wq,
                                                             const float* __restrict__ scales,
                                                             const float* __restrict__ zeros,
                                                             half* __restrict__ c,
                                                             int m,
                                                             int k,
                                                             int n) {
    __shared__ half a_smem[64][16];
    __shared__ half b_smem[16][32];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int row_base = blockIdx.y * 64;
    const int col_base = blockIdx.x * 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
    #pragma unroll
    for (int tile = 0; tile < 2; ++tile) {
        wmma::fill_fragment(acc[tile], 0.0f);
    }

    constexpr int bytes_per_group = 16;
    const int groups = k / 128;
    for (int group = 0; group < groups; ++group) {
        const int k_group_base = group * 128;
        #pragma unroll
        for (int inner = 0; inner < 128; inner += 16) {
            for (int idx = tid; idx < 64 * 16; idx += 128) {
                const int r = idx / 16;
                const int kk = idx - r * 16;
                a_smem[r][kk] = a[static_cast<size_t>(row_base + r) * static_cast<size_t>(k) + static_cast<size_t>(k_group_base + inner + kk)];
            }
            for (int idx = tid; idx < 16 * 32; idx += 128) {
                const int kk = idx / 32;
                const int cn = idx - kk * 32;
                const int logical_k = inner + kk;
                const int col = col_base + cn;
                const int byte_idx = logical_k >> 3;
                const int bit_offset = logical_k & 7;
                const uint8_t bits = wq[(static_cast<size_t>(group * bytes_per_group + byte_idx) * static_cast<size_t>(n)) + static_cast<size_t>(col)];
                const int bit = (bits >> bit_offset) & 1;
                const size_t meta_index = static_cast<size_t>(group) * static_cast<size_t>(n) + static_cast<size_t>(col);
                const float weight = fmaf(static_cast<float>(bit), scales[meta_index], zeros[meta_index]);
                b_smem[kk][cn] = __float2half(weight);
            }
            __syncthreads();

            #pragma unroll
            for (int tile = 0; tile < 2; ++tile) {
                const int linear_tile = warp_id + tile * 4;
                const int frag_m = linear_tile >> 1;
                const int frag_n = linear_tile & 1;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(a_frag, &a_smem[frag_m * 16][0], 16);
                wmma::load_matrix_sync(b_frag, &b_smem[0][frag_n * 16], 32);
                wmma::mma_sync(acc[tile], a_frag, b_frag, acc[tile]);
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int tile = 0; tile < 2; ++tile) {
        const int linear_tile = warp_id + tile * 4;
        const int frag_m = linear_tile >> 1;
        const int frag_n = linear_tile & 1;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> out_frag;
        #pragma unroll
        for (int i = 0; i < out_frag.num_elements; ++i) {
            out_frag.x[i] = __float2half(acc[tile].x[i]);
        }
        wmma::store_matrix_sync(&c[(static_cast<size_t>(row_base + frag_m * 16) * static_cast<size_t>(n)) + static_cast<size_t>(col_base + frag_n * 16)],
                                out_frag,
                                n,
                                wmma::mem_row_major);
    }
}

__global__ void bonsai_int1_linear_gemlite_tc_linear2_async_kernel(const half* __restrict__ a,
                                                                   const uint8_t* __restrict__ wq,
                                                                   const float* __restrict__ scales,
                                                                   const float* __restrict__ zeros,
                                                                   half* __restrict__ c,
                                                                   int m,
                                                                   int k,
                                                                   int n) {
    __shared__ half a_smem[3][64][16];
    __shared__ half b_smem[16][32];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int row_base = blockIdx.y * 64;
    const int col_base = blockIdx.x * 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
    #pragma unroll
    for (int tile = 0; tile < 2; ++tile) {
        wmma::fill_fragment(acc[tile], 0.0f);
    }

    const int total_tiles = k / 16;
    auto prefetch_a_tile = [&](int tile_index) {
        const int stage = tile_index % 3;
        const int k_base = tile_index * 16;
        if (tid < 128) {
            const int r = tid >> 1;
            const int kk = (tid & 1) * 8;
            bonsai_cp_async_cg_16(&a_smem[stage][r][kk],
                                  &a[static_cast<size_t>(row_base + r) * static_cast<size_t>(k) + static_cast<size_t>(k_base + kk)]);
        }
        bonsai_cp_async_commit_group();
    };

    const int prefetch_count = total_tiles < 3 ? total_tiles : 3;
    for (int tile = 0; tile < prefetch_count; ++tile) {
        prefetch_a_tile(tile);
    }

    constexpr int bytes_per_group = 16;
    for (int tile_index = 0; tile_index < total_tiles; ++tile_index) {
        bonsai_cp_async_wait_group_2();
        __syncthreads();

        const int stage = tile_index % 3;
        const int group = tile_index >> 3;
        const int inner = (tile_index & 7) * 16;
        #pragma unroll
        for (int idx = tid; idx < 16 * 32; idx += 128) {
            const int kk = idx / 32;
            const int cn = idx - kk * 32;
            const int logical_k = inner + kk;
            const int col = col_base + cn;
            const int byte_idx = logical_k >> 3;
            const int bit_offset = logical_k & 7;
            const uint8_t bits = wq[(static_cast<size_t>(group * bytes_per_group + byte_idx) * static_cast<size_t>(n)) + static_cast<size_t>(col)];
            const int bit = (bits >> bit_offset) & 1;
            const size_t meta_index = static_cast<size_t>(group) * static_cast<size_t>(n) + static_cast<size_t>(col);
            const float weight = fmaf(static_cast<float>(bit), scales[meta_index], zeros[meta_index]);
            b_smem[kk][cn] = __float2half(weight);
        }
        __syncthreads();

        #pragma unroll
        for (int tile = 0; tile < 2; ++tile) {
            const int linear_tile = warp_id + tile * 4;
            const int frag_m = linear_tile >> 1;
            const int frag_n = linear_tile & 1;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
            wmma::load_matrix_sync(a_frag, &a_smem[stage][frag_m * 16][0], 16);
            wmma::load_matrix_sync(b_frag, &b_smem[0][frag_n * 16], 32);
            wmma::mma_sync(acc[tile], a_frag, b_frag, acc[tile]);
        }
        __syncthreads();

        const int next_tile = tile_index + 3;
        if (next_tile < total_tiles) {
            prefetch_a_tile(next_tile);
        }
    }
    bonsai_cp_async_wait_all();
    __syncthreads();

    #pragma unroll
    for (int tile = 0; tile < 2; ++tile) {
        const int linear_tile = warp_id + tile * 4;
        const int frag_m = linear_tile >> 1;
        const int frag_n = linear_tile & 1;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> out_frag;
        #pragma unroll
        for (int i = 0; i < out_frag.num_elements; ++i) {
            out_frag.x[i] = __float2half(acc[tile].x[i]);
        }
        wmma::store_matrix_sync(&c[(static_cast<size_t>(row_base + frag_m * 16) * static_cast<size_t>(n)) + static_cast<size_t>(col_base + frag_n * 16)],
                                out_frag,
                                n,
                                wmma::mem_row_major);
    }
}

__global__ void bonsai_int1_linear_gemlite_tc_linear2_async_b_kernel(const half* __restrict__ a,
                                                                     const uint8_t* __restrict__ wq,
                                                                     const float* __restrict__ scales,
                                                                     const float* __restrict__ zeros,
                                                                     half* __restrict__ c,
                                                                     int m,
                                                                     int k,
                                                                     int n) {
    extern __shared__ unsigned char smem[];

    constexpr int BM = 64;
    constexpr int BN = 32;
    constexpr int BK = 128;
    constexpr int B_PACKED_STAGE_BYTES = BK * BN;
    constexpr int A_STAGE_BYTES = BM * BK * static_cast<int>(sizeof(half));
    constexpr int B_DEQ_OFFSET = 2 * A_STAGE_BYTES + 2 * B_PACKED_STAGE_BYTES;
    constexpr int SCALE_OFFSET = B_DEQ_OFFSET + BK * BN * static_cast<int>(sizeof(half));
    constexpr int ZERO_OFFSET = SCALE_OFFSET + 2 * BN * static_cast<int>(sizeof(float));

    half* a_stage = reinterpret_cast<half*>(smem);
    uint8_t* b_stage = reinterpret_cast<uint8_t*>(smem + 2 * A_STAGE_BYTES);
    half* b_deq = reinterpret_cast<half*>(smem + B_DEQ_OFFSET);
    float* scale_stage = reinterpret_cast<float*>(smem + SCALE_OFFSET);
    float* zero_stage = reinterpret_cast<float*>(smem + ZERO_OFFSET);

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int row_base = blockIdx.y * BM;
    const int col_base = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
    #pragma unroll
    for (int tile = 0; tile < 2; ++tile) {
        wmma::fill_fragment(acc[tile], 0.0f);
    }

    constexpr int bytes_per_group = BK / 8;
    const int groups = k / BK;

    auto prefetch_group = [&](int stage, int group) {
        const int k_group_base = group * BK;
        for (int chunk = tid; chunk < (BM * BK * static_cast<int>(sizeof(half))) / 16; chunk += blockDim.x) {
            const int r = chunk / 16;
            const int kk = (chunk - r * 16) * 8;
            bonsai_cp_async_cg_16(&a_stage[(stage * BM + r) * BK + kk],
                                  &a[static_cast<size_t>(row_base + r) * static_cast<size_t>(k) + static_cast<size_t>(k_group_base + kk)]);
        }
        bonsai_cp_async_commit_group();

        for (int chunk = tid; chunk < (BK * BN) / 16; chunk += blockDim.x) {
            const int logical_k = chunk >> 1;
            const int cn = (chunk & 1) * 16;
            const int packed_row = group * bytes_per_group + (logical_k >> 3);
            bonsai_cp_async_cg_16(&b_stage[(stage * BK + logical_k) * BN + cn],
                                  &wq[static_cast<size_t>(packed_row) * static_cast<size_t>(n) + static_cast<size_t>(col_base + cn)]);
        }
        bonsai_cp_async_commit_group();

        if (tid < 8) {
            const int cn = tid * 4;
            bonsai_cp_async_cg_16(&scale_stage[stage * BN + cn],
                                  &scales[static_cast<size_t>(group) * static_cast<size_t>(n) + static_cast<size_t>(col_base + cn)]);
        }
        bonsai_cp_async_commit_group();

        if (tid < 8) {
            const int cn = tid * 4;
            bonsai_cp_async_cg_16(&zero_stage[stage * BN + cn],
                                  &zeros[static_cast<size_t>(group) * static_cast<size_t>(n) + static_cast<size_t>(col_base + cn)]);
        }
        bonsai_cp_async_commit_group();
    };

    prefetch_group(0, 0);
    prefetch_group(1, 1);

    for (int group = 0; group < groups; ++group) {
        const int stage = group & 1;
        bonsai_cp_async_wait_group_4();
        __syncthreads();

        for (int idx = tid; idx < BK * BN; idx += blockDim.x) {
            const int logical_k = idx / BN;
            const int cn = idx - logical_k * BN;
            const uint8_t bits = b_stage[(stage * BK + logical_k) * BN + cn];
            const int bit = (bits >> (logical_k & 7)) & 1;
            const float weight = fmaf(static_cast<float>(bit), scale_stage[stage * BN + cn], zero_stage[stage * BN + cn]);
            b_deq[logical_k * BN + cn] = __float2half(weight);
        }
        __syncthreads();

        #pragma unroll
        for (int inner = 0; inner < BK; inner += 16) {
            #pragma unroll
            for (int tile = 0; tile < 2; ++tile) {
                const int linear_tile = warp_id + tile * 4;
                const int frag_m = linear_tile >> 1;
                const int frag_n = linear_tile & 1;
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(a_frag, &a_stage[(stage * BM + frag_m * 16) * BK + inner], BK);
                wmma::load_matrix_sync(b_frag, &b_deq[inner * BN + frag_n * 16], BN);
                wmma::mma_sync(acc[tile], a_frag, b_frag, acc[tile]);
            }
        }
        __syncthreads();

        const int next_group = group + 2;
        if (next_group < groups) {
            prefetch_group(stage, next_group);
        }
    }
    bonsai_cp_async_wait_all();
    __syncthreads();

    #pragma unroll
    for (int tile = 0; tile < 2; ++tile) {
        const int linear_tile = warp_id + tile * 4;
        const int frag_m = linear_tile >> 1;
        const int frag_n = linear_tile & 1;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> out_frag;
        #pragma unroll
        for (int i = 0; i < out_frag.num_elements; ++i) {
            out_frag.x[i] = __float2half(acc[tile].x[i]);
        }
        wmma::store_matrix_sync(&c[(static_cast<size_t>(row_base + frag_m * 16) * static_cast<size_t>(n)) + static_cast<size_t>(col_base + frag_n * 16)],
                                out_frag,
                                n,
                                wmma::mem_row_major);
    }
}

__global__ void bonsai_int1_linear_gemlite_tc_linear2_ldmatrix_kernel(const half* __restrict__ a,
                                                                      const uint8_t* __restrict__ wq,
                                                                      const float* __restrict__ scales,
                                                                      const float* __restrict__ zeros,
                                                                      half* __restrict__ c,
                                                                      int m,
                                                                      int k,
                                                                      int n) {
    __shared__ __align__(16) half a_smem[64][128];
    __shared__ __align__(16) half b_deq_smem[128 * 32];

    constexpr int BM = 64;
    constexpr int BN = 32;
    constexpr int BK = 128;
    constexpr int bytes_per_group = BK / 8;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int row_base = blockIdx.y * BM;
    const int col_base = blockIdx.x * BN;

    float acc[4][4];
    #pragma unroll
    for (int nf = 0; nf < 4; ++nf) {
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            acc[nf][l] = 0.0f;
        }
    }

    const int groups = k / BK;
    for (int group = 0; group < groups; ++group) {
        const int k_group_base = group * BK;

        for (int idx = tid; idx < BK * BN; idx += blockDim.x) {
            const int logical_k = idx / BN;
            const int cn = idx - logical_k * BN;
            const int col = col_base + cn;
            const int byte_idx = logical_k >> 3;
            const int bit_offset = logical_k & 7;
            const uint8_t bits = wq[(static_cast<size_t>(group * bytes_per_group + byte_idx) * static_cast<size_t>(n)) +
                                    static_cast<size_t>(col)];
            const int bit = (bits >> bit_offset) & 1;
            const size_t meta_index = static_cast<size_t>(group) * static_cast<size_t>(n) + static_cast<size_t>(col);
            const float weight = fmaf(static_cast<float>(bit), scales[meta_index], zeros[meta_index]);
            const uint32_t offset_bytes = bonsai_triton_shared2_b_f16_offset_bytes(logical_k, cn);
            b_deq_smem[offset_bytes >> 1] = __float2half(weight);
        }
        for (int idx = tid; idx < BM * BK; idx += blockDim.x) {
            const int r = idx / BK;
            const int kk = idx - r * BK;
            a_smem[r][kk] = a[(static_cast<size_t>(row_base + r) * static_cast<size_t>(k)) +
                               static_cast<size_t>(k_group_base + kk)];
        }
        __syncthreads();

        #pragma unroll
        for (int inner = 0; inner < BK; inner += 16) {
            const int a_row = warp_id * 16 + (lane & 15);
            const int a_col = inner + ((lane >> 4) << 3);
            uint32_t a_reg[4];
            bonsai_ldmatrix_x4_b16(a_reg[0], a_reg[1], a_reg[2], a_reg[3],
                                   bonsai_smem_addr(&a_smem[a_row][a_col]));

            const int b_matrix = lane >> 3;
            const int b_row_in_matrix = lane & 7;
            uint32_t b_top[4];
            uint32_t b_bottom[4];
            const uint32_t top_addr = bonsai_smem_addr(reinterpret_cast<const char*>(b_deq_smem) +
                                                       bonsai_triton_shared2_b_f16_offset_bytes(inner + b_row_in_matrix,
                                                                                                b_matrix << 3));
            const uint32_t bottom_addr = bonsai_smem_addr(reinterpret_cast<const char*>(b_deq_smem) +
                                                          bonsai_triton_shared2_b_f16_offset_bytes(inner + 8 + b_row_in_matrix,
                                                                                                   b_matrix << 3));
            bonsai_ldmatrix_x4_trans_b16(b_top[0], b_top[1], b_top[2], b_top[3], top_addr);
            bonsai_ldmatrix_x4_trans_b16(b_bottom[0], b_bottom[1], b_bottom[2], b_bottom[3], bottom_addr);

            #pragma unroll
            for (int nf = 0; nf < 4; ++nf) {
                bonsai_mma_m16n8k16_f32(a_reg[0], a_reg[1], a_reg[2], a_reg[3],
                                         b_top[nf], b_bottom[nf],
                                         acc[nf][0], acc[nf][1], acc[nf][2], acc[nf][3]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int nf = 0; nf < 4; ++nf) {
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int out_i = ((l >> 1) << 3) + (lane >> 2);
            const int out_j = ((lane & 3) << 1) + (l & 1);
            const int row = row_base + warp_id * 16 + out_i;
            const int col = col_base + nf * 8 + out_j;
            c[static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col)] = __float2half(acc[nf][l]);
        }
    }
}

struct BonsaiDeviceLinear {
    uint8_t* wq = nullptr;
    float* scales = nullptr;
    float* zeros = nullptr;
    uint8_t* wq_tile = nullptr;
    float* scales_tile = nullptr;
    float* zeros_tile = nullptr;
    int k = 0;
    int n = 0;
    int n_tiles = 0;
    size_t wq_bytes = 0;
    size_t meta_bytes = 0;
    size_t tile_bytes = 0;
    bool has_tile_major = false;
};

struct BonsaiStreamCounters {
    std::atomic<uint64_t> custom_kernel_launches{0};
    std::atomic<uint64_t> launches_on_backend_stream{0};
    std::atomic<uint64_t> device_sync_calls{0};
    std::atomic<uint64_t> event_sync_calls{0};
    std::atomic<uint64_t> default_stream_launches{0};
};

BonsaiStreamCounters g_stream_counters;

struct BonsaiLinearProfileEntry {
    uint64_t calls = 0;
    double total_ms = 0.0;
    double max_ms = 0.0;
    int last_m = 0;
    int last_k = 0;
    int last_n = 0;
};

struct BonsaiFamilyProfileEntry {
    uint64_t calls = 0;
    uint64_t timed_calls = 0;
    double total_ms = 0.0;
    double max_ms = 0.0;
    std::vector<double> times_ms;
    std::string representative_name;
    std::string family;
    std::string kernel;
    int last_m = 0;
    int last_k = 0;
    int last_n = 0;
    uint64_t device_bytes = 0;
    bool uses_prepack = false;
};

struct BonsaiShapeCensusEntry {
    uint64_t calls = 0;
    std::string families;
    std::string kernels;
    bool tc_v2_supported = false;
    bool prepack_supported = false;
};

struct BonsaiFamilyEvent {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    std::string family;
    std::string name;
    std::string kernel;
    int m = 0;
    int k = 0;
    int n = 0;
    uint64_t device_bytes = 0;
    bool uses_prepack = false;
};

std::mutex g_profile_mutex;
std::map<std::string, BonsaiLinearProfileEntry> g_profile_entries;
std::atomic<bool> g_profile_atexit_registered{false};
std::mutex g_family_profile_mutex;
std::map<std::string, BonsaiFamilyProfileEntry> g_family_profile_entries;
std::map<std::string, BonsaiShapeCensusEntry> g_shape_census_entries;
std::atomic<uint64_t> g_family_timed_call_attempts{0};
std::atomic<bool> g_family_profile_atexit_registered{false};
std::mutex g_splitk_mutex;
float* g_splitk_partial = nullptr;
size_t g_splitk_partial_bytes = 0;

void set_error(char* error, size_t error_size, const char* message) {
    if (error != nullptr && error_size > 0) {
        std::snprintf(error, error_size, "%s", message);
    }
}

bool force_bonsai_device_sync_enabled();
cudaStream_t current_bonsai_backend_stream();
void log_stream_counters(const char* where, cudaStream_t stream);

bool check_cuda(cudaError_t status, char* error, size_t error_size, const char* where) {
    if (status == cudaSuccess) {
        return true;
    }
    char buffer[256];
    std::snprintf(buffer, sizeof(buffer), "%s: %s", where, cudaGetErrorString(status));
    set_error(error, error_size, buffer);
    return false;
}

bool ensure_splitk_partial(size_t bytes, char* error, size_t error_size) {
    std::lock_guard<std::mutex> lock(g_splitk_mutex);
    if (g_splitk_partial != nullptr && g_splitk_partial_bytes >= bytes) {
        return true;
    }
    if (g_splitk_partial != nullptr) {
        cudaFree(g_splitk_partial);
        g_splitk_partial = nullptr;
        g_splitk_partial_bytes = 0;
    }
    if (!check_cuda(cudaMalloc(&g_splitk_partial, bytes), error, error_size, "cudaMalloc Bonsai split-K partial")) {
        return false;
    }
    g_splitk_partial_bytes = bytes;
    return true;
}

bool maybe_force_device_sync(char* error, size_t error_size, const char* where) {
    if (!force_bonsai_device_sync_enabled()) {
        return true;
    }
    g_stream_counters.device_sync_calls.fetch_add(1);
    return check_cuda(cudaDeviceSynchronize(), error, error_size, where);
}

bool trace_bonsai_int1_stats_enabled() {
    const char* value = std::getenv("SDCPP_TRACE_BONSAI_INT1_STATS");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool dump_bonsai_tensors_enabled() {
    const char* value = std::getenv("SDCPP_BONSAI_DUMP_TENSORS");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

int bonsai_dump_tensor_limit() {
    const char* value = std::getenv("SDCPP_BONSAI_DUMP_TENSOR_LIMIT");
    if (value == nullptr || value[0] == '\0') {
        return 64;
    }
    char* end = nullptr;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0) {
        return 64;
    }
    return static_cast<int>(std::min<long>(parsed, 4096));
}

bool trace_bonsai_streams_enabled() {
    const char* value = std::getenv("SDCPP_TRACE_BONSAI_STREAMS");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool profile_bonsai_int1_enabled() {
    const char* value = std::getenv("SDCPP_PROFILE_BONSAI_INT1");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool profile_bonsai_int1_families_enabled() {
    const char* value = std::getenv("SDCPP_PROFILE_BONSAI_INT1_FAMILIES");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool profile_bonsai_int1_family_events_enabled() {
    const char* value = std::getenv("SDCPP_PROFILE_BONSAI_INT1_RECORD_EVENTS");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

std::string profile_bonsai_int1_filter() {
    const char* value = std::getenv("SDCPP_PROFILE_BONSAI_INT1_FILTER");
    return value != nullptr ? std::string(value) : std::string();
}

uint64_t profile_bonsai_int1_max_calls() {
    const char* value = std::getenv("SDCPP_PROFILE_BONSAI_INT1_MAX_CALLS");
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }
    char* end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value) {
        return 0;
    }
    return static_cast<uint64_t>(parsed);
}

bool tiled_bonsai_int1_kernel_enabled() {
    const char* value = std::getenv("SDCPP_BONSAI_INT1_TILED_KERNEL");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool gemlite_shape_kernel_enabled() {
    const char* value = std::getenv("SDCPP_BONSAI_INT1_GEMLITE_SHAPE_KERNEL");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool gemlite_gemm_v2_kernel_enabled() {
    const char* value = std::getenv("SDCPP_BONSAI_INT1_GEMLITE_GEMM_V2");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool gemlite_tc_gemm_kernel_enabled() {
    const char* value = std::getenv("SDCPP_BONSAI_INT1_GEMLITE_TC_GEMM");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool gemlite_tc_gemm_v2_kernel_enabled() {
    const char* value = std::getenv("SDCPP_BONSAI_INT1_GEMLITE_TC_GEMM_V2");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool linear1_bankfix_kernel_enabled() {
    const char* value = std::getenv("SDCPP_BONSAI_INT1_LINEAR1_BANKFIX");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool gemlite_linear1_largesmem_kernel_enabled() {
    const char* value = std::getenv("SDCPP_BONSAI_INT1_GEMLITE_LINEAR1_LARGESMEM");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool gemlite_tc_linear2_kernel_enabled() {
    const char* value = std::getenv("SDCPP_BONSAI_INT1_GEMLITE_TC_LINEAR2");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool gemlite_linear2_async_kernel_enabled() {
    const char* value = std::getenv("SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_ASYNC");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool gemlite_linear2_async_b_kernel_enabled() {
    const char* value = std::getenv("SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_ASYNC_B");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool gemlite_linear2_ldmatrix_kernel_enabled() {
    const char* value = std::getenv("SDCPP_BONSAI_INT1_GEMLITE_LINEAR2_LDMATRIX");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool gemlite_tc_img_mlp0_kernel_enabled() {
    const char* value = std::getenv("SDCPP_BONSAI_INT1_GEMLITE_TC_IMG_MLP0");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool gemlite_tc_img_mlp2_kernel_enabled() {
    const char* value = std::getenv("SDCPP_BONSAI_INT1_GEMLITE_TC_IMG_MLP2");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool gemlite_tc_qkv_kernel_enabled() {
    const char* value = std::getenv("SDCPP_BONSAI_INT1_GEMLITE_TC_QKV");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool gemlite_tc_attn_proj_kernel_enabled() {
    const char* value = std::getenv("SDCPP_BONSAI_INT1_GEMLITE_TC_ATTN_PROJ");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool gemlite_tc_gemm_v2_dequant_only_enabled() {
    const char* value = std::getenv("SDCPP_BONSAI_INT1_TC_V2_DEQUANT_ONLY");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool gemlite_tc_gemm_v2_mma_only_enabled() {
    const char* value = std::getenv("SDCPP_BONSAI_INT1_TC_V2_MMA_ONLY");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool bonsai_tile_major_prepack_enabled() {
    const char* value = std::getenv("SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

int bonsai_tile_major_prepack_variant() {
    const char* value = std::getenv("SDCPP_BONSAI_INT1_TILE_MAJOR_PREPACK_VARIANT");
    if (value == nullptr || value[0] == '\0') {
        return 2;
    }
    return std::atoi(value);
}

int gemlite_gemm_v2_variant() {
    const char* value = std::getenv("SDCPP_BONSAI_INT1_GEMM_V2_VARIANT");
    if (value == nullptr || value[0] == '\0') {
        return 9;
    }
    return std::atoi(value);
}

bool force_bonsai_device_sync_enabled() {
    const char* value = std::getenv("SDCPP_BONSAI_FORCE_DEVICE_SYNC");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

void print_bonsai_profile_summary() {
    if (!profile_bonsai_int1_enabled()) {
        return;
    }
    std::vector<std::pair<std::string, BonsaiLinearProfileEntry>> entries;
    {
        std::lock_guard<std::mutex> lock(g_profile_mutex);
        entries.assign(g_profile_entries.begin(), g_profile_entries.end());
    }
    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
        return a.second.total_ms > b.second.total_ms;
    });
    double total_ms = 0.0;
    uint64_t total_calls = 0;
    for (const auto& kv : entries) {
        total_ms += kv.second.total_ms;
        total_calls += kv.second.calls;
    }
    fprintf(stderr,
            "[BonsaiProfile] custom_op_aggregate calls=%llu total_ms=%.3f unique_linears=%zu\n",
            static_cast<unsigned long long>(total_calls),
            total_ms,
            entries.size());
    const size_t limit = std::min<size_t>(20, entries.size());
    for (size_t i = 0; i < limit; ++i) {
        const auto& name = entries[i].first;
        const auto& e = entries[i].second;
        fprintf(stderr,
                "[BonsaiProfile] rank=%zu name=%s calls=%llu total_ms=%.3f mean_ms=%.3f max_ms=%.3f last_shape=%dx%dx%d\n",
                i + 1,
                name.c_str(),
                static_cast<unsigned long long>(e.calls),
                e.total_ms,
                e.calls > 0 ? e.total_ms / static_cast<double>(e.calls) : 0.0,
                e.max_ms,
                e.last_m,
                e.last_k,
                e.last_n);
    }
}

void ensure_bonsai_profile_atexit() {
    bool expected = false;
    if (g_profile_atexit_registered.compare_exchange_strong(expected, true)) {
        std::atexit(print_bonsai_profile_summary);
    }
}

double median_double(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if ((values.size() & 1u) != 0u) {
        return values[mid];
    }
    return 0.5 * (values[mid - 1] + values[mid]);
}

bool append_unique_token(std::string& list, const std::string& token) {
    if (token.empty()) {
        return false;
    }
    if (list.empty()) {
        list = token;
        return true;
    }
    size_t pos = 0;
    while (pos <= list.size()) {
        const size_t end = list.find(',', pos);
        const std::string existing = list.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
        if (existing == token) {
            return false;
        }
        if (end == std::string::npos) {
            break;
        }
        pos = end + 1;
    }
    list += ",";
    list += token;
    return true;
}

std::string json_escape_string(const std::string& value) {
    std::ostringstream out;
    for (char ch : value) {
        switch (ch) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << ch; break;
        }
    }
    return out.str();
}

std::string bonsai_linear_family(const char* raw_name) {
    const std::string name = raw_name != nullptr ? std::string(raw_name) : std::string("unknown");
    const bool is_single = name.find("single_blocks.") != std::string::npos || name.find("single_transformer_blocks.") != std::string::npos;
    const bool is_double = name.find("double_blocks.") != std::string::npos || name.find("transformer_blocks.") != std::string::npos;
    if (is_single) {
        if (name.find("linear1") != std::string::npos || name.find("to_qkv_mlp_proj") != std::string::npos) return "single_blocks.linear1";
        if (name.find("linear2") != std::string::npos || name.find("proj_out") != std::string::npos) return "single_blocks.linear2";
        if (name.find("norm") != std::string::npos) return "single_blocks.norm";
        return "single_blocks.other";
    }
    if (is_double) {
        if (name.find("img_attn.qkv") != std::string::npos) return "double_blocks.img_attn.qkv";
        if (name.find("txt_attn.qkv") != std::string::npos) return "double_blocks.txt_attn.qkv";
        if (name.find("img_attn.proj") != std::string::npos) return "double_blocks.img_attn.proj";
        if (name.find("txt_attn.proj") != std::string::npos) return "double_blocks.txt_attn.proj";
        if (name.find("img_mlp.0") != std::string::npos) return "double_blocks.img_mlp.0";
        if (name.find("img_mlp.2") != std::string::npos) return "double_blocks.img_mlp.2";
        if (name.find("txt_mlp.0") != std::string::npos) return "double_blocks.txt_mlp.0";
        if (name.find("txt_mlp.2") != std::string::npos) return "double_blocks.txt_mlp.2";
        return "double_blocks.other";
    }
    if (name.find("final") != std::string::npos) {
        return "final_layer";
    }
    return "other";
}

bool bonsai_profile_name_matches(const std::string& family, const char* raw_name) {
    const std::string filter = profile_bonsai_int1_filter();
    if (filter.empty()) {
        return true;
    }
    const std::string name = raw_name != nullptr ? std::string(raw_name) : std::string();
    return family.find(filter) != std::string::npos || name.find(filter) != std::string::npos;
}

uint64_t bonsai_linear_device_bytes(const BonsaiDeviceLinear* linear) {
    if (linear == nullptr) {
        return 0;
    }
    return static_cast<uint64_t>(linear->wq_bytes + 2 * linear->meta_bytes + linear->tile_bytes);
}

const char* bonsai_selected_kernel_label(const BonsaiDeviceLinear* linear, int m, int k, int n, bool strided) {
    if (strided) {
        if (gemlite_tc_qkv_kernel_enabled() && k == 3072 && n == 3072 && (m % 64) == 0 && (n % 128) == 0 && (k % 128) == 0) {
            return "tc_qkv";
        }
        return tiled_bonsai_int1_kernel_enabled() ? "tiled_strided" : "native_strided";
    }
    if (gemlite_tc_gemm_v2_dequant_only_enabled() && k == 3072 && n == 27648 && (m % 64) == 0 && (n % 128) == 0 && (k % 128) == 0) {
        return "tc_v2_dequant_only";
    }
    if (gemlite_linear1_largesmem_kernel_enabled() && k == 3072 && n == 27648 && (m % 64) == 0 && (n % 128) == 0 && (k % 128) == 0) {
        return "tc_linear1_largesmem";
    }
    if (gemlite_linear2_ldmatrix_kernel_enabled() && k == 12288 && n == 3072 && (m % 64) == 0 && (n % 32) == 0 && (k % 128) == 0) {
        return "tc_linear2_ldmatrix";
    }
    if (gemlite_linear2_async_b_kernel_enabled() && k == 12288 && n == 3072 && (m % 64) == 0 && (n % 32) == 0 && (k % 128) == 0) {
        return "tc_linear2_async_b";
    }
    if (gemlite_linear2_async_kernel_enabled() && k == 12288 && n == 3072 && (m % 64) == 0 && (n % 32) == 0 && (k % 128) == 0) {
        return "tc_linear2_async";
    }
    if (gemlite_tc_linear2_kernel_enabled() && k == 12288 && n == 3072 && (m % 64) == 0 && (n % 32) == 0 && (k % 128) == 0) {
        return "tc_linear2";
    }
    if (gemlite_tc_img_mlp0_kernel_enabled() && k == 3072 && n == 18432 && (m % 64) == 0 && (n % 128) == 0 && (k % 128) == 0) {
        return "tc_img_mlp0";
    }
    if (gemlite_tc_img_mlp2_kernel_enabled() && k == 9216 && n == 3072 && (m % 64) == 0 && (n % 128) == 0 && (k % 128) == 0) {
        return "tc_img_mlp2";
    }
    if (gemlite_tc_attn_proj_kernel_enabled() && k == 3072 && n == 3072 && (m % 64) == 0 && (k % 128) == 0) {
        if (m == 512 && (n % 64) == 0) {
            return "tc_txt_attn_proj";
        }
        if ((n % 128) == 0) {
            return "tc_img_attn_proj";
        }
    }
    if (gemlite_tc_gemm_v2_mma_only_enabled() && k == 3072 && n == 27648 && (m % 64) == 0 && (n % 128) == 0 && (k % 128) == 0) {
        return "tc_v2_mma_only";
    }
    if (bonsai_tile_major_prepack_enabled() && linear != nullptr && linear->has_tile_major && k == 3072 && n == 27648 &&
        (m % 64) == 0 && (n % 128) == 0 && (k % 128) == 0) {
        if (linear1_bankfix_kernel_enabled()) return "tc_v2_prepack_v3_bankfix";
        const int variant = bonsai_tile_major_prepack_variant();
        if (variant == 1) return "tc_v2_prepack_v1";
        if (variant == 3) return "tc_v2_prepack_v3";
        return "tc_v2_prepack_v2";
    }
    if (gemlite_tc_gemm_v2_kernel_enabled() && k == 3072 && n == 27648 && (m % 64) == 0 && (n % 128) == 0 && (k % 128) == 0) {
        return "tc_v2";
    }
    if (gemlite_tc_gemm_kernel_enabled() && k == 3072 && n == 27648 && (m % 64) == 0 && (n % 128) == 0 && (k % 128) == 0) {
        return "tc_v1";
    }
    if (gemlite_gemm_v2_kernel_enabled() && k == 3072 && n == 27648 && (k % 128) == 0) {
        return "shape_v2";
    }
    if (gemlite_shape_kernel_enabled() && k == 3072 && n == 27648 && (k % 128) == 0) {
        return "shape_v1";
    }
    return tiled_bonsai_int1_kernel_enabled() ? "tiled" : "native";
}

void print_bonsai_family_profile_summary() {
    if (!profile_bonsai_int1_families_enabled()) {
        return;
    }
    std::vector<std::pair<std::string, BonsaiFamilyProfileEntry>> families;
    std::vector<std::pair<std::string, BonsaiShapeCensusEntry>> shapes;
    {
        std::lock_guard<std::mutex> lock(g_family_profile_mutex);
        families.assign(g_family_profile_entries.begin(), g_family_profile_entries.end());
        shapes.assign(g_shape_census_entries.begin(), g_shape_census_entries.end());
    }
    double total_timed_ms = 0.0;
    for (const auto& kv : families) {
        total_timed_ms += kv.second.total_ms;
    }
    std::sort(families.begin(), families.end(), [](const auto& a, const auto& b) {
        return a.second.total_ms > b.second.total_ms;
    });
    fprintf(stderr,
            "[BonsaiFamilyProfile] aggregate families=%zu total_timed_ms=%.3f record_events=%s filter=\"%s\" max_calls=%llu\n",
            families.size(),
            total_timed_ms,
            profile_bonsai_int1_family_events_enabled() ? "true" : "false",
            profile_bonsai_int1_filter().c_str(),
            static_cast<unsigned long long>(profile_bonsai_int1_max_calls()));
    for (const auto& kv : families) {
        const BonsaiFamilyProfileEntry& e = kv.second;
        const double pct = total_timed_ms > 0.0 ? (100.0 * e.total_ms / total_timed_ms) : 0.0;
        fprintf(stderr,
                "[BonsaiFamilyProfile] family=%s calls=%llu timed_calls=%llu total_ms=%.3f median_ms=%.3f mean_ms=%.3f max_ms=%.3f pct=%.2f shape=%dx%dx%d kernel=%s uses_prepack=%s device_mb=%.3f representative=\"%s\"\n",
                kv.first.c_str(),
                static_cast<unsigned long long>(e.calls),
                static_cast<unsigned long long>(e.timed_calls),
                e.total_ms,
                median_double(e.times_ms),
                e.timed_calls > 0 ? e.total_ms / static_cast<double>(e.timed_calls) : 0.0,
                e.max_ms,
                pct,
                e.last_m,
                e.last_k,
                e.last_n,
                e.kernel.c_str(),
                e.uses_prepack ? "true" : "false",
                static_cast<double>(e.device_bytes) / (1024.0 * 1024.0),
                e.representative_name.c_str());
    }
    std::sort(shapes.begin(), shapes.end(), [](const auto& a, const auto& b) {
        return a.second.calls > b.second.calls;
    });
    for (const auto& kv : shapes) {
        const BonsaiShapeCensusEntry& e = kv.second;
        fprintf(stderr,
                "[BonsaiShapeCensus] shape=%s calls=%llu families=%s kernels=%s tc_v2_supported=%s prepack_supported=%s\n",
                kv.first.c_str(),
                static_cast<unsigned long long>(e.calls),
                e.families.c_str(),
                e.kernels.c_str(),
                e.tc_v2_supported ? "true" : "false",
                e.prepack_supported ? "true" : "false");
    }
    const char* census_path = std::getenv("SDCPP_PROFILE_BONSAI_INT1_SHAPE_CENSUS_JSON");
    if (census_path != nullptr && census_path[0] != '\0') {
        std::ofstream out(census_path, std::ios::binary);
        if (out) {
            out << "{\n";
            out << "  \"families\": [\n";
            for (size_t i = 0; i < families.size(); ++i) {
                const auto& kv = families[i];
                const BonsaiFamilyProfileEntry& e = kv.second;
                out << "    {\"family\":\"" << json_escape_string(kv.first)
                    << "\",\"calls\":" << e.calls
                    << ",\"timed_calls\":" << e.timed_calls
                    << ",\"total_ms\":" << e.total_ms
                    << ",\"median_ms\":" << median_double(e.times_ms)
                    << ",\"max_ms\":" << e.max_ms
                    << ",\"shape\":{\"m\":" << e.last_m << ",\"k\":" << e.last_k << ",\"n\":" << e.last_n << "}"
                    << ",\"kernel\":\"" << json_escape_string(e.kernel)
                    << "\",\"uses_prepack\":" << (e.uses_prepack ? "true" : "false")
                    << ",\"device_bytes\":" << e.device_bytes
                    << ",\"representative\":\"" << json_escape_string(e.representative_name) << "\"}";
                out << (i + 1 < families.size() ? ",\n" : "\n");
            }
            out << "  ],\n";
            out << "  \"shapes\": [\n";
            for (size_t i = 0; i < shapes.size(); ++i) {
                const auto& kv = shapes[i];
                const BonsaiShapeCensusEntry& e = kv.second;
                out << "    {\"shape\":\"" << json_escape_string(kv.first)
                    << "\",\"calls\":" << e.calls
                    << ",\"families\":\"" << json_escape_string(e.families)
                    << "\",\"kernels\":\"" << json_escape_string(e.kernels)
                    << "\",\"tc_v2_supported\":" << (e.tc_v2_supported ? "true" : "false")
                    << ",\"prepack_supported\":" << (e.prepack_supported ? "true" : "false") << "}";
                out << (i + 1 < shapes.size() ? ",\n" : "\n");
            }
            out << "  ]\n";
            out << "}\n";
            fprintf(stderr, "[BonsaiShapeCensus] json=%s\n", census_path);
        } else {
            fprintf(stderr, "[BonsaiShapeCensus] json_write_failed path=%s\n", census_path);
        }
    }
}

void ensure_bonsai_family_profile_atexit() {
    bool expected = false;
    if (g_family_profile_atexit_registered.compare_exchange_strong(expected, true)) {
        std::atexit(print_bonsai_family_profile_summary);
    }
}

void record_bonsai_family_call(const char* name, int m, int k, int n, const char* kernel, const BonsaiDeviceLinear* linear, bool uses_prepack) {
    if (!profile_bonsai_int1_families_enabled()) {
        return;
    }
    ensure_bonsai_family_profile_atexit();
    const std::string family = bonsai_linear_family(name);
    if (!bonsai_profile_name_matches(family, name)) {
        return;
    }
    const std::string kernel_text = kernel != nullptr ? std::string(kernel) : std::string("unknown");
    char shape_key_buf[96];
    std::snprintf(shape_key_buf, sizeof(shape_key_buf), "M=%d,K=%d,N=%d", m, k, n);
    std::lock_guard<std::mutex> lock(g_family_profile_mutex);
    BonsaiFamilyProfileEntry& entry = g_family_profile_entries[family];
    entry.calls++;
    entry.family = family;
    if (entry.representative_name.empty() && name != nullptr) entry.representative_name = name;
    entry.kernel = kernel_text;
    entry.last_m = m;
    entry.last_k = k;
    entry.last_n = n;
    entry.device_bytes = bonsai_linear_device_bytes(linear);
    entry.uses_prepack = entry.uses_prepack || uses_prepack;
    BonsaiShapeCensusEntry& shape = g_shape_census_entries[shape_key_buf];
    shape.calls++;
    append_unique_token(shape.families, family);
    append_unique_token(shape.kernels, kernel_text);
    shape.tc_v2_supported = shape.tc_v2_supported ||
                            (k == 3072 && n == 27648 && (m % 64) == 0 && (n % 128) == 0 && (k % 128) == 0) ||
                            (k == 12288 && n == 3072 && (m % 64) == 0 && (n % 32) == 0 && (k % 128) == 0) ||
                            (k == 3072 && n == 18432 && (m % 64) == 0 && (n % 128) == 0 && (k % 128) == 0) ||
                            (k == 9216 && n == 3072 && (m % 64) == 0 && (n % 128) == 0 && (k % 128) == 0) ||
                            (k == 3072 && n == 3072 && (m % 64) == 0 && (n % 128) == 0 && (k % 128) == 0);
    shape.prepack_supported = shape.prepack_supported || (linear != nullptr && linear->has_tile_major && k == 3072 && n == 27648);
}

void record_bonsai_family_elapsed(const BonsaiFamilyEvent& event, float elapsed_ms) {
    if (!profile_bonsai_int1_families_enabled() || event.family.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_family_profile_mutex);
    BonsaiFamilyProfileEntry& entry = g_family_profile_entries[event.family];
    entry.timed_calls++;
    entry.total_ms += elapsed_ms;
    entry.max_ms = std::max(entry.max_ms, static_cast<double>(elapsed_ms));
    entry.times_ms.push_back(elapsed_ms);
    if (entry.representative_name.empty()) entry.representative_name = event.name;
    entry.kernel = event.kernel;
    entry.last_m = event.m;
    entry.last_k = event.k;
    entry.last_n = event.n;
    entry.device_bytes = event.device_bytes;
    entry.uses_prepack = entry.uses_prepack || event.uses_prepack;
}

bool begin_bonsai_family_event(BonsaiFamilyEvent& event,
                               const char* name,
                               int m,
                               int k,
                               int n,
                               const char* kernel,
                               const BonsaiDeviceLinear* linear,
                               bool uses_prepack,
                               cudaStream_t stream,
                               char* error,
                               size_t error_size) {
    if (!profile_bonsai_int1_families_enabled()) {
        return true;
    }
    record_bonsai_family_call(name, m, k, n, kernel, linear, uses_prepack);
    if (!profile_bonsai_int1_family_events_enabled()) {
        return true;
    }
    const std::string family = bonsai_linear_family(name);
    if (!bonsai_profile_name_matches(family, name)) {
        return true;
    }
    const uint64_t max_calls = profile_bonsai_int1_max_calls();
    const uint64_t attempt = g_family_timed_call_attempts.fetch_add(1) + 1;
    if (max_calls > 0 && attempt > max_calls) {
        return true;
    }
    event.family = family;
    event.name = name != nullptr ? std::string(name) : std::string("unknown");
    event.kernel = kernel != nullptr ? std::string(kernel) : std::string("unknown");
    event.m = m;
    event.k = k;
    event.n = n;
    event.device_bytes = bonsai_linear_device_bytes(linear);
    event.uses_prepack = uses_prepack;
    if (!check_cuda(cudaEventCreate(&event.start), error, error_size, "Bonsai family profile start event create") ||
        !check_cuda(cudaEventCreate(&event.stop), error, error_size, "Bonsai family profile stop event create") ||
        !check_cuda(cudaEventRecord(event.start, stream), error, error_size, "Bonsai family profile start event record")) {
        if (event.start != nullptr) cudaEventDestroy(event.start);
        if (event.stop != nullptr) cudaEventDestroy(event.stop);
        event.start = nullptr;
        event.stop = nullptr;
        return false;
    }
    return true;
}

bool finish_bonsai_family_event(BonsaiFamilyEvent& event, cudaStream_t stream, char* error, size_t error_size) {
    if (event.start == nullptr || event.stop == nullptr) {
        return true;
    }
    float elapsed_ms = 0.0f;
    bool ok = check_cuda(cudaEventRecord(event.stop, stream), error, error_size, "Bonsai family profile stop event record") &&
              check_cuda(cudaEventSynchronize(event.stop), error, error_size, "Bonsai family profile stop event sync") &&
              check_cuda(cudaEventElapsedTime(&elapsed_ms, event.start, event.stop), error, error_size, "Bonsai family profile elapsed");
    if (ok) {
        record_bonsai_family_elapsed(event, elapsed_ms);
    }
    cudaEventDestroy(event.start);
    cudaEventDestroy(event.stop);
    event.start = nullptr;
    event.stop = nullptr;
    return ok;
}

void record_bonsai_profile(const char* name, int m, int k, int n, float elapsed_ms) {
    if (!profile_bonsai_int1_enabled()) {
        return;
    }
    ensure_bonsai_profile_atexit();
    std::lock_guard<std::mutex> lock(g_profile_mutex);
    BonsaiLinearProfileEntry& entry = g_profile_entries[name != nullptr ? name : "unknown"];
    entry.calls++;
    entry.total_ms += elapsed_ms;
    entry.max_ms = std::max(entry.max_ms, static_cast<double>(elapsed_ms));
    entry.last_m = m;
    entry.last_k = k;
    entry.last_n = n;
}

bool create_profile_events(cudaEvent_t* start, cudaEvent_t* stop, char* error, size_t error_size) {
    if (!profile_bonsai_int1_enabled()) {
        *start = nullptr;
        *stop = nullptr;
        return true;
    }
    ensure_bonsai_profile_atexit();
    return check_cuda(cudaEventCreate(start), error, error_size, "Bonsai profile start event create") &&
           check_cuda(cudaEventCreate(stop), error, error_size, "Bonsai profile stop event create");
}

bool finish_profile_events(cudaEvent_t start,
                           cudaEvent_t stop,
                           cudaStream_t stream,
                           const char* name,
                           int m,
                           int k,
                           int n,
                           char* error,
                           size_t error_size) {
    if (start == nullptr || stop == nullptr) {
        return true;
    }
    float elapsed_ms = 0.0f;
    bool ok = check_cuda(cudaEventRecord(stop, stream), error, error_size, "Bonsai profile stop event record") &&
              check_cuda(cudaEventSynchronize(stop), error, error_size, "Bonsai profile stop event sync") &&
              check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), error, error_size, "Bonsai profile elapsed");
    if (ok) {
        record_bonsai_profile(name, m, k, n, elapsed_ms);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ok;
}

void log_stream_counters(const char* where, cudaStream_t stream) {
    if (!trace_bonsai_streams_enabled()) {
        return;
    }
    fprintf(stderr,
            "[BonsaiStream] where=%s stream=%p custom_kernel_launches=%llu bonsai_launches_on_backend_stream=%llu bonsai_device_sync_calls=%llu bonsai_event_sync_calls=%llu bonsai_default_stream_launches=%llu force_device_sync=%s\n",
            where != nullptr ? where : "unknown",
            reinterpret_cast<void*>(stream),
            static_cast<unsigned long long>(g_stream_counters.custom_kernel_launches.load()),
            static_cast<unsigned long long>(g_stream_counters.launches_on_backend_stream.load()),
            static_cast<unsigned long long>(g_stream_counters.device_sync_calls.load()),
            static_cast<unsigned long long>(g_stream_counters.event_sync_calls.load()),
            static_cast<unsigned long long>(g_stream_counters.default_stream_launches.load()),
            force_bonsai_device_sync_enabled() ? "true" : "false");
}

cudaStream_t current_bonsai_backend_stream() {
    return ggml_cuda_get_current_custom_op_stream();
}

std::filesystem::path bonsai_dump_dir() {
    const char* value = std::getenv("SDCPP_BONSAI_DUMP_DIR");
    if (value != nullptr && value[0] != '\0') {
        return std::filesystem::path(value);
    }
    return std::filesystem::path("bonsai-tensor-dumps");
}

std::string sanitize_dump_label(const char* label) {
    std::string out = label != nullptr && label[0] != '\0' ? label : "tensor";
    for (char& ch : out) {
        const bool ok = (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_' || ch == '-';
        if (!ok) {
            ch = '_';
        }
    }
    return out;
}

std::string json_path_string(const std::filesystem::path& path) {
    return path.generic_string();
}

void maybe_log_or_dump_half_tensor(const char* label, const half* dev, size_t count, int64_t dim0, int64_t dim1) {
    if ((!trace_bonsai_int1_stats_enabled() && !dump_bonsai_tensors_enabled()) || dev == nullptr || count == 0) {
        return;
    }
    static int logged = 0;
    if (logged >= bonsai_dump_tensor_limit()) {
        return;
    }
    const int dump_index = logged++;
    cudaStream_t stream = current_bonsai_backend_stream();
    if (stream != nullptr) {
        cudaError_t sync_status = cudaStreamSynchronize(stream);
        if (sync_status != cudaSuccess) {
            fprintf(stderr, "[BonsaiGemLiteINT1] %s stats stream sync failed: %s\n", label, cudaGetErrorString(sync_status));
            return;
        }
    }
    std::vector<half> host(count);
    cudaError_t status = cudaMemcpy(host.data(), dev, count * sizeof(half), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        fprintf(stderr, "[BonsaiGemLiteINT1] %s stats copy failed: %s\n", label, cudaGetErrorString(status));
        return;
    }
    float mn = INFINITY;
    float mx = -INFINITY;
    double sum = 0.0;
    size_t finite = 0;
    size_t nan_count = 0;
    size_t inf_count = 0;
    for (half h : host) {
        const float v = __half2float(h);
        if (std::isnan(v)) {
            nan_count++;
            continue;
        }
        if (!std::isfinite(v)) {
            inf_count++;
            continue;
        }
        mn = std::min(mn, v);
        mx = std::max(mx, v);
        sum += v;
        finite++;
    }
    fprintf(stderr,
            "[BonsaiGemLiteINT1] %s output_stats count=%zu finite=%zu nan=%zu inf=%zu min=%g max=%g mean=%g\n",
            label,
            count,
            finite,
            nan_count,
            inf_count,
            finite > 0 ? mn : 0.0f,
            finite > 0 ? mx : 0.0f,
            finite > 0 ? sum / (double)finite : 0.0);
    if (dump_bonsai_tensors_enabled()) {
        std::error_code ec;
        const std::filesystem::path dir = bonsai_dump_dir();
        std::filesystem::create_directories(dir, ec);
        if (ec) {
            fprintf(stderr, "[BonsaiGemLiteINT1] %s dump create_dir failed: %s\n", label, ec.message().c_str());
            return;
        }
        std::ostringstream stem;
        stem << std::setw(4) << std::setfill('0') << dump_index << "_" << sanitize_dump_label(label);
        const std::filesystem::path bin_path = dir / (stem.str() + ".f16.bin");
        const std::filesystem::path json_path = dir / (stem.str() + ".json");
        {
            std::ofstream out(bin_path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(host.data()), static_cast<std::streamsize>(host.size() * sizeof(half)));
        }
        {
            std::ofstream meta(json_path, std::ios::binary);
            meta << "{\n"
                 << "  \"label\": \"" << sanitize_dump_label(label) << "\",\n"
                 << "  \"dtype\": \"f16\",\n"
                 << "  \"shape\": [" << dim0 << ", " << dim1 << "],\n"
                 << "  \"count\": " << count << ",\n"
                 << "  \"finite\": " << finite << ",\n"
                 << "  \"nan\": " << nan_count << ",\n"
                 << "  \"inf\": " << inf_count << ",\n"
                 << "  \"min\": " << (finite > 0 ? mn : 0.0f) << ",\n"
                 << "  \"max\": " << (finite > 0 ? mx : 0.0f) << ",\n"
                 << "  \"mean\": " << (finite > 0 ? sum / (double)finite : 0.0) << ",\n"
                 << "  \"binary\": \"" << json_path_string(bin_path) << "\"\n"
                 << "}\n";
        }
        fprintf(stderr, "[BonsaiGemLiteINT1] %s tensor_dump path=%s metadata=%s\n",
                label,
                bin_path.string().c_str(),
                json_path.string().c_str());
    }
}

void maybe_log_or_dump_float_tensor(const char* label, const float* dev, size_t count, int64_t dim0, int64_t dim1) {
    if ((!trace_bonsai_int1_stats_enabled() && !dump_bonsai_tensors_enabled()) || dev == nullptr || count == 0) {
        return;
    }
    static int logged = 0;
    if (logged >= bonsai_dump_tensor_limit()) {
        return;
    }
    const int dump_index = logged++;
    cudaStream_t stream = current_bonsai_backend_stream();
    if (stream != nullptr) {
        cudaError_t sync_status = cudaStreamSynchronize(stream);
        if (sync_status != cudaSuccess) {
            fprintf(stderr, "[BonsaiGemLiteINT1] %s stats stream sync failed: %s\n", label, cudaGetErrorString(sync_status));
            return;
        }
    }
    std::vector<float> host(count);
    cudaError_t status = cudaMemcpy(host.data(), dev, count * sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        fprintf(stderr, "[BonsaiGemLiteINT1] %s stats copy failed: %s\n", label, cudaGetErrorString(status));
        return;
    }
    float mn = INFINITY;
    float mx = -INFINITY;
    double sum = 0.0;
    size_t finite = 0;
    size_t nan_count = 0;
    size_t inf_count = 0;
    for (float v : host) {
        if (std::isnan(v)) {
            nan_count++;
            continue;
        }
        if (!std::isfinite(v)) {
            inf_count++;
            continue;
        }
        mn = std::min(mn, v);
        mx = std::max(mx, v);
        sum += v;
        finite++;
    }
    fprintf(stderr,
            "[BonsaiGemLiteINT1] %s output_stats count=%zu finite=%zu nan=%zu inf=%zu min=%g max=%g mean=%g\n",
            label,
            count,
            finite,
            nan_count,
            inf_count,
            finite > 0 ? mn : 0.0f,
            finite > 0 ? mx : 0.0f,
            finite > 0 ? sum / (double)finite : 0.0);
    if (dump_bonsai_tensors_enabled()) {
        std::error_code ec;
        const std::filesystem::path dir = bonsai_dump_dir();
        std::filesystem::create_directories(dir, ec);
        if (ec) {
            fprintf(stderr, "[BonsaiGemLiteINT1] %s dump create_dir failed: %s\n", label, ec.message().c_str());
            return;
        }
        std::ostringstream stem;
        stem << "dbg_" << std::setw(4) << std::setfill('0') << dump_index << "_" << sanitize_dump_label(label);
        const std::filesystem::path bin_path = dir / (stem.str() + ".f32.bin");
        const std::filesystem::path json_path = dir / (stem.str() + ".json");
        {
            std::ofstream out(bin_path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(host.data()), static_cast<std::streamsize>(host.size() * sizeof(float)));
        }
        {
            std::ofstream meta(json_path, std::ios::binary);
            meta << "{\n"
                 << "  \"label\": \"" << sanitize_dump_label(label) << "\",\n"
                 << "  \"dtype\": \"f32\",\n"
                 << "  \"shape\": [" << dim0 << ", " << dim1 << "],\n"
                 << "  \"count\": " << count << ",\n"
                 << "  \"finite\": " << finite << ",\n"
                 << "  \"nan\": " << nan_count << ",\n"
                 << "  \"inf\": " << inf_count << ",\n"
                 << "  \"min\": " << (finite > 0 ? mn : 0.0f) << ",\n"
                 << "  \"max\": " << (finite > 0 ? mx : 0.0f) << ",\n"
                 << "  \"mean\": " << (finite > 0 ? sum / (double)finite : 0.0) << ",\n"
                 << "  \"binary\": \"" << json_path_string(bin_path) << "\"\n"
                 << "}\n";
        }
        fprintf(stderr, "[BonsaiGemLiteINT1] %s tensor_dump path=%s metadata=%s\n",
                label,
                bin_path.string().c_str(),
                json_path.string().c_str());
    }
}

}  // namespace

extern "C" bool sd_bonsai_gemlite_int1_device_linear_create(const uint8_t* wq,
                                                             size_t wq_bytes,
                                                             const float* scales,
                                                             const float* zeros,
                                                             int k,
                                                             int n,
                                                             void** out,
                                                             uint64_t* device_bytes,
                                                             char* error,
                                                             size_t error_size) {
    if (out == nullptr || wq == nullptr || scales == nullptr || zeros == nullptr || k <= 0 || n <= 0) {
        set_error(error, error_size, "invalid Bonsai device linear create arguments");
        return false;
    }
    if ((k % 128) != 0 || wq_bytes != static_cast<size_t>(k / 8) * static_cast<size_t>(n)) {
        set_error(error, error_size, "unsupported Bonsai device linear shape");
        return false;
    }
    BonsaiDeviceLinear* linear = new BonsaiDeviceLinear();
    linear->k = k;
    linear->n = n;
    linear->wq_bytes = wq_bytes;
    linear->meta_bytes = static_cast<size_t>(k / 128) * static_cast<size_t>(n) * sizeof(float);
    bool ok = true;
    ok = ok && check_cuda(cudaMalloc(&linear->wq, linear->wq_bytes), error, error_size, "cudaMalloc Bonsai W_q");
    ok = ok && check_cuda(cudaMalloc(&linear->scales, linear->meta_bytes), error, error_size, "cudaMalloc Bonsai scales");
    ok = ok && check_cuda(cudaMalloc(&linear->zeros, linear->meta_bytes), error, error_size, "cudaMalloc Bonsai zeros");
    ok = ok && check_cuda(cudaMemcpy(linear->wq, wq, linear->wq_bytes, cudaMemcpyHostToDevice), error, error_size, "copy Bonsai W_q");
    ok = ok && check_cuda(cudaMemcpy(linear->scales, scales, linear->meta_bytes, cudaMemcpyHostToDevice), error, error_size, "copy Bonsai scales");
    ok = ok && check_cuda(cudaMemcpy(linear->zeros, zeros, linear->meta_bytes, cudaMemcpyHostToDevice), error, error_size, "copy Bonsai zeros");
    if (ok && bonsai_tile_major_prepack_enabled() && k == 3072 && n == 27648 && (k % 128) == 0 && (n % 128) == 0) {
        const int groups = k / 128;
        const int n_tiles = n / 128;
        constexpr int bytes_per_group = 16;
        const size_t tile_wq_bytes = static_cast<size_t>(groups) * static_cast<size_t>(n_tiles) * bytes_per_group * 128ull;
        const size_t tile_meta_elems = static_cast<size_t>(groups) * static_cast<size_t>(n_tiles) * 128ull;
        const size_t tile_meta_bytes = tile_meta_elems * sizeof(float);
        std::vector<uint8_t> host_wq_tile(tile_wq_bytes);
        std::vector<float> host_scales_tile(tile_meta_elems);
        std::vector<float> host_zeros_tile(tile_meta_elems);
        for (int group = 0; group < groups; ++group) {
            for (int tile_n = 0; tile_n < n_tiles; ++tile_n) {
                const int col_base = tile_n * 128;
                const size_t packed_tile_base = (static_cast<size_t>(group) * static_cast<size_t>(n_tiles) + static_cast<size_t>(tile_n)) *
                                                bytes_per_group * 128ull;
                const size_t meta_tile_base = (static_cast<size_t>(group) * static_cast<size_t>(n_tiles) + static_cast<size_t>(tile_n)) * 128ull;
                for (int byte_idx = 0; byte_idx < bytes_per_group; ++byte_idx) {
                    const size_t raw_row = static_cast<size_t>(group * bytes_per_group + byte_idx) * static_cast<size_t>(n);
                    const size_t tile_row = packed_tile_base + static_cast<size_t>(byte_idx) * 128ull;
                    std::memcpy(host_wq_tile.data() + tile_row, wq + raw_row + static_cast<size_t>(col_base), 128);
                }
                const size_t raw_meta = static_cast<size_t>(group) * static_cast<size_t>(n) + static_cast<size_t>(col_base);
                std::memcpy(host_scales_tile.data() + meta_tile_base, scales + raw_meta, 128 * sizeof(float));
                std::memcpy(host_zeros_tile.data() + meta_tile_base, zeros + raw_meta, 128 * sizeof(float));
            }
        }
        ok = ok && check_cuda(cudaMalloc(&linear->wq_tile, tile_wq_bytes), error, error_size, "cudaMalloc Bonsai tile-major W_q");
        ok = ok && check_cuda(cudaMalloc(&linear->scales_tile, tile_meta_bytes), error, error_size, "cudaMalloc Bonsai tile-major scales");
        ok = ok && check_cuda(cudaMalloc(&linear->zeros_tile, tile_meta_bytes), error, error_size, "cudaMalloc Bonsai tile-major zeros");
        ok = ok && check_cuda(cudaMemcpy(linear->wq_tile, host_wq_tile.data(), tile_wq_bytes, cudaMemcpyHostToDevice), error, error_size, "copy Bonsai tile-major W_q");
        ok = ok && check_cuda(cudaMemcpy(linear->scales_tile, host_scales_tile.data(), tile_meta_bytes, cudaMemcpyHostToDevice), error, error_size, "copy Bonsai tile-major scales");
        ok = ok && check_cuda(cudaMemcpy(linear->zeros_tile, host_zeros_tile.data(), tile_meta_bytes, cudaMemcpyHostToDevice), error, error_size, "copy Bonsai tile-major zeros");
        if (ok) {
            linear->has_tile_major = true;
            linear->n_tiles = n_tiles;
            linear->tile_bytes = tile_wq_bytes + 2 * tile_meta_bytes;
        }
    }
    if (!ok) {
        cudaFree(linear->wq);
        cudaFree(linear->scales);
        cudaFree(linear->zeros);
        cudaFree(linear->wq_tile);
        cudaFree(linear->scales_tile);
        cudaFree(linear->zeros_tile);
        delete linear;
        return false;
    }
    *out = linear;
    if (device_bytes != nullptr) {
        *device_bytes = static_cast<uint64_t>(linear->wq_bytes + 2 * linear->meta_bytes + linear->tile_bytes);
    }
    return true;
}

extern "C" void sd_bonsai_gemlite_int1_device_linear_destroy(void* ptr) {
    BonsaiDeviceLinear* linear = reinterpret_cast<BonsaiDeviceLinear*>(ptr);
    if (linear == nullptr) {
        return;
    }
    cudaFree(linear->wq);
    cudaFree(linear->scales);
    cudaFree(linear->zeros);
    cudaFree(linear->wq_tile);
    cudaFree(linear->scales_tile);
    cudaFree(linear->zeros_tile);
    delete linear;
}

extern "C" bool sd_bonsai_gemlite_int1_cuda_forward(void* ptr,
                                                     const void* a_dev,
                                                     void* c_dev,
                                                     int m,
                                                     int k,
                                                     int n,
                                                     const char* debug_name,
                                                     char* error,
                                                     size_t error_size) {
    BonsaiDeviceLinear* linear = reinterpret_cast<BonsaiDeviceLinear*>(ptr);
    if (linear == nullptr || a_dev == nullptr || c_dev == nullptr) {
        set_error(error, error_size, "invalid Bonsai INT1 forward arguments");
        return false;
    }
    if (linear->k != k || linear->n != n) {
        set_error(error, error_size, "Bonsai INT1 forward shape mismatch");
        return false;
    }
    cudaStream_t stream = current_bonsai_backend_stream();
    if (stream == nullptr && !force_bonsai_device_sync_enabled()) {
        set_error(error, error_size, "Bonsai INT1 custom op has no active ggml CUDA stream");
        g_stream_counters.default_stream_launches.fetch_add(1);
        log_stream_counters("linear_missing_backend_stream", stream);
        return false;
    }
    if (!maybe_force_device_sync(error, error_size, "Bonsai INT1 pre-kernel synchronize")) {
        return false;
    }
    if (dump_bonsai_tensors_enabled()) {
        cudaMemsetAsync(c_dev, 0, static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(half), stream);
    }
    const char* kernel_label = bonsai_selected_kernel_label(linear, m, k, n, false);
    const bool uses_prepack = std::strstr(kernel_label, "prepack") != nullptr;
    cudaEvent_t profile_start = nullptr;
    cudaEvent_t profile_stop = nullptr;
    if (!create_profile_events(&profile_start, &profile_stop, error, error_size)) {
        return false;
    }
    if (profile_start != nullptr && !check_cuda(cudaEventRecord(profile_start, stream), error, error_size, "Bonsai profile start event record")) {
        cudaEventDestroy(profile_start);
        cudaEventDestroy(profile_stop);
        return false;
    }
    BonsaiFamilyEvent family_event;
    if (!begin_bonsai_family_event(family_event, debug_name, m, k, n, kernel_label, linear, uses_prepack, stream, error, error_size)) {
        if (profile_start != nullptr) {
            cudaEventDestroy(profile_start);
            cudaEventDestroy(profile_stop);
        }
        return false;
    }
    if (gemlite_tc_gemm_v2_dequant_only_enabled() && k == 3072 && n == 27648 && (m % 64) == 0 && (n % 128) == 0 && (k % 128) == 0) {
        const dim3 block(128);
        const dim3 grid(n / 128, m / 64);
        bonsai_int1_linear_gemlite_tc_gemm_v2_dequant_only_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                              linear->wq,
                                                                                              linear->scales,
                                                                                              linear->zeros,
                                                                                              reinterpret_cast<half*>(c_dev),
                                                                                              m,
                                                                                              k,
                                                                                              n);
    } else if (gemlite_linear1_largesmem_kernel_enabled() && k == 3072 && n == 27648 && (m % 64) == 0 && (n % 128) == 0 && (k % 128) == 0) {
        const dim3 block(128);
        const dim3 grid(n / 128, m / 64);
        constexpr size_t shared_bytes = (64 * 128 + 128 * 128) * sizeof(half);
        (void)cudaFuncSetAttribute(bonsai_int1_linear_gemlite_tc_linear1_largesmem_kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(shared_bytes));
        bonsai_int1_linear_gemlite_tc_linear1_largesmem_kernel<<<grid, block, shared_bytes, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                                      linear->wq,
                                                                                                      linear->scales,
                                                                                                      linear->zeros,
                                                                                                      reinterpret_cast<half*>(c_dev),
                                                                                                      m,
                                                                                                      k,
                                                                                                      n);
    } else if (gemlite_linear2_ldmatrix_kernel_enabled() && k == 12288 && n == 3072 && (m % 64) == 0 && (n % 32) == 0 && (k % 128) == 0) {
        const dim3 block(128);
        const dim3 grid(n / 32, m / 64);
        bonsai_int1_linear_gemlite_tc_linear2_ldmatrix_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                         linear->wq,
                                                                                         linear->scales,
                                                                                         linear->zeros,
                                                                                         reinterpret_cast<half*>(c_dev),
                                                                                         m,
                                                                                         k,
                                                                                         n);
    } else if (gemlite_linear2_async_b_kernel_enabled() && k == 12288 && n == 3072 && (m % 64) == 0 && (n % 32) == 0 && (k % 128) == 0) {
        const dim3 block(128);
        const dim3 grid(n / 32, m / 64);
        constexpr size_t shared_bytes = 49664;
        (void)cudaFuncSetAttribute(bonsai_int1_linear_gemlite_tc_linear2_async_b_kernel,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   static_cast<int>(shared_bytes));
        bonsai_int1_linear_gemlite_tc_linear2_async_b_kernel<<<grid, block, shared_bytes, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                                    linear->wq,
                                                                                                    linear->scales,
                                                                                                    linear->zeros,
                                                                                                    reinterpret_cast<half*>(c_dev),
                                                                                                    m,
                                                                                                    k,
                                                                                                    n);
    } else if (gemlite_linear2_async_kernel_enabled() && k == 12288 && n == 3072 && (m % 64) == 0 && (n % 32) == 0 && (k % 128) == 0) {
        const dim3 block(128);
        const dim3 grid(n / 32, m / 64);
        bonsai_int1_linear_gemlite_tc_linear2_async_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                      linear->wq,
                                                                                      linear->scales,
                                                                                      linear->zeros,
                                                                                      reinterpret_cast<half*>(c_dev),
                                                                                      m,
                                                                                      k,
                                                                                      n);
    } else if (gemlite_tc_linear2_kernel_enabled() && k == 12288 && n == 3072 && (m % 64) == 0 && (n % 32) == 0 && (k % 128) == 0) {
        const dim3 block(128);
        const dim3 grid(n / 32, m / 64);
        bonsai_int1_linear_gemlite_tc_linear2_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                linear->wq,
                                                                                linear->scales,
                                                                                linear->zeros,
                                                                                reinterpret_cast<half*>(c_dev),
                                                                                m,
                                                                                k,
                                                                                n);
    } else if (gemlite_tc_img_mlp0_kernel_enabled() && k == 3072 && n == 18432 && (m % 64) == 0 && (n % 128) == 0 && (k % 128) == 0) {
        const dim3 block(128);
        const dim3 grid(n / 128, m / 64);
        bonsai_int1_linear_gemlite_tc_gemm_v2_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                 linear->wq,
                                                                                 linear->scales,
                                                                                 linear->zeros,
                                                                                 reinterpret_cast<half*>(c_dev),
                                                                                 m,
                                                                                 k,
                                                                                 n);
    } else if (gemlite_tc_img_mlp2_kernel_enabled() && k == 9216 && n == 3072 && (m % 64) == 0 && (n % 128) == 0 && (k % 128) == 0) {
        const dim3 block(128);
        const dim3 grid(n / 128, m / 64);
        bonsai_int1_linear_gemlite_tc_gemm_v2_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                 linear->wq,
                                                                                 linear->scales,
                                                                                 linear->zeros,
                                                                                 reinterpret_cast<half*>(c_dev),
                                                                                 m,
                                                                                 k,
                                                                                 n);
    } else if (gemlite_tc_attn_proj_kernel_enabled() && k == 3072 && n == 3072 && m == 512 && (n % 64) == 0) {
        const dim3 block(128);
        const dim3 grid(n / 64, m / 64);
        bonsai_int1_linear_gemlite_tc_attn_proj_txt_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                      linear->wq,
                                                                                      linear->scales,
                                                                                      linear->zeros,
                                                                                      reinterpret_cast<half*>(c_dev),
                                                                                      m,
                                                                                      k,
                                                                                      n);
    } else if (gemlite_tc_attn_proj_kernel_enabled() && k == 3072 && n == 3072 && (m % 64) == 0 && (n % 128) == 0) {
        const dim3 block(128);
        const dim3 grid(n / 128, m / 64);
        bonsai_int1_linear_gemlite_tc_gemm_v2_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                 linear->wq,
                                                                                 linear->scales,
                                                                                 linear->zeros,
                                                                                 reinterpret_cast<half*>(c_dev),
                                                                                 m,
                                                                                 k,
                                                                                 n);
    } else if (gemlite_tc_gemm_v2_mma_only_enabled() && k == 3072 && n == 27648 && (m % 64) == 0 && (n % 128) == 0 && (k % 128) == 0) {
        const dim3 block(128);
        const dim3 grid(n / 128, m / 64);
        bonsai_int1_linear_gemlite_tc_gemm_v2_mma_only_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                         reinterpret_cast<half*>(c_dev),
                                                                                         m,
                                                                                         k,
                                                                                         n);
    } else if (bonsai_tile_major_prepack_enabled() && k == 3072 && n == 27648 && (m % 64) == 0 && (n % 128) == 0 && (k % 128) == 0) {
        if (!linear->has_tile_major) {
            if (profile_start != nullptr) {
                cudaEventDestroy(profile_start);
                cudaEventDestroy(profile_stop);
            }
            set_error(error, error_size, "Bonsai tile-major prepack requested but linear has no tile-major buffers");
            return false;
        }
        const dim3 block(128);
        const dim3 grid(n / 128, m / 64);
        const int prepack_variant = bonsai_tile_major_prepack_variant();
        if (linear1_bankfix_kernel_enabled()) {
            bonsai_int1_linear_gemlite_tc_gemm_v2_prepack_bankfix_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                                    linear->wq_tile,
                                                                                                    linear->scales_tile,
                                                                                                    linear->zeros_tile,
                                                                                                    reinterpret_cast<half*>(c_dev),
                                                                                                    m,
                                                                                                    k,
                                                                                                    n,
                                                                                                    linear->n_tiles);
        } else if (prepack_variant == 1) {
            bonsai_int1_linear_gemlite_tc_gemm_v2_prepack_wq_only_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                                    linear->wq_tile,
                                                                                                    linear->scales,
                                                                                                    linear->zeros,
                                                                                                    reinterpret_cast<half*>(c_dev),
                                                                                                    m,
                                                                                                    k,
                                                                                                    n,
                                                                                                    linear->n_tiles);
        } else if (prepack_variant == 3) {
            bonsai_int1_linear_gemlite_tc_gemm_v2_prepack_padded_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                                   linear->wq_tile,
                                                                                                   linear->scales_tile,
                                                                                                   linear->zeros_tile,
                                                                                                   reinterpret_cast<half*>(c_dev),
                                                                                                   m,
                                                                                                   k,
                                                                                                   n,
                                                                                                   linear->n_tiles);
        } else {
            bonsai_int1_linear_gemlite_tc_gemm_v2_prepack_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                            linear->wq_tile,
                                                                                            linear->scales_tile,
                                                                                            linear->zeros_tile,
                                                                                            reinterpret_cast<half*>(c_dev),
                                                                                            m,
                                                                                            k,
                                                                                            n,
                                                                                            linear->n_tiles);
        }
    } else if (gemlite_tc_gemm_v2_kernel_enabled() && k == 3072 && n == 27648 && (m % 64) == 0 && (n % 128) == 0 && (k % 128) == 0) {
        const dim3 block(128);
        const dim3 grid(n / 128, m / 64);
        bonsai_int1_linear_gemlite_tc_gemm_v2_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                 linear->wq,
                                                                                 linear->scales,
                                                                                 linear->zeros,
                                                                                 reinterpret_cast<half*>(c_dev),
                                                                                 m,
                                                                                 k,
                                                                                 n);
    } else if (gemlite_tc_gemm_kernel_enabled() && k == 3072 && n == 27648 && (m % 64) == 0 && (n % 128) == 0 && (k % 128) == 0) {
        const dim3 block(128);
        const dim3 grid(n / 128, m / 64);
        bonsai_int1_linear_gemlite_tc_gemm_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                              linear->wq,
                                                                              linear->scales,
                                                                              linear->zeros,
                                                                              reinterpret_cast<half*>(c_dev),
                                                                              m,
                                                                              k,
                                                                              n);
    } else if (gemlite_gemm_v2_kernel_enabled() && k == 3072 && n == 27648 && (k % 128) == 0) {
        const int variant = gemlite_gemm_v2_variant();
        if (variant == 11) {
            constexpr int BM = 4;
            constexpr int BN = 128;
            constexpr int COLS_PER_THREAD = 8;
            constexpr int SPLITS = 4;
            const size_t partial_bytes = static_cast<size_t>(SPLITS) * static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(float);
            if (!ensure_splitk_partial(partial_bytes, error, error_size)) {
                if (profile_start != nullptr) {
                    cudaEventDestroy(profile_start);
                    cudaEventDestroy(profile_stop);
                }
                return false;
            }
            const dim3 block(BN / COLS_PER_THREAD, BM);
            const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM, SPLITS);
            bonsai_int1_linear_gemlite_splitk_partial_kernel<BM, BN, COLS_PER_THREAD, SPLITS><<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                                                        linear->wq,
                                                                                                                        linear->scales,
                                                                                                                        g_splitk_partial,
                                                                                                                        m,
                                                                                                                        k,
                                                                                                                        n);
            const int total = m * n;
            const int reduce_threads = 256;
            const int reduce_blocks = (total + reduce_threads - 1) / reduce_threads;
            bonsai_int1_linear_gemlite_splitk_reduce_kernel<SPLITS><<<reduce_blocks, reduce_threads, 0, stream>>>(g_splitk_partial,
                                                                                                                  reinterpret_cast<half*>(c_dev),
                                                                                                                  m,
                                                                                                                  n);
        } else if (variant == 10) {
            constexpr int BM = 8;
            constexpr int BN = 128;
            constexpr int COLS_PER_THREAD = 8;
            const dim3 block(BN / COLS_PER_THREAD, BM);
            const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
            bonsai_int1_linear_gemlite_scale_only_kernel<BM, BN, COLS_PER_THREAD><<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                                             linear->wq,
                                                                                                             linear->scales,
                                                                                                             reinterpret_cast<half*>(c_dev),
                                                                                                             m,
                                                                                                             k,
                                                                                                             n);
        } else if (variant == 9) {
            constexpr int BM = 4;
            constexpr int BN = 128;
            constexpr int COLS_PER_THREAD = 8;
            const dim3 block(BN / COLS_PER_THREAD, BM);
            const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
            bonsai_int1_linear_gemlite_scale_only_kernel<BM, BN, COLS_PER_THREAD><<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                                             linear->wq,
                                                                                                             linear->scales,
                                                                                                             reinterpret_cast<half*>(c_dev),
                                                                                                             m,
                                                                                                             k,
                                                                                                             n);
        } else if (variant == 8) {
            constexpr int BM = 4;
            constexpr int BN = 256;
            constexpr int COLS_PER_THREAD = 16;
            const dim3 block(BN / COLS_PER_THREAD, BM);
            const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
            bonsai_int1_linear_gemlite_gemm_v2_kernel<BM, BN, COLS_PER_THREAD><<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                                           linear->wq,
                                                                                                           linear->scales,
                                                                                                           linear->zeros,
                                                                                                           reinterpret_cast<half*>(c_dev),
                                                                                                           m,
                                                                                                           k,
                                                                                                           n);
        } else if (variant == 7) {
            constexpr int BM = 8;
            constexpr int BN = 256;
            constexpr int COLS_PER_THREAD = 8;
            const dim3 block(BN / COLS_PER_THREAD, BM);
            const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
            bonsai_int1_linear_gemlite_gemm_v2_kernel<BM, BN, COLS_PER_THREAD><<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                                          linear->wq,
                                                                                                          linear->scales,
                                                                                                          linear->zeros,
                                                                                                          reinterpret_cast<half*>(c_dev),
                                                                                                          m,
                                                                                                          k,
                                                                                                          n);
        } else if (variant == 6) {
            constexpr int BM = 16;
            constexpr int BN = 128;
            constexpr int COLS_PER_THREAD = 8;
            const dim3 block(BN / COLS_PER_THREAD, BM);
            const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
            bonsai_int1_linear_gemlite_gemm_v2_kernel<BM, BN, COLS_PER_THREAD><<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                                          linear->wq,
                                                                                                          linear->scales,
                                                                                                          linear->zeros,
                                                                                                          reinterpret_cast<half*>(c_dev),
                                                                                                          m,
                                                                                                          k,
                                                                                                          n);
        } else if (variant == 5) {
            constexpr int BM = 8;
            constexpr int BN = 128;
            constexpr int COLS_PER_THREAD = 16;
            const dim3 block(BN / COLS_PER_THREAD, BM);
            const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
            bonsai_int1_linear_gemlite_gemm_v2_kernel<BM, BN, COLS_PER_THREAD><<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                                           linear->wq,
                                                                                                           linear->scales,
                                                                                                           linear->zeros,
                                                                                                           reinterpret_cast<half*>(c_dev),
                                                                                                           m,
                                                                                                           k,
                                                                                                           n);
        } else if (variant == 4) {
            constexpr int BM = 4;
            constexpr int BN = 128;
            constexpr int COLS_PER_THREAD = 16;
            const dim3 block(BN / COLS_PER_THREAD, BM);
            const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
            bonsai_int1_linear_gemlite_gemm_v2_kernel<BM, BN, COLS_PER_THREAD><<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                                           linear->wq,
                                                                                                           linear->scales,
                                                                                                           linear->zeros,
                                                                                                           reinterpret_cast<half*>(c_dev),
                                                                                                           m,
                                                                                                           k,
                                                                                                           n);
        } else if (variant == 3) {
            constexpr int BM = 8;
            constexpr int BN = 128;
            constexpr int COLS_PER_THREAD = 8;
            const dim3 block(BN / COLS_PER_THREAD, BM);
            const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
            bonsai_int1_linear_gemlite_gemm_v2_kernel<BM, BN, COLS_PER_THREAD><<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                                          linear->wq,
                                                                                                          linear->scales,
                                                                                                          linear->zeros,
                                                                                                          reinterpret_cast<half*>(c_dev),
                                                                                                          m,
                                                                                                          k,
                                                                                                          n);
        } else if (variant == 2) {
            constexpr int BM = 4;
            constexpr int BN = 128;
            constexpr int COLS_PER_THREAD = 8;
            const dim3 block(BN / COLS_PER_THREAD, BM);
            const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
            bonsai_int1_linear_gemlite_gemm_v2_kernel<BM, BN, COLS_PER_THREAD><<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                                          linear->wq,
                                                                                                          linear->scales,
                                                                                                          linear->zeros,
                                                                                                          reinterpret_cast<half*>(c_dev),
                                                                                                          m,
                                                                                                          k,
                                                                                                          n);
        } else {
            constexpr int BM = 4;
            constexpr int BN = 64;
            constexpr int COLS_PER_THREAD = 4;
            const dim3 block(BN / COLS_PER_THREAD, BM);
            const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
            bonsai_int1_linear_gemlite_gemm_v2_kernel<BM, BN, COLS_PER_THREAD><<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                                          linear->wq,
                                                                                                          linear->scales,
                                                                                                          linear->zeros,
                                                                                                          reinterpret_cast<half*>(c_dev),
                                                                                                          m,
                                                                                                          k,
                                                                                                          n);
        }
    } else if (gemlite_shape_kernel_enabled() && k == 3072 && n == 27648 && (k % 128) == 0) {
        const dim3 block(32, 4);
        const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
        bonsai_int1_linear_gemlite_shape_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                            linear->wq,
                                                                            linear->scales,
                                                                            linear->zeros,
                                                                            reinterpret_cast<half*>(c_dev),
                                                                            m,
                                                                            k,
                                                                            n);
    } else if (!tiled_bonsai_int1_kernel_enabled()) {
        const dim3 block(16, 16);
        const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
        bonsai_int1_linear_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                              linear->wq,
                                                              linear->scales,
                                                              linear->zeros,
                                                              reinterpret_cast<half*>(c_dev),
                                                              m,
                                                              k,
                                                              n);
    } else {
        constexpr int BM = 16;
        constexpr int BN = 16;
        constexpr int BK = 128;
        const dim3 block(BN, BM);
        const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
        bonsai_int1_linear_tiled_kernel<BM, BN, BK><<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                linear->wq,
                                                                                linear->scales,
                                                                                linear->zeros,
                                                                                reinterpret_cast<half*>(c_dev),
                                                                                m,
                                                                                k,
                                                                                n);
    }
    g_stream_counters.custom_kernel_launches.fetch_add(1);
    if (stream != nullptr) {
        g_stream_counters.launches_on_backend_stream.fetch_add(1);
    } else {
        g_stream_counters.default_stream_launches.fetch_add(1);
    }
    log_stream_counters("linear_launch", stream);
    return check_cuda(cudaGetLastError(), error, error_size, "Bonsai INT1 runtime kernel launch") &&
           finish_bonsai_family_event(family_event, stream, error, error_size) &&
           finish_profile_events(profile_start, profile_stop, stream, debug_name, m, k, n, error, error_size) &&
           maybe_force_device_sync(error, error_size, "Bonsai INT1 post-kernel synchronize");
}

static bool bonsai_forward_strided(void* ptr,
                                   const void* a_dev,
                                   void* c_dev,
                                   int m,
                                   int k,
                                   int n,
                                   int c_stride,
                                   int c_col_offset,
                                   const char* debug_name,
                                   char* error,
                                   size_t error_size) {
    BonsaiDeviceLinear* linear = reinterpret_cast<BonsaiDeviceLinear*>(ptr);
    if (linear == nullptr || a_dev == nullptr || c_dev == nullptr) {
        set_error(error, error_size, "invalid Bonsai INT1 strided forward arguments");
        return false;
    }
    if (linear->k != k || linear->n != n || c_stride < n || c_col_offset < 0 || c_col_offset + n > c_stride) {
        set_error(error, error_size, "Bonsai INT1 strided forward shape mismatch");
        return false;
    }
    cudaStream_t stream = current_bonsai_backend_stream();
    if (stream == nullptr && !force_bonsai_device_sync_enabled()) {
        set_error(error, error_size, "Bonsai INT1 QKV custom op has no active ggml CUDA stream");
        g_stream_counters.default_stream_launches.fetch_add(1);
        log_stream_counters("qkv_missing_backend_stream", stream);
        return false;
    }
    if (!maybe_force_device_sync(error, error_size, "Bonsai INT1 strided pre-kernel synchronize")) {
        return false;
    }
    const char* kernel_label = bonsai_selected_kernel_label(linear, m, k, n, true);
    cudaEvent_t profile_start = nullptr;
    cudaEvent_t profile_stop = nullptr;
    if (!create_profile_events(&profile_start, &profile_stop, error, error_size)) {
        return false;
    }
    if (profile_start != nullptr && !check_cuda(cudaEventRecord(profile_start, stream), error, error_size, "Bonsai profile strided start event record")) {
        cudaEventDestroy(profile_start);
        cudaEventDestroy(profile_stop);
        return false;
    }
    BonsaiFamilyEvent family_event;
    if (!begin_bonsai_family_event(family_event, debug_name, m, k, n, kernel_label, linear, false, stream, error, error_size)) {
        if (profile_start != nullptr) {
            cudaEventDestroy(profile_start);
            cudaEventDestroy(profile_stop);
        }
        return false;
    }
    if (gemlite_tc_qkv_kernel_enabled() && k == 3072 && n == 3072 && (m % 64) == 0 && (n % 128) == 0 && (k % 128) == 0) {
        const dim3 block(128);
        const dim3 grid(n / 128, m / 64);
        bonsai_int1_linear_gemlite_tc_gemm_v2_strided_out_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                            linear->wq,
                                                                                            linear->scales,
                                                                                            linear->zeros,
                                                                                            reinterpret_cast<half*>(c_dev),
                                                                                            m,
                                                                                            k,
                                                                                            n,
                                                                                            c_stride,
                                                                                            c_col_offset);
    } else if (!tiled_bonsai_int1_kernel_enabled()) {
        const dim3 block(16, 16);
        const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
        bonsai_int1_linear_strided_out_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                          linear->wq,
                                                                          linear->scales,
                                                                          linear->zeros,
                                                                          reinterpret_cast<half*>(c_dev),
                                                                          m,
                                                                          k,
                                                                          n,
                                                                          c_stride,
                                                                          c_col_offset);
    } else {
        constexpr int BM = 16;
        constexpr int BN = 16;
        constexpr int BK = 128;
        const dim3 block(BN, BM);
        const dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
        bonsai_int1_linear_tiled_strided_out_kernel<BM, BN, BK><<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(a_dev),
                                                                                            linear->wq,
                                                                                            linear->scales,
                                                                                            linear->zeros,
                                                                                            reinterpret_cast<half*>(c_dev),
                                                                                            m,
                                                                                            k,
                                                                                            n,
                                                                                            c_stride,
                                                                                            c_col_offset);
    }
    g_stream_counters.custom_kernel_launches.fetch_add(1);
    if (stream != nullptr) {
        g_stream_counters.launches_on_backend_stream.fetch_add(1);
    } else {
        g_stream_counters.default_stream_launches.fetch_add(1);
    }
    log_stream_counters("qkv_strided_launch", stream);
    return check_cuda(cudaGetLastError(), error, error_size, "Bonsai INT1 strided runtime kernel launch") &&
           finish_bonsai_family_event(family_event, stream, error, error_size) &&
           finish_profile_events(profile_start, profile_stop, stream, debug_name, m, k, n, error, error_size) &&
           maybe_force_device_sync(error, error_size, "Bonsai INT1 strided post-kernel synchronize");
}

extern "C" void sd_bonsai_gemlite_int1_ggml_custom_forward(ggml_tensor* dst, int ith, int nth, void* userdata) {
    if (ith != 0) {
        return;
    }
    (void)nth;
    if (dst == nullptr || dst->src[0] == nullptr) {
        return;
    }
    const ggml_tensor* x = dst->src[0];
    sd::BonsaiGemliteLinearCustomUserdata* custom = reinterpret_cast<sd::BonsaiGemliteLinearCustomUserdata*>(userdata);
    void* device_linear = custom != nullptr ? custom->device_linear : userdata;
    const char* debug_name = custom != nullptr && custom->debug_name != nullptr ? custom->debug_name : "linear";
    const int k = static_cast<int>(x->ne[0]);
    const int m = static_cast<int>(ggml_nelements(x) / k);
    const int n = static_cast<int>(dst->ne[0]);
    if (trace_bonsai_int1_stats_enabled() || dump_bonsai_tensors_enabled()) {
        fprintf(stderr,
                "[BonsaiGemLiteINT1] %s ptrs input=%p output=%p m=%d k=%d n=%d x_nb0=%zu x_nb1=%zu dst_nb0=%zu dst_nb1=%zu\n",
                debug_name,
                x->data,
                dst->data,
                m,
                k,
                n,
                static_cast<size_t>(x->nb[0]),
                static_cast<size_t>(x->nb[1]),
                static_cast<size_t>(dst->nb[0]),
                static_cast<size_t>(dst->nb[1]));
        BonsaiDeviceLinear* linear = reinterpret_cast<BonsaiDeviceLinear*>(device_linear);
        if (linear != nullptr && std::strstr(debug_name, "double_blocks.0.txt_mlp.2.weight") != nullptr) {
            const size_t meta_count = static_cast<size_t>(linear->k / 128) * static_cast<size_t>(linear->n);
            maybe_log_or_dump_float_tensor("block00.double.txt_mlp2.scales_device", linear->scales, meta_count, linear->k / 128, linear->n);
            maybe_log_or_dump_float_tensor("block00.double.txt_mlp2.zeros_device", linear->zeros, meta_count, linear->k / 128, linear->n);
        }
    }
    char error[256] = {};
    if (!sd_bonsai_gemlite_int1_cuda_forward(device_linear, x->data, dst->data, m, k, n, debug_name, error, sizeof(error))) {
        fprintf(stderr, "Bonsai INT1 custom op failed: %s\n", error);
        return;
    }
    std::string input_label = std::string(debug_name) + ".input";
    std::string output_label = std::string(debug_name) + ".output";
    maybe_log_or_dump_half_tensor(input_label.c_str(), reinterpret_cast<const half*>(x->data), static_cast<size_t>(m) * static_cast<size_t>(k), m, k);
    maybe_log_or_dump_half_tensor(output_label.c_str(), reinterpret_cast<const half*>(dst->data), static_cast<size_t>(m) * static_cast<size_t>(n), m, n);
}

extern "C" void sd_bonsai_gemlite_int1_ggml_custom_qkv_forward(ggml_tensor* dst, int ith, int nth, void* userdata) {
    if (ith != 0) {
        return;
    }
    (void)nth;
    if (dst == nullptr || dst->src[0] == nullptr || userdata == nullptr) {
        return;
    }
    const ggml_tensor* x = dst->src[0];
    sd::BonsaiGemliteQkvCustomUserdata* custom = reinterpret_cast<sd::BonsaiGemliteQkvCustomUserdata*>(userdata);
    BonsaiDeviceLinear* q = reinterpret_cast<BonsaiDeviceLinear*>(custom->q);
    BonsaiDeviceLinear* k_linear = reinterpret_cast<BonsaiDeviceLinear*>(custom->k);
    BonsaiDeviceLinear* v = reinterpret_cast<BonsaiDeviceLinear*>(custom->v);
    const char* debug_name = custom->debug_name != nullptr ? custom->debug_name : "qkv";
    if (q == nullptr || k_linear == nullptr || v == nullptr) {
        fprintf(stderr, "Bonsai INT1 QKV custom op failed: missing Q/K/V device linear\n");
        return;
    }
    if (q->k != k_linear->k || q->k != v->k || q->n != k_linear->n || q->n != v->n) {
        fprintf(stderr, "Bonsai INT1 QKV custom op failed: inconsistent Q/K/V shapes\n");
        return;
    }
    const int input_k = static_cast<int>(x->ne[0]);
    const int m = static_cast<int>(ggml_nelements(x) / input_k);
    const int n_each = q->n;
    if (input_k != q->k || static_cast<int>(dst->ne[0]) != 3 * n_each) {
        fprintf(stderr, "Bonsai INT1 QKV custom op failed: output shape mismatch\n");
        return;
    }

    char error[256] = {};
    const int n_total = 3 * n_each;
    std::string q_label = std::string(debug_name) + ".q";
    std::string k_label = std::string(debug_name) + ".k";
    std::string v_label = std::string(debug_name) + ".v";
    bool ok = bonsai_forward_strided(q, x->data, dst->data, m, input_k, n_each, n_total, 0, q_label.c_str(), error, sizeof(error));
    if (!ok) {
        fprintf(stderr, "Bonsai INT1 Q custom op failed: %s\n", error);
        return;
    }
    ok = bonsai_forward_strided(k_linear, x->data, dst->data, m, input_k, n_each, n_total, n_each, k_label.c_str(), error, sizeof(error));
    if (!ok) {
        fprintf(stderr, "Bonsai INT1 K custom op failed: %s\n", error);
        return;
    }
    ok = bonsai_forward_strided(v, x->data, dst->data, m, input_k, n_each, n_total, 2 * n_each, v_label.c_str(), error, sizeof(error));
    if (!ok) {
        fprintf(stderr, "Bonsai INT1 V custom op failed: %s\n", error);
        return;
    }
    std::string input_label = std::string(debug_name) + ".input";
    std::string output_label = std::string(debug_name) + ".output";
    maybe_log_or_dump_half_tensor(input_label.c_str(), reinterpret_cast<const half*>(x->data), static_cast<size_t>(m) * static_cast<size_t>(input_k), m, input_k);
    maybe_log_or_dump_half_tensor(output_label.c_str(), reinterpret_cast<const half*>(dst->data), static_cast<size_t>(m) * static_cast<size_t>(n_total), m, n_total);
}

extern "C" void sd_bonsai_gemlite_int1_ggml_debug_identity_forward(ggml_tensor* dst, int ith, int nth, void* userdata) {
    if (ith != 0) {
        return;
    }
    (void)nth;
    if (dst == nullptr || dst->src[0] == nullptr) {
        return;
    }
    const ggml_tensor* x = dst->src[0];
    const size_t bytes = ggml_nbytes(x);
    cudaStream_t stream = current_bonsai_backend_stream();
    if (stream == nullptr && !force_bonsai_device_sync_enabled()) {
        fprintf(stderr, "[BonsaiGemLiteINT1] debug identity has no active ggml CUDA stream\n");
        g_stream_counters.default_stream_launches.fetch_add(1);
        log_stream_counters("debug_identity_missing_backend_stream", stream);
        return;
    }
    if (x->data != dst->data && bytes > 0) {
        cudaError_t status = cudaMemcpyAsync(dst->data, x->data, bytes, cudaMemcpyDeviceToDevice, stream);
        if (status != cudaSuccess) {
            fprintf(stderr, "[BonsaiGemLiteINT1] debug identity copy failed: %s\n", cudaGetErrorString(status));
            return;
        }
        if (force_bonsai_device_sync_enabled()) {
            g_stream_counters.device_sync_calls.fetch_add(1);
            cudaStreamSynchronize(stream);
        }
    }
    const char* label = reinterpret_cast<const char*>(userdata);
    if (label == nullptr) {
        label = "debug_identity";
    }
    const int64_t cols = x->ne[0] > 0 ? x->ne[0] : 1;
    const int64_t rows = ggml_nelements(x) / cols;
    if (x->type == GGML_TYPE_F16) {
        maybe_log_or_dump_half_tensor(label, reinterpret_cast<const half*>(dst->data), static_cast<size_t>(ggml_nelements(x)), rows, cols);
    } else if (x->type == GGML_TYPE_F32) {
        maybe_log_or_dump_float_tensor(label, reinterpret_cast<const float*>(dst->data), static_cast<size_t>(ggml_nelements(x)), rows, cols);
    } else {
        fprintf(stderr, "[BonsaiGemLiteINT1] %s debug identity unsupported dtype=%d count=%" PRId64 "\n",
                label,
                static_cast<int>(x->type),
                ggml_nelements(x));
    }
}

extern "C" bool sd_bonsai_gemlite_int1_cuda_probe_run(const uint8_t* wq,
                                                       size_t wq_bytes,
                                                       const float* scales,
                                                       const float* zeros,
                                                       int m,
                                                       int k,
                                                       int n,
                                                       float* elapsed_ms,
                                                       float* output_min,
                                                       float* output_max,
                                                       double* output_sum,
                                                       uint64_t* device_bytes,
                                                       char* error,
                                                       size_t error_size) {
    if (wq == nullptr || scales == nullptr || zeros == nullptr || m <= 0 || k <= 0 || n <= 0) {
        set_error(error, error_size, "invalid Bonsai INT1 probe arguments");
        return false;
    }
    if ((k % 128) != 0 || wq_bytes != static_cast<size_t>(k / 8) * static_cast<size_t>(n)) {
        set_error(error, error_size, "unsupported Bonsai INT1 probe shape");
        return false;
    }

    const size_t a_bytes = static_cast<size_t>(m) * static_cast<size_t>(k) * sizeof(half);
    const size_t c_bytes = static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(half);
    const size_t meta_bytes = static_cast<size_t>(k / 128) * static_cast<size_t>(n) * sizeof(float);

    std::vector<half> host_a(static_cast<size_t>(m) * static_cast<size_t>(k));
    for (size_t i = 0; i < host_a.size(); ++i) {
        const float v = static_cast<float>((static_cast<int>(i % 29) - 14)) / 29.0f;
        host_a[i] = __float2half(v);
    }
    std::vector<half> host_c(static_cast<size_t>(m) * static_cast<size_t>(n));

    half* dev_a = nullptr;
    half* dev_c = nullptr;
    uint8_t* dev_wq = nullptr;
    float* dev_scales = nullptr;
    float* dev_zeros = nullptr;

    if (!check_cuda(cudaMalloc(&dev_a, a_bytes), error, error_size, "cudaMalloc input")) return false;
    if (!check_cuda(cudaMalloc(&dev_c, c_bytes), error, error_size, "cudaMalloc output")) return false;
    if (!check_cuda(cudaMalloc(&dev_wq, wq_bytes), error, error_size, "cudaMalloc W_q")) return false;
    if (!check_cuda(cudaMalloc(&dev_scales, meta_bytes), error, error_size, "cudaMalloc scales")) return false;
    if (!check_cuda(cudaMalloc(&dev_zeros, meta_bytes), error, error_size, "cudaMalloc zeros")) return false;

    bool ok = true;
    ok = ok && check_cuda(cudaMemcpy(dev_a, host_a.data(), a_bytes, cudaMemcpyHostToDevice), error, error_size, "copy input");
    ok = ok && check_cuda(cudaMemcpy(dev_wq, wq, wq_bytes, cudaMemcpyHostToDevice), error, error_size, "copy W_q");
    ok = ok && check_cuda(cudaMemcpy(dev_scales, scales, meta_bytes, cudaMemcpyHostToDevice), error, error_size, "copy scales");
    ok = ok && check_cuda(cudaMemcpy(dev_zeros, zeros, meta_bytes, cudaMemcpyHostToDevice), error, error_size, "copy zeros");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    ok = ok && check_cuda(cudaEventCreate(&start), error, error_size, "event create start");
    ok = ok && check_cuda(cudaEventCreate(&stop), error, error_size, "event create stop");
    if (ok) {
        const dim3 block(16, 8);
        const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
        ok = ok && check_cuda(cudaEventRecord(start), error, error_size, "event record start");
        bonsai_int1_linear_kernel<<<grid, block>>>(dev_a, dev_wq, dev_scales, dev_zeros, dev_c, m, k, n);
        ok = ok && check_cuda(cudaGetLastError(), error, error_size, "Bonsai INT1 kernel launch");
        ok = ok && check_cuda(cudaEventRecord(stop), error, error_size, "event record stop");
        ok = ok && check_cuda(cudaEventSynchronize(stop), error, error_size, "kernel synchronize");
        if (ok && elapsed_ms != nullptr) {
            check_cuda(cudaEventElapsedTime(elapsed_ms, start, stop), error, error_size, "event elapsed");
        }
        ok = ok && check_cuda(cudaMemcpy(host_c.data(), dev_c, c_bytes, cudaMemcpyDeviceToHost), error, error_size, "copy output");
    }

    if (ok) {
        float mn = 0.0f;
        float mx = 0.0f;
        double sum = 0.0;
        for (size_t i = 0; i < host_c.size(); ++i) {
            const float v = __half2float(host_c[i]);
            if (i == 0 || v < mn) mn = v;
            if (i == 0 || v > mx) mx = v;
            sum += v;
        }
        if (output_min != nullptr) *output_min = mn;
        if (output_max != nullptr) *output_max = mx;
        if (output_sum != nullptr) *output_sum = sum;
        if (device_bytes != nullptr) *device_bytes = static_cast<uint64_t>(a_bytes + c_bytes + wq_bytes + 2 * meta_bytes);
    }

    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
    cudaFree(dev_a);
    cudaFree(dev_c);
    cudaFree(dev_wq);
    cudaFree(dev_scales);
    cudaFree(dev_zeros);
    return ok;
}
