#include <cuda_fp16.h>
#include <cuda_tile.h>

namespace ct = cuda::tiles;

extern "C" __tile_global__ void tile_cpp_gemm_smoke(const half * a, const half * b, float * c) {
    using matrix_shape = ct::extents<uint32_t, 16, 16>;
    using tile_shape = ct::shape<16, 16>;

    ct::tensor_span<const half, matrix_shape> a_span(a, matrix_shape{});
    ct::tensor_span<const half, matrix_shape> b_span(b, matrix_shape{});
    ct::tensor_span<float, matrix_shape> c_span(c, matrix_shape{});

    ct::partition_view a_view(a_span, tile_shape{});
    ct::partition_view b_view(b_span, tile_shape{});
    ct::partition_view c_view(c_span, tile_shape{});

    auto a_tile = a_view.load(0, 0);
    auto b_tile = b_view.load(0, 0);
    auto acc_tile = ct::broadcast(0.0f, tile_shape{});
    auto c_tile = ct::mma(a_tile, b_tile, acc_tile);
    c_view.store(c_tile, 0, 0);
}
