// Copyright (C) 2024 ByteDance and/or its affiliates
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//          http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "common/base.h"
#include "common/memory.h"
#include "mma_any/bmma.h"

template <
    // Quantization bit width of X and W and signed/unsigned
    typename QuantType,
    // tiling shapes
    typename ThreadBlockShape, typename WarpShape, typename MmaShape,
    // threadblock level pipeline stage
    int kThreadBlockStage,
    // type of accumulator
    typename AccumulatorType,
    // type of shared memory swizzling
    typename ASwizzle, typename BSwizzle, typename CSwizzle,
    // pipeline configuration
    bool UseRegisterDoubleBuffer = true, bool UseMinimumSync = true, bool GridMappingXYToMN = false
    // // layout of A, B, C matrix; (MxK rowmajor) * (KxN colmajor) == (MxN rowmajor)
    // typename LayoutA = Layout::RowMajor,
    // typename LayoutB = Layout::ColumnMajor,
    // typename LayoutC = Layout::RowMajor
    >
struct AqBMMAKernel {
    static constexpr int X_BITS = QuantType::X_BITS;
    static constexpr int W_BITS = QuantType::W_BITS;
    static constexpr int BLOCK_M = ThreadBlockShape::M;
    static constexpr int BLOCK_N = ThreadBlockShape::N;
    static constexpr int BLOCK_K = ThreadBlockShape::K;
    static constexpr int WARP_M = WarpShape::M;
    static constexpr int WARP_N = WarpShape::N;
    static constexpr int WARP_K = WarpShape::K;
    static constexpr int MMA_M = MmaShape::M;
    static constexpr int MMA_N = MmaShape::N;
    static constexpr int MMA_K = MmaShape::K;
    static constexpr int SKEW = W_BITS * BLOCK_N % 16 == 0 ? 8 : 0;
    static constexpr bool quant_signed = QuantType::SIGNED;
    static constexpr int WARP_M_TILES = WARP_M / MMA_M;
    static constexpr int WARP_N_TILES = WARP_N / MMA_N;
    static constexpr int X_WARPS_NUMS = CEIL(BLOCK_M * X_BITS, MMA_M) / WARP_M_TILES;
    static constexpr int W_WARPS_NUMS = CEIL(BLOCK_N * W_BITS, MMA_N) / WARP_N_TILES;
    static_assert(WARP_K == MMA_K, "Only support warp shape K == Mma shape K.\n");
    static_assert(WARP_M % MMA_M == 0, "WARP_M must be an integer multiple of MMA_M.\n");
    static_assert(WARP_N % MMA_N == 0, "WARP_N must be an integer multiple of MMA_N.\n");
    static_assert(BLOCK_K % WARP_K == 0, "BLOCK_K must be an integer multiple of WARP_K.\n");
    static_assert(kThreadBlockStage > 1, "kThreadBlockStage must be greater than 1.\n");
    static_assert(WARP_K % 128 == 0, "Only support warp shape WARP_K>=128 for performance.\n");
    // static_assert(WARP_M % 16 == 0, "Only support warp shape M>=16 for performance.\n");
    // precompute constants
    static constexpr bool GridMapping = GridMappingXYToMN;
    // determine the number of threads
    static constexpr int blockDims = 32 * X_WARPS_NUMS * W_WARPS_NUMS;
#if GPU_ARCH >= 80
    // use multi-stage shared-mem buffer (needed by async copy)
    // data is copied directly from globalmem to shared memory without going through registers.
    static constexpr size_t input_buffer_size_static =
        kThreadBlockStage * BLOCK_M * BLOCK_K * X_BITS / 8 +
        kThreadBlockStage * BLOCK_N * BLOCK_K * W_BITS / 8;
#else
    // use ping-pong buffer and multi-stage regfile buffering
    static constexpr size_t input_buffer_size_static =
        2 * BLOCK_M * BLOCK_K * X_BITS / 8 + 2 * BLOCK_N * BLOCK_K * W_BITS / 8;
#endif // GPU_ARCH >= 80

    // The output results need to be stored in shem for scaling processing.
    static constexpr size_t output_buffer_size_static =
        (MMA_M * WARP_M_TILES * X_WARPS_NUMS) * (MMA_N * WARP_N_TILES * W_WARPS_NUMS + SKEW) *
        sizeof(int32_t);
    // mainloop interface
    __device__ __forceinline__ void mainLoop(const int M, const int N, const int K, const int *X,
                                             const int *W, int *shared_mem_workspace);

    __device__ __forceinline__ void epilogue(const int M, const int N, half *D,
                                             int *shared_mem_workspace, const half *C,
                                             bool bias = false);
};

#if GPU_ARCH >= 80
// async-copy multi-stage kernel
// RowMajor * ColMajor(A:activation,  B:weight)
template <
    // Quantization bit width of X and W and signed/unsigned
    typename QuantType,
    // tiling shapes
    typename ThreadBlockShape, typename WarpShape, typename MmaShape,
    // threadblock level pipeline stage
    int kThreadBlockStage,
    // type of accumulator
    typename AccumulatorType,
    // type of shared memory swizzling
    typename ASwizzle, typename BSwizzle, typename CSwizzle,
    // pipeline configuration
    bool UseRegisterDoubleBuffer, bool UseMinimumSync, bool GridMappingXYToMN>
__device__ __forceinline__ void
AqBMMAKernel<QuantType, ThreadBlockShape, WarpShape, MmaShape, kThreadBlockStage, AccumulatorType,
             ASwizzle, BSwizzle, CSwizzle, UseRegisterDoubleBuffer, UseMinimumSync,
             GridMappingXYToMN>::mainLoop(const int M, const int N, const int K, const int *X,
                                          const int *W, int *shared_mem_workspace)
{
    // compute some global ids,
    const unsigned int idx_block_M = GridMappingXYToMN ? blockIdx.x : blockIdx.y;
    const unsigned int idx_block_N = GridMappingXYToMN ? blockIdx.y : blockIdx.x;
    const unsigned int warp_id = threadIdx.x >> 5;

    // compute global offsets: 0 Bit component position
    int row_offset = K / 32;
    int x_bit_offset = M * row_offset;
    int w_bit_offset = N * row_offset;
    const int *x_panel = X + idx_block_M * BLOCK_M * row_offset;
    // int ldx = row_offset;
    const int *w_panel = W + idx_block_N * BLOCK_N * row_offset;
    // int ldw = row_offset;

    // compute shared memory buffer addresses
    constexpr int NStage = kThreadBlockStage;
    constexpr int size_of_tile_x = X_BITS * BLOCK_M * BLOCK_K; // b1
    constexpr int size_of_tile_w = W_BITS * BLOCK_N * BLOCK_K; // b1
    int *shared_x = shared_mem_workspace;
    int *shared_w = shared_x + size_of_tile_x * kThreadBlockStage / 32;
    constexpr int kAccess = 128;
    constexpr int iter_copy_x = CEIL(size_of_tile_x / kAccess, blockDims);
    constexpr int iter_copy_w = CEIL(size_of_tile_w / kAccess, blockDims);
    // Assume that N is divisible by CTA
    // bool is_residue =
    //     (M % BLOCK_M != 0) && (idx_block_M == (GridMappingXYToMN ? gridDim.x : gridDim.y) - 1);

    // compute shared memory offsets: Xbit and W bit
    // shared_mem X: [kThreadBlockStage, X_BITS, BLOCK_M, BLOCK_K]
    // shared_mem W: [kThreadBlockStage, W_BITS, BLOCK_N, BLOCK_K]
    // output shared_mem: [X_BITS * BLOCK_M, W_BITS * BLOCK_N]
    // Each warp is responsible for the calculation of [WARP_M, WARP_N] in the output
    // This corresponds to [WARP_M_TILES, WARP_N_TILES] MMA Tiles.
    // Warp is organized as row-first, so the area that the current warp is responsible for calculating can be calculated, and then the corresponding area is read from shardmem according to the Bit component/row information of X, W corresponding to the output area
    // x_bit = x_row / BLOCK_M;  x_bit in [0, X_BITS-1]
    // w_bit = w_row / BLOCK_N;  w_bit in [0, W_BITS-1]
    int x_row = warp_id / W_WARPS_NUMS * WARP_M;
    int w_row = warp_id % W_WARPS_NUMS * WARP_N;
    // smem X row major
    const int smem_ldx = BLOCK_K / 32;
    // smem W col major [K, N] colmajor = [N, K] rowmajor
    const int smem_ldw = BLOCK_K / 32;

    // ASwizzle aSwizzle;
    // BSwizzle bSwizzle;
    // define mma buffers
    typedef typename aq_bmma::fragment_a_rowmajor<MmaShape> FragmentA;
    typedef typename aq_bmma::fragment_b_colmajor<MmaShape> FragmentB;
    typedef typename aq_bmma::fragment_c<MmaShape, AccumulatorType> FragmentC;
    const int kWarpStage = (UseRegisterDoubleBuffer ? 2 : 1);
    FragmentA afrag[kWarpStage][WARP_M_TILES];
    FragmentB bfrag[kWarpStage][WARP_N_TILES];
    FragmentC cfrag[WARP_M_TILES][WARP_N_TILES];
    const int num_tile = CEIL(K, BLOCK_K);
    Pipeline<NStage, UseMinimumSync> pipe;
    int fetch = 0, compute = 0;

#pragma unroll
    for (; compute < num_tile; compute++) {
        for (; fetch < compute + NStage; fetch++) {
            pipe.acquireWriter();
            // fetch data
            if (fetch < num_tile) {
                // current fetch stage`s src global mem: 0 bit position
                const int *tile_x = x_panel + fetch * BLOCK_K / 32;
                const int *tile_w = w_panel + fetch * BLOCK_K / 32;
                // current fetch stage`s dst shared_mem
                int *shared_tile_x = shared_x + (fetch % NStage) * size_of_tile_x / 32;
                int *shared_tile_w = shared_w + (fetch % NStage) * size_of_tile_w / 32;
#pragma unroll
                for (int i = 0; i < iter_copy_x; i++) {
                    // [X_BITS, BLOCK_M, BLOCK_K]
                    int idx = (threadIdx.x + blockDims * i) * kAccess;
                    int idx_bit = idx / (BLOCK_M * BLOCK_K);
                    int idx_m = idx / BLOCK_K % BLOCK_M;
                    int idx_k = idx % BLOCK_K;
                    bool valid = (idx < size_of_tile_x);
                    // residue handling
                    bool zfill = ((fetch * BLOCK_K + idx_k) >= K) ||
                                 (idx_m >= (M - idx_block_M * BLOCK_M)) || (idx_bit >= X_BITS);
                    // [idx_bit, idx_m, idx_k] and The starting address of the current CTA block in the 0bit matrix is ​​tile_x
                    const int *src =
                        tile_x + idx_bit * x_bit_offset + idx_m * row_offset + idx_k / 32;
                    int *dst = shared_tile_x + idx / 32;
                    // Global mem loads data to shard mem, copy 128 b1 = int4 = 16 bytes
                    cpAsyncPredZfill<16>(dst, src, valid, zfill);
                }
#pragma unroll
                for (int i = 0; i < iter_copy_w; i++) {
                    // [W_BITS, BLOCK_N, BLOCK_K]
                    int idx = (threadIdx.x + blockDims * i) * kAccess;
                    int idx_bit = idx / (BLOCK_N * BLOCK_K);
                    int idx_n = idx / BLOCK_K % BLOCK_N;
                    int idx_k = idx % BLOCK_K;
                    bool valid = (idx < size_of_tile_w);
                    bool zfill = ((fetch * BLOCK_K + idx_k) >= K) ||
                                 (idx_n >= (N - idx_block_N * BLOCK_N)) || (idx_bit >= W_BITS);
                    // [idx_bit, idx_n, idx_k] and The starting address of the current CTA block in the 0bit matrix is ​​tile_w
                    const int *src =
                        tile_w + idx_bit * w_bit_offset + idx_n * row_offset + idx_k / 32;
                    int *dst = shared_tile_w + idx / 32;
                    // Global mem loads data to shard mem, copy 128 b1 = int4 = 16 bytes
                    cpAsyncPredZfill<16>(dst, src, valid, zfill);
                }
            }
            pipe.commitStage();
        }
        pipe.acquireReader();

        int *shared_tile_w = shared_w + (compute % NStage) * size_of_tile_w / 32;
        int *shared_tile_x = shared_x + (compute % NStage) * size_of_tile_x / 32;

#pragma unroll
        // compute [WARP_M_TILES, WARP_N_TILES]`s [MMA_M, MMA_N] in [X_BITS * BLOCK_M, W_BITS * block_N]
        // MMA_K == WARP_K Double_buffer is performed inside block
        for (int k = 0; k < BLOCK_K / MMA_K; k++) {
            // Load into afrag
#pragma unroll
            for (int m = 0; m < WARP_M_TILES; m++) {
                int offset = (x_row + m * MMA_M) * smem_ldx + k * MMA_K / 32;
                aq_bmma::loadMatrixSync(afrag[k % kWarpStage][m], shared_tile_x, offset, smem_ldx);
            }
            // Load into bfrag
#pragma unroll
            for (int n = 0; n < WARP_N_TILES; n++) {
                int offset = (w_row + n * MMA_N) * smem_ldw + k * MMA_K / 32;
                aq_bmma::loadMatrixSync(bfrag[k % kWarpStage][n], shared_tile_w, offset, smem_ldw);
            }
#pragma unroll
            for (int m = 0; m < WARP_M_TILES; m++) {
#pragma unroll
                for (int n = 0; n < WARP_N_TILES; n++) {
                    aq_bmma::bmmaSync(cfrag[m][n], afrag[k % kWarpStage][m],
                                      bfrag[k % kWarpStage][n], cfrag[m][n]);
                }
            }
        }
        pipe.releaseReader();
    }
    __syncthreads();
    // store each warp`s cfrag to shared memory
    // output shared_mem: [X_BITS * BLOCK_M, W_BITS * BLOCK_N]
    // Each warp is responsible for the calculation of [WARP_M, WARP_N] in the output
    // This corresponds to [WARP_M_TILES, WARP_N_TILES] MMA Tiles.
    int *shared_c = shared_mem_workspace;
    const int smem_ldc = W_BITS * BLOCK_N + SKEW;
    int c_warp_offset = x_row * smem_ldc + w_row;
#pragma unroll
    for (int m = 0; m < WARP_M_TILES; m++) {
#pragma unroll
        for (int n = 0; n < WARP_N_TILES; n++) {
            int offset = c_warp_offset + m * MMA_M * smem_ldc + n * MMA_N;
            storeMatrixSync(cfrag[m][n], shared_c, offset, smem_ldc);
        }
    }
    __syncthreads();
}

#else // __CUDA_ARCH__ < 80
// shared-memory ping-pong buffering, regfile buffered multi-stage kernel
// RowMajor * ColMajor(A:activation,  B:weight)
template <
    // Quantization bit width of X and W and signed/unsigned
    typename QuantType,
    // tiling shapes
    typename ThreadBlockShape, typename WarpShape, typename MmaShape,
    // threadblock level pipeline stage
    int kThreadBlockStage,
    // type of accumulator
    typename AccumulatorType,
    // type of shared memory swizzling
    typename ASwizzle, typename BSwizzle, typename CSwizzle,
    // pipeline configuration
    bool UseRegisterDoubleBuffer, bool UseMinimumSync, bool GridMappingXYToMN>
__device__ __forceinline__ void
AqBMMAKernel<QuantType, ThreadBlockShape, WarpShape, MmaShape, kThreadBlockStage, AccumulatorType,
             ASwizzle, BSwizzle, CSwizzle, UseRegisterDoubleBuffer, UseMinimumSync,
             GridMappingXYToMN>::mainLoop(const int M, const int N, const int K, const int *X,
                                          const int *W, int *shared_mem_workspace)
{
    // compute some ids
    int idx_block_M = GridMappingXYToMN ? blockIdx.x : blockIdx.y;
    int idx_block_N = GridMappingXYToMN ? blockIdx.y : blockIdx.x;
    const unsigned int warp_id = threadIdx.x >> 5;

    // compute global offsets: 0 Bit component position
    int row_offset = K / 32;
    int x_bit_offset = M * row_offset;
    int w_bit_offset = N * row_offset;
    const int *x_panel = X + idx_block_M * BLOCK_M * row_offset;
    int ldx = row_offset;
    const int *w_panel = W + idx_block_N * BLOCK_N * row_offset;
    int ldw = row_offset;

    // compute global to shared copy constants
    constexpr int NStage = kThreadBlockStage;
    constexpr int size_of_tile_x = X_BITS * BLOCK_M * BLOCK_K; // b1
    constexpr int size_of_tile_w = W_BITS * BLOCK_N * BLOCK_K; // b1
    constexpr int kAccess = 128;
    constexpr int iter_copy_x = CEIL(size_of_tile_x / kAccess, blockDims);
    constexpr int iter_copy_w = CEIL(size_of_tile_w / kAccess, blockDims);
    bool is_residue =
        (M % BLOCK_M != 0) && (idx_block_M == (GridMappingXYToMN ? gridDim.x : gridDim.y) - 1);
    int4 copy_buffer_x[NStage][iter_copy_x];
    int4 copy_buffer_w[NStage][iter_copy_w];
    constexpr int NBuffer = 2;
    int *shared_x = shared_mem_workspace;
    int *shared_w = shared_x + size_of_tile_x * NBuffer / 32;
    // compute shared memory offsets: Xbit and W bit
    // shared_mem X: [kThreadBlockStage, X_BITS, BLOCK_M, BLOCK_K]
    // shared_mem W: [kThreadBlockStage, W_BITS, BLOCK_N, BLOCK_K]
    // output shared_mem: [X_BITS * BLOCK_M, W_BITS * BLOCK_N]
    // Each warp is responsible for the calculation of [WARP_M, WARP_N] in the output
    // This corresponds to [WARP_M_TILES, WARP_N_TILES] MMA Tiles.
    // Warp is organized as row-first, so the area that the current warp is responsible for calculating can be calculated, and then the corresponding area is read from shardmem according to the Bit component/row information of X, W corresponding to the output area
    // x_bit = x_row / BLOCK_M;  x_bit in [0, X_BITS-1]
    // w_bit = w_row / BLOCK_N;  w_bit in [0, W_BITS-1]
    int x_row = warp_id / W_WARPS_NUMS * WARP_M;
    int w_row = warp_id % W_WARPS_NUMS * WARP_N;
    // smem X row major
    const int smem_ldx = BLOCK_K / 32;
    // smem W col major [K, N] colmajor = [N, K] rowmajor
    const int smem_ldw = BLOCK_K / 32;

    // compute shared memory offsets
    ASwizzle aSwizzle;
    BSwizzle bSwizzle;
    // define mma buffers
    typedef typename aq_bmma::fragment_a_rowmajor<MmaShape> FragmentA;
    typedef typename aq_bmma::fragment_b_colmajor<MmaShape> FragmentB;
    typedef typename aq_bmma::fragment_c<MmaShape, AccumulatorType> FragmentC;
    const int kWarpStage = (UseRegisterDoubleBuffer ? 2 : 1);
    FragmentA afrag[kWarpStage][WARP_M_TILES];
    FragmentB bfrag[kWarpStage][WARP_N_TILES];
    FragmentC cfrag[WARP_M_TILES][WARP_N_TILES];
    const int num_tile = CEIL(K, BLOCK_K);
    int fetch = 0, compute = -NStage;
    while (compute < num_tile) {
#pragma unroll
        for (int stage = 0; stage < NStage; stage++) {
            if (compute >= 0 && compute < num_tile) {
                // store data: local mem --> shared mem
                int *shared_tile_w = shared_w + (compute % NBuffer) * size_of_tile_w / 32;
                int *shared_tile_x = shared_x + (compute % NBuffer) * size_of_tile_x / 32;
#pragma unroll
                for (int i = 0; i < iter_copy_x; i++) {
                    // [X_BITS, BLOCK_M, BLOCK_K]
                    int idx = (threadIdx.x + blockDims * i) * kAccess;
                    int idx_bit = idx / (BLOCK_M * BLOCK_K);
                    int idx_m = idx / BLOCK_K % BLOCK_M;
                    int idx_k = idx % BLOCK_K;
                    bool valid = (idx < size_of_tile_x);
                    int *dst = shared_tile_x + idx;
                    if (valid)
                        *(int4 *)dst = copy_buffer_x[stage][i];
                }
#pragma unroll
                for (int i = 0; i < iter_copy_w; i++) {
                    // [W_BITS, BLOCK_N, BLOCK_K]
                    int idx = (threadIdx.x + blockDims * i) * kAccess;
                    int idx_bit = idx / (BLOCK_N * BLOCK_K);
                    int idx_n = idx / BLOCK_K % BLOCK_N;
                    int idx_k = idx % BLOCK_K;
                    bool valid = (idx < size_of_tile_w);
                    int *dst = shared_tile_w + idx;
                    if (valid)
                        *(int4 *)dst = copy_buffer_w[stage][i];
                }
                __syncthreads();

                // compute
#pragma unroll
                for (int k = 0; k < BLOCK_K / MMA_K; k++) {
                    // Load into afrag
#pragma unroll
                    for (int m = 0; m < WARP_M_TILES; m++) {
                        int offset = (x_row + m * MMA_M) * smem_ldx + k * MMA_K / 32;
                        aq_bmma::loadMatrixSync(afrag[k % kWarpStage][m], shared_tile_x, offset,
                                                smem_ldx);
                    }
                    // Load into bfrag
#pragma unroll
                    for (int n = 0; n < WARP_N_TILES; n++) {
                        int offset = (w_row + n * MMA_N) * smem_ldw + k * MMA_K / 32;
                        aq_bmma::loadMatrixSync(bfrag[k % kWarpStage][n], shared_tile_w, offset,
                                                smem_ldw);
                    }
#pragma unroll
                    for (int m = 0; m < WARP_M_TILES; m++) {
#pragma unroll
                        for (int n = 0; n < WARP_N_TILES; n++) {
                            aq_bmma::bmmaSync(cfrag[m][n], afrag[k % kWarpStage][m],
                                              bfrag[k % kWarpStage][n], cfrag[m][n]);
                        }
                    }
                }
            }
            compute++;
            // dataflow: global memory --> local memory
            if (fetch < num_tile) {
                // current fetch stage`s src global mem: 0 bit position
                const int *tile_x = x_panel + fetch * BLOCK_K / 32;
                const int *tile_w = w_panel + fetch * BLOCK_K / 32;
#pragma unroll
                for (int i = 0; i < iter_copy_w; i++) {
                    // [W_BITS, BLOCK_N, BLOCK_K]
                    int idx = (threadIdx.x + blockDims * i) * kAccess;
                    int idx_bit = idx / (BLOCK_N * BLOCK_K);
                    int idx_n = idx / BLOCK_K % BLOCK_N;
                    int idx_k = idx % BLOCK_K;
                    bool valid = (idx < size_of_tile_w);
                    bool zfill =
                        (fetch * BLOCK_K + idx_k >= K) || (idx_n + idx_block_N * BLOCK_N >= N);
                    // [idx_bit, idx_n, idx_k] and The starting address of the current CTA block in the 0bit matrix is ​​tile_w
                    const int *src =
                        tile_w + idx_bit * w_bit_offset + idx_n * row_offset + idx_k / 32;
                    // Global mem loads data to local mem, copy 128 b1 = int4 = 16 bytes
                    if (valid)
                        if (!zfill)
                            copy_buffer_w[stage][i] = *(const int4 *)(src);
                        else
                            copy_buffer_w[stage][i] = { 0, 0, 0, 0 };
                }
#pragma unroll
                for (int i = 0; i < iter_copy_x; i++) {
                    // [X_BITS, BLOCK_M, BLOCK_K]
                    int idx = (threadIdx.x + blockDims * i) * kAccess;
                    int idx_bit = idx / (BLOCK_M * BLOCK_K);
                    int idx_m = idx / BLOCK_K % BLOCK_M;
                    int idx_k = idx % BLOCK_K;
                    bool valid = (idx < size_of_tile_x);
                    bool zfill = ((fetch * BLOCK_K + idx_k) >= K) ||
                                 (idx_m >= (M - idx_block_M * BLOCK_M)) || (idx_bit >= X_BITS);
                    // [idx_bit, idx_m, idx_k] and The starting address of the current CTA block in the 0bit matrix is ​​tile_x
                    const int *src =
                        tile_x + idx_bit * x_bit_offset + idx_m * row_offset + idx_k / 32;
                    if (valid)
                        if (!zfill)
                            copy_buffer_x[stage][i] = *(const int4 *)src;
                        else
                            copy_buffer_x[stage][i] = { 0, 0, 0, 0 };
                }
            }
            fetch++;
        }
    }

    // store C to shared memory
    __syncthreads();
    int *shared_c = shared_mem_workspace;
    const int smem_ldc = W_BITS * BLOCK_N + SKEW;
    int c_warp_offset = x_row * smem_ldc + w_row;
#pragma unroll
    for (int m = 0; m < WARP_M_TILES; m++) {
#pragma unroll
        for (int n = 0; n < WARP_N_TILES; n++) {
            int offset = c_warp_offset + m * MMA_M * smem_ldc + n * MMA_N;
            storeMatrixSync(cfrag[m][n], shared_c, offset, smem_ldc);
        }
    }
    __syncthreads();
}

#endif

// epilogue processing, perform parameter scale and write back to global mem
template <typename QuantType, typename ThreadBlockShape, typename WarpShape, typename MmaShape,
          int kThreadBlockStage, typename AccumulatorType, typename ASwizzle, typename BSwizzle,
          typename CSwizzle, bool UseRegisterDoubleBuffer, bool UseMinimumSync,
          bool GridMappingXYToMN>
__device__ __forceinline__ void
AqBMMAKernel<QuantType, ThreadBlockShape, WarpShape, MmaShape, kThreadBlockStage, AccumulatorType,
             ASwizzle, BSwizzle, CSwizzle, UseRegisterDoubleBuffer, UseMinimumSync,
             GridMappingXYToMN>::epilogue(const int M, const int N, half *D,
                                          int *shared_mem_workspace, const half *C, bool bias)
{
    int idx_block_M = GridMappingXYToMN ? blockIdx.x : blockIdx.y;
    int idx_block_N = GridMappingXYToMN ? blockIdx.y : blockIdx.x;
    half scale = C[0];

    // Parallel reading and writing implementation
    IntVector<4> buffer;
    constexpr int CAccess = 4;
    constexpr int smem_ldc = W_BITS * BLOCK_N + SKEW;
    int idx =
        threadIdx.x / (BLOCK_N / CAccess) * smem_ldc + threadIdx.x % (BLOCK_N / CAccess) * CAccess;
    bool valid = (threadIdx.x * CAccess < BLOCK_M * BLOCK_N);
    if (valid) {
        // load CAccess
        int *shmem_stream_ptr = (int *)shared_mem_workspace + idx;
        int base_multiplier = 1;
#pragma unroll
        for (int i = 0; i < X_BITS; i++) {
            int cur_multiplier =
                quant_signed && (i == X_BITS - 1) ? -1 * base_multiplier : base_multiplier;
#pragma unroll
            for (int j = 0; j < W_BITS; j++) {
                int4 *tmp = reinterpret_cast<int4 *>(shmem_stream_ptr + BLOCK_N * j);
                buffer.x[0] += (cur_multiplier * tmp->x);
                buffer.x[1] += (cur_multiplier * tmp->y);
                buffer.x[2] += (cur_multiplier * tmp->z);
                buffer.x[3] += (cur_multiplier * tmp->w);
                cur_multiplier =
                    quant_signed && (j == W_BITS - 2) ? -2 * cur_multiplier : 2 * cur_multiplier;
            }
            base_multiplier *= 2;
            shmem_stream_ptr += BLOCK_M * smem_ldc;
        }
    }
    __syncthreads();
    // store to global mem
    int gmem_idx = idx_block_M * BLOCK_M * N + idx_block_N * BLOCK_N +
                   threadIdx.x / (BLOCK_N / CAccess) * N +
                   threadIdx.x % (BLOCK_N / CAccess) * CAccess;
    if (gmem_idx < M * N) {
        D[gmem_idx + 0] = __int2half_rn(buffer.x[0]) * scale;
        D[gmem_idx + 1] = __int2half_rn(buffer.x[1]) * scale;
        D[gmem_idx + 2] = __int2half_rn(buffer.x[2]) * scale;
        D[gmem_idx + 3] = __int2half_rn(buffer.x[3]) * scale;
        // FIXME: missaligned address
        // int4 *d_ptr = reinterpret_cast<int4 *>(D + gmem_idx);
        // *d_ptr = *((int4 *)buffer.x);
    }
}
