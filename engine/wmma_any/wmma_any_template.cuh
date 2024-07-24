
#pragma once
// GPU configuration.
#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include "common/base.h"
#include "common/memory.h"


namespace aq_wmma{

#define C_LAYOUT wmma::mem_row_major
using namespace nvcuda;
using namespace nvcuda::wmma::experimental;


// #if GPU_ARCH >= 80

// Calculate the size of shardmem
inline size_t smemSizeInBytes(int kThreadBlockStage,
                                 int block_M, int block_N, int block_K,
                                 int x_bits, int w_bits)
{
    // use multi-stage shared-mem buffer (needed by async copy)
    // data is copied directly from globalmem to shared memory without going through registers.
    size_t input_buffer_size_static =
        kThreadBlockStage * block_M * block_K * x_bits / 8 + 
        kThreadBlockStage * block_N * block_K * w_bits / 8;
    // The output results need to be stored in shem for scaling processing. 
    size_t output_buffer_size_static = (block_M * block_N) * x_bits * w_bits * sizeof(int32_t);
    size_t true_size = max(input_buffer_size_static, output_buffer_size_static);
    return true_size;
}


// Global mem data is loaded directly into shard mem, 
// requiring the architecture to be larger than sm_80
template <
    // Quantization bit width of X and W
    int X_BITS,
    int W_BITS,
    bool SIGNED,
    // tiling shapes
    int BLOCK_M,
    int BLOCK_N,
    int BLOCK_K,
    int WARP_M,
    int WARP_N,
    int WARP_K,
    int MMA_M,
    int MMA_N,
    int MMA_K,
    // therads num
    int BLOCK_DIMS,
    // threadblock level pipeline stage
    int kThreadBlockStage,
    // pipeline configuration
    bool UseRegisterDoubleBuffer = true, 
    bool UseMinimumSync = true, 
    bool GridMappingXYToMN = false>
__global__ void bmmaArbitrarilyQuant(
    const int M, const int N, const int K, 
    const int *X, 
    const int *W, 
    int *D,
    const half *C, 
    bool bias
){
    extern __shared__ int shared_mem_workspace[];
    // Compute Current Block, Warp and lane Id.
    // compute some global ids, 
    int idx_block_M = GridMappingXYToMN ? blockIdx.x : blockIdx.y;
    int idx_block_N = GridMappingXYToMN ? blockIdx.y : blockIdx.x;
    int warp_id = threadIdx.x >> 5;
    constexpr int WARP_M_TILES = WARP_M / MMA_M;
    constexpr int WARP_N_TILES = WARP_N / MMA_N;
    constexpr int X_WARPS_NUMS = BLOCK_M * X_BITS / WARP_M;
    constexpr int W_WARPS_NUMS = BLOCK_N * W_BITS / WARP_N;
    // compute global offsets: 0 Bit component position
    int row_offset = K / 32;
    int x_bit_offset = M * row_offset;
    int w_bit_offset = N * row_offset;
    const int *x_panel = X + idx_block_M * BLOCK_M * row_offset;
    int ldx = row_offset;
    const int *w_panel = W + idx_block_N * BLOCK_N * row_offset;
    int ldw = row_offset;

    // compute shared memory buffer addresses
    constexpr int NStage = kThreadBlockStage;
    constexpr int size_of_tile_x = X_BITS * BLOCK_M * BLOCK_K ; // b1
    constexpr int size_of_tile_w = W_BITS * BLOCK_N * BLOCK_K ; // b1
    int *shared_x = shared_mem_workspace;
    int *shared_w = shared_x + size_of_tile_x * kThreadBlockStage / 32;
    constexpr int kAccess = 128;
    constexpr int iter_copy_x = CEIL(size_of_tile_x / kAccess, BLOCK_DIMS);
    constexpr int iter_copy_w = CEIL(size_of_tile_w / kAccess, BLOCK_DIMS);
    bool is_residue =
        (M % BLOCK_M != 0) && (idx_block_M == (GridMappingXYToMN ? gridDim.x : gridDim.y) - 1);

    // compute shared memory offsets: Xbit and W bit
    // shared_mem X: [kThreadBlockStage, X_BITS, BLOCK_M, BLOCK_K] 
    // shared_mem W: [kThreadBlockStage, W_BITS, BLOCK_N, BLOCK_K] 
    // output shared_mem: [X_BITS * BLOCK_M, W_BITS * BLOCK_N]
    // Each warp is responsible for the calculation of [warp_M, warp_N] in the output
    // This corresponds to [WARP_M_TILES, WARP_N_TILES] MMA Tiles.
    // Warp is organized as row-first, so the area that the current warp is responsible for calculating can be calculated, and then the corresponding area is read from shardmem according to the Bit component/row information of X, W corresponding to the output area
    // x_bit = x_row / BLOCK_M;  x_bit in [0, X_BITS-1]
    // w_bit = w_row / BLOCK_N;  w_bit in [0, W_BITS-1]
    int x_row = warp_id / W_WARPS_NUMS * WARP_M;
    int w_row = warp_id % W_WARPS_NUMS * WARP_N;
    // smem X row major
    constexpr int smem_ldx = BLOCK_K / 32;
    // smem W col major [K, N] colmajor = [N, K] rowmajor
    constexpr int smem_ldw = BLOCK_K / 32;
    
    // define wmma
    const int kWarpStage = (UseRegisterDoubleBuffer ? 2 : 1);
    wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, precision::b1, wmma::row_major>
                afrag[kWarpStage][WARP_M_TILES];
    wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, precision::b1, wmma::col_major>
                bfrag[kWarpStage][WARP_N_TILES];
    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, int> 
                cfrag[WARP_M_TILES][WARP_N_TILES];

#pragma unroll
    for (int i = 0; i < WARP_M_TILES; i++)
#pragma unroll
      for (int j = 0; j < WARP_N_TILES; j++) wmma::fill_fragment(cfrag[i][j], 0);

    const int num_tile = K / BLOCK_K;
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
                    int idx = (threadIdx.x + BLOCK_DIMS * i) * kAccess;
                    int idx_bit = idx / (BLOCK_M * BLOCK_K);
                    int idx_m = idx / BLOCK_K % BLOCK_M;
                    int idx_k = idx % BLOCK_K;
                    bool valid = (idx < size_of_tile_x);
                    bool zfill = (fetch * BLOCK_K + idx_k) >= K;
                    // residue handling
                    if (is_residue) {
                        valid = valid && (idx_m < (M - idx_block_M * BLOCK_M)) && (idx_bit < X_BITS);
                        zfill = zfill || (idx_m >= (M - idx_block_M * BLOCK_M)) || (idx_bit >= X_BITS);
                    }
                    // [idx_bit, idx_m, idx_k] and The starting address of the current CTA block in the 0bit matrix is ​​tile_x
                    const int *src = tile_x + idx_bit * x_bit_offset + idx_m * row_offset + idx_k / 32;
                    int *dst = shared_tile_x + idx / 32;
                    // Global mem loads data to shard mem, copy 128 b1 = int4 = 16 bytes
                    cpAsyncPredZfill<16>(dst, src, valid, zfill);
                }
#pragma unroll
                for (int i = 0; i < iter_copy_w; i++) {
                    // [W_BITS, BLOCK_N, BLOCK_K]
                    int idx = (threadIdx.x + BLOCK_DIMS * i) * kAccess;
                    int idx_bit = idx / (BLOCK_N * BLOCK_K);
                    int idx_n = idx / BLOCK_K % BLOCK_N;
                    int idx_k = idx % BLOCK_K;
                    bool valid = (idx < size_of_tile_w);
                    bool zfill = (fetch * BLOCK_K + idx_k) >= K;
                    // [idx_bit, idx_n, idx_k] and The starting address of the current CTA block in the 0bit matrix is ​​tile_w
                    const int *src = tile_w + idx_bit * w_bit_offset + idx_n * row_offset + idx_k / 32;
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
        // compute [WARP_M_TILES, WARP_N_TILES]`s [mma_M, mma_N] in [X_BITS * BLOCK_M, W_BITS * BLOCK_N]
        // mma_K == warp_K Double_buffer is performed inside block
        for (int k = 0; k < BLOCK_K / MMA_K; k++) {
            // Load into afrag
#pragma unroll
            for (int m = 0; m < WARP_M_TILES; m++) {
                int offset = (x_row + m * MMA_M) * smem_ldx + k * MMA_K / 32;
                wmma::load_matrix_sync(afrag[k % kWarpStage][m], shared_tile_x + offset, BLOCK_K);
            }
            // Load into bfrag
#pragma unroll
            for (int n = 0; n < WARP_N_TILES; n++) {
                int offset = (w_row + n * MMA_N) * smem_ldw + k * MMA_K / 32;
                wmma::load_matrix_sync(bfrag[k % kWarpStage][n], shared_tile_w + offset, BLOCK_K);
            }
#pragma unroll
            for (int m = 0; m < WARP_M_TILES; m++) {
#pragma unroll
                for (int n = 0; n < WARP_N_TILES; n++) {
                    wmma::bmma_sync(cfrag[m][n], afrag[k % kWarpStage][m], bfrag[k % kWarpStage][n], cfrag[m][n], bmmaBitOpAND);
                }
            }
        }
        pipe.releaseReader();
    }
    __syncthreads();
    // store each warp`s cfrag to shared memory  
    // output shared_mem: [X_BITS * BLOCK_M, W_BITS * BLOCK_N]
    // Each warp is responsible for the calculation of [warp_M, warp_N] in the output
    // This corresponds to [WARP_M_TILES, WARP_N_TILES] MMA Tiles.
    int *shared_c = shared_mem_workspace;
    const int smem_ldc = W_BITS * BLOCK_N;
    int c_warp_offset = x_row * smem_ldc + w_row;
#pragma unroll
    for (int m = 0; m < WARP_M_TILES; m++) {
#pragma unroll
        for (int n = 0; n < WARP_N_TILES; n++) {
            int offset = c_warp_offset + m * MMA_M * smem_ldc + n * MMA_N;
            wmma::store_matrix_sync(shared_c+offset, cfrag[m][n], smem_ldc, C_LAYOUT);
        }
    }
    __syncthreads();
    // According to the x bit component and w bit component 
    // information corresponding to each Tile block, 
    // and the calculation symbol confirmation scaling coefficient

    //     constexpr int CAccess = 1;
    //     constexpr int iter_scale_c = CEIL(BLOCK_M * BLOCK_N / CAccess, BLOCK_DIMS);
    //     int* d_tile = D + idx_block_M * BLOCK_M * N + idx_block_N * BLOCK_N;
    // #pragma unroll
    //     for (size_t i = 0; i < iter_scale_c; i++)
    //     {
    //         int idx = (threadIdx.x + BLOCK_DIMS * i) * CAccess;
    //         bool valid = (idx < BLOCK_M * BLOCK_N);
    //         if (is_residue)
    //             valid = valid && ((idx % BLOCK_N)< (N - BLOCK_N * idx_block_N)) && ((idx / BLOCK_N)< ( M - BLOCK_M*idx_block_M));
    //         if (valid)
    //         {
    //             int *shmem_stream_ptr = (int *)shared_mem_workspace + idx;
    //             int val = 0;
    //             int base_multiplier = 1;
    // #pragma unroll
    //             for (int i = 0; i < X_BITS; i++) {
    //                 int cur_multiplier = SIGNED && (i == X_BITS -1) ? -1 * base_multiplier : base_multiplier;
    // #pragma unroll
    //                 for (int j = 0; j < W_BITS; j++) {
    //                     int tmp = *(shmem_stream_ptr + BLOCK_N * j);
    //                     val += (cur_multiplier * tmp);
    //                     cur_multiplier =  SIGNED && (j == W_BITS - 2) ? -2 * cur_multiplier : 2 * cur_multiplier;
    //                 }
    //                 base_multiplier *= 2;
    //                 shmem_stream_ptr += BLOCK_M * BLOCK_N * W_BITS;
    //             }
    //             // store to global mem
    //             d_tile[(idx / BLOCK_N) * N + (idx % BLOCK_N)] = val;
    //         }
    //     }
    //     __syncthreads();

    // Parallel reading and writing implementation 
    IntVector<4> buffer;
    constexpr int CAccess = 4;
    constexpr int iter_scale_c = CEIL(BLOCK_M * BLOCK_N / CAccess, BLOCK_DIMS);
    int* d_tile = D + idx_block_M * BLOCK_M * N + idx_block_N * BLOCK_N;
#pragma unroll
    for (size_t i = 0; i < iter_scale_c; i++)
    {
        int idx = (threadIdx.x + BLOCK_DIMS * i) * CAccess;
        bool valid = (idx < BLOCK_M * BLOCK_N);
        if (is_residue)
            valid = valid && ((idx % BLOCK_N)< (N - BLOCK_N * idx_block_N)) && ((idx / BLOCK_N)< ( M - BLOCK_M*idx_block_M));
        if (valid)
        {   
            // load CAccess 
            int *shmem_stream_ptr = (int *)shared_mem_workspace + idx;
            buffer.reset();
            int base_multiplier = 1;
#pragma unroll
            for (int i = 0; i < X_BITS; i++) {
                int cur_multiplier = SIGNED && (i == X_BITS -1) ? -1 * base_multiplier : base_multiplier;
#pragma unroll
                for (int j = 0; j < W_BITS; j++) {
                    int tmp0 = *(shmem_stream_ptr + 0 + BLOCK_N * j);
                    buffer.x[0] += (cur_multiplier * tmp0);
                    int tmp1 = *(shmem_stream_ptr + 1 + BLOCK_N * j);
                    buffer.x[1] += (cur_multiplier * tmp1);
                    int tmp2 = *(shmem_stream_ptr + 2 + BLOCK_N * j);
                    buffer.x[2] += (cur_multiplier * tmp2);
                    int tmp3 = *(shmem_stream_ptr + 3 + BLOCK_N * j);
                    buffer.x[3] += (cur_multiplier * tmp3);
                    cur_multiplier =  SIGNED && (j == W_BITS - 2) ? -2 * cur_multiplier : 2 * cur_multiplier;
                }
                base_multiplier *= 2;
                shmem_stream_ptr += BLOCK_M * BLOCK_N * W_BITS;
            }
            // store to global mem
            buffer.st(d_tile + idx);
        }
    }
    __syncthreads();
}

// #endif // GPU_ARCH

} // aq_wmma