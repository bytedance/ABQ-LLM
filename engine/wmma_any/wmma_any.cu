// Copyright (C) ABQ.2024 (liusongwei.zju@bytedance.com)
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

#include <iostream>
#include <cuda_runtime.h>
#include "wmma_any/wmma_any_template.cuh"
#include "wmma_any/wmma_any.h"

////////////////////////////////////////////////////////////////////////////////////////////////////
template <
    // quant info
    typename QuantType,
    // tiling shapes
    typename ThreadBlockShape, typename WarpShape, typename MmaShape,
    // threadblock level pipeline stage
    int kThreadBlockStage>
cudaError_t launchWmmaAnyKernel(const AnyQuantParams &params, const cudaStream_t &stream)
{
    constexpr int WARP_M_TILES = WarpShape::M / MmaShape::M;
    constexpr int WARP_N_TILES = WarpShape::N / MmaShape::N;
    constexpr int X_WARPS_NUMS =
        ThreadBlockShape::M / MmaShape::M * QuantType::X_BITS / WARP_M_TILES;
    constexpr int W_WARPS_NUMS =
        ThreadBlockShape::N / MmaShape::N * QuantType::W_BITS / WARP_N_TILES;
    // static_assert(WarpShape::K == MmaShape::K, "Only support warp shape K == Mma shape K.\n");
    // static_assert(ThreadBlockShape::M * QuantType::X_BITS % WarpShape::M == 0);
    // static_assert(ThreadBlockShape::N * QuantType::W_BITS % WarpShape::N == 0);
    // static_assert(WarpShape::M % MmaShape::M == 0);
    // static_assert(WarpShape::N % MmaShape::N == 0);
    // static_assert(ThreadBlockShape::K % WarpShape::K == 0);
    // static_assert(kThreadBlockStage > 0);
    static_assert(MmaShape::K == 128, "Only support MmaShape::K == 128.\n");
    // static_assert(WarpShape::M % 16 == 0, "Only support warp shape M>=16 for performance.\n");
    // determine the number of threads
    constexpr int blockDims = 32 * X_WARPS_NUMS * W_WARPS_NUMS;
    size_t smem_sz =
        aq_wmma::smemSizeInBytes(kThreadBlockStage, ThreadBlockShape::M, ThreadBlockShape::N,
                                 ThreadBlockShape::K, QuantType::X_BITS, QuantType::W_BITS);
    // printf("shared_mem_size:%d\n", smem_sz);
    if (smem_sz >= 32 * 1024) {
        // set kernel attribute
        if (cudaSuccess !=
                cudaFuncSetAttribute(
                    aq_wmma::bmmaArbitrarilyQuant<
                        QuantType::X_BITS, QuantType::W_BITS, QuantType::SIGNED,
                        ThreadBlockShape::M, ThreadBlockShape::N, ThreadBlockShape::K, WarpShape::M,
                        WarpShape::N, WarpShape::K, MmaShape::M, MmaShape::N, MmaShape::K,
                        blockDims, kThreadBlockStage, true, true, false>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_sz) ||
            cudaSuccess !=
                cudaFuncSetAttribute(
                    aq_wmma::bmmaArbitrarilyQuant<
                        QuantType::X_BITS, QuantType::W_BITS, QuantType::SIGNED,
                        ThreadBlockShape::M, ThreadBlockShape::N, ThreadBlockShape::K, WarpShape::M,
                        WarpShape::N, WarpShape::K, MmaShape::M, MmaShape::N, MmaShape::K,
                        blockDims, kThreadBlockStage, true, true, false>,
                    cudaFuncAttributePreferredSharedMemoryCarveout, 100)) {
            cudaError_t err = cudaGetLastError();
            std::cerr << "Set kernel attribute failed: " << cudaGetErrorString(err);
        }
    }
    // printf("dyn shared_mem_size:%d\n", this->_state.shared_mem_size);
    // calculate launch configuration
    int gdimX = CEIL(params.N, ThreadBlockShape::N);
    int gdimY = CEIL(params.M, ThreadBlockShape::M);
    dim3 gridDim = dim3(gdimX, gdimY, 1);
    dim3 blockDim = dim3(blockDims, 1, 1);
    aq_wmma::bmmaArbitrarilyQuant<QuantType::X_BITS, QuantType::W_BITS, QuantType::SIGNED,
                                  ThreadBlockShape::M, ThreadBlockShape::N, ThreadBlockShape::K,
                                  WarpShape::M, WarpShape::N, WarpShape::K, MmaShape::M, MmaShape::N,
                                  MmaShape::K, blockDims, kThreadBlockStage, true, true, false>
        <<<gridDim, blockDim, smem_sz, stream>>>(params.M, params.N, params.K, params.X, params.W,
                                                 params.D, params.C, params.bias);
    cudaError_t ret = cudaGetLastError();
    return ret;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
#define INSTANTIATE_FUN(X_BITS, W_BITS, SIGNED, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K,  \
                        MMA_M, MMA_N, MMA_K, NSTAGE)                                                \
    template cudaError_t                                                                            \
    launchWmmaAnyKernel<QuantType<X_BITS, W_BITS, SIGNED>, ShapeBase<BLOCK_M, BLOCK_N, BLOCK_K>,    \
                        ShapeBase<WARP_M, WARP_N, WARP_K>, ShapeBase<MMA_M, MMA_N, MMA_K>, NSTAGE>( \
        const AnyQuantParams &params, const cudaStream_t &stream)

////////////////////////////////////////////////////////////////////////////////////////////////////

// W4A4 int
// cta<8,8,128> warp<8,32,128> mma<8,8,128>   WARPS[4x1]
INSTANTIATE_FUN(4, 4, true, 8, 8, 128, 8, 32, 128, 8, 8, 128, 2);
INSTANTIATE_FUN(4, 4, true, 8, 8, 128, 8, 32, 128, 8, 8, 128, 3);
INSTANTIATE_FUN(4, 4, true, 8, 8, 128, 8, 32, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<16,16,128> mma<8,8,128>   WARPS[2x2]
INSTANTIATE_FUN(4, 4, true, 8, 8, 128, 16, 16, 128, 8, 8, 128, 2);
INSTANTIATE_FUN(4, 4, true, 8, 8, 128, 16, 16, 128, 8, 8, 128, 3);
INSTANTIATE_FUN(4, 4, true, 8, 8, 128, 16, 16, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<16,32,128> mma<8,8,128>   WARPS[2x1]
INSTANTIATE_FUN(4, 4, true, 8, 8, 128, 16, 32, 128, 8, 8, 128, 2);
INSTANTIATE_FUN(4, 4, true, 8, 8, 128, 16, 32, 128, 8, 8, 128, 3);
INSTANTIATE_FUN(4, 4, true, 8, 8, 128, 16, 32, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<32,32,128> mma<8,8,128>   WARPS[1x1]
INSTANTIATE_FUN(4, 4, true, 8, 8, 128, 32, 32, 128, 8, 8, 128, 2);
INSTANTIATE_FUN(4, 4, true, 8, 8, 128, 32, 32, 128, 8, 8, 128, 3);
INSTANTIATE_FUN(4, 4, true, 8, 8, 128, 32, 32, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<16,16,128> mma<8,8,128>   WARPS[2x2] split_k = 2
INSTANTIATE_FUN(4, 4, true, 8, 8, 256, 16, 16, 128, 8, 8, 128, 2);
INSTANTIATE_FUN(4, 4, true, 8, 8, 256, 16, 16, 128, 8, 8, 128, 3);
INSTANTIATE_FUN(4, 4, true, 8, 8, 256, 16, 16, 128, 8, 8, 128, 4);

// cta<16,16,128> warp<16,32,128> mma<8,8,128>   WARPS[4x2]
INSTANTIATE_FUN(4, 4, true, 16, 16, 128, 16, 32, 128, 8, 8, 128, 2);
INSTANTIATE_FUN(4, 4, true, 16, 16, 128, 16, 32, 128, 8, 8, 128, 3);
INSTANTIATE_FUN(4, 4, true, 16, 16, 128, 16, 32, 128, 8, 8, 128, 4);
// cta<16,16,128> warp<16,64,128> mma<8,8,128>   WARPS[4x1]
INSTANTIATE_FUN(4, 4, true, 16, 16, 128, 16, 64, 128, 8, 8, 128, 2);
INSTANTIATE_FUN(4, 4, true, 16, 16, 128, 16, 64, 128, 8, 8, 128, 3);
INSTANTIATE_FUN(4, 4, true, 16, 16, 128, 16, 64, 128, 8, 8, 128, 4);
// cta<16,16,128> warp<32,32,128> mma<8,8,128>   WARPS[2x2]
INSTANTIATE_FUN(4, 4, true, 16, 16, 128, 32, 32, 128, 8, 8, 128, 2);
INSTANTIATE_FUN(4, 4, true, 16, 16, 128, 32, 32, 128, 8, 8, 128, 3);
INSTANTIATE_FUN(4, 4, true, 16, 16, 128, 32, 32, 128, 8, 8, 128, 4);
// cta<16,16,256> warp<16,32,128> mma<8,8,128>   WARPS[4x2] split_k = 2
INSTANTIATE_FUN(4, 4, true, 16, 16, 256, 16, 32, 128, 8, 8, 128, 2);
INSTANTIATE_FUN(4, 4, true, 16, 16, 256, 16, 32, 128, 8, 8, 128, 3);
INSTANTIATE_FUN(4, 4, true, 16, 16, 256, 16, 32, 128, 8, 8, 128, 4);

// W4A4 uint
// cta<8,8,128> warp<8,32,128> mma<8,8,128>   WARPS[4x1]
INSTANTIATE_FUN(4, 4, false, 8, 8, 128, 8, 32, 128, 8, 8, 128, 2);
INSTANTIATE_FUN(4, 4, false, 8, 8, 128, 8, 32, 128, 8, 8, 128, 3);
INSTANTIATE_FUN(4, 4, false, 8, 8, 128, 8, 32, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<16,16,128> mma<8,8,128>   WARPS[2x2]
INSTANTIATE_FUN(4, 4, false, 8, 8, 128, 16, 16, 128, 8, 8, 128, 2);
INSTANTIATE_FUN(4, 4, false, 8, 8, 128, 16, 16, 128, 8, 8, 128, 3);
INSTANTIATE_FUN(4, 4, false, 8, 8, 128, 16, 16, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<16,32,128> mma<8,8,128>   WARPS[2x1]
INSTANTIATE_FUN(4, 4, false, 8, 8, 128, 16, 32, 128, 8, 8, 128, 2);
INSTANTIATE_FUN(4, 4, false, 8, 8, 128, 16, 32, 128, 8, 8, 128, 3);
INSTANTIATE_FUN(4, 4, false, 8, 8, 128, 16, 32, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<32,32,128> mma<8,8,128>   WARPS[1x1]
INSTANTIATE_FUN(4, 4, false, 8, 8, 128, 32, 32, 128, 8, 8, 128, 2);
INSTANTIATE_FUN(4, 4, false, 8, 8, 128, 32, 32, 128, 8, 8, 128, 3);
INSTANTIATE_FUN(4, 4, false, 8, 8, 128, 32, 32, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<16,16,128> mma<8,8,128>   WARPS[2x2] split_k = 2
INSTANTIATE_FUN(4, 4, false, 8, 8, 256, 16, 16, 128, 8, 8, 128, 2);
INSTANTIATE_FUN(4, 4, false, 8, 8, 256, 16, 16, 128, 8, 8, 128, 3);
INSTANTIATE_FUN(4, 4, false, 8, 8, 256, 16, 16, 128, 8, 8, 128, 4);

// cta<16,16,128> warp<16,32,128> mma<8,8,128>   WARPS[4x2]
INSTANTIATE_FUN(4, 4, false, 16, 16, 128, 16, 32, 128, 8, 8, 128, 2);
INSTANTIATE_FUN(4, 4, false, 16, 16, 128, 16, 32, 128, 8, 8, 128, 3);
INSTANTIATE_FUN(4, 4, false, 16, 16, 128, 16, 32, 128, 8, 8, 128, 4);
// cta<16,16,128> warp<16,64,128> mma<8,8,128>   WARPS[4x1]
INSTANTIATE_FUN(4, 4, false, 16, 16, 128, 16, 64, 128, 8, 8, 128, 2);
INSTANTIATE_FUN(4, 4, false, 16, 16, 128, 16, 64, 128, 8, 8, 128, 3);
INSTANTIATE_FUN(4, 4, false, 16, 16, 128, 16, 64, 128, 8, 8, 128, 4);
// cta<16,16,128> warp<32,32,128> mma<8,8,128>   WARPS[2x2]
INSTANTIATE_FUN(4, 4, false, 16, 16, 128, 32, 32, 128, 8, 8, 128, 2);
INSTANTIATE_FUN(4, 4, false, 16, 16, 128, 32, 32, 128, 8, 8, 128, 3);
INSTANTIATE_FUN(4, 4, false, 16, 16, 128, 32, 32, 128, 8, 8, 128, 4);
// cta<16,16,256> warp<16,32,128> mma<8,8,128>   WARPS[4x2] split_k = 2
INSTANTIATE_FUN(4, 4, false, 16, 16, 256, 16, 32, 128, 8, 8, 128, 2);
INSTANTIATE_FUN(4, 4, false, 16, 16, 256, 16, 32, 128, 8, 8, 128, 3);
INSTANTIATE_FUN(4, 4, false, 16, 16, 256, 16, 32, 128, 8, 8, 128, 4);