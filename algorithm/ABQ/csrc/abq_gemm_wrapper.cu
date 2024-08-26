// Copyright 2024 ByteDance and/or its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "abq_gemm_wrapper.h"
#include "mma_any/aq_bmma_library.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/utils/memory_utils.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void pack_int(const uint4* in_data, unsigned int* packed_data, const int M, const int K)
{
    const unsigned int bit = blockIdx.y;
    const int          L   = M * (K / 32);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < L; idx += blockDim.x) {
        unsigned int pack_val = 0;
        // Each threads read thirty two 32-bit int elements to pack one 32-bit element
        // Read 16B at a time and loop 8 times
        for (int i = 0; i < 8; ++i) {
            const uint4 val = in_data[idx * 8 + i];
            pack_val |= ((val.x >> bit) & 0x1) << (32 - 1 - i * 4);
            pack_val |= ((val.y >> bit) & 0x1) << (32 - 2 - i * 4);
            pack_val |= ((val.z >> bit) & 0x1) << (32 - 3 - i * 4);
            pack_val |= ((val.w >> bit) & 0x1) << (32 - 4 - i * 4);
        }
        // printf("M = %d, K = %d, L = %d, bit = %d, pack_val = %x\n", M, K, L, bit, pack_val);
        packed_data[bit * L + idx] = pack_val;
    }
}

__global__ void pack_half(const half* in_data, const float* scale, unsigned int* packed_data, const int M, const int K)
{
    const unsigned int bit         = blockIdx.y;
    const int          L           = M * (K / 32);
    const half         local_scale = __float2half(scale[0]);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < L; idx += blockDim.x) {
        unsigned int pack_val = 0;
        // Each threads read thirty two 16-bit half elements to pack one 32-bit element
        // Read 16B at a time and loop 4 times
        for (int i = 0; i < 4; ++i) {
            HalfVector<8> val;
            val.ld(in_data + idx * 32 + i * 8);
            // half val = in_data[idx * 32 + i];
            pack_val |= ((static_cast<int>(val.x[0] * local_scale) >> bit) & 0x1) << (32 - 1 - i * 8);
            pack_val |= ((static_cast<int>(val.x[1] * local_scale) >> bit) & 0x1) << (32 - 2 - i * 8);
            pack_val |= ((static_cast<int>(val.x[2] * local_scale) >> bit) & 0x1) << (32 - 3 - i * 8);
            pack_val |= ((static_cast<int>(val.x[3] * local_scale) >> bit) & 0x1) << (32 - 4 - i * 8);
            pack_val |= ((static_cast<int>(val.x[4] * local_scale) >> bit) & 0x1) << (32 - 5 - i * 8);
            pack_val |= ((static_cast<int>(val.x[5] * local_scale) >> bit) & 0x1) << (32 - 6 - i * 8);
            pack_val |= ((static_cast<int>(val.x[6] * local_scale) >> bit) & 0x1) << (32 - 7 - i * 8);
            pack_val |= ((static_cast<int>(val.x[7] * local_scale) >> bit) & 0x1) << (32 - 8 - i * 8);
        }
        // printf("M = %d, K = %d, L = %d, bit = %d, pack_val = %x\n", M, K, L, bit, pack_val);
        packed_data[bit * L + idx] = pack_val;
    }
}

ABQGEMMWrapper::ABQGEMMWrapper(int X_BITS, int W_BITS, bool SIGNED): x_bits_(X_BITS), w_bits_(W_BITS), signed_(SIGNED)
{
}

ABQGEMMWrapper::~ABQGEMMWrapper() {}

void ABQGEMMWrapper::pack(
    const half* in_data, const float* scale, int* packed_data, int M, int K, int BIT, cudaStream_t stream)
{
    // printf("[ABQ][DEBUG] M,K,BIT=%d,%d,%d\n", M, K, BIT);
    dim3 threads(min(max(32, M * (K / 32)), 512));
    dim3 blocks((M * (K / 32) + threads.x - 1) / threads.x, BIT);
    pack_half<<<blocks, threads, 0, stream>>>(in_data, scale, reinterpret_cast<unsigned int*>(packed_data), M, K);
}

void ABQGEMMWrapper::pack(const int* in_data, int* packed_data, int M, int K, int BIT, cudaStream_t stream = NULL)
{
    dim3 threads(min(max(32, M * (K / 32)), 512));
    dim3 blocks((M * (K / 32) + threads.x - 1) / threads.x, BIT);
    pack_int<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint4*>(in_data), reinterpret_cast<unsigned int*>(packed_data), M, K);
}

void ABQGEMMWrapper::gemm(const int    M,
                          const int    N,
                          const int    K,
                          const half*  A,
                          const int*   B,
                          const half*  C,
                          half*        D,
                          const float* scale,
                          const float* scale_inter,
                          const float* scale_out,
                          bool         bias,
                          char*        abq_gemm_workspace,
                          size_t       abq_gemm_ws_bytes,
                          cudaStream_t stream = NULL)
{
    // quant+packing
    pack(A, scale, reinterpret_cast<int*>(abq_gemm_workspace), M, K, x_bits_, stream);
    // printf("[ABQ][DEBUG] M,N,K=%d,%d,%d\n", M, N, K);
    AqBMMAInitFn_t init_fn;
    AqBMMAExecFn_t exec_fn;
    AqBMMAOpState  state;
    if (K < 128 || K % 128 != 0) {
        printf("[ABQ][Error] unsupport K = %d\n", K);
        return;
    }
    // w2a8
    if (w_bits_ == 2 && x_bits_ == 8) {
        if(M == 1){
            if (signed_) {
                init_fn = AqBMMA_8x2xtrue_1x48x512_8x24x128_8x8x128_4_InitFn;
                exec_fn = AqBMMA_8x2xtrue_1x48x512_8x24x128_8x8x128_4_ExecFn;
            }
            else {
                init_fn = AqBMMA_8x2xfalse_1x48x512_8x24x128_8x8x128_4_InitFn;
                exec_fn = AqBMMA_8x2xfalse_1x48x512_8x24x128_8x8x128_4_ExecFn;
            }
        }else{
            if (signed_) {
                init_fn = AqBMMA_8x2xtrue_8x48x256_32x48x128_8x8x128_5_InitFn;
                exec_fn = AqBMMA_8x2xtrue_8x48x256_32x48x128_8x8x128_5_ExecFn;
            }
            else {
                init_fn = AqBMMA_8x2xfalse_8x48x256_32x48x128_8x8x128_5_InitFn;
                exec_fn = AqBMMA_8x2xfalse_8x48x256_32x48x128_8x8x128_5_ExecFn;
            }
        }
        state = (*init_fn)(
            reinterpret_cast<int*>(abq_gemm_workspace), B, M, N, K, D, reinterpret_cast<const half*>(scale_out), bias);
    }
    else {
        printf("[ABQ][Error] unsupport w%da%d\n", w_bits_, x_bits_);
        return;
    }

    if (!state.initSuccess) {
        printf("[ABQ][Error] return due to unsuccessful initialization.\n");
        return;
    }
    (*exec_fn)(state, stream);
}
