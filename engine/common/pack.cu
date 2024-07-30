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


#include "common/pack.h"

__global__ void pack4(const uint4 *in_data, unsigned int *pack_data, const int m, const int k)
{
    const unsigned int bit = blockIdx.y;
    const int L = m * k / 32;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < L; idx += blockDim.x) {
        unsigned int pack_val = 0;
        const uint4 val = in_data[idx]; // 32 * int4
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            pack_val |= ((((val.x >> ((8 - 1 - b) * 4)) >> bit) & 0x1) << (32 - 1 - b));
        }
#pragma unroll
        for (int b = 8; b < 16; ++b) {
            pack_val |= ((((val.y >> ((16 - 1 - b) * 4)) >> bit) & 0x1) << (32 - 1 - b));
        }
#pragma unroll
        for (int b = 16; b < 24; ++b) {
            pack_val |= ((((val.z >> ((24 - 1 - b) * 4)) >> bit) & 0x1) << (32 - 1 - b));
        }
#pragma unroll
        for (int b = 24; b < 32; ++b) {
            pack_val |= ((((val.w >> ((32 - 1 - b) * 4)) >> bit) & 0x1) << (32 - 1 - b));
        }
        // printf("m = %d, k = %d, L = %d, bit = %d, pack_val = %x\n", m, k, L, bit, pack_val);
        pack_data[bit * L + idx] = pack_val;
    }
}

__global__ void pack(const uint4 *in_data, unsigned int *pack_data, const int m, const int k)
{
    const unsigned int bit = blockIdx.y;
    const int L = m * (k / 32);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < L; idx += blockDim.x) {
        unsigned int pack_val = 0;
        // 1个线程需要读32个32-bit int来packing成一个32-bit int
        // 每次以128-bit int4读16Bytes, 循环8次
        for (int i = 0; i < 8; ++i) {
            const uint4 val = in_data[idx * 8 + i];
            pack_val |= ((val.x >> bit) & 0x1) << (32 - 1 - i * 4);
            pack_val |= ((val.y >> bit) & 0x1) << (32 - 2 - i * 4);
            pack_val |= ((val.z >> bit) & 0x1) << (32 - 3 - i * 4);
            pack_val |= ((val.w >> bit) & 0x1) << (32 - 4 - i * 4);
        }
        //printf("m = %d, k = %d, L = %d, bit = %d, pack_val = %x\n", m, k, L, bit, pack_val);
        pack_data[bit * L + idx] = pack_val;
    }
}

/// @brief k >= 128
/// @param in_data (m * k) int4 <-> (m * (k / 8)) int32
/// @param pack_data (BIT * m * (k / 32)) int 32, BIT=4
/// @param m
/// @param k
/// @param BIT
/// @param stream
cudaError_t launch_pack4(const int *in_data, int *pack_data, int m, int k, int BIT,
                         cudaStream_t stream)
{
    dim3 threads(min(max(32, m * (k / 32)), 512));
    dim3 blocks((m * (k / 32) + threads.x - 1) / threads.x, BIT);
    // printf("launch_pack4, m=%d, k=%d, blocks.x=%d, blocks.y=%d,
    // threads.x=%d\n", m, k, blocks.x, blocks.y, threads.x);
    pack4<<<blocks, threads, 0, stream>>>(reinterpret_cast<const uint4 *>(in_data),
                                          reinterpret_cast<unsigned int *>(pack_data), m, k);
    cudaError_t ret = cudaGetLastError();
    return ret;
}

/// @brief k >= 128
/// @param in_data (m * k) int4 <-> (m * (k / 8)) int32
/// @param pack_data (BIT * m * (k / 32)) int 32, BIT=4
/// @param m
/// @param k
/// @param BIT
/// @param stream
cudaError_t launch_pack(const int *in_data, int *pack_data, int m, int k, int BIT,
                        cudaStream_t stream)
{
    dim3 threads(min(max(32, m * (k / 32)), 512));
    dim3 blocks((m * (k / 32) + threads.x - 1) / threads.x, BIT);
    // printf("launch_pack, m=%d, k=%d, blocks.x=%d, blocks.y=%d,
    // threads.x=%d\n", m, k, blocks.x, blocks.y, threads.x);
    pack<<<blocks, threads, 0, stream>>>(reinterpret_cast<const uint4 *>(in_data),
                                         reinterpret_cast<unsigned int *>(pack_data), m, k);
    cudaError_t ret = cudaGetLastError();
    return ret;
}