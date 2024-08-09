// Copyright (C) ABQ-LLM (liusongwei.zju@bytedance.com)
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

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <vector>
using namespace std;

int main(int argc, char **argv)
{
    if (argc < 5) {
        printf("Usage: ./gen_kernel X_BITS W_BITS SIGNED K_STAGE\n");
        return -1;
    }
    bool use_mma = true;
    int x_bits = atoi(argv[1]);
    int w_bits = atoi(argv[2]);
    bool quant_sign = atoi(argv[3]) == 1;
    int k_stages = atoi(argv[4]);
    printf("////// W%dA%d %s\n", w_bits, x_bits, quant_sign ? "int" : "uint");
    constexpr int SHARED_MEM_PER_SM = 102400; // bytes
    constexpr int MAX_SHARED_MEM_PER_BLOCK_OPTIN = 101376; //bytes
    constexpr int REGS_PER_THREAD = 255;
    unordered_set<string> st;
    unordered_map<int, vector<int>> BMS{ { 2, { 4, 8 } },   { 3, { 2, 8 } },    { 4, { 2, 4, 8 } },
                                         { 5, { 1, 8 } },   { 6, { 1, 4, 8 } }, { 7, { 1, 8 } },
                                         { 8, { 1, 4, 8 } } };
    // BK = 128 is inefficient
    vector<int> BKS{ 256, 384, 512 };
    //vector<int> BKS{ 512 };
    vector<int> batchs{ 1, 4, 8 };
    for (auto BLOCK_K : BKS) {
        for (int X_WARPS_NUMS = 1; X_WARPS_NUMS <= 32; X_WARPS_NUMS *= 2) {
            for (int W_WARPS_NUMS = 1; W_WARPS_NUMS <= 32; W_WARPS_NUMS *= 2) {
                if (X_WARPS_NUMS * W_WARPS_NUMS > 32) // out of warps in one block
                    continue;
                if (X_WARPS_NUMS * W_WARPS_NUMS == 1) // inefficient
                    continue;
                if (X_WARPS_NUMS * W_WARPS_NUMS >= 8) // inefficient
                    continue;
                for (int WARP_M = 8; WARP_M <= 64; WARP_M += 8) {
                    for (int WARP_N = 8; WARP_N <= 128; WARP_N += 8) {
                        int MMA_M = 8;
                        int MMA_N = 8;
                        int MMA_K = 128;
                        if (WARP_M % 16 == 0) {
                            MMA_M = 16;
                            MMA_K = 256;
                        }
                        int WARP_K = MMA_K;
                        if (BLOCK_K % MMA_K != 0)
                            continue;
                        int WARP_M_TILES = WARP_M / MMA_M;
                        int WARP_N_TILES = WARP_N / MMA_N;

                        int REGS_A = WARP_M_TILES * MMA_M * MMA_K / 32 / 32;
                        int REGS_B = WARP_N_TILES * MMA_N * MMA_K / 32 / 32;
                        int REGS_C = WARP_M_TILES * WARP_N_TILES * MMA_M * MMA_N / 32;
                        if (REGS_A + REGS_B + REGS_C > REGS_PER_THREAD) // out of regs in one block
                            continue;

                        int BLOCK_M = X_WARPS_NUMS * WARP_M_TILES * MMA_M / x_bits;
                        int BLOCK_N = W_WARPS_NUMS * WARP_N_TILES * MMA_N / w_bits;
                        if (BLOCK_M > 8) // inefficient for small batch
                            continue;

                        bool efficient_m = false;
                        for (auto b : BMS[x_bits]) {
                            if (BLOCK_M == b) {
                                efficient_m = true;
                                break;
                            }
                        }
                        if (!efficient_m)
                            continue;

                        if (BLOCK_N < 32 ||
                            BLOCK_N % 16 != 0) // BLOCK_N < 32 is inefficient for large N,K
                            continue;
                        if (BLOCK_N > MMA_K) {
                            if (BLOCK_N % MMA_K != 0)
                                continue;
                        }

                        int SKEW = w_bits * BLOCK_N % 16 == 0 ? 8 : 0;
                        size_t input_buffer_size =
                            2 * BLOCK_M * BLOCK_K * x_bits / 8 + 2 * BLOCK_N * BLOCK_K * w_bits / 8;
                        size_t output_buffer_size = (MMA_M * WARP_M_TILES * X_WARPS_NUMS) *
                                                    (MMA_N * WARP_N_TILES * W_WARPS_NUMS + SKEW) *
                                                    sizeof(int);
                        size_t max_stages;
                        if (output_buffer_size > input_buffer_size) {
                            max_stages = output_buffer_size / (BLOCK_M * BLOCK_K * x_bits / 8 +
                                                               BLOCK_N * BLOCK_K * w_bits / 8);
                        } else {
                            max_stages = k_stages;
                        }
                        size_t shared_mem_size =
                            max(input_buffer_size, output_buffer_size); // bytes
                        if (shared_mem_size >= SHARED_MEM_PER_SM ||
                            shared_mem_size >=
                                MAX_SHARED_MEM_PER_BLOCK_OPTIN) // out of shared memory
                            continue;

                        string key = to_string(BLOCK_M) + "," + to_string(BLOCK_N) + "," +
                                     to_string(BLOCK_K) + "," + to_string(WARP_M) + "," +
                                     to_string(WARP_N) + "," + to_string(WARP_K);
                        if (st.find(key) != st.end()) // alread exists
                            continue;

                        st.insert(key);
                        printf("// cta<%d,%d,%d> warp<%d,%d,%d> mma<%d,%d,%d>   WARPS[%dx%d]\n",
                               BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N,
                               MMA_K, X_WARPS_NUMS, W_WARPS_NUMS);
                        for (int kThreadBlockStage = 2; kThreadBlockStage <= max_stages;
                             ++kThreadBlockStage) {
                            size_t input_buffer_size =
                                kThreadBlockStage * BLOCK_M * BLOCK_K * x_bits / 8 +
                                kThreadBlockStage * BLOCK_N * BLOCK_K * w_bits / 8;
                            size_t output_buffer_size =
                                (MMA_M * WARP_M_TILES * X_WARPS_NUMS) *
                                (MMA_N * WARP_N_TILES * W_WARPS_NUMS + SKEW) * sizeof(int);
                            size_t shared_mem_size =
                                max(input_buffer_size, output_buffer_size); // bytes
                            if (shared_mem_size >= SHARED_MEM_PER_SM ||
                                shared_mem_size >=
                                    MAX_SHARED_MEM_PER_BLOCK_OPTIN) // out of shared memory
                                continue;
                            printf(
                                "AQ_INSTANTIATE_FUN(%s, %d, %d, %s, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d);\n",
                                use_mma ? "AqBMMA" : "AqBWMMA", x_bits, w_bits,
                                quant_sign ? "true" : "false", BLOCK_M, BLOCK_N, BLOCK_K, WARP_M,
                                WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, kThreadBlockStage);
                        }
                    }
                }
            }
        }
    }

    return 0;
}