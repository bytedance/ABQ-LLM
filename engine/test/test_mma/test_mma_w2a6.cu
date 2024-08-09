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

#include <string>
#include <sstream>
#include "mma_any/aq_bmma_library.h"
#include "mma_any/aq_bmma_op.h"
#include "test/test_mma/test_mma.h"

void test_mma_w2a6(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream)
{
#ifdef W2A6
    std::string config_str;
    std::stringstream s;
    s << x_bits << " " << w_bits << " " << m << " " << n << " " << k << " ";
    if (quant_sign) {
        s << "sign ";
    } else {
        s << "unsigned ";
    }
    config_str = s.str();
    float exec_dur = 0;
    float pack_dur = 0;
    float true_gflop_count = (float)m / 1e9 * n * k * 2 * x_bits * w_bits;
    float gflop_count = (float)m / 1e9 * n * k * 2;
    float max_gflop = 0;
    std::stringstream best_config;

    if (quant_sign) {
        ////// W2A6 int
        // cta<1,32,256> warp<8,32,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 1, 32, 256, 8, 32, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 32, 256, 8, 32, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 32, 256, 8, 32, 128, 8, 8, 128, 4);
        // cta<1,48,256> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 1, 48, 256, 8, 48, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 48, 256, 8, 48, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 48, 256, 8, 48, 128, 8, 8, 128, 4);
        // cta<1,64,256> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 1, 64, 256, 8, 64, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 64, 256, 8, 64, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 64, 256, 8, 64, 128, 8, 8, 128, 4);
        // cta<1,80,256> warp<8,80,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 1, 80, 256, 8, 80, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 80, 256, 8, 80, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 80, 256, 8, 80, 128, 8, 8, 128, 4);
        // cta<1,96,256> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 1, 96, 256, 8, 96, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 96, 256, 8, 96, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 96, 256, 8, 96, 128, 8, 8, 128, 4);
        // cta<1,112,256> warp<8,112,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 1, 112, 256, 8, 112, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 112, 256, 8, 112, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 112, 256, 8, 112, 128, 8, 8, 128, 4);
        // cta<1,128,256> warp<8,128,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 1, 128, 256, 8, 128, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 128, 256, 8, 128, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 128, 256, 8, 128, 128, 8, 8, 128, 4);
        // cta<4,32,256> warp<24,32,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 4, 32, 256, 24, 32, 128, 8, 8, 128, 2);
        // cta<4,48,256> warp<24,48,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 4, 48, 256, 24, 48, 128, 8, 8, 128, 2);
        // cta<4,64,256> warp<24,64,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 4, 64, 256, 24, 64, 128, 8, 8, 128, 2);
        // cta<4,80,256> warp<24,80,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 4, 80, 256, 24, 80, 128, 8, 8, 128, 2);
        // cta<4,96,256> warp<24,96,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 4, 96, 256, 24, 96, 128, 8, 8, 128, 2);
        // cta<4,112,256> warp<24,112,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 4, 112, 256, 24, 112, 128, 8, 8, 128, 2);
        // cta<4,128,256> warp<24,128,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 4, 128, 256, 24, 128, 128, 8, 8, 128, 2);
        // cta<8,32,256> warp<48,32,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 8, 32, 256, 48, 32, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 32, 256, 48, 32, 128, 8, 8, 128, 3);
        // cta<8,48,256> warp<48,48,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 8, 48, 256, 48, 48, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 48, 256, 48, 48, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 8, 48, 256, 48, 48, 128, 8, 8, 128, 4);
        // cta<8,64,256> warp<48,64,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 8, 64, 256, 48, 64, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 64, 256, 48, 64, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 8, 64, 256, 48, 64, 128, 8, 8, 128, 4);
        // cta<8,80,256> warp<48,80,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 8, 80, 256, 48, 80, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 80, 256, 48, 80, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 8, 80, 256, 48, 80, 128, 8, 8, 128, 4);
        // cta<8,96,256> warp<48,96,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 8, 96, 256, 48, 96, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 96, 256, 48, 96, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 8, 96, 256, 48, 96, 128, 8, 8, 128, 4);
        TEST(6, 2, true, 8, 96, 256, 48, 96, 128, 8, 8, 128, 5);
        // cta<8,112,256> warp<48,112,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 8, 112, 256, 48, 112, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 112, 256, 48, 112, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 8, 112, 256, 48, 112, 128, 8, 8, 128, 4);
        TEST(6, 2, true, 8, 112, 256, 48, 112, 128, 8, 8, 128, 5);
        // cta<8,128,256> warp<48,128,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 8, 128, 256, 48, 128, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 128, 256, 48, 128, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 8, 128, 256, 48, 128, 128, 8, 8, 128, 4);
        TEST(6, 2, true, 8, 128, 256, 48, 128, 128, 8, 8, 128, 5);
        // cta<1,32,256> warp<8,16,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 1, 32, 256, 8, 16, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 32, 256, 8, 16, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 32, 256, 8, 16, 128, 8, 8, 128, 4);
        // cta<1,48,256> warp<8,24,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 1, 48, 256, 8, 24, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 48, 256, 8, 24, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 48, 256, 8, 24, 128, 8, 8, 128, 4);
        // cta<1,64,256> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 1, 64, 256, 8, 32, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 64, 256, 8, 32, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 64, 256, 8, 32, 128, 8, 8, 128, 4);
        // cta<1,80,256> warp<8,40,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 1, 80, 256, 8, 40, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 80, 256, 8, 40, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 80, 256, 8, 40, 128, 8, 8, 128, 4);
        // cta<1,96,256> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 1, 96, 256, 8, 48, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 96, 256, 8, 48, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 96, 256, 8, 48, 128, 8, 8, 128, 4);
        // cta<1,112,256> warp<8,56,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 1, 112, 256, 8, 56, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 112, 256, 8, 56, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 112, 256, 8, 56, 128, 8, 8, 128, 4);
        // cta<1,128,256> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 1, 128, 256, 8, 64, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 128, 256, 8, 64, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 128, 256, 8, 64, 128, 8, 8, 128, 4);
        // cta<1,256,256> warp<8,128,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 1, 256, 256, 8, 128, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 256, 256, 8, 128, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 256, 256, 8, 128, 128, 8, 8, 128, 4);
        // cta<4,32,256> warp<24,16,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 4, 32, 256, 24, 16, 128, 8, 8, 128, 2);
        // cta<4,48,256> warp<24,24,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 4, 48, 256, 24, 24, 128, 8, 8, 128, 2);
        // cta<4,64,256> warp<24,32,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 4, 64, 256, 24, 32, 128, 8, 8, 128, 2);
        // cta<4,80,256> warp<24,40,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 4, 80, 256, 24, 40, 128, 8, 8, 128, 2);
        // cta<4,96,256> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 4, 96, 256, 24, 48, 128, 8, 8, 128, 2);
        // cta<4,112,256> warp<24,56,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 4, 112, 256, 24, 56, 128, 8, 8, 128, 2);
        // cta<4,128,256> warp<24,64,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 4, 128, 256, 24, 64, 128, 8, 8, 128, 2);
        // cta<4,256,256> warp<24,128,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 4, 256, 256, 24, 128, 128, 8, 8, 128, 2);
        // cta<8,32,256> warp<48,16,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 8, 32, 256, 48, 16, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 32, 256, 48, 16, 128, 8, 8, 128, 3);
        // cta<8,48,256> warp<48,24,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 8, 48, 256, 48, 24, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 48, 256, 48, 24, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 8, 48, 256, 48, 24, 128, 8, 8, 128, 4);
        // cta<8,64,256> warp<48,32,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 8, 64, 256, 48, 32, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 64, 256, 48, 32, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 8, 64, 256, 48, 32, 128, 8, 8, 128, 4);
        // cta<8,80,256> warp<48,40,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 8, 80, 256, 48, 40, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 80, 256, 48, 40, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 8, 80, 256, 48, 40, 128, 8, 8, 128, 4);
        // cta<8,96,256> warp<48,48,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 8, 96, 256, 48, 48, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 96, 256, 48, 48, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 8, 96, 256, 48, 48, 128, 8, 8, 128, 4);
        TEST(6, 2, true, 8, 96, 256, 48, 48, 128, 8, 8, 128, 5);
        // cta<8,112,256> warp<48,56,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 8, 112, 256, 48, 56, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 112, 256, 48, 56, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 8, 112, 256, 48, 56, 128, 8, 8, 128, 4);
        TEST(6, 2, true, 8, 112, 256, 48, 56, 128, 8, 8, 128, 5);
        // cta<8,128,256> warp<48,64,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 8, 128, 256, 48, 64, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 128, 256, 48, 64, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 8, 128, 256, 48, 64, 128, 8, 8, 128, 4);
        TEST(6, 2, true, 8, 128, 256, 48, 64, 128, 8, 8, 128, 5);
        // cta<8,256,256> warp<48,128,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 8, 256, 256, 48, 128, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 256, 256, 48, 128, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 8, 256, 256, 48, 128, 128, 8, 8, 128, 4);
        TEST(6, 2, true, 8, 256, 256, 48, 128, 128, 8, 8, 128, 5);
        // cta<8,32,256> warp<24,64,128> mma<8,8,128>   WARPS[2x1]
        TEST(6, 2, true, 8, 32, 256, 24, 64, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 32, 256, 24, 64, 128, 8, 8, 128, 3);
        // cta<8,48,256> warp<24,96,128> mma<8,8,128>   WARPS[2x1]
        TEST(6, 2, true, 8, 48, 256, 24, 96, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 48, 256, 24, 96, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 8, 48, 256, 24, 96, 128, 8, 8, 128, 4);
        // cta<8,64,256> warp<24,128,128> mma<8,8,128>   WARPS[2x1]
        TEST(6, 2, true, 8, 64, 256, 24, 128, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 64, 256, 24, 128, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 8, 64, 256, 24, 128, 128, 8, 8, 128, 4);
        // cta<8,32,256> warp<24,32,128> mma<8,8,128>   WARPS[2x2]
        TEST(6, 2, true, 8, 32, 256, 24, 32, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 32, 256, 24, 32, 128, 8, 8, 128, 3);
        // cta<8,48,256> warp<24,48,128> mma<8,8,128>   WARPS[2x2]
        TEST(6, 2, true, 8, 48, 256, 24, 48, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 48, 256, 24, 48, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 8, 48, 256, 24, 48, 128, 8, 8, 128, 4);
        // cta<8,64,256> warp<24,64,128> mma<8,8,128>   WARPS[2x2]
        TEST(6, 2, true, 8, 64, 256, 24, 64, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 64, 256, 24, 64, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 8, 64, 256, 24, 64, 128, 8, 8, 128, 4);
        // cta<8,80,256> warp<24,80,128> mma<8,8,128>   WARPS[2x2]
        TEST(6, 2, true, 8, 80, 256, 24, 80, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 80, 256, 24, 80, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 8, 80, 256, 24, 80, 128, 8, 8, 128, 4);
        // cta<8,96,256> warp<24,96,128> mma<8,8,128>   WARPS[2x2]
        TEST(6, 2, true, 8, 96, 256, 24, 96, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 96, 256, 24, 96, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 8, 96, 256, 24, 96, 128, 8, 8, 128, 4);
        TEST(6, 2, true, 8, 96, 256, 24, 96, 128, 8, 8, 128, 5);
        // cta<8,112,256> warp<24,112,128> mma<8,8,128>   WARPS[2x2]
        TEST(6, 2, true, 8, 112, 256, 24, 112, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 112, 256, 24, 112, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 8, 112, 256, 24, 112, 128, 8, 8, 128, 4);
        TEST(6, 2, true, 8, 112, 256, 24, 112, 128, 8, 8, 128, 5);
        // cta<8,128,256> warp<24,128,128> mma<8,8,128>   WARPS[2x2]
        TEST(6, 2, true, 8, 128, 256, 24, 128, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 128, 256, 24, 128, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 8, 128, 256, 24, 128, 128, 8, 8, 128, 4);
        TEST(6, 2, true, 8, 128, 256, 24, 128, 128, 8, 8, 128, 5);
        // cta<1,32,384> warp<8,32,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 1, 32, 384, 8, 32, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 32, 384, 8, 32, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 32, 384, 8, 32, 128, 8, 8, 128, 4);
        // cta<1,48,384> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 1, 48, 384, 8, 48, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 48, 384, 8, 48, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 48, 384, 8, 48, 128, 8, 8, 128, 4);
        // cta<1,64,384> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 1, 64, 384, 8, 64, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 64, 384, 8, 64, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 64, 384, 8, 64, 128, 8, 8, 128, 4);
        // cta<1,80,384> warp<8,80,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 1, 80, 384, 8, 80, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 80, 384, 8, 80, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 80, 384, 8, 80, 128, 8, 8, 128, 4);
        // cta<1,96,384> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 1, 96, 384, 8, 96, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 96, 384, 8, 96, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 96, 384, 8, 96, 128, 8, 8, 128, 4);
        // cta<1,112,384> warp<8,112,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 1, 112, 384, 8, 112, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 112, 384, 8, 112, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 112, 384, 8, 112, 128, 8, 8, 128, 4);
        // cta<1,128,384> warp<8,128,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 1, 128, 384, 8, 128, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 128, 384, 8, 128, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 128, 384, 8, 128, 128, 8, 8, 128, 4);
        // cta<4,32,384> warp<24,32,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 4, 32, 384, 24, 32, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 32, 384, 24, 32, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 32, 384, 24, 32, 128, 8, 8, 128, 4);
        // cta<4,48,384> warp<24,48,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 4, 48, 384, 24, 48, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 48, 384, 24, 48, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 48, 384, 24, 48, 128, 8, 8, 128, 4);
        // cta<4,64,384> warp<24,64,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 4, 64, 384, 24, 64, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 64, 384, 24, 64, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 64, 384, 24, 64, 128, 8, 8, 128, 4);
        // cta<4,80,384> warp<24,80,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 4, 80, 384, 24, 80, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 80, 384, 24, 80, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 80, 384, 24, 80, 128, 8, 8, 128, 4);
        // cta<4,96,384> warp<24,96,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 4, 96, 384, 24, 96, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 96, 384, 24, 96, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 96, 384, 24, 96, 128, 8, 8, 128, 4);
        // cta<4,112,384> warp<24,112,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 4, 112, 384, 24, 112, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 112, 384, 24, 112, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 112, 384, 24, 112, 128, 8, 8, 128, 4);
        // cta<4,128,384> warp<24,128,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 4, 128, 384, 24, 128, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 128, 384, 24, 128, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 128, 384, 24, 128, 128, 8, 8, 128, 4);
        // cta<8,32,384> warp<48,32,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 8, 32, 384, 48, 32, 128, 8, 8, 128, 2);
        // cta<8,48,384> warp<48,48,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 8, 48, 384, 48, 48, 128, 8, 8, 128, 2);
        // cta<8,64,384> warp<48,64,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 8, 64, 384, 48, 64, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 64, 384, 48, 64, 128, 8, 8, 128, 3);
        // cta<8,80,384> warp<48,80,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 8, 80, 384, 48, 80, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 80, 384, 48, 80, 128, 8, 8, 128, 3);
        // cta<8,96,384> warp<48,96,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 8, 96, 384, 48, 96, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 96, 384, 48, 96, 128, 8, 8, 128, 3);
        // cta<8,112,384> warp<48,112,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 8, 112, 384, 48, 112, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 112, 384, 48, 112, 128, 8, 8, 128, 3);
        // cta<8,128,384> warp<48,128,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 8, 128, 384, 48, 128, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 128, 384, 48, 128, 128, 8, 8, 128, 3);
        // cta<1,32,384> warp<8,16,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 1, 32, 384, 8, 16, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 32, 384, 8, 16, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 32, 384, 8, 16, 128, 8, 8, 128, 4);
        // cta<1,48,384> warp<8,24,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 1, 48, 384, 8, 24, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 48, 384, 8, 24, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 48, 384, 8, 24, 128, 8, 8, 128, 4);
        // cta<1,64,384> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 1, 64, 384, 8, 32, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 64, 384, 8, 32, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 64, 384, 8, 32, 128, 8, 8, 128, 4);
        // cta<1,80,384> warp<8,40,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 1, 80, 384, 8, 40, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 80, 384, 8, 40, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 80, 384, 8, 40, 128, 8, 8, 128, 4);
        // cta<1,96,384> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 1, 96, 384, 8, 48, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 96, 384, 8, 48, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 96, 384, 8, 48, 128, 8, 8, 128, 4);
        // cta<1,112,384> warp<8,56,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 1, 112, 384, 8, 56, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 112, 384, 8, 56, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 112, 384, 8, 56, 128, 8, 8, 128, 4);
        // cta<1,128,384> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 1, 128, 384, 8, 64, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 128, 384, 8, 64, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 128, 384, 8, 64, 128, 8, 8, 128, 4);
        // cta<1,256,384> warp<8,128,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 1, 256, 384, 8, 128, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 256, 384, 8, 128, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 256, 384, 8, 128, 128, 8, 8, 128, 4);
        // cta<4,32,384> warp<24,16,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 4, 32, 384, 24, 16, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 32, 384, 24, 16, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 32, 384, 24, 16, 128, 8, 8, 128, 4);
        // cta<4,48,384> warp<24,24,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 4, 48, 384, 24, 24, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 48, 384, 24, 24, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 48, 384, 24, 24, 128, 8, 8, 128, 4);
        // cta<4,64,384> warp<24,32,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 4, 64, 384, 24, 32, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 64, 384, 24, 32, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 64, 384, 24, 32, 128, 8, 8, 128, 4);
        // cta<4,80,384> warp<24,40,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 4, 80, 384, 24, 40, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 80, 384, 24, 40, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 80, 384, 24, 40, 128, 8, 8, 128, 4);
        // cta<4,96,384> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 4, 96, 384, 24, 48, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 96, 384, 24, 48, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 96, 384, 24, 48, 128, 8, 8, 128, 4);
        // cta<4,112,384> warp<24,56,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 4, 112, 384, 24, 56, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 112, 384, 24, 56, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 112, 384, 24, 56, 128, 8, 8, 128, 4);
        // cta<4,128,384> warp<24,64,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 4, 128, 384, 24, 64, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 128, 384, 24, 64, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 128, 384, 24, 64, 128, 8, 8, 128, 4);
        // cta<4,256,384> warp<24,128,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 4, 256, 384, 24, 128, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 256, 384, 24, 128, 128, 8, 8, 128, 3);
        // cta<8,32,384> warp<48,16,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 8, 32, 384, 48, 16, 128, 8, 8, 128, 2);
        // cta<8,48,384> warp<48,24,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 8, 48, 384, 48, 24, 128, 8, 8, 128, 2);
        // cta<8,64,384> warp<48,32,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 8, 64, 384, 48, 32, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 64, 384, 48, 32, 128, 8, 8, 128, 3);
        // cta<8,80,384> warp<48,40,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 8, 80, 384, 48, 40, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 80, 384, 48, 40, 128, 8, 8, 128, 3);
        // cta<8,96,384> warp<48,48,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 8, 96, 384, 48, 48, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 96, 384, 48, 48, 128, 8, 8, 128, 3);
        // cta<8,112,384> warp<48,56,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 8, 112, 384, 48, 56, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 112, 384, 48, 56, 128, 8, 8, 128, 3);
        // cta<8,128,384> warp<48,64,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 8, 128, 384, 48, 64, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 128, 384, 48, 64, 128, 8, 8, 128, 3);
        // cta<8,256,384> warp<48,128,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 8, 256, 384, 48, 128, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 256, 384, 48, 128, 128, 8, 8, 128, 3);
        // cta<8,32,384> warp<24,64,128> mma<8,8,128>   WARPS[2x1]
        TEST(6, 2, true, 8, 32, 384, 24, 64, 128, 8, 8, 128, 2);
        // cta<8,48,384> warp<24,96,128> mma<8,8,128>   WARPS[2x1]
        TEST(6, 2, true, 8, 48, 384, 24, 96, 128, 8, 8, 128, 2);
        // cta<8,64,384> warp<24,128,128> mma<8,8,128>   WARPS[2x1]
        TEST(6, 2, true, 8, 64, 384, 24, 128, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 64, 384, 24, 128, 128, 8, 8, 128, 3);
        // cta<8,32,384> warp<24,32,128> mma<8,8,128>   WARPS[2x2]
        TEST(6, 2, true, 8, 32, 384, 24, 32, 128, 8, 8, 128, 2);
        // cta<8,48,384> warp<24,48,128> mma<8,8,128>   WARPS[2x2]
        TEST(6, 2, true, 8, 48, 384, 24, 48, 128, 8, 8, 128, 2);
        // cta<8,64,384> warp<24,64,128> mma<8,8,128>   WARPS[2x2]
        TEST(6, 2, true, 8, 64, 384, 24, 64, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 64, 384, 24, 64, 128, 8, 8, 128, 3);
        // cta<8,80,384> warp<24,80,128> mma<8,8,128>   WARPS[2x2]
        TEST(6, 2, true, 8, 80, 384, 24, 80, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 80, 384, 24, 80, 128, 8, 8, 128, 3);
        // cta<8,96,384> warp<24,96,128> mma<8,8,128>   WARPS[2x2]
        TEST(6, 2, true, 8, 96, 384, 24, 96, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 96, 384, 24, 96, 128, 8, 8, 128, 3);
        // cta<8,112,384> warp<24,112,128> mma<8,8,128>   WARPS[2x2]
        TEST(6, 2, true, 8, 112, 384, 24, 112, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 112, 384, 24, 112, 128, 8, 8, 128, 3);
        // cta<8,128,384> warp<24,128,128> mma<8,8,128>   WARPS[2x2]
        TEST(6, 2, true, 8, 128, 384, 24, 128, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 128, 384, 24, 128, 128, 8, 8, 128, 3);
        // cta<1,32,512> warp<8,32,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 1, 32, 512, 8, 32, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 32, 512, 8, 32, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 32, 512, 8, 32, 128, 8, 8, 128, 4);
        // cta<1,48,512> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 1, 48, 512, 8, 48, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 48, 512, 8, 48, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 48, 512, 8, 48, 128, 8, 8, 128, 4);
        // cta<1,64,512> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 1, 64, 512, 8, 64, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 64, 512, 8, 64, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 64, 512, 8, 64, 128, 8, 8, 128, 4);
        // cta<1,80,512> warp<8,80,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 1, 80, 512, 8, 80, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 80, 512, 8, 80, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 80, 512, 8, 80, 128, 8, 8, 128, 4);
        // cta<1,96,512> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 1, 96, 512, 8, 96, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 96, 512, 8, 96, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 96, 512, 8, 96, 128, 8, 8, 128, 4);
        // cta<1,112,512> warp<8,112,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 1, 112, 512, 8, 112, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 112, 512, 8, 112, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 112, 512, 8, 112, 128, 8, 8, 128, 4);
        // cta<1,128,512> warp<8,128,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 1, 128, 512, 8, 128, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 128, 512, 8, 128, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 128, 512, 8, 128, 128, 8, 8, 128, 4);
        // cta<4,32,512> warp<24,32,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 4, 32, 512, 24, 32, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 32, 512, 24, 32, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 32, 512, 24, 32, 128, 8, 8, 128, 4);
        // cta<4,48,512> warp<24,48,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 4, 48, 512, 24, 48, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 48, 512, 24, 48, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 48, 512, 24, 48, 128, 8, 8, 128, 4);
        // cta<4,64,512> warp<24,64,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 4, 64, 512, 24, 64, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 64, 512, 24, 64, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 64, 512, 24, 64, 128, 8, 8, 128, 4);
        // cta<4,80,512> warp<24,80,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 4, 80, 512, 24, 80, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 80, 512, 24, 80, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 80, 512, 24, 80, 128, 8, 8, 128, 4);
        // cta<4,96,512> warp<24,96,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 4, 96, 512, 24, 96, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 96, 512, 24, 96, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 96, 512, 24, 96, 128, 8, 8, 128, 4);
        // cta<4,112,512> warp<24,112,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 4, 112, 512, 24, 112, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 112, 512, 24, 112, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 112, 512, 24, 112, 128, 8, 8, 128, 4);
        // cta<4,128,512> warp<24,128,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 4, 128, 512, 24, 128, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 128, 512, 24, 128, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 128, 512, 24, 128, 128, 8, 8, 128, 4);
        // cta<8,32,512> warp<48,32,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 8, 32, 512, 48, 32, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 32, 512, 48, 32, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 8, 32, 512, 48, 32, 128, 8, 8, 128, 4);
        // cta<8,48,512> warp<48,48,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 8, 48, 512, 48, 48, 128, 8, 8, 128, 2);
        // cta<8,64,512> warp<48,64,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 8, 64, 512, 48, 64, 128, 8, 8, 128, 2);
        // cta<8,80,512> warp<48,80,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 8, 80, 512, 48, 80, 128, 8, 8, 128, 2);
        // cta<8,96,512> warp<48,96,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 8, 96, 512, 48, 96, 128, 8, 8, 128, 2);
        // cta<8,112,512> warp<48,112,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 8, 112, 512, 48, 112, 128, 8, 8, 128, 2);
        // cta<8,128,512> warp<48,128,128> mma<8,8,128>   WARPS[1x2]
        TEST(6, 2, true, 8, 128, 512, 48, 128, 128, 8, 8, 128, 2);
        // cta<1,32,512> warp<8,16,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 1, 32, 512, 8, 16, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 32, 512, 8, 16, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 32, 512, 8, 16, 128, 8, 8, 128, 4);
        // cta<1,48,512> warp<8,24,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 1, 48, 512, 8, 24, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 48, 512, 8, 24, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 48, 512, 8, 24, 128, 8, 8, 128, 4);
        // cta<1,64,512> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 1, 64, 512, 8, 32, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 64, 512, 8, 32, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 64, 512, 8, 32, 128, 8, 8, 128, 4);
        // cta<1,80,512> warp<8,40,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 1, 80, 512, 8, 40, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 80, 512, 8, 40, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 80, 512, 8, 40, 128, 8, 8, 128, 4);
        // cta<1,96,512> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 1, 96, 512, 8, 48, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 96, 512, 8, 48, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 96, 512, 8, 48, 128, 8, 8, 128, 4);
        // cta<1,112,512> warp<8,56,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 1, 112, 512, 8, 56, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 112, 512, 8, 56, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 112, 512, 8, 56, 128, 8, 8, 128, 4);
        // cta<1,128,512> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 1, 128, 512, 8, 64, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 128, 512, 8, 64, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 1, 128, 512, 8, 64, 128, 8, 8, 128, 4);
        // cta<1,256,512> warp<8,128,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 1, 256, 512, 8, 128, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 1, 256, 512, 8, 128, 128, 8, 8, 128, 3);
        // cta<4,32,512> warp<24,16,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 4, 32, 512, 24, 16, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 32, 512, 24, 16, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 32, 512, 24, 16, 128, 8, 8, 128, 4);
        // cta<4,48,512> warp<24,24,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 4, 48, 512, 24, 24, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 48, 512, 24, 24, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 48, 512, 24, 24, 128, 8, 8, 128, 4);
        // cta<4,64,512> warp<24,32,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 4, 64, 512, 24, 32, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 64, 512, 24, 32, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 64, 512, 24, 32, 128, 8, 8, 128, 4);
        // cta<4,80,512> warp<24,40,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 4, 80, 512, 24, 40, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 80, 512, 24, 40, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 80, 512, 24, 40, 128, 8, 8, 128, 4);
        // cta<4,96,512> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 4, 96, 512, 24, 48, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 96, 512, 24, 48, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 96, 512, 24, 48, 128, 8, 8, 128, 4);
        // cta<4,112,512> warp<24,56,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 4, 112, 512, 24, 56, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 112, 512, 24, 56, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 112, 512, 24, 56, 128, 8, 8, 128, 4);
        // cta<4,128,512> warp<24,64,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 4, 128, 512, 24, 64, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 4, 128, 512, 24, 64, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 4, 128, 512, 24, 64, 128, 8, 8, 128, 4);
        // cta<4,256,512> warp<24,128,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 4, 256, 512, 24, 128, 128, 8, 8, 128, 2);
        // cta<8,32,512> warp<48,16,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 8, 32, 512, 48, 16, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 32, 512, 48, 16, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 8, 32, 512, 48, 16, 128, 8, 8, 128, 4);
        // cta<8,48,512> warp<48,24,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 8, 48, 512, 48, 24, 128, 8, 8, 128, 2);
        // cta<8,64,512> warp<48,32,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 8, 64, 512, 48, 32, 128, 8, 8, 128, 2);
        // cta<8,80,512> warp<48,40,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 8, 80, 512, 48, 40, 128, 8, 8, 128, 2);
        // cta<8,96,512> warp<48,48,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 8, 96, 512, 48, 48, 128, 8, 8, 128, 2);
        // cta<8,112,512> warp<48,56,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 8, 112, 512, 48, 56, 128, 8, 8, 128, 2);
        // cta<8,128,512> warp<48,64,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 8, 128, 512, 48, 64, 128, 8, 8, 128, 2);
        // cta<8,256,512> warp<48,128,128> mma<8,8,128>   WARPS[1x4]
        TEST(6, 2, true, 8, 256, 512, 48, 128, 128, 8, 8, 128, 2);
        // cta<8,32,512> warp<24,64,128> mma<8,8,128>   WARPS[2x1]
        TEST(6, 2, true, 8, 32, 512, 24, 64, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 32, 512, 24, 64, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 8, 32, 512, 24, 64, 128, 8, 8, 128, 4);
        // cta<8,48,512> warp<24,96,128> mma<8,8,128>   WARPS[2x1]
        TEST(6, 2, true, 8, 48, 512, 24, 96, 128, 8, 8, 128, 2);
        // cta<8,64,512> warp<24,128,128> mma<8,8,128>   WARPS[2x1]
        TEST(6, 2, true, 8, 64, 512, 24, 128, 128, 8, 8, 128, 2);
        // cta<8,32,512> warp<24,32,128> mma<8,8,128>   WARPS[2x2]
        TEST(6, 2, true, 8, 32, 512, 24, 32, 128, 8, 8, 128, 2);
        TEST(6, 2, true, 8, 32, 512, 24, 32, 128, 8, 8, 128, 3);
        TEST(6, 2, true, 8, 32, 512, 24, 32, 128, 8, 8, 128, 4);
        // cta<8,48,512> warp<24,48,128> mma<8,8,128>   WARPS[2x2]
        TEST(6, 2, true, 8, 48, 512, 24, 48, 128, 8, 8, 128, 2);
        // cta<8,64,512> warp<24,64,128> mma<8,8,128>   WARPS[2x2]
        TEST(6, 2, true, 8, 64, 512, 24, 64, 128, 8, 8, 128, 2);
        // cta<8,80,512> warp<24,80,128> mma<8,8,128>   WARPS[2x2]
        TEST(6, 2, true, 8, 80, 512, 24, 80, 128, 8, 8, 128, 2);
        // cta<8,96,512> warp<24,96,128> mma<8,8,128>   WARPS[2x2]
        TEST(6, 2, true, 8, 96, 512, 24, 96, 128, 8, 8, 128, 2);
        // cta<8,112,512> warp<24,112,128> mma<8,8,128>   WARPS[2x2]
        TEST(6, 2, true, 8, 112, 512, 24, 112, 128, 8, 8, 128, 2);
        // cta<8,128,512> warp<24,128,128> mma<8,8,128>   WARPS[2x2]
        TEST(6, 2, true, 8, 128, 512, 24, 128, 128, 8, 8, 128, 2);
    } else {
    }

    printf("The best kernel config is %s with %f TOPS\n", best_config.str().c_str(), max_gflop);
#else
    printf("unsupport w%da%d\n", w_bits, x_bits);
#endif
}