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

void test_mma_w5a5(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream)
{
#ifdef W5A5
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
        ////// W5A5 int
        // cta<1,32,256> warp<8,80,128> mma<8,8,128>   WARPS[1x2]
        TEST(5, 5, true, 1, 32, 256, 8, 80, 128, 8, 8, 128, 2);
        TEST(5, 5, true, 1, 32, 256, 8, 80, 128, 8, 8, 128, 3);
        TEST(5, 5, true, 1, 32, 256, 8, 80, 128, 8, 8, 128, 4);
        // cta<1,48,256> warp<8,120,128> mma<8,8,128>   WARPS[1x2]
        TEST(5, 5, true, 1, 48, 256, 8, 120, 128, 8, 8, 128, 2);
        TEST(5, 5, true, 1, 48, 256, 8, 120, 128, 8, 8, 128, 3);
        TEST(5, 5, true, 1, 48, 256, 8, 120, 128, 8, 8, 128, 4);
        // cta<8,32,256> warp<40,80,128> mma<8,8,128>   WARPS[1x2]
        TEST(5, 5, true, 8, 32, 256, 40, 80, 128, 8, 8, 128, 2);
        TEST(5, 5, true, 8, 32, 256, 40, 80, 128, 8, 8, 128, 3);
        TEST(5, 5, true, 8, 32, 256, 40, 80, 128, 8, 8, 128, 4);
        // cta<8,48,256> warp<40,120,128> mma<8,8,128>   WARPS[1x2]
        TEST(5, 5, true, 8, 48, 256, 40, 120, 128, 8, 8, 128, 2);
        TEST(5, 5, true, 8, 48, 256, 40, 120, 128, 8, 8, 128, 3);
        TEST(5, 5, true, 8, 48, 256, 40, 120, 128, 8, 8, 128, 4);
        // cta<1,32,256> warp<8,40,128> mma<8,8,128>   WARPS[1x4]
        TEST(5, 5, true, 1, 32, 256, 8, 40, 128, 8, 8, 128, 2);
        TEST(5, 5, true, 1, 32, 256, 8, 40, 128, 8, 8, 128, 3);
        TEST(5, 5, true, 1, 32, 256, 8, 40, 128, 8, 8, 128, 4);
        // cta<1,64,256> warp<8,80,128> mma<8,8,128>   WARPS[1x4]
        TEST(5, 5, true, 1, 64, 256, 8, 80, 128, 8, 8, 128, 2);
        TEST(5, 5, true, 1, 64, 256, 8, 80, 128, 8, 8, 128, 3);
        TEST(5, 5, true, 1, 64, 256, 8, 80, 128, 8, 8, 128, 4);
        // cta<1,96,256> warp<8,120,128> mma<8,8,128>   WARPS[1x4]
        TEST(5, 5, true, 1, 96, 256, 8, 120, 128, 8, 8, 128, 2);
        TEST(5, 5, true, 1, 96, 256, 8, 120, 128, 8, 8, 128, 3);
        TEST(5, 5, true, 1, 96, 256, 8, 120, 128, 8, 8, 128, 4);
        // cta<8,32,256> warp<40,40,128> mma<8,8,128>   WARPS[1x4]
        TEST(5, 5, true, 8, 32, 256, 40, 40, 128, 8, 8, 128, 2);
        TEST(5, 5, true, 8, 32, 256, 40, 40, 128, 8, 8, 128, 3);
        TEST(5, 5, true, 8, 32, 256, 40, 40, 128, 8, 8, 128, 4);
        // cta<8,64,256> warp<40,80,128> mma<8,8,128>   WARPS[1x4]
        TEST(5, 5, true, 8, 64, 256, 40, 80, 128, 8, 8, 128, 2);
        TEST(5, 5, true, 8, 64, 256, 40, 80, 128, 8, 8, 128, 3);
        TEST(5, 5, true, 8, 64, 256, 40, 80, 128, 8, 8, 128, 4);
        // cta<8,96,256> warp<40,120,128> mma<8,8,128>   WARPS[1x4]
        TEST(5, 5, true, 8, 96, 256, 40, 120, 128, 8, 8, 128, 2);
        TEST(5, 5, true, 8, 96, 256, 40, 120, 128, 8, 8, 128, 3);
        TEST(5, 5, true, 8, 96, 256, 40, 120, 128, 8, 8, 128, 4);
        // cta<1,32,384> warp<8,80,128> mma<8,8,128>   WARPS[1x2]
        TEST(5, 5, true, 1, 32, 384, 8, 80, 128, 8, 8, 128, 2);
        TEST(5, 5, true, 1, 32, 384, 8, 80, 128, 8, 8, 128, 3);
        TEST(5, 5, true, 1, 32, 384, 8, 80, 128, 8, 8, 128, 4);
        // cta<1,48,384> warp<8,120,128> mma<8,8,128>   WARPS[1x2]
        TEST(5, 5, true, 1, 48, 384, 8, 120, 128, 8, 8, 128, 2);
        TEST(5, 5, true, 1, 48, 384, 8, 120, 128, 8, 8, 128, 3);
        TEST(5, 5, true, 1, 48, 384, 8, 120, 128, 8, 8, 128, 4);
        // cta<8,32,384> warp<40,80,128> mma<8,8,128>   WARPS[1x2]
        TEST(5, 5, true, 8, 32, 384, 40, 80, 128, 8, 8, 128, 2);
        // cta<8,48,384> warp<40,120,128> mma<8,8,128>   WARPS[1x2]
        TEST(5, 5, true, 8, 48, 384, 40, 120, 128, 8, 8, 128, 2);
        // cta<1,32,384> warp<8,40,128> mma<8,8,128>   WARPS[1x4]
        TEST(5, 5, true, 1, 32, 384, 8, 40, 128, 8, 8, 128, 2);
        TEST(5, 5, true, 1, 32, 384, 8, 40, 128, 8, 8, 128, 3);
        TEST(5, 5, true, 1, 32, 384, 8, 40, 128, 8, 8, 128, 4);
        // cta<1,64,384> warp<8,80,128> mma<8,8,128>   WARPS[1x4]
        TEST(5, 5, true, 1, 64, 384, 8, 80, 128, 8, 8, 128, 2);
        TEST(5, 5, true, 1, 64, 384, 8, 80, 128, 8, 8, 128, 3);
        TEST(5, 5, true, 1, 64, 384, 8, 80, 128, 8, 8, 128, 4);
        // cta<1,96,384> warp<8,120,128> mma<8,8,128>   WARPS[1x4]
        TEST(5, 5, true, 1, 96, 384, 8, 120, 128, 8, 8, 128, 2);
        TEST(5, 5, true, 1, 96, 384, 8, 120, 128, 8, 8, 128, 3);
        TEST(5, 5, true, 1, 96, 384, 8, 120, 128, 8, 8, 128, 4);
        // cta<8,32,384> warp<40,40,128> mma<8,8,128>   WARPS[1x4]
        TEST(5, 5, true, 8, 32, 384, 40, 40, 128, 8, 8, 128, 2);
        // cta<8,64,384> warp<40,80,128> mma<8,8,128>   WARPS[1x4]
        TEST(5, 5, true, 8, 64, 384, 40, 80, 128, 8, 8, 128, 2);
        TEST(5, 5, true, 8, 64, 384, 40, 80, 128, 8, 8, 128, 3);
        // cta<8,96,384> warp<40,120,128> mma<8,8,128>   WARPS[1x4]
        TEST(5, 5, true, 8, 96, 384, 40, 120, 128, 8, 8, 128, 2);
        TEST(5, 5, true, 8, 96, 384, 40, 120, 128, 8, 8, 128, 3);
        // cta<1,32,512> warp<8,80,128> mma<8,8,128>   WARPS[1x2]
        TEST(5, 5, true, 1, 32, 512, 8, 80, 128, 8, 8, 128, 2);
        TEST(5, 5, true, 1, 32, 512, 8, 80, 128, 8, 8, 128, 3);
        TEST(5, 5, true, 1, 32, 512, 8, 80, 128, 8, 8, 128, 4);
        // cta<1,48,512> warp<8,120,128> mma<8,8,128>   WARPS[1x2]
        TEST(5, 5, true, 1, 48, 512, 8, 120, 128, 8, 8, 128, 2);
        TEST(5, 5, true, 1, 48, 512, 8, 120, 128, 8, 8, 128, 3);
        TEST(5, 5, true, 1, 48, 512, 8, 120, 128, 8, 8, 128, 4);
        // cta<8,32,512> warp<40,80,128> mma<8,8,128>   WARPS[1x2]
        TEST(5, 5, true, 8, 32, 512, 40, 80, 128, 8, 8, 128, 2);
        // cta<8,48,512> warp<40,120,128> mma<8,8,128>   WARPS[1x2]
        TEST(5, 5, true, 8, 48, 512, 40, 120, 128, 8, 8, 128, 2);
        // cta<1,32,512> warp<8,40,128> mma<8,8,128>   WARPS[1x4]
        TEST(5, 5, true, 1, 32, 512, 8, 40, 128, 8, 8, 128, 2);
        TEST(5, 5, true, 1, 32, 512, 8, 40, 128, 8, 8, 128, 3);
        TEST(5, 5, true, 1, 32, 512, 8, 40, 128, 8, 8, 128, 4);
        // cta<1,64,512> warp<8,80,128> mma<8,8,128>   WARPS[1x4]
        TEST(5, 5, true, 1, 64, 512, 8, 80, 128, 8, 8, 128, 2);
        TEST(5, 5, true, 1, 64, 512, 8, 80, 128, 8, 8, 128, 3);
        TEST(5, 5, true, 1, 64, 512, 8, 80, 128, 8, 8, 128, 4);
        // cta<1,96,512> warp<8,120,128> mma<8,8,128>   WARPS[1x4]
        TEST(5, 5, true, 1, 96, 512, 8, 120, 128, 8, 8, 128, 2);
        TEST(5, 5, true, 1, 96, 512, 8, 120, 128, 8, 8, 128, 3);
        // cta<8,32,512> warp<40,40,128> mma<8,8,128>   WARPS[1x4]
        TEST(5, 5, true, 8, 32, 512, 40, 40, 128, 8, 8, 128, 2);
        // cta<8,64,512> warp<40,80,128> mma<8,8,128>   WARPS[1x4]
        TEST(5, 5, true, 8, 64, 512, 40, 80, 128, 8, 8, 128, 2);
        // cta<8,96,512> warp<40,120,128> mma<8,8,128>   WARPS[1x4]
        TEST(5, 5, true, 8, 96, 512, 40, 120, 128, 8, 8, 128, 2);
    } else {
    }

    printf("The best kernel config is %s with %f TOPS\n", best_config.str().c_str(), max_gflop);
#else
    printf("unsupport w%da%d\n", w_bits, x_bits);
#endif
}