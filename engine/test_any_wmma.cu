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

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include "common/base.h"
#include "common/pack.h"
#include "common/timer.h"
#include "mma_any/aq_wmma_library.h"
#include "mma_any/aq_wmma_op.h"

/// benchmark func for wmma
inline bool isCudaSuccess(cudaError_t status)
{
    cudaError_t error = status;
    if (error != cudaSuccess) {
        std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    return true;
}

bool check(const int *ref_out, const int *out, int m, int n)
{
    for (int i = 0; i < m * n; ++i) {
        if (ref_out[i] != out[i]) {
            return false;
        }
    }
    return true;
}


void print_matrix(int *matrix, int m, int n, bool hex)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (hex)
                printf("%x,", matrix[i * n + j]);
            else
                printf("%d,", matrix[i * n + j]);
        }
        printf("\n");
    }
}

int int_pow(int base, int exp)
{
    int result = 1;
    while (exp) {
        if (exp % 2)
            result *= base;
        exp /= 2;
        base *= base;
    }
    return result;
}

void init_matrix(int *matrix, int m, int n, int bits)
{
    for (int i = 0; i < m * n; ++i) {
        matrix[i] = rand() % int_pow(2, bits);
    }
}

void compute_ref(int *w, int *x, int *ref_c, int M, int N, int K, int W_BIT, int X_BIT, bool SIGNED)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int tmp = 0;
            for (int xb = 0; xb < X_BIT; xb++) {
                int X_Multiplier =
                    SIGNED && (xb == X_BIT - 1) ? -1 * int_pow(2, xb) : int_pow(2, xb);
                for (int wb = 0; wb < W_BIT; wb++) {
                    int W_Multiplier =
                        SIGNED && (wb == W_BIT - 1) ? -1 * int_pow(2, wb) : int_pow(2, wb);
                    for (int k_tile = 0; k_tile < K / 32; k_tile++) {
                        int w_int = w[wb * N * K / 32 + n * K / 32 + k_tile];
                        int x_int = x[xb * M * K / 32 + m * K / 32 + k_tile];
                        for (int k = 0; k < 32; k++) {
                            int mask = 1;
                            int x_val = ((mask << k) & x_int) >> k;
                            int w_val = ((mask << k) & w_int) >> k;
                            tmp += X_Multiplier * W_Multiplier * x_val * w_val;
                        }
                    }
                }
            }
            ref_c[m * N + n] = tmp;
        }
    }
}

#define TEST(X_BITS, W_BITS, SIGNED, BM, BN, BK, WM, WN, WK, MMA_M, MMA_N, MMA_K, NSTAGE)      \
    {                                                                                          \
        std::cout << GPU_ARCH << " " << config_str << " ";                                     \
        printf("%d %d %d %d %d %d %d %d %d %d ", BM, BN, BK, WM, WN, WK, MMA_M, MMA_N, MMA_K,  \
               NSTAGE);                                                                        \
        int ret = benchmark<AQ_INIT_FUN(AqBWMMA), AQ_EXEC_FUN(AqBWMMA), AQ_OP_STATE(AqBWMMA)>( \
            AQ_NAME_FUN(AqBWMMA, Init, X_BITS, W_BITS, SIGNED, BM, BN, BK, WM, WN, WK, MMA_M,  \
                        MMA_N, MMA_K, NSTAGE),                                                 \
            AQ_NAME_FUN(AqBWMMA, Exec, X_BITS, W_BITS, SIGNED, BM, BN, BK, WM, WN, WK, MMA_M,  \
                        MMA_N, MMA_K, NSTAGE),                                                 \
            x_bits, w_bits, d_x, d_w, d_x_pack, d_w_pack, m, n, k, d_out, nullptr, h_out,      \
            h_ref_out, false, SIGNED, exec_dur, pack_dur, stream, warmup, repeat);             \
        if (ret == 0 && gflop_count / exec_dur > max_gflop) {                                  \
            max_gflop = gflop_count / exec_dur;                                                \
            min_latency = exec_dur * 1e3;                                                      \
            best_config.str("");                                                               \
            best_config << BM << ", " << BN << ", " << BK << ", " << WM << ", " << WN << ", "  \
                        << WK << ", " << MMA_M << ", " << MMA_N << ", " << MMA_K << ", "       \
                        << NSTAGE;                                                             \
        }                                                                                      \
        printf("packing %f (us) exec %f (us) %f TOPS | %f B-TOPS | %s\n", pack_dur * 1e3,      \
               exec_dur * 1e3, gflop_count / exec_dur, true_gflop_count / exec_dur,            \
               ret == 0  ? "PASSED" :                                                          \
               ret == -1 ? "ERROR" :                                                           \
                           "FAILED");                                                          \
    }

int main(int argc, char **argv)
{
    if (argc < 7) {
        printf("Usage: ./test_any_wmma M N K X_BITS W_BITS SIGNED\n");
        return -1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    int x_bits = atoi(argv[4]);
    int w_bits = atoi(argv[5]);
    bool quant_sign = atoi(argv[6]) == 1;
    if (k < 128 || k % 128 != 0) {
        printf("Error, k must >= 128 and k % 128 == 0!");
        return -1;
    }
    int repeat = 1000;
    int warmup = 10;
    float exec_dur = 0;
    float pack_dur = 0;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::string config_str;
    std::stringstream s;
    s << x_bits << " " << w_bits << " " << m << " " << n << " " << k << " ";
    if (quant_sign) {
        s << "sign ";
    } else {
        s << "unsigned ";
    }
    config_str = s.str();
    float true_gflop_count = (float)m / 1e9 * n * k * 2 * x_bits * w_bits;
    float gflop_count = (float)m / 1e9 * n * k * 2;
    float max_gflop = 0;
    float min_latency = 1e9;
    std::stringstream best_config;

    int *h_x = (int *)malloc(m * k * sizeof(int));
    int *h_w = (int *)malloc(n * k * sizeof(int));
    int *h_x_pack = (int *)malloc(x_bits * m * (k / 32) * sizeof(int));
    int *h_w_pack = (int *)malloc(w_bits * n * (k / 32) * sizeof(int));
    int *h_out = (int *)malloc(m * n * sizeof(int));
    int *h_ref_out = (int *)malloc(m * n * sizeof(int));

    int *d_x;
    int *d_x_pack;
    int *d_w;
    int *d_w_pack;
    int *d_out;
    cudaMalloc(&d_x, m * k * sizeof(int));
    cudaMalloc(&d_w, n * k * sizeof(int));
    cudaMalloc(&d_x_pack, x_bits * m * (k / 32) * sizeof(int));
    cudaMalloc(&d_w_pack, w_bits * n * (k / 32) * sizeof(int));
    cudaMalloc(&d_out, m * n * sizeof(int));
    // INIT HOST TENSOR
    init_matrix(h_x, m, k, x_bits);
    init_matrix(h_w, n, k, w_bits);
    cudaMemcpy(d_x, h_x, sizeof(int) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, sizeof(int) * n * k, cudaMemcpyHostToDevice);

    if (w_bits <= 32) {
        cudaError_t err = launch_pack(d_w, d_w_pack, n, k, w_bits);
        if (err != cudaSuccess) {
            printf("Line %d: 'weight launch_pack' failed: %s\n", __LINE__, cudaGetErrorString(err));
            return -1;
        }
    } else {
        printf("unsupport w_bits %d: for launch_pack func \n", w_bits);
        return -1;
    }

    if (x_bits <= 32) {
        cudaError_t err = launch_pack(d_x, d_x_pack, m, k, x_bits);
        if (err != cudaSuccess) {
            printf("Line %d: 'activation launch_pack' failed: %s\n", __LINE__,
                   cudaGetErrorString(err));
            return -1;
        }
    } else {
        printf("unsupport x_bits %d: for launch_pack func \n", x_bits);
        return -1;
    }

    cudaMemcpy(h_x_pack, d_x_pack, x_bits * m * (k / 32) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w_pack, d_w_pack, w_bits * n * (k / 32) * sizeof(int), cudaMemcpyDeviceToHost);

    compute_ref(h_w_pack, h_x_pack, h_ref_out, m, n, k, w_bits, x_bits, quant_sign);

    switch (x_bits) {
    case 2:
        switch (w_bits) {
        #ifdef W2A2
        case 2:
            if (quant_sign) {
                ////// W2A2 int
                // cta<4,32,256> warp<8,32,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 4, 32, 256, 8, 32, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 32, 256, 8, 32, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 32, 256, 8, 32, 128, 8, 8, 128, 4);
                // cta<4,48,256> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 4, 48, 256, 8, 48, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 48, 256, 8, 48, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 48, 256, 8, 48, 128, 8, 8, 128, 4);
                // cta<4,64,256> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 4, 64, 256, 8, 64, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 64, 256, 8, 64, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 64, 256, 8, 64, 128, 8, 8, 128, 4);
                // cta<4,80,256> warp<8,80,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 4, 80, 256, 8, 80, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 80, 256, 8, 80, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 80, 256, 8, 80, 128, 8, 8, 128, 4);
                // cta<4,96,256> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 4, 96, 256, 8, 96, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 96, 256, 8, 96, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 96, 256, 8, 96, 128, 8, 8, 128, 4);
                // cta<4,112,256> warp<8,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 4, 112, 256, 8, 112, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 112, 256, 8, 112, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 112, 256, 8, 112, 128, 8, 8, 128, 4);
                // cta<4,128,256> warp<8,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 4, 128, 256, 8, 128, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 128, 256, 8, 128, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 128, 256, 8, 128, 128, 8, 8, 128, 4);
                // cta<8,32,256> warp<16,32,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 8, 32, 256, 16, 32, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 32, 256, 16, 32, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 32, 256, 16, 32, 128, 8, 8, 128, 4);
                // cta<8,48,256> warp<16,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 8, 48, 256, 16, 48, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 48, 256, 16, 48, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 48, 256, 16, 48, 128, 8, 8, 128, 4);
                // cta<8,64,256> warp<16,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 8, 64, 256, 16, 64, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 64, 256, 16, 64, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 64, 256, 16, 64, 128, 8, 8, 128, 4);
                // cta<8,80,256> warp<16,80,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 8, 80, 256, 16, 80, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 80, 256, 16, 80, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 80, 256, 16, 80, 128, 8, 8, 128, 4);
                // cta<8,96,256> warp<16,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 8, 96, 256, 16, 96, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 96, 256, 16, 96, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 96, 256, 16, 96, 128, 8, 8, 128, 4);
                // cta<8,112,256> warp<16,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 8, 112, 256, 16, 112, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 112, 256, 16, 112, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 112, 256, 16, 112, 128, 8, 8, 128, 4);
                // cta<8,128,256> warp<16,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 8, 128, 256, 16, 128, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 128, 256, 16, 128, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 128, 256, 16, 128, 128, 8, 8, 128, 4);
                // cta<4,32,256> warp<8,16,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 4, 32, 256, 8, 16, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 32, 256, 8, 16, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 32, 256, 8, 16, 128, 8, 8, 128, 4);
                // cta<4,48,256> warp<8,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 4, 48, 256, 8, 24, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 48, 256, 8, 24, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 48, 256, 8, 24, 128, 8, 8, 128, 4);
                // cta<4,64,256> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 4, 64, 256, 8, 32, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 64, 256, 8, 32, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 64, 256, 8, 32, 128, 8, 8, 128, 4);
                // cta<4,80,256> warp<8,40,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 4, 80, 256, 8, 40, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 80, 256, 8, 40, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 80, 256, 8, 40, 128, 8, 8, 128, 4);
                // cta<4,96,256> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 4, 96, 256, 8, 48, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 96, 256, 8, 48, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 96, 256, 8, 48, 128, 8, 8, 128, 4);
                // cta<4,112,256> warp<8,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 4, 112, 256, 8, 56, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 112, 256, 8, 56, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 112, 256, 8, 56, 128, 8, 8, 128, 4);
                // cta<4,128,256> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 4, 128, 256, 8, 64, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 128, 256, 8, 64, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 128, 256, 8, 64, 128, 8, 8, 128, 4);
                // cta<4,256,256> warp<8,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 4, 256, 256, 8, 128, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 256, 256, 8, 128, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 256, 256, 8, 128, 128, 8, 8, 128, 4);
                // cta<8,32,256> warp<16,16,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 8, 32, 256, 16, 16, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 32, 256, 16, 16, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 32, 256, 16, 16, 128, 8, 8, 128, 4);
                // cta<8,48,256> warp<16,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 8, 48, 256, 16, 24, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 48, 256, 16, 24, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 48, 256, 16, 24, 128, 8, 8, 128, 4);
                // cta<8,64,256> warp<16,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 8, 64, 256, 16, 32, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 64, 256, 16, 32, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 64, 256, 16, 32, 128, 8, 8, 128, 4);
                // cta<8,80,256> warp<16,40,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 8, 80, 256, 16, 40, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 80, 256, 16, 40, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 80, 256, 16, 40, 128, 8, 8, 128, 4);
                // cta<8,96,256> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 8, 96, 256, 16, 48, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 96, 256, 16, 48, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 96, 256, 16, 48, 128, 8, 8, 128, 4);
                // cta<8,112,256> warp<16,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 8, 112, 256, 16, 56, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 112, 256, 16, 56, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 112, 256, 16, 56, 128, 8, 8, 128, 4);
                // cta<8,128,256> warp<16,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 8, 128, 256, 16, 64, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 128, 256, 16, 64, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 128, 256, 16, 64, 128, 8, 8, 128, 4);
                // cta<8,256,256> warp<16,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 8, 256, 256, 16, 128, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 256, 256, 16, 128, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 256, 256, 16, 128, 128, 8, 8, 128, 4);
                // cta<8,32,256> warp<8,64,128> mma<8,8,128>   WARPS[2x1]
                TEST(2, 2, true, 8, 32, 256, 8, 64, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 32, 256, 8, 64, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 32, 256, 8, 64, 128, 8, 8, 128, 4);
                // cta<8,48,256> warp<8,96,128> mma<8,8,128>   WARPS[2x1]
                TEST(2, 2, true, 8, 48, 256, 8, 96, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 48, 256, 8, 96, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 48, 256, 8, 96, 128, 8, 8, 128, 4);
                // cta<8,64,256> warp<8,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(2, 2, true, 8, 64, 256, 8, 128, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 64, 256, 8, 128, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 64, 256, 8, 128, 128, 8, 8, 128, 4);
                // cta<8,32,256> warp<8,32,128> mma<8,8,128>   WARPS[2x2]
                TEST(2, 2, true, 8, 32, 256, 8, 32, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 32, 256, 8, 32, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 32, 256, 8, 32, 128, 8, 8, 128, 4);
                // cta<8,48,256> warp<8,48,128> mma<8,8,128>   WARPS[2x2]
                TEST(2, 2, true, 8, 48, 256, 8, 48, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 48, 256, 8, 48, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 48, 256, 8, 48, 128, 8, 8, 128, 4);
                // cta<8,64,256> warp<8,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(2, 2, true, 8, 64, 256, 8, 64, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 64, 256, 8, 64, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 64, 256, 8, 64, 128, 8, 8, 128, 4);
                // cta<8,80,256> warp<8,80,128> mma<8,8,128>   WARPS[2x2]
                TEST(2, 2, true, 8, 80, 256, 8, 80, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 80, 256, 8, 80, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 80, 256, 8, 80, 128, 8, 8, 128, 4);
                // cta<8,96,256> warp<8,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(2, 2, true, 8, 96, 256, 8, 96, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 96, 256, 8, 96, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 96, 256, 8, 96, 128, 8, 8, 128, 4);
                // cta<8,112,256> warp<8,112,128> mma<8,8,128>   WARPS[2x2]
                TEST(2, 2, true, 8, 112, 256, 8, 112, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 112, 256, 8, 112, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 112, 256, 8, 112, 128, 8, 8, 128, 4);
                // cta<8,128,256> warp<8,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(2, 2, true, 8, 128, 256, 8, 128, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 128, 256, 8, 128, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 128, 256, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,384> warp<8,32,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 4, 32, 384, 8, 32, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 32, 384, 8, 32, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 32, 384, 8, 32, 128, 8, 8, 128, 4);
                // cta<4,48,384> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 4, 48, 384, 8, 48, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 48, 384, 8, 48, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 48, 384, 8, 48, 128, 8, 8, 128, 4);
                // cta<4,64,384> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 4, 64, 384, 8, 64, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 64, 384, 8, 64, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 64, 384, 8, 64, 128, 8, 8, 128, 4);
                // cta<4,80,384> warp<8,80,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 4, 80, 384, 8, 80, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 80, 384, 8, 80, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 80, 384, 8, 80, 128, 8, 8, 128, 4);
                // cta<4,96,384> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 4, 96, 384, 8, 96, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 96, 384, 8, 96, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 96, 384, 8, 96, 128, 8, 8, 128, 4);
                // cta<4,112,384> warp<8,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 4, 112, 384, 8, 112, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 112, 384, 8, 112, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 112, 384, 8, 112, 128, 8, 8, 128, 4);
                // cta<4,128,384> warp<8,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 4, 128, 384, 8, 128, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 128, 384, 8, 128, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 128, 384, 8, 128, 128, 8, 8, 128, 4);
                // cta<8,32,384> warp<16,32,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 8, 32, 384, 16, 32, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 32, 384, 16, 32, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 32, 384, 16, 32, 128, 8, 8, 128, 4);
                // cta<8,48,384> warp<16,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 8, 48, 384, 16, 48, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 48, 384, 16, 48, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 48, 384, 16, 48, 128, 8, 8, 128, 4);
                // cta<8,64,384> warp<16,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 8, 64, 384, 16, 64, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 64, 384, 16, 64, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 64, 384, 16, 64, 128, 8, 8, 128, 4);
                // cta<8,80,384> warp<16,80,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 8, 80, 384, 16, 80, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 80, 384, 16, 80, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 80, 384, 16, 80, 128, 8, 8, 128, 4);
                // cta<8,96,384> warp<16,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 8, 96, 384, 16, 96, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 96, 384, 16, 96, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 96, 384, 16, 96, 128, 8, 8, 128, 4);
                // cta<8,112,384> warp<16,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 8, 112, 384, 16, 112, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 112, 384, 16, 112, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 112, 384, 16, 112, 128, 8, 8, 128, 4);
                // cta<8,128,384> warp<16,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 8, 128, 384, 16, 128, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 128, 384, 16, 128, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 128, 384, 16, 128, 128, 8, 8, 128, 4);
                // cta<4,32,384> warp<8,16,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 4, 32, 384, 8, 16, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 32, 384, 8, 16, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 32, 384, 8, 16, 128, 8, 8, 128, 4);
                // cta<4,48,384> warp<8,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 4, 48, 384, 8, 24, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 48, 384, 8, 24, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 48, 384, 8, 24, 128, 8, 8, 128, 4);
                // cta<4,64,384> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 4, 64, 384, 8, 32, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 64, 384, 8, 32, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 64, 384, 8, 32, 128, 8, 8, 128, 4);
                // cta<4,80,384> warp<8,40,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 4, 80, 384, 8, 40, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 80, 384, 8, 40, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 80, 384, 8, 40, 128, 8, 8, 128, 4);
                // cta<4,96,384> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 4, 96, 384, 8, 48, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 96, 384, 8, 48, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 96, 384, 8, 48, 128, 8, 8, 128, 4);
                // cta<4,112,384> warp<8,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 4, 112, 384, 8, 56, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 112, 384, 8, 56, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 112, 384, 8, 56, 128, 8, 8, 128, 4);
                // cta<4,128,384> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 4, 128, 384, 8, 64, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 128, 384, 8, 64, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 128, 384, 8, 64, 128, 8, 8, 128, 4);
                // cta<4,256,384> warp<8,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 4, 256, 384, 8, 128, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 256, 384, 8, 128, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 256, 384, 8, 128, 128, 8, 8, 128, 4);
                // cta<8,32,384> warp<16,16,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 8, 32, 384, 16, 16, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 32, 384, 16, 16, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 32, 384, 16, 16, 128, 8, 8, 128, 4);
                // cta<8,48,384> warp<16,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 8, 48, 384, 16, 24, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 48, 384, 16, 24, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 48, 384, 16, 24, 128, 8, 8, 128, 4);
                // cta<8,64,384> warp<16,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 8, 64, 384, 16, 32, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 64, 384, 16, 32, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 64, 384, 16, 32, 128, 8, 8, 128, 4);
                // cta<8,80,384> warp<16,40,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 8, 80, 384, 16, 40, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 80, 384, 16, 40, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 80, 384, 16, 40, 128, 8, 8, 128, 4);
                // cta<8,96,384> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 8, 96, 384, 16, 48, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 96, 384, 16, 48, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 96, 384, 16, 48, 128, 8, 8, 128, 4);
                // cta<8,112,384> warp<16,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 8, 112, 384, 16, 56, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 112, 384, 16, 56, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 112, 384, 16, 56, 128, 8, 8, 128, 4);
                // cta<8,128,384> warp<16,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 8, 128, 384, 16, 64, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 128, 384, 16, 64, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 128, 384, 16, 64, 128, 8, 8, 128, 4);
                // cta<8,256,384> warp<16,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 8, 256, 384, 16, 128, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 256, 384, 16, 128, 128, 8, 8, 128, 3);
                // cta<8,32,384> warp<8,64,128> mma<8,8,128>   WARPS[2x1]
                TEST(2, 2, true, 8, 32, 384, 8, 64, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 32, 384, 8, 64, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 32, 384, 8, 64, 128, 8, 8, 128, 4);
                // cta<8,48,384> warp<8,96,128> mma<8,8,128>   WARPS[2x1]
                TEST(2, 2, true, 8, 48, 384, 8, 96, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 48, 384, 8, 96, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 48, 384, 8, 96, 128, 8, 8, 128, 4);
                // cta<8,64,384> warp<8,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(2, 2, true, 8, 64, 384, 8, 128, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 64, 384, 8, 128, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 64, 384, 8, 128, 128, 8, 8, 128, 4);
                // cta<8,32,384> warp<8,32,128> mma<8,8,128>   WARPS[2x2]
                TEST(2, 2, true, 8, 32, 384, 8, 32, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 32, 384, 8, 32, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 32, 384, 8, 32, 128, 8, 8, 128, 4);
                // cta<8,48,384> warp<8,48,128> mma<8,8,128>   WARPS[2x2]
                TEST(2, 2, true, 8, 48, 384, 8, 48, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 48, 384, 8, 48, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 48, 384, 8, 48, 128, 8, 8, 128, 4);
                // cta<8,64,384> warp<8,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(2, 2, true, 8, 64, 384, 8, 64, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 64, 384, 8, 64, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 64, 384, 8, 64, 128, 8, 8, 128, 4);
                // cta<8,80,384> warp<8,80,128> mma<8,8,128>   WARPS[2x2]
                TEST(2, 2, true, 8, 80, 384, 8, 80, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 80, 384, 8, 80, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 80, 384, 8, 80, 128, 8, 8, 128, 4);
                // cta<8,96,384> warp<8,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(2, 2, true, 8, 96, 384, 8, 96, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 96, 384, 8, 96, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 96, 384, 8, 96, 128, 8, 8, 128, 4);
                // cta<8,112,384> warp<8,112,128> mma<8,8,128>   WARPS[2x2]
                TEST(2, 2, true, 8, 112, 384, 8, 112, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 112, 384, 8, 112, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 112, 384, 8, 112, 128, 8, 8, 128, 4);
                // cta<8,128,384> warp<8,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(2, 2, true, 8, 128, 384, 8, 128, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 128, 384, 8, 128, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 128, 384, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,512> warp<8,32,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 4, 32, 512, 8, 32, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 32, 512, 8, 32, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 32, 512, 8, 32, 128, 8, 8, 128, 4);
                // cta<4,48,512> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 4, 48, 512, 8, 48, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 48, 512, 8, 48, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 48, 512, 8, 48, 128, 8, 8, 128, 4);
                // cta<4,64,512> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 4, 64, 512, 8, 64, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 64, 512, 8, 64, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 64, 512, 8, 64, 128, 8, 8, 128, 4);
                // cta<4,80,512> warp<8,80,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 4, 80, 512, 8, 80, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 80, 512, 8, 80, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 80, 512, 8, 80, 128, 8, 8, 128, 4);
                // cta<4,96,512> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 4, 96, 512, 8, 96, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 96, 512, 8, 96, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 96, 512, 8, 96, 128, 8, 8, 128, 4);
                // cta<4,112,512> warp<8,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 4, 112, 512, 8, 112, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 112, 512, 8, 112, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 112, 512, 8, 112, 128, 8, 8, 128, 4);
                // cta<4,128,512> warp<8,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 4, 128, 512, 8, 128, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 128, 512, 8, 128, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 128, 512, 8, 128, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<16,32,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 8, 32, 512, 16, 32, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 32, 512, 16, 32, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 32, 512, 16, 32, 128, 8, 8, 128, 4);
                // cta<8,48,512> warp<16,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 8, 48, 512, 16, 48, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 48, 512, 16, 48, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 48, 512, 16, 48, 128, 8, 8, 128, 4);
                // cta<8,64,512> warp<16,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 8, 64, 512, 16, 64, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 64, 512, 16, 64, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 64, 512, 16, 64, 128, 8, 8, 128, 4);
                // cta<8,80,512> warp<16,80,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 8, 80, 512, 16, 80, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 80, 512, 16, 80, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 80, 512, 16, 80, 128, 8, 8, 128, 4);
                // cta<8,96,512> warp<16,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 8, 96, 512, 16, 96, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 96, 512, 16, 96, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 96, 512, 16, 96, 128, 8, 8, 128, 4);
                // cta<8,112,512> warp<16,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 8, 112, 512, 16, 112, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 112, 512, 16, 112, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 112, 512, 16, 112, 128, 8, 8, 128, 4);
                // cta<8,128,512> warp<16,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(2, 2, true, 8, 128, 512, 16, 128, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 128, 512, 16, 128, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 128, 512, 16, 128, 128, 8, 8, 128, 4);
                // cta<4,32,512> warp<8,16,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 4, 32, 512, 8, 16, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 32, 512, 8, 16, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 32, 512, 8, 16, 128, 8, 8, 128, 4);
                // cta<4,48,512> warp<8,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 4, 48, 512, 8, 24, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 48, 512, 8, 24, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 48, 512, 8, 24, 128, 8, 8, 128, 4);
                // cta<4,64,512> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 4, 64, 512, 8, 32, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 64, 512, 8, 32, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 64, 512, 8, 32, 128, 8, 8, 128, 4);
                // cta<4,80,512> warp<8,40,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 4, 80, 512, 8, 40, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 80, 512, 8, 40, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 80, 512, 8, 40, 128, 8, 8, 128, 4);
                // cta<4,96,512> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 4, 96, 512, 8, 48, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 96, 512, 8, 48, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 96, 512, 8, 48, 128, 8, 8, 128, 4);
                // cta<4,112,512> warp<8,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 4, 112, 512, 8, 56, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 112, 512, 8, 56, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 112, 512, 8, 56, 128, 8, 8, 128, 4);
                // cta<4,128,512> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 4, 128, 512, 8, 64, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 128, 512, 8, 64, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 4, 128, 512, 8, 64, 128, 8, 8, 128, 4);
                // cta<4,256,512> warp<8,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 4, 256, 512, 8, 128, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 4, 256, 512, 8, 128, 128, 8, 8, 128, 3);
                // cta<8,32,512> warp<16,16,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 8, 32, 512, 16, 16, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 32, 512, 16, 16, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 32, 512, 16, 16, 128, 8, 8, 128, 4);
                // cta<8,48,512> warp<16,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 8, 48, 512, 16, 24, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 48, 512, 16, 24, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 48, 512, 16, 24, 128, 8, 8, 128, 4);
                // cta<8,64,512> warp<16,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 8, 64, 512, 16, 32, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 64, 512, 16, 32, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 64, 512, 16, 32, 128, 8, 8, 128, 4);
                // cta<8,80,512> warp<16,40,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 8, 80, 512, 16, 40, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 80, 512, 16, 40, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 80, 512, 16, 40, 128, 8, 8, 128, 4);
                // cta<8,96,512> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 8, 96, 512, 16, 48, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 96, 512, 16, 48, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 96, 512, 16, 48, 128, 8, 8, 128, 4);
                // cta<8,112,512> warp<16,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 8, 112, 512, 16, 56, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 112, 512, 16, 56, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 112, 512, 16, 56, 128, 8, 8, 128, 4);
                // cta<8,128,512> warp<16,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 8, 128, 512, 16, 64, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 128, 512, 16, 64, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 128, 512, 16, 64, 128, 8, 8, 128, 4);
                // cta<8,256,512> warp<16,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(2, 2, true, 8, 256, 512, 16, 128, 128, 8, 8, 128, 2);
                // cta<8,32,512> warp<8,64,128> mma<8,8,128>   WARPS[2x1]
                TEST(2, 2, true, 8, 32, 512, 8, 64, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 32, 512, 8, 64, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 32, 512, 8, 64, 128, 8, 8, 128, 4);
                // cta<8,48,512> warp<8,96,128> mma<8,8,128>   WARPS[2x1]
                TEST(2, 2, true, 8, 48, 512, 8, 96, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 48, 512, 8, 96, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 48, 512, 8, 96, 128, 8, 8, 128, 4);
                // cta<8,64,512> warp<8,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(2, 2, true, 8, 64, 512, 8, 128, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 64, 512, 8, 128, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 64, 512, 8, 128, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<8,32,128> mma<8,8,128>   WARPS[2x2]
                TEST(2, 2, true, 8, 32, 512, 8, 32, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 32, 512, 8, 32, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 32, 512, 8, 32, 128, 8, 8, 128, 4);
                // cta<8,48,512> warp<8,48,128> mma<8,8,128>   WARPS[2x2]
                TEST(2, 2, true, 8, 48, 512, 8, 48, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 48, 512, 8, 48, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 48, 512, 8, 48, 128, 8, 8, 128, 4);
                // cta<8,64,512> warp<8,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(2, 2, true, 8, 64, 512, 8, 64, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 64, 512, 8, 64, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 64, 512, 8, 64, 128, 8, 8, 128, 4);
                // cta<8,80,512> warp<8,80,128> mma<8,8,128>   WARPS[2x2]
                TEST(2, 2, true, 8, 80, 512, 8, 80, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 80, 512, 8, 80, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 80, 512, 8, 80, 128, 8, 8, 128, 4);
                // cta<8,96,512> warp<8,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(2, 2, true, 8, 96, 512, 8, 96, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 96, 512, 8, 96, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 96, 512, 8, 96, 128, 8, 8, 128, 4);
                // cta<8,112,512> warp<8,112,128> mma<8,8,128>   WARPS[2x2]
                TEST(2, 2, true, 8, 112, 512, 8, 112, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 112, 512, 8, 112, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 112, 512, 8, 112, 128, 8, 8, 128, 4);
                // cta<8,128,512> warp<8,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(2, 2, true, 8, 128, 512, 8, 128, 128, 8, 8, 128, 2);
                TEST(2, 2, true, 8, 128, 512, 8, 128, 128, 8, 8, 128, 3);
                TEST(2, 2, true, 8, 128, 512, 8, 128, 128, 8, 8, 128, 4);
            } else {
            }
            break;
        #endif
        default:
            printf("unsupport w%da%d\n", w_bits, x_bits);
        }
        break;
    case 3:
        switch (w_bits) {
        #ifdef W3A3
        case 3:
            if (quant_sign) {
                ////// W3A3 int
                // cta<2,32,256> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(3, 3, true, 2, 32, 256, 8, 48, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 2, 32, 256, 8, 48, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 2, 32, 256, 8, 48, 128, 8, 8, 128, 4);
                // cta<2,48,256> warp<8,72,128> mma<8,8,128>   WARPS[1x2]
                TEST(3, 3, true, 2, 48, 256, 8, 72, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 2, 48, 256, 8, 72, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 2, 48, 256, 8, 72, 128, 8, 8, 128, 4);
                // cta<2,64,256> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(3, 3, true, 2, 64, 256, 8, 96, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 2, 64, 256, 8, 96, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 2, 64, 256, 8, 96, 128, 8, 8, 128, 4);
                // cta<2,80,256> warp<8,120,128> mma<8,8,128>   WARPS[1x2]
                TEST(3, 3, true, 2, 80, 256, 8, 120, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 2, 80, 256, 8, 120, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 2, 80, 256, 8, 120, 128, 8, 8, 128, 4);
                // cta<8,32,256> warp<24,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(3, 3, true, 8, 32, 256, 24, 48, 128, 8, 8, 128, 2);
                // cta<8,48,256> warp<24,72,128> mma<8,8,128>   WARPS[1x2]
                TEST(3, 3, true, 8, 48, 256, 24, 72, 128, 8, 8, 128, 2);
                // cta<8,64,256> warp<24,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(3, 3, true, 8, 64, 256, 24, 96, 128, 8, 8, 128, 2);
                // cta<8,80,256> warp<24,120,128> mma<8,8,128>   WARPS[1x2]
                TEST(3, 3, true, 8, 80, 256, 24, 120, 128, 8, 8, 128, 2);
                // cta<2,32,256> warp<8,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(3, 3, true, 2, 32, 256, 8, 24, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 2, 32, 256, 8, 24, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 2, 32, 256, 8, 24, 128, 8, 8, 128, 4);
                // cta<2,64,256> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(3, 3, true, 2, 64, 256, 8, 48, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 2, 64, 256, 8, 48, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 2, 64, 256, 8, 48, 128, 8, 8, 128, 4);
                // cta<2,96,256> warp<8,72,128> mma<8,8,128>   WARPS[1x4]
                TEST(3, 3, true, 2, 96, 256, 8, 72, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 2, 96, 256, 8, 72, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 2, 96, 256, 8, 72, 128, 8, 8, 128, 4);
                // cta<2,128,256> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(3, 3, true, 2, 128, 256, 8, 96, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 2, 128, 256, 8, 96, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 2, 128, 256, 8, 96, 128, 8, 8, 128, 4);
                // cta<8,32,256> warp<24,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(3, 3, true, 8, 32, 256, 24, 24, 128, 8, 8, 128, 2);
                // cta<8,64,256> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(3, 3, true, 8, 64, 256, 24, 48, 128, 8, 8, 128, 2);
                // cta<8,96,256> warp<24,72,128> mma<8,8,128>   WARPS[1x4]
                TEST(3, 3, true, 8, 96, 256, 24, 72, 128, 8, 8, 128, 2);
                // cta<8,128,256> warp<24,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(3, 3, true, 8, 128, 256, 24, 96, 128, 8, 8, 128, 2);
                // cta<2,32,384> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(3, 3, true, 2, 32, 384, 8, 48, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 2, 32, 384, 8, 48, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 2, 32, 384, 8, 48, 128, 8, 8, 128, 4);
                // cta<2,48,384> warp<8,72,128> mma<8,8,128>   WARPS[1x2]
                TEST(3, 3, true, 2, 48, 384, 8, 72, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 2, 48, 384, 8, 72, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 2, 48, 384, 8, 72, 128, 8, 8, 128, 4);
                // cta<2,64,384> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(3, 3, true, 2, 64, 384, 8, 96, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 2, 64, 384, 8, 96, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 2, 64, 384, 8, 96, 128, 8, 8, 128, 4);
                // cta<2,80,384> warp<8,120,128> mma<8,8,128>   WARPS[1x2]
                TEST(3, 3, true, 2, 80, 384, 8, 120, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 2, 80, 384, 8, 120, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 2, 80, 384, 8, 120, 128, 8, 8, 128, 4);
                // cta<8,32,384> warp<24,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(3, 3, true, 8, 32, 384, 24, 48, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 8, 32, 384, 24, 48, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 8, 32, 384, 24, 48, 128, 8, 8, 128, 4);
                // cta<8,48,384> warp<24,72,128> mma<8,8,128>   WARPS[1x2]
                TEST(3, 3, true, 8, 48, 384, 24, 72, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 8, 48, 384, 24, 72, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 8, 48, 384, 24, 72, 128, 8, 8, 128, 4);
                // cta<8,64,384> warp<24,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(3, 3, true, 8, 64, 384, 24, 96, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 8, 64, 384, 24, 96, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 8, 64, 384, 24, 96, 128, 8, 8, 128, 4);
                // cta<8,80,384> warp<24,120,128> mma<8,8,128>   WARPS[1x2]
                TEST(3, 3, true, 8, 80, 384, 24, 120, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 8, 80, 384, 24, 120, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 8, 80, 384, 24, 120, 128, 8, 8, 128, 4);
                // cta<2,32,384> warp<8,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(3, 3, true, 2, 32, 384, 8, 24, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 2, 32, 384, 8, 24, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 2, 32, 384, 8, 24, 128, 8, 8, 128, 4);
                // cta<2,64,384> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(3, 3, true, 2, 64, 384, 8, 48, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 2, 64, 384, 8, 48, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 2, 64, 384, 8, 48, 128, 8, 8, 128, 4);
                // cta<2,96,384> warp<8,72,128> mma<8,8,128>   WARPS[1x4]
                TEST(3, 3, true, 2, 96, 384, 8, 72, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 2, 96, 384, 8, 72, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 2, 96, 384, 8, 72, 128, 8, 8, 128, 4);
                // cta<2,128,384> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(3, 3, true, 2, 128, 384, 8, 96, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 2, 128, 384, 8, 96, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 2, 128, 384, 8, 96, 128, 8, 8, 128, 4);
                // cta<8,32,384> warp<24,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(3, 3, true, 8, 32, 384, 24, 24, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 8, 32, 384, 24, 24, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 8, 32, 384, 24, 24, 128, 8, 8, 128, 4);
                // cta<8,64,384> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(3, 3, true, 8, 64, 384, 24, 48, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 8, 64, 384, 24, 48, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 8, 64, 384, 24, 48, 128, 8, 8, 128, 4);
                // cta<8,96,384> warp<24,72,128> mma<8,8,128>   WARPS[1x4]
                TEST(3, 3, true, 8, 96, 384, 24, 72, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 8, 96, 384, 24, 72, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 8, 96, 384, 24, 72, 128, 8, 8, 128, 4);
                // cta<8,128,384> warp<24,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(3, 3, true, 8, 128, 384, 24, 96, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 8, 128, 384, 24, 96, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 8, 128, 384, 24, 96, 128, 8, 8, 128, 4);
                // cta<2,32,512> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(3, 3, true, 2, 32, 512, 8, 48, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 2, 32, 512, 8, 48, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 2, 32, 512, 8, 48, 128, 8, 8, 128, 4);
                // cta<2,48,512> warp<8,72,128> mma<8,8,128>   WARPS[1x2]
                TEST(3, 3, true, 2, 48, 512, 8, 72, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 2, 48, 512, 8, 72, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 2, 48, 512, 8, 72, 128, 8, 8, 128, 4);
                // cta<2,64,512> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(3, 3, true, 2, 64, 512, 8, 96, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 2, 64, 512, 8, 96, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 2, 64, 512, 8, 96, 128, 8, 8, 128, 4);
                // cta<2,80,512> warp<8,120,128> mma<8,8,128>   WARPS[1x2]
                TEST(3, 3, true, 2, 80, 512, 8, 120, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 2, 80, 512, 8, 120, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 2, 80, 512, 8, 120, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<24,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(3, 3, true, 8, 32, 512, 24, 48, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 8, 32, 512, 24, 48, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 8, 32, 512, 24, 48, 128, 8, 8, 128, 4);
                // cta<8,48,512> warp<24,72,128> mma<8,8,128>   WARPS[1x2]
                TEST(3, 3, true, 8, 48, 512, 24, 72, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 8, 48, 512, 24, 72, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 8, 48, 512, 24, 72, 128, 8, 8, 128, 4);
                // cta<8,64,512> warp<24,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(3, 3, true, 8, 64, 512, 24, 96, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 8, 64, 512, 24, 96, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 8, 64, 512, 24, 96, 128, 8, 8, 128, 4);
                // cta<8,80,512> warp<24,120,128> mma<8,8,128>   WARPS[1x2]
                TEST(3, 3, true, 8, 80, 512, 24, 120, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 8, 80, 512, 24, 120, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 8, 80, 512, 24, 120, 128, 8, 8, 128, 4);
                // cta<2,32,512> warp<8,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(3, 3, true, 2, 32, 512, 8, 24, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 2, 32, 512, 8, 24, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 2, 32, 512, 8, 24, 128, 8, 8, 128, 4);
                // cta<2,64,512> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(3, 3, true, 2, 64, 512, 8, 48, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 2, 64, 512, 8, 48, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 2, 64, 512, 8, 48, 128, 8, 8, 128, 4);
                // cta<2,96,512> warp<8,72,128> mma<8,8,128>   WARPS[1x4]
                TEST(3, 3, true, 2, 96, 512, 8, 72, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 2, 96, 512, 8, 72, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 2, 96, 512, 8, 72, 128, 8, 8, 128, 4);
                // cta<2,128,512> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(3, 3, true, 2, 128, 512, 8, 96, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 2, 128, 512, 8, 96, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 2, 128, 512, 8, 96, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<24,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(3, 3, true, 8, 32, 512, 24, 24, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 8, 32, 512, 24, 24, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 8, 32, 512, 24, 24, 128, 8, 8, 128, 4);
                // cta<8,64,512> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(3, 3, true, 8, 64, 512, 24, 48, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 8, 64, 512, 24, 48, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 8, 64, 512, 24, 48, 128, 8, 8, 128, 4);
                // cta<8,96,512> warp<24,72,128> mma<8,8,128>   WARPS[1x4]
                TEST(3, 3, true, 8, 96, 512, 24, 72, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 8, 96, 512, 24, 72, 128, 8, 8, 128, 3);
                TEST(3, 3, true, 8, 96, 512, 24, 72, 128, 8, 8, 128, 4);
                // cta<8,128,512> warp<24,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(3, 3, true, 8, 128, 512, 24, 96, 128, 8, 8, 128, 2);
                TEST(3, 3, true, 8, 128, 512, 24, 96, 128, 8, 8, 128, 3);
            } else {
            }
            break;
        #endif
        default:
            printf("unsupport w%da%d\n", w_bits, x_bits);
        }
        break;
    case 4:
        switch (w_bits) {
        #ifdef W2A4
        case 2:
            if (quant_sign) {
                ////// W2A4 int
                // cta<2,32,256> warp<8,32,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 2, 32, 256, 8, 32, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 32, 256, 8, 32, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 32, 256, 8, 32, 128, 8, 8, 128, 4);
                // cta<2,48,256> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 2, 48, 256, 8, 48, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 48, 256, 8, 48, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 48, 256, 8, 48, 128, 8, 8, 128, 4);
                // cta<2,64,256> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 2, 64, 256, 8, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 64, 256, 8, 64, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 64, 256, 8, 64, 128, 8, 8, 128, 4);
                // cta<2,80,256> warp<8,80,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 2, 80, 256, 8, 80, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 80, 256, 8, 80, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 80, 256, 8, 80, 128, 8, 8, 128, 4);
                // cta<2,96,256> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 2, 96, 256, 8, 96, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 96, 256, 8, 96, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 96, 256, 8, 96, 128, 8, 8, 128, 4);
                // cta<2,112,256> warp<8,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 2, 112, 256, 8, 112, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 112, 256, 8, 112, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 112, 256, 8, 112, 128, 8, 8, 128, 4);
                // cta<2,128,256> warp<8,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 2, 128, 256, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 128, 256, 8, 128, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 128, 256, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,256> warp<16,32,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 4, 32, 256, 16, 32, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 32, 256, 16, 32, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 32, 256, 16, 32, 128, 8, 8, 128, 4);
                // cta<4,48,256> warp<16,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 4, 48, 256, 16, 48, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 48, 256, 16, 48, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 48, 256, 16, 48, 128, 8, 8, 128, 4);
                // cta<4,64,256> warp<16,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 4, 64, 256, 16, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 64, 256, 16, 64, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 64, 256, 16, 64, 128, 8, 8, 128, 4);
                // cta<4,80,256> warp<16,80,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 4, 80, 256, 16, 80, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 80, 256, 16, 80, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 80, 256, 16, 80, 128, 8, 8, 128, 4);
                // cta<4,96,256> warp<16,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 4, 96, 256, 16, 96, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 96, 256, 16, 96, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 96, 256, 16, 96, 128, 8, 8, 128, 4);
                // cta<4,112,256> warp<16,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 4, 112, 256, 16, 112, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 112, 256, 16, 112, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 112, 256, 16, 112, 128, 8, 8, 128, 4);
                // cta<4,128,256> warp<16,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 4, 128, 256, 16, 128, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 128, 256, 16, 128, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 128, 256, 16, 128, 128, 8, 8, 128, 4);
                // cta<8,32,256> warp<32,32,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 8, 32, 256, 32, 32, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 32, 256, 32, 32, 128, 8, 8, 128, 3);
                // cta<8,48,256> warp<32,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 8, 48, 256, 32, 48, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 48, 256, 32, 48, 128, 8, 8, 128, 3);
                // cta<8,64,256> warp<32,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 8, 64, 256, 32, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 64, 256, 32, 64, 128, 8, 8, 128, 3);
                // cta<8,80,256> warp<32,80,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 8, 80, 256, 32, 80, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 80, 256, 32, 80, 128, 8, 8, 128, 3);
                // cta<8,96,256> warp<32,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 8, 96, 256, 32, 96, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 96, 256, 32, 96, 128, 8, 8, 128, 3);
                // cta<8,112,256> warp<32,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 8, 112, 256, 32, 112, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 112, 256, 32, 112, 128, 8, 8, 128, 3);
                // cta<8,128,256> warp<32,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 8, 128, 256, 32, 128, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 128, 256, 32, 128, 128, 8, 8, 128, 3);
                // cta<2,32,256> warp<8,16,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 2, 32, 256, 8, 16, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 32, 256, 8, 16, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 32, 256, 8, 16, 128, 8, 8, 128, 4);
                // cta<2,48,256> warp<8,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 2, 48, 256, 8, 24, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 48, 256, 8, 24, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 48, 256, 8, 24, 128, 8, 8, 128, 4);
                // cta<2,64,256> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 2, 64, 256, 8, 32, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 64, 256, 8, 32, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 64, 256, 8, 32, 128, 8, 8, 128, 4);
                // cta<2,80,256> warp<8,40,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 2, 80, 256, 8, 40, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 80, 256, 8, 40, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 80, 256, 8, 40, 128, 8, 8, 128, 4);
                // cta<2,96,256> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 2, 96, 256, 8, 48, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 96, 256, 8, 48, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 96, 256, 8, 48, 128, 8, 8, 128, 4);
                // cta<2,112,256> warp<8,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 2, 112, 256, 8, 56, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 112, 256, 8, 56, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 112, 256, 8, 56, 128, 8, 8, 128, 4);
                // cta<2,128,256> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 2, 128, 256, 8, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 128, 256, 8, 64, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 128, 256, 8, 64, 128, 8, 8, 128, 4);
                // cta<2,256,256> warp<8,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 2, 256, 256, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 256, 256, 8, 128, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 256, 256, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,256> warp<16,16,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 4, 32, 256, 16, 16, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 32, 256, 16, 16, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 32, 256, 16, 16, 128, 8, 8, 128, 4);
                // cta<4,48,256> warp<16,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 4, 48, 256, 16, 24, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 48, 256, 16, 24, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 48, 256, 16, 24, 128, 8, 8, 128, 4);
                // cta<4,64,256> warp<16,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 4, 64, 256, 16, 32, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 64, 256, 16, 32, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 64, 256, 16, 32, 128, 8, 8, 128, 4);
                // cta<4,80,256> warp<16,40,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 4, 80, 256, 16, 40, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 80, 256, 16, 40, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 80, 256, 16, 40, 128, 8, 8, 128, 4);
                // cta<4,96,256> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 4, 96, 256, 16, 48, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 96, 256, 16, 48, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 96, 256, 16, 48, 128, 8, 8, 128, 4);
                // cta<4,112,256> warp<16,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 4, 112, 256, 16, 56, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 112, 256, 16, 56, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 112, 256, 16, 56, 128, 8, 8, 128, 4);
                // cta<4,128,256> warp<16,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 4, 128, 256, 16, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 128, 256, 16, 64, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 128, 256, 16, 64, 128, 8, 8, 128, 4);
                // cta<4,256,256> warp<16,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 4, 256, 256, 16, 128, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 256, 256, 16, 128, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 256, 256, 16, 128, 128, 8, 8, 128, 4);
                // cta<8,32,256> warp<32,16,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 8, 32, 256, 32, 16, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 32, 256, 32, 16, 128, 8, 8, 128, 3);
                // cta<8,48,256> warp<32,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 8, 48, 256, 32, 24, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 48, 256, 32, 24, 128, 8, 8, 128, 3);
                // cta<8,64,256> warp<32,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 8, 64, 256, 32, 32, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 64, 256, 32, 32, 128, 8, 8, 128, 3);
                // cta<8,80,256> warp<32,40,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 8, 80, 256, 32, 40, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 80, 256, 32, 40, 128, 8, 8, 128, 3);
                // cta<8,96,256> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 8, 96, 256, 32, 48, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 96, 256, 32, 48, 128, 8, 8, 128, 3);
                // cta<8,112,256> warp<32,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 8, 112, 256, 32, 56, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 112, 256, 32, 56, 128, 8, 8, 128, 3);
                // cta<8,128,256> warp<32,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 8, 128, 256, 32, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 128, 256, 32, 64, 128, 8, 8, 128, 3);
                // cta<8,256,256> warp<32,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 8, 256, 256, 32, 128, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 256, 256, 32, 128, 128, 8, 8, 128, 3);
                // cta<4,32,256> warp<8,64,128> mma<8,8,128>   WARPS[2x1]
                TEST(4, 2, true, 4, 32, 256, 8, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 32, 256, 8, 64, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 32, 256, 8, 64, 128, 8, 8, 128, 4);
                // cta<4,48,256> warp<8,96,128> mma<8,8,128>   WARPS[2x1]
                TEST(4, 2, true, 4, 48, 256, 8, 96, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 48, 256, 8, 96, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 48, 256, 8, 96, 128, 8, 8, 128, 4);
                // cta<4,64,256> warp<8,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(4, 2, true, 4, 64, 256, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 64, 256, 8, 128, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 64, 256, 8, 128, 128, 8, 8, 128, 4);
                // cta<8,32,256> warp<16,64,128> mma<8,8,128>   WARPS[2x1]
                TEST(4, 2, true, 8, 32, 256, 16, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 32, 256, 16, 64, 128, 8, 8, 128, 3);
                // cta<8,48,256> warp<16,96,128> mma<8,8,128>   WARPS[2x1]
                TEST(4, 2, true, 8, 48, 256, 16, 96, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 48, 256, 16, 96, 128, 8, 8, 128, 3);
                // cta<8,64,256> warp<16,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(4, 2, true, 8, 64, 256, 16, 128, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 64, 256, 16, 128, 128, 8, 8, 128, 3);
                // cta<4,32,256> warp<8,32,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 4, 32, 256, 8, 32, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 32, 256, 8, 32, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 32, 256, 8, 32, 128, 8, 8, 128, 4);
                // cta<4,48,256> warp<8,48,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 4, 48, 256, 8, 48, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 48, 256, 8, 48, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 48, 256, 8, 48, 128, 8, 8, 128, 4);
                // cta<4,64,256> warp<8,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 4, 64, 256, 8, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 64, 256, 8, 64, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 64, 256, 8, 64, 128, 8, 8, 128, 4);
                // cta<4,80,256> warp<8,80,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 4, 80, 256, 8, 80, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 80, 256, 8, 80, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 80, 256, 8, 80, 128, 8, 8, 128, 4);
                // cta<4,96,256> warp<8,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 4, 96, 256, 8, 96, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 96, 256, 8, 96, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 96, 256, 8, 96, 128, 8, 8, 128, 4);
                // cta<4,112,256> warp<8,112,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 4, 112, 256, 8, 112, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 112, 256, 8, 112, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 112, 256, 8, 112, 128, 8, 8, 128, 4);
                // cta<4,128,256> warp<8,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 4, 128, 256, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 128, 256, 8, 128, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 128, 256, 8, 128, 128, 8, 8, 128, 4);
                // cta<8,32,256> warp<16,32,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 8, 32, 256, 16, 32, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 32, 256, 16, 32, 128, 8, 8, 128, 3);
                // cta<8,48,256> warp<16,48,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 8, 48, 256, 16, 48, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 48, 256, 16, 48, 128, 8, 8, 128, 3);
                // cta<8,64,256> warp<16,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 8, 64, 256, 16, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 64, 256, 16, 64, 128, 8, 8, 128, 3);
                // cta<8,80,256> warp<16,80,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 8, 80, 256, 16, 80, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 80, 256, 16, 80, 128, 8, 8, 128, 3);
                // cta<8,96,256> warp<16,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 8, 96, 256, 16, 96, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 96, 256, 16, 96, 128, 8, 8, 128, 3);
                // cta<8,112,256> warp<16,112,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 8, 112, 256, 16, 112, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 112, 256, 16, 112, 128, 8, 8, 128, 3);
                // cta<8,128,256> warp<16,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 8, 128, 256, 16, 128, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 128, 256, 16, 128, 128, 8, 8, 128, 3);
                // cta<8,32,256> warp<8,64,128> mma<8,8,128>   WARPS[4x1]
                TEST(4, 2, true, 8, 32, 256, 8, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 32, 256, 8, 64, 128, 8, 8, 128, 3);
                // cta<8,48,256> warp<8,96,128> mma<8,8,128>   WARPS[4x1]
                TEST(4, 2, true, 8, 48, 256, 8, 96, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 48, 256, 8, 96, 128, 8, 8, 128, 3);
                // cta<8,64,256> warp<8,128,128> mma<8,8,128>   WARPS[4x1]
                TEST(4, 2, true, 8, 64, 256, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 64, 256, 8, 128, 128, 8, 8, 128, 3);
                // cta<2,32,384> warp<8,32,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 2, 32, 384, 8, 32, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 32, 384, 8, 32, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 32, 384, 8, 32, 128, 8, 8, 128, 4);
                // cta<2,48,384> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 2, 48, 384, 8, 48, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 48, 384, 8, 48, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 48, 384, 8, 48, 128, 8, 8, 128, 4);
                // cta<2,64,384> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 2, 64, 384, 8, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 64, 384, 8, 64, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 64, 384, 8, 64, 128, 8, 8, 128, 4);
                // cta<2,80,384> warp<8,80,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 2, 80, 384, 8, 80, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 80, 384, 8, 80, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 80, 384, 8, 80, 128, 8, 8, 128, 4);
                // cta<2,96,384> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 2, 96, 384, 8, 96, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 96, 384, 8, 96, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 96, 384, 8, 96, 128, 8, 8, 128, 4);
                // cta<2,112,384> warp<8,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 2, 112, 384, 8, 112, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 112, 384, 8, 112, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 112, 384, 8, 112, 128, 8, 8, 128, 4);
                // cta<2,128,384> warp<8,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 2, 128, 384, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 128, 384, 8, 128, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 128, 384, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,384> warp<16,32,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 4, 32, 384, 16, 32, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 32, 384, 16, 32, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 32, 384, 16, 32, 128, 8, 8, 128, 4);
                // cta<4,48,384> warp<16,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 4, 48, 384, 16, 48, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 48, 384, 16, 48, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 48, 384, 16, 48, 128, 8, 8, 128, 4);
                // cta<4,64,384> warp<16,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 4, 64, 384, 16, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 64, 384, 16, 64, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 64, 384, 16, 64, 128, 8, 8, 128, 4);
                // cta<4,80,384> warp<16,80,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 4, 80, 384, 16, 80, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 80, 384, 16, 80, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 80, 384, 16, 80, 128, 8, 8, 128, 4);
                // cta<4,96,384> warp<16,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 4, 96, 384, 16, 96, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 96, 384, 16, 96, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 96, 384, 16, 96, 128, 8, 8, 128, 4);
                // cta<4,112,384> warp<16,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 4, 112, 384, 16, 112, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 112, 384, 16, 112, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 112, 384, 16, 112, 128, 8, 8, 128, 4);
                // cta<4,128,384> warp<16,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 4, 128, 384, 16, 128, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 128, 384, 16, 128, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 128, 384, 16, 128, 128, 8, 8, 128, 4);
                // cta<8,32,384> warp<32,32,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 8, 32, 384, 32, 32, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 32, 384, 32, 32, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 32, 384, 32, 32, 128, 8, 8, 128, 4);
                // cta<8,48,384> warp<32,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 8, 48, 384, 32, 48, 128, 8, 8, 128, 2);
                // cta<8,64,384> warp<32,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 8, 64, 384, 32, 64, 128, 8, 8, 128, 2);
                // cta<8,80,384> warp<32,80,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 8, 80, 384, 32, 80, 128, 8, 8, 128, 2);
                // cta<8,96,384> warp<32,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 8, 96, 384, 32, 96, 128, 8, 8, 128, 2);
                // cta<8,112,384> warp<32,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 8, 112, 384, 32, 112, 128, 8, 8, 128, 2);
                // cta<8,128,384> warp<32,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 8, 128, 384, 32, 128, 128, 8, 8, 128, 2);
                // cta<2,32,384> warp<8,16,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 2, 32, 384, 8, 16, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 32, 384, 8, 16, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 32, 384, 8, 16, 128, 8, 8, 128, 4);
                // cta<2,48,384> warp<8,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 2, 48, 384, 8, 24, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 48, 384, 8, 24, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 48, 384, 8, 24, 128, 8, 8, 128, 4);
                // cta<2,64,384> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 2, 64, 384, 8, 32, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 64, 384, 8, 32, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 64, 384, 8, 32, 128, 8, 8, 128, 4);
                // cta<2,80,384> warp<8,40,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 2, 80, 384, 8, 40, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 80, 384, 8, 40, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 80, 384, 8, 40, 128, 8, 8, 128, 4);
                // cta<2,96,384> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 2, 96, 384, 8, 48, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 96, 384, 8, 48, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 96, 384, 8, 48, 128, 8, 8, 128, 4);
                // cta<2,112,384> warp<8,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 2, 112, 384, 8, 56, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 112, 384, 8, 56, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 112, 384, 8, 56, 128, 8, 8, 128, 4);
                // cta<2,128,384> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 2, 128, 384, 8, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 128, 384, 8, 64, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 128, 384, 8, 64, 128, 8, 8, 128, 4);
                // cta<2,256,384> warp<8,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 2, 256, 384, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 256, 384, 8, 128, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 256, 384, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,384> warp<16,16,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 4, 32, 384, 16, 16, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 32, 384, 16, 16, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 32, 384, 16, 16, 128, 8, 8, 128, 4);
                // cta<4,48,384> warp<16,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 4, 48, 384, 16, 24, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 48, 384, 16, 24, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 48, 384, 16, 24, 128, 8, 8, 128, 4);
                // cta<4,64,384> warp<16,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 4, 64, 384, 16, 32, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 64, 384, 16, 32, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 64, 384, 16, 32, 128, 8, 8, 128, 4);
                // cta<4,80,384> warp<16,40,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 4, 80, 384, 16, 40, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 80, 384, 16, 40, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 80, 384, 16, 40, 128, 8, 8, 128, 4);
                // cta<4,96,384> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 4, 96, 384, 16, 48, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 96, 384, 16, 48, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 96, 384, 16, 48, 128, 8, 8, 128, 4);
                // cta<4,112,384> warp<16,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 4, 112, 384, 16, 56, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 112, 384, 16, 56, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 112, 384, 16, 56, 128, 8, 8, 128, 4);
                // cta<4,128,384> warp<16,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 4, 128, 384, 16, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 128, 384, 16, 64, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 128, 384, 16, 64, 128, 8, 8, 128, 4);
                // cta<4,256,384> warp<16,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 4, 256, 384, 16, 128, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 256, 384, 16, 128, 128, 8, 8, 128, 3);
                // cta<8,32,384> warp<32,16,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 8, 32, 384, 32, 16, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 32, 384, 32, 16, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 32, 384, 32, 16, 128, 8, 8, 128, 4);
                // cta<8,48,384> warp<32,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 8, 48, 384, 32, 24, 128, 8, 8, 128, 2);
                // cta<8,64,384> warp<32,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 8, 64, 384, 32, 32, 128, 8, 8, 128, 2);
                // cta<8,80,384> warp<32,40,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 8, 80, 384, 32, 40, 128, 8, 8, 128, 2);
                // cta<8,96,384> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 8, 96, 384, 32, 48, 128, 8, 8, 128, 2);
                // cta<8,112,384> warp<32,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 8, 112, 384, 32, 56, 128, 8, 8, 128, 2);
                // cta<8,128,384> warp<32,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 8, 128, 384, 32, 64, 128, 8, 8, 128, 2);
                // cta<8,256,384> warp<32,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 8, 256, 384, 32, 128, 128, 8, 8, 128, 2);
                // cta<4,32,384> warp<8,64,128> mma<8,8,128>   WARPS[2x1]
                TEST(4, 2, true, 4, 32, 384, 8, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 32, 384, 8, 64, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 32, 384, 8, 64, 128, 8, 8, 128, 4);
                // cta<4,48,384> warp<8,96,128> mma<8,8,128>   WARPS[2x1]
                TEST(4, 2, true, 4, 48, 384, 8, 96, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 48, 384, 8, 96, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 48, 384, 8, 96, 128, 8, 8, 128, 4);
                // cta<4,64,384> warp<8,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(4, 2, true, 4, 64, 384, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 64, 384, 8, 128, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 64, 384, 8, 128, 128, 8, 8, 128, 4);
                // cta<8,32,384> warp<16,64,128> mma<8,8,128>   WARPS[2x1]
                TEST(4, 2, true, 8, 32, 384, 16, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 32, 384, 16, 64, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 32, 384, 16, 64, 128, 8, 8, 128, 4);
                // cta<8,48,384> warp<16,96,128> mma<8,8,128>   WARPS[2x1]
                TEST(4, 2, true, 8, 48, 384, 16, 96, 128, 8, 8, 128, 2);
                // cta<8,64,384> warp<16,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(4, 2, true, 8, 64, 384, 16, 128, 128, 8, 8, 128, 2);
                // cta<4,32,384> warp<8,32,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 4, 32, 384, 8, 32, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 32, 384, 8, 32, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 32, 384, 8, 32, 128, 8, 8, 128, 4);
                // cta<4,48,384> warp<8,48,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 4, 48, 384, 8, 48, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 48, 384, 8, 48, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 48, 384, 8, 48, 128, 8, 8, 128, 4);
                // cta<4,64,384> warp<8,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 4, 64, 384, 8, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 64, 384, 8, 64, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 64, 384, 8, 64, 128, 8, 8, 128, 4);
                // cta<4,80,384> warp<8,80,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 4, 80, 384, 8, 80, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 80, 384, 8, 80, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 80, 384, 8, 80, 128, 8, 8, 128, 4);
                // cta<4,96,384> warp<8,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 4, 96, 384, 8, 96, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 96, 384, 8, 96, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 96, 384, 8, 96, 128, 8, 8, 128, 4);
                // cta<4,112,384> warp<8,112,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 4, 112, 384, 8, 112, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 112, 384, 8, 112, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 112, 384, 8, 112, 128, 8, 8, 128, 4);
                // cta<4,128,384> warp<8,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 4, 128, 384, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 128, 384, 8, 128, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 128, 384, 8, 128, 128, 8, 8, 128, 4);
                // cta<8,32,384> warp<16,32,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 8, 32, 384, 16, 32, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 32, 384, 16, 32, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 32, 384, 16, 32, 128, 8, 8, 128, 4);
                // cta<8,48,384> warp<16,48,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 8, 48, 384, 16, 48, 128, 8, 8, 128, 2);
                // cta<8,64,384> warp<16,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 8, 64, 384, 16, 64, 128, 8, 8, 128, 2);
                // cta<8,80,384> warp<16,80,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 8, 80, 384, 16, 80, 128, 8, 8, 128, 2);
                // cta<8,96,384> warp<16,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 8, 96, 384, 16, 96, 128, 8, 8, 128, 2);
                // cta<8,112,384> warp<16,112,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 8, 112, 384, 16, 112, 128, 8, 8, 128, 2);
                // cta<8,128,384> warp<16,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 8, 128, 384, 16, 128, 128, 8, 8, 128, 2);
                // cta<8,32,384> warp<8,64,128> mma<8,8,128>   WARPS[4x1]
                TEST(4, 2, true, 8, 32, 384, 8, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 32, 384, 8, 64, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 32, 384, 8, 64, 128, 8, 8, 128, 4);
                // cta<8,48,384> warp<8,96,128> mma<8,8,128>   WARPS[4x1]
                TEST(4, 2, true, 8, 48, 384, 8, 96, 128, 8, 8, 128, 2);
                // cta<8,64,384> warp<8,128,128> mma<8,8,128>   WARPS[4x1]
                TEST(4, 2, true, 8, 64, 384, 8, 128, 128, 8, 8, 128, 2);
                // cta<2,32,512> warp<8,32,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 2, 32, 512, 8, 32, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 32, 512, 8, 32, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 32, 512, 8, 32, 128, 8, 8, 128, 4);
                // cta<2,48,512> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 2, 48, 512, 8, 48, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 48, 512, 8, 48, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 48, 512, 8, 48, 128, 8, 8, 128, 4);
                // cta<2,64,512> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 2, 64, 512, 8, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 64, 512, 8, 64, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 64, 512, 8, 64, 128, 8, 8, 128, 4);
                // cta<2,80,512> warp<8,80,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 2, 80, 512, 8, 80, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 80, 512, 8, 80, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 80, 512, 8, 80, 128, 8, 8, 128, 4);
                // cta<2,96,512> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 2, 96, 512, 8, 96, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 96, 512, 8, 96, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 96, 512, 8, 96, 128, 8, 8, 128, 4);
                // cta<2,112,512> warp<8,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 2, 112, 512, 8, 112, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 112, 512, 8, 112, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 112, 512, 8, 112, 128, 8, 8, 128, 4);
                // cta<2,128,512> warp<8,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 2, 128, 512, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 128, 512, 8, 128, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 128, 512, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,512> warp<16,32,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 4, 32, 512, 16, 32, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 32, 512, 16, 32, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 32, 512, 16, 32, 128, 8, 8, 128, 4);
                // cta<4,48,512> warp<16,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 4, 48, 512, 16, 48, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 48, 512, 16, 48, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 48, 512, 16, 48, 128, 8, 8, 128, 4);
                // cta<4,64,512> warp<16,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 4, 64, 512, 16, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 64, 512, 16, 64, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 64, 512, 16, 64, 128, 8, 8, 128, 4);
                // cta<4,80,512> warp<16,80,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 4, 80, 512, 16, 80, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 80, 512, 16, 80, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 80, 512, 16, 80, 128, 8, 8, 128, 4);
                // cta<4,96,512> warp<16,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 4, 96, 512, 16, 96, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 96, 512, 16, 96, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 96, 512, 16, 96, 128, 8, 8, 128, 4);
                // cta<4,112,512> warp<16,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 4, 112, 512, 16, 112, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 112, 512, 16, 112, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 112, 512, 16, 112, 128, 8, 8, 128, 4);
                // cta<4,128,512> warp<16,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 4, 128, 512, 16, 128, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 128, 512, 16, 128, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 128, 512, 16, 128, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<32,32,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 8, 32, 512, 32, 32, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 32, 512, 32, 32, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 32, 512, 32, 32, 128, 8, 8, 128, 4);
                // cta<8,48,512> warp<32,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 8, 48, 512, 32, 48, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 48, 512, 32, 48, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 48, 512, 32, 48, 128, 8, 8, 128, 4);
                // cta<8,64,512> warp<32,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 8, 64, 512, 32, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 64, 512, 32, 64, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 64, 512, 32, 64, 128, 8, 8, 128, 4);
                // cta<8,80,512> warp<32,80,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 8, 80, 512, 32, 80, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 80, 512, 32, 80, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 80, 512, 32, 80, 128, 8, 8, 128, 4);
                // cta<8,96,512> warp<32,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 8, 96, 512, 32, 96, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 96, 512, 32, 96, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 96, 512, 32, 96, 128, 8, 8, 128, 4);
                // cta<8,112,512> warp<32,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 8, 112, 512, 32, 112, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 112, 512, 32, 112, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 112, 512, 32, 112, 128, 8, 8, 128, 4);
                // cta<8,128,512> warp<32,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 2, true, 8, 128, 512, 32, 128, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 128, 512, 32, 128, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 128, 512, 32, 128, 128, 8, 8, 128, 4);
                // cta<2,32,512> warp<8,16,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 2, 32, 512, 8, 16, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 32, 512, 8, 16, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 32, 512, 8, 16, 128, 8, 8, 128, 4);
                // cta<2,48,512> warp<8,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 2, 48, 512, 8, 24, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 48, 512, 8, 24, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 48, 512, 8, 24, 128, 8, 8, 128, 4);
                // cta<2,64,512> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 2, 64, 512, 8, 32, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 64, 512, 8, 32, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 64, 512, 8, 32, 128, 8, 8, 128, 4);
                // cta<2,80,512> warp<8,40,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 2, 80, 512, 8, 40, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 80, 512, 8, 40, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 80, 512, 8, 40, 128, 8, 8, 128, 4);
                // cta<2,96,512> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 2, 96, 512, 8, 48, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 96, 512, 8, 48, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 96, 512, 8, 48, 128, 8, 8, 128, 4);
                // cta<2,112,512> warp<8,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 2, 112, 512, 8, 56, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 112, 512, 8, 56, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 112, 512, 8, 56, 128, 8, 8, 128, 4);
                // cta<2,128,512> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 2, 128, 512, 8, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 128, 512, 8, 64, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 2, 128, 512, 8, 64, 128, 8, 8, 128, 4);
                // cta<2,256,512> warp<8,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 2, 256, 512, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 2, 256, 512, 8, 128, 128, 8, 8, 128, 3);
                // cta<4,32,512> warp<16,16,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 4, 32, 512, 16, 16, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 32, 512, 16, 16, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 32, 512, 16, 16, 128, 8, 8, 128, 4);
                // cta<4,48,512> warp<16,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 4, 48, 512, 16, 24, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 48, 512, 16, 24, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 48, 512, 16, 24, 128, 8, 8, 128, 4);
                // cta<4,64,512> warp<16,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 4, 64, 512, 16, 32, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 64, 512, 16, 32, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 64, 512, 16, 32, 128, 8, 8, 128, 4);
                // cta<4,80,512> warp<16,40,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 4, 80, 512, 16, 40, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 80, 512, 16, 40, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 80, 512, 16, 40, 128, 8, 8, 128, 4);
                // cta<4,96,512> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 4, 96, 512, 16, 48, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 96, 512, 16, 48, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 96, 512, 16, 48, 128, 8, 8, 128, 4);
                // cta<4,112,512> warp<16,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 4, 112, 512, 16, 56, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 112, 512, 16, 56, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 112, 512, 16, 56, 128, 8, 8, 128, 4);
                // cta<4,128,512> warp<16,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 4, 128, 512, 16, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 128, 512, 16, 64, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 128, 512, 16, 64, 128, 8, 8, 128, 4);
                // cta<4,256,512> warp<16,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 4, 256, 512, 16, 128, 128, 8, 8, 128, 2);
                // cta<8,32,512> warp<32,16,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 8, 32, 512, 32, 16, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 32, 512, 32, 16, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 32, 512, 32, 16, 128, 8, 8, 128, 4);
                // cta<8,48,512> warp<32,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 8, 48, 512, 32, 24, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 48, 512, 32, 24, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 48, 512, 32, 24, 128, 8, 8, 128, 4);
                // cta<8,64,512> warp<32,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 8, 64, 512, 32, 32, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 64, 512, 32, 32, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 64, 512, 32, 32, 128, 8, 8, 128, 4);
                // cta<8,80,512> warp<32,40,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 8, 80, 512, 32, 40, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 80, 512, 32, 40, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 80, 512, 32, 40, 128, 8, 8, 128, 4);
                // cta<8,96,512> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 8, 96, 512, 32, 48, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 96, 512, 32, 48, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 96, 512, 32, 48, 128, 8, 8, 128, 4);
                // cta<8,112,512> warp<32,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 8, 112, 512, 32, 56, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 112, 512, 32, 56, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 112, 512, 32, 56, 128, 8, 8, 128, 4);
                // cta<8,128,512> warp<32,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 8, 128, 512, 32, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 128, 512, 32, 64, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 128, 512, 32, 64, 128, 8, 8, 128, 4);
                // cta<8,256,512> warp<32,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 2, true, 8, 256, 512, 32, 128, 128, 8, 8, 128, 2);
                // cta<4,32,512> warp<8,64,128> mma<8,8,128>   WARPS[2x1]
                TEST(4, 2, true, 4, 32, 512, 8, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 32, 512, 8, 64, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 32, 512, 8, 64, 128, 8, 8, 128, 4);
                // cta<4,48,512> warp<8,96,128> mma<8,8,128>   WARPS[2x1]
                TEST(4, 2, true, 4, 48, 512, 8, 96, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 48, 512, 8, 96, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 48, 512, 8, 96, 128, 8, 8, 128, 4);
                // cta<4,64,512> warp<8,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(4, 2, true, 4, 64, 512, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 64, 512, 8, 128, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 64, 512, 8, 128, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<16,64,128> mma<8,8,128>   WARPS[2x1]
                TEST(4, 2, true, 8, 32, 512, 16, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 32, 512, 16, 64, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 32, 512, 16, 64, 128, 8, 8, 128, 4);
                // cta<8,48,512> warp<16,96,128> mma<8,8,128>   WARPS[2x1]
                TEST(4, 2, true, 8, 48, 512, 16, 96, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 48, 512, 16, 96, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 48, 512, 16, 96, 128, 8, 8, 128, 4);
                // cta<8,64,512> warp<16,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(4, 2, true, 8, 64, 512, 16, 128, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 64, 512, 16, 128, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 64, 512, 16, 128, 128, 8, 8, 128, 4);
                // cta<4,32,512> warp<8,32,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 4, 32, 512, 8, 32, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 32, 512, 8, 32, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 32, 512, 8, 32, 128, 8, 8, 128, 4);
                // cta<4,48,512> warp<8,48,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 4, 48, 512, 8, 48, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 48, 512, 8, 48, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 48, 512, 8, 48, 128, 8, 8, 128, 4);
                // cta<4,64,512> warp<8,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 4, 64, 512, 8, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 64, 512, 8, 64, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 64, 512, 8, 64, 128, 8, 8, 128, 4);
                // cta<4,80,512> warp<8,80,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 4, 80, 512, 8, 80, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 80, 512, 8, 80, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 80, 512, 8, 80, 128, 8, 8, 128, 4);
                // cta<4,96,512> warp<8,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 4, 96, 512, 8, 96, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 96, 512, 8, 96, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 96, 512, 8, 96, 128, 8, 8, 128, 4);
                // cta<4,112,512> warp<8,112,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 4, 112, 512, 8, 112, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 112, 512, 8, 112, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 112, 512, 8, 112, 128, 8, 8, 128, 4);
                // cta<4,128,512> warp<8,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 4, 128, 512, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 4, 128, 512, 8, 128, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 4, 128, 512, 8, 128, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<16,32,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 8, 32, 512, 16, 32, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 32, 512, 16, 32, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 32, 512, 16, 32, 128, 8, 8, 128, 4);
                // cta<8,48,512> warp<16,48,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 8, 48, 512, 16, 48, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 48, 512, 16, 48, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 48, 512, 16, 48, 128, 8, 8, 128, 4);
                // cta<8,64,512> warp<16,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 8, 64, 512, 16, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 64, 512, 16, 64, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 64, 512, 16, 64, 128, 8, 8, 128, 4);
                // cta<8,80,512> warp<16,80,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 8, 80, 512, 16, 80, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 80, 512, 16, 80, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 80, 512, 16, 80, 128, 8, 8, 128, 4);
                // cta<8,96,512> warp<16,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 8, 96, 512, 16, 96, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 96, 512, 16, 96, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 96, 512, 16, 96, 128, 8, 8, 128, 4);
                // cta<8,112,512> warp<16,112,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 8, 112, 512, 16, 112, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 112, 512, 16, 112, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 112, 512, 16, 112, 128, 8, 8, 128, 4);
                // cta<8,128,512> warp<16,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 2, true, 8, 128, 512, 16, 128, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 128, 512, 16, 128, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 128, 512, 16, 128, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<8,64,128> mma<8,8,128>   WARPS[4x1]
                TEST(4, 2, true, 8, 32, 512, 8, 64, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 32, 512, 8, 64, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 32, 512, 8, 64, 128, 8, 8, 128, 4);
                // cta<8,48,512> warp<8,96,128> mma<8,8,128>   WARPS[4x1]
                TEST(4, 2, true, 8, 48, 512, 8, 96, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 48, 512, 8, 96, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 48, 512, 8, 96, 128, 8, 8, 128, 4);
                // cta<8,64,512> warp<8,128,128> mma<8,8,128>   WARPS[4x1]
                TEST(4, 2, true, 8, 64, 512, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 2, true, 8, 64, 512, 8, 128, 128, 8, 8, 128, 3);
                TEST(4, 2, true, 8, 64, 512, 8, 128, 128, 8, 8, 128, 4);
            } else {
            }
            break;
        #endif

        #ifdef W4A4
        case 4:
            if (quant_sign) {
                ////// W4A4 int
                // cta<2,32,256> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 2, 32, 256, 8, 64, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 32, 256, 8, 64, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 32, 256, 8, 64, 128, 8, 8, 128, 4);
                // cta<2,48,256> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 2, 48, 256, 8, 96, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 48, 256, 8, 96, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 48, 256, 8, 96, 128, 8, 8, 128, 4);
                // cta<2,64,256> warp<8,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 2, 64, 256, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 64, 256, 8, 128, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 64, 256, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,256> warp<16,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 4, 32, 256, 16, 64, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 32, 256, 16, 64, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 32, 256, 16, 64, 128, 8, 8, 128, 4);
                // cta<4,48,256> warp<16,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 4, 48, 256, 16, 96, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 48, 256, 16, 96, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 48, 256, 16, 96, 128, 8, 8, 128, 4);
                // cta<4,64,256> warp<16,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 4, 64, 256, 16, 128, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 64, 256, 16, 128, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 64, 256, 16, 128, 128, 8, 8, 128, 4);
                // cta<8,32,256> warp<32,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 8, 32, 256, 32, 64, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 32, 256, 32, 64, 128, 8, 8, 128, 3);
                // cta<8,48,256> warp<32,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 8, 48, 256, 32, 96, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 48, 256, 32, 96, 128, 8, 8, 128, 3);
                // cta<8,64,256> warp<32,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 8, 64, 256, 32, 128, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 64, 256, 32, 128, 128, 8, 8, 128, 3);
                // cta<2,32,256> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 2, 32, 256, 8, 32, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 32, 256, 8, 32, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 32, 256, 8, 32, 128, 8, 8, 128, 4);
                // cta<2,48,256> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 2, 48, 256, 8, 48, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 48, 256, 8, 48, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 48, 256, 8, 48, 128, 8, 8, 128, 4);
                // cta<2,64,256> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 2, 64, 256, 8, 64, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 64, 256, 8, 64, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 64, 256, 8, 64, 128, 8, 8, 128, 4);
                // cta<2,80,256> warp<8,80,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 2, 80, 256, 8, 80, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 80, 256, 8, 80, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 80, 256, 8, 80, 128, 8, 8, 128, 4);
                // cta<2,96,256> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 2, 96, 256, 8, 96, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 96, 256, 8, 96, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 96, 256, 8, 96, 128, 8, 8, 128, 4);
                // cta<2,112,256> warp<8,112,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 2, 112, 256, 8, 112, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 112, 256, 8, 112, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 112, 256, 8, 112, 128, 8, 8, 128, 4);
                // cta<2,128,256> warp<8,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 2, 128, 256, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 128, 256, 8, 128, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 128, 256, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,256> warp<16,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 4, 32, 256, 16, 32, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 32, 256, 16, 32, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 32, 256, 16, 32, 128, 8, 8, 128, 4);
                // cta<4,48,256> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 4, 48, 256, 16, 48, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 48, 256, 16, 48, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 48, 256, 16, 48, 128, 8, 8, 128, 4);
                // cta<4,64,256> warp<16,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 4, 64, 256, 16, 64, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 64, 256, 16, 64, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 64, 256, 16, 64, 128, 8, 8, 128, 4);
                // cta<4,80,256> warp<16,80,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 4, 80, 256, 16, 80, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 80, 256, 16, 80, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 80, 256, 16, 80, 128, 8, 8, 128, 4);
                // cta<4,96,256> warp<16,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 4, 96, 256, 16, 96, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 96, 256, 16, 96, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 96, 256, 16, 96, 128, 8, 8, 128, 4);
                // cta<4,112,256> warp<16,112,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 4, 112, 256, 16, 112, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 112, 256, 16, 112, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 112, 256, 16, 112, 128, 8, 8, 128, 4);
                // cta<4,128,256> warp<16,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 4, 128, 256, 16, 128, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 128, 256, 16, 128, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 128, 256, 16, 128, 128, 8, 8, 128, 4);
                // cta<8,32,256> warp<32,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 8, 32, 256, 32, 32, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 32, 256, 32, 32, 128, 8, 8, 128, 3);
                // cta<8,48,256> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 8, 48, 256, 32, 48, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 48, 256, 32, 48, 128, 8, 8, 128, 3);
                // cta<8,64,256> warp<32,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 8, 64, 256, 32, 64, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 64, 256, 32, 64, 128, 8, 8, 128, 3);
                // cta<8,80,256> warp<32,80,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 8, 80, 256, 32, 80, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 80, 256, 32, 80, 128, 8, 8, 128, 3);
                // cta<8,96,256> warp<32,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 8, 96, 256, 32, 96, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 96, 256, 32, 96, 128, 8, 8, 128, 3);
                // cta<8,112,256> warp<32,112,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 8, 112, 256, 32, 112, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 112, 256, 32, 112, 128, 8, 8, 128, 3);
                // cta<8,128,256> warp<32,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 8, 128, 256, 32, 128, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 128, 256, 32, 128, 128, 8, 8, 128, 3);
                // cta<4,32,256> warp<8,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(4, 4, true, 4, 32, 256, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 32, 256, 8, 128, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 32, 256, 8, 128, 128, 8, 8, 128, 4);
                // cta<8,32,256> warp<16,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(4, 4, true, 8, 32, 256, 16, 128, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 32, 256, 16, 128, 128, 8, 8, 128, 3);
                // cta<4,32,256> warp<8,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 4, true, 4, 32, 256, 8, 64, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 32, 256, 8, 64, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 32, 256, 8, 64, 128, 8, 8, 128, 4);
                // cta<4,48,256> warp<8,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 4, true, 4, 48, 256, 8, 96, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 48, 256, 8, 96, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 48, 256, 8, 96, 128, 8, 8, 128, 4);
                // cta<4,64,256> warp<8,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 4, true, 4, 64, 256, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 64, 256, 8, 128, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 64, 256, 8, 128, 128, 8, 8, 128, 4);
                // cta<8,32,256> warp<16,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 4, true, 8, 32, 256, 16, 64, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 32, 256, 16, 64, 128, 8, 8, 128, 3);
                // cta<8,48,256> warp<16,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 4, true, 8, 48, 256, 16, 96, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 48, 256, 16, 96, 128, 8, 8, 128, 3);
                // cta<8,64,256> warp<16,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 4, true, 8, 64, 256, 16, 128, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 64, 256, 16, 128, 128, 8, 8, 128, 3);
                // cta<8,32,256> warp<8,128,128> mma<8,8,128>   WARPS[4x1]
                TEST(4, 4, true, 8, 32, 256, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 32, 256, 8, 128, 128, 8, 8, 128, 3);
                // cta<2,32,384> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 2, 32, 384, 8, 64, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 32, 384, 8, 64, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 32, 384, 8, 64, 128, 8, 8, 128, 4);
                // cta<2,48,384> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 2, 48, 384, 8, 96, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 48, 384, 8, 96, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 48, 384, 8, 96, 128, 8, 8, 128, 4);
                // cta<2,64,384> warp<8,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 2, 64, 384, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 64, 384, 8, 128, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 64, 384, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,384> warp<16,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 4, 32, 384, 16, 64, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 32, 384, 16, 64, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 32, 384, 16, 64, 128, 8, 8, 128, 4);
                // cta<4,48,384> warp<16,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 4, 48, 384, 16, 96, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 48, 384, 16, 96, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 48, 384, 16, 96, 128, 8, 8, 128, 4);
                // cta<4,64,384> warp<16,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 4, 64, 384, 16, 128, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 64, 384, 16, 128, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 64, 384, 16, 128, 128, 8, 8, 128, 4);
                // cta<8,32,384> warp<32,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 8, 32, 384, 32, 64, 128, 8, 8, 128, 2);
                // cta<8,48,384> warp<32,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 8, 48, 384, 32, 96, 128, 8, 8, 128, 2);
                // cta<8,64,384> warp<32,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 8, 64, 384, 32, 128, 128, 8, 8, 128, 2);
                // cta<2,32,384> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 2, 32, 384, 8, 32, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 32, 384, 8, 32, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 32, 384, 8, 32, 128, 8, 8, 128, 4);
                // cta<2,48,384> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 2, 48, 384, 8, 48, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 48, 384, 8, 48, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 48, 384, 8, 48, 128, 8, 8, 128, 4);
                // cta<2,64,384> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 2, 64, 384, 8, 64, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 64, 384, 8, 64, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 64, 384, 8, 64, 128, 8, 8, 128, 4);
                // cta<2,80,384> warp<8,80,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 2, 80, 384, 8, 80, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 80, 384, 8, 80, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 80, 384, 8, 80, 128, 8, 8, 128, 4);
                // cta<2,96,384> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 2, 96, 384, 8, 96, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 96, 384, 8, 96, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 96, 384, 8, 96, 128, 8, 8, 128, 4);
                // cta<2,112,384> warp<8,112,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 2, 112, 384, 8, 112, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 112, 384, 8, 112, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 112, 384, 8, 112, 128, 8, 8, 128, 4);
                // cta<2,128,384> warp<8,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 2, 128, 384, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 128, 384, 8, 128, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 128, 384, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,384> warp<16,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 4, 32, 384, 16, 32, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 32, 384, 16, 32, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 32, 384, 16, 32, 128, 8, 8, 128, 4);
                // cta<4,48,384> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 4, 48, 384, 16, 48, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 48, 384, 16, 48, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 48, 384, 16, 48, 128, 8, 8, 128, 4);
                // cta<4,64,384> warp<16,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 4, 64, 384, 16, 64, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 64, 384, 16, 64, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 64, 384, 16, 64, 128, 8, 8, 128, 4);
                // cta<4,80,384> warp<16,80,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 4, 80, 384, 16, 80, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 80, 384, 16, 80, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 80, 384, 16, 80, 128, 8, 8, 128, 4);
                // cta<4,96,384> warp<16,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 4, 96, 384, 16, 96, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 96, 384, 16, 96, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 96, 384, 16, 96, 128, 8, 8, 128, 4);
                // cta<4,112,384> warp<16,112,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 4, 112, 384, 16, 112, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 112, 384, 16, 112, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 112, 384, 16, 112, 128, 8, 8, 128, 4);
                // cta<4,128,384> warp<16,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 4, 128, 384, 16, 128, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 128, 384, 16, 128, 128, 8, 8, 128, 3);
                // cta<8,32,384> warp<32,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 8, 32, 384, 32, 32, 128, 8, 8, 128, 2);
                // cta<8,48,384> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 8, 48, 384, 32, 48, 128, 8, 8, 128, 2);
                // cta<8,64,384> warp<32,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 8, 64, 384, 32, 64, 128, 8, 8, 128, 2);
                // cta<8,80,384> warp<32,80,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 8, 80, 384, 32, 80, 128, 8, 8, 128, 2);
                // cta<8,96,384> warp<32,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 8, 96, 384, 32, 96, 128, 8, 8, 128, 2);
                // cta<8,112,384> warp<32,112,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 8, 112, 384, 32, 112, 128, 8, 8, 128, 2);
                // cta<8,128,384> warp<32,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 8, 128, 384, 32, 128, 128, 8, 8, 128, 2);
                // cta<4,32,384> warp<8,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(4, 4, true, 4, 32, 384, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 32, 384, 8, 128, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 32, 384, 8, 128, 128, 8, 8, 128, 4);
                // cta<8,32,384> warp<16,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(4, 4, true, 8, 32, 384, 16, 128, 128, 8, 8, 128, 2);
                // cta<4,32,384> warp<8,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 4, true, 4, 32, 384, 8, 64, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 32, 384, 8, 64, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 32, 384, 8, 64, 128, 8, 8, 128, 4);
                // cta<4,48,384> warp<8,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 4, true, 4, 48, 384, 8, 96, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 48, 384, 8, 96, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 48, 384, 8, 96, 128, 8, 8, 128, 4);
                // cta<4,64,384> warp<8,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 4, true, 4, 64, 384, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 64, 384, 8, 128, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 64, 384, 8, 128, 128, 8, 8, 128, 4);
                // cta<8,32,384> warp<16,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 4, true, 8, 32, 384, 16, 64, 128, 8, 8, 128, 2);
                // cta<8,48,384> warp<16,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 4, true, 8, 48, 384, 16, 96, 128, 8, 8, 128, 2);
                // cta<8,64,384> warp<16,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 4, true, 8, 64, 384, 16, 128, 128, 8, 8, 128, 2);
                // cta<8,32,384> warp<8,128,128> mma<8,8,128>   WARPS[4x1]
                TEST(4, 4, true, 8, 32, 384, 8, 128, 128, 8, 8, 128, 2);
                // cta<2,32,512> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 2, 32, 512, 8, 64, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 32, 512, 8, 64, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 32, 512, 8, 64, 128, 8, 8, 128, 4);
                // cta<2,48,512> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 2, 48, 512, 8, 96, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 48, 512, 8, 96, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 48, 512, 8, 96, 128, 8, 8, 128, 4);
                // cta<2,64,512> warp<8,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 2, 64, 512, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 64, 512, 8, 128, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 64, 512, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,512> warp<16,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 4, 32, 512, 16, 64, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 32, 512, 16, 64, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 32, 512, 16, 64, 128, 8, 8, 128, 4);
                // cta<4,48,512> warp<16,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 4, 48, 512, 16, 96, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 48, 512, 16, 96, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 48, 512, 16, 96, 128, 8, 8, 128, 4);
                // cta<4,64,512> warp<16,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 4, 64, 512, 16, 128, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 64, 512, 16, 128, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 64, 512, 16, 128, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<32,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 8, 32, 512, 32, 64, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 32, 512, 32, 64, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 8, 32, 512, 32, 64, 128, 8, 8, 128, 4);
                // cta<8,48,512> warp<32,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 8, 48, 512, 32, 96, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 48, 512, 32, 96, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 8, 48, 512, 32, 96, 128, 8, 8, 128, 4);
                // cta<8,64,512> warp<32,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(4, 4, true, 8, 64, 512, 32, 128, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 64, 512, 32, 128, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 8, 64, 512, 32, 128, 128, 8, 8, 128, 4);
                // cta<2,32,512> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 2, 32, 512, 8, 32, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 32, 512, 8, 32, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 32, 512, 8, 32, 128, 8, 8, 128, 4);
                // cta<2,48,512> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 2, 48, 512, 8, 48, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 48, 512, 8, 48, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 48, 512, 8, 48, 128, 8, 8, 128, 4);
                // cta<2,64,512> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 2, 64, 512, 8, 64, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 64, 512, 8, 64, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 64, 512, 8, 64, 128, 8, 8, 128, 4);
                // cta<2,80,512> warp<8,80,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 2, 80, 512, 8, 80, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 80, 512, 8, 80, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 80, 512, 8, 80, 128, 8, 8, 128, 4);
                // cta<2,96,512> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 2, 96, 512, 8, 96, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 96, 512, 8, 96, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 2, 96, 512, 8, 96, 128, 8, 8, 128, 4);
                // cta<2,112,512> warp<8,112,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 2, 112, 512, 8, 112, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 112, 512, 8, 112, 128, 8, 8, 128, 3);
                // cta<2,128,512> warp<8,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 2, 128, 512, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 2, 128, 512, 8, 128, 128, 8, 8, 128, 3);
                // cta<4,32,512> warp<16,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 4, 32, 512, 16, 32, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 32, 512, 16, 32, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 32, 512, 16, 32, 128, 8, 8, 128, 4);
                // cta<4,48,512> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 4, 48, 512, 16, 48, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 48, 512, 16, 48, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 48, 512, 16, 48, 128, 8, 8, 128, 4);
                // cta<4,64,512> warp<16,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 4, 64, 512, 16, 64, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 64, 512, 16, 64, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 64, 512, 16, 64, 128, 8, 8, 128, 4);
                // cta<4,80,512> warp<16,80,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 4, 80, 512, 16, 80, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 80, 512, 16, 80, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 80, 512, 16, 80, 128, 8, 8, 128, 4);
                // cta<4,96,512> warp<16,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 4, 96, 512, 16, 96, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 96, 512, 16, 96, 128, 8, 8, 128, 3);
                // cta<4,112,512> warp<16,112,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 4, 112, 512, 16, 112, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 112, 512, 16, 112, 128, 8, 8, 128, 3);
                // cta<4,128,512> warp<16,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 4, 128, 512, 16, 128, 128, 8, 8, 128, 2);
                // cta<8,32,512> warp<32,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 8, 32, 512, 32, 32, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 32, 512, 32, 32, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 8, 32, 512, 32, 32, 128, 8, 8, 128, 4);
                // cta<8,48,512> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 8, 48, 512, 32, 48, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 48, 512, 32, 48, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 8, 48, 512, 32, 48, 128, 8, 8, 128, 4);
                // cta<8,64,512> warp<32,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 8, 64, 512, 32, 64, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 64, 512, 32, 64, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 8, 64, 512, 32, 64, 128, 8, 8, 128, 4);
                // cta<8,80,512> warp<32,80,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 8, 80, 512, 32, 80, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 80, 512, 32, 80, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 8, 80, 512, 32, 80, 128, 8, 8, 128, 4);
                // cta<8,96,512> warp<32,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 8, 96, 512, 32, 96, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 96, 512, 32, 96, 128, 8, 8, 128, 3);
                // cta<8,112,512> warp<32,112,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 8, 112, 512, 32, 112, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 112, 512, 32, 112, 128, 8, 8, 128, 3);
                // cta<8,128,512> warp<32,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(4, 4, true, 8, 128, 512, 32, 128, 128, 8, 8, 128, 2);
                // cta<4,32,512> warp<8,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(4, 4, true, 4, 32, 512, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 32, 512, 8, 128, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 32, 512, 8, 128, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<16,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(4, 4, true, 8, 32, 512, 16, 128, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 32, 512, 16, 128, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 8, 32, 512, 16, 128, 128, 8, 8, 128, 4);
                // cta<4,32,512> warp<8,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 4, true, 4, 32, 512, 8, 64, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 32, 512, 8, 64, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 32, 512, 8, 64, 128, 8, 8, 128, 4);
                // cta<4,48,512> warp<8,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 4, true, 4, 48, 512, 8, 96, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 48, 512, 8, 96, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 48, 512, 8, 96, 128, 8, 8, 128, 4);
                // cta<4,64,512> warp<8,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 4, true, 4, 64, 512, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 4, 64, 512, 8, 128, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 4, 64, 512, 8, 128, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<16,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 4, true, 8, 32, 512, 16, 64, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 32, 512, 16, 64, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 8, 32, 512, 16, 64, 128, 8, 8, 128, 4);
                // cta<8,48,512> warp<16,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 4, true, 8, 48, 512, 16, 96, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 48, 512, 16, 96, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 8, 48, 512, 16, 96, 128, 8, 8, 128, 4);
                // cta<8,64,512> warp<16,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(4, 4, true, 8, 64, 512, 16, 128, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 64, 512, 16, 128, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 8, 64, 512, 16, 128, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<8,128,128> mma<8,8,128>   WARPS[4x1]
                TEST(4, 4, true, 8, 32, 512, 8, 128, 128, 8, 8, 128, 2);
                TEST(4, 4, true, 8, 32, 512, 8, 128, 128, 8, 8, 128, 3);
                TEST(4, 4, true, 8, 32, 512, 8, 128, 128, 8, 8, 128, 4);
            } else {
            }
            break;
        #endif
        default:
            printf("unsupport w%da%d\n", w_bits, x_bits);
        }
        break;
    case 5:
        switch (w_bits) {
        #ifdef W5A5
        case 5:
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
            break;
        #endif
        default:
            printf("unsupport w%da%d\n", w_bits, x_bits);
        }
        break;
    case 6:
        switch (w_bits) {
        #ifdef W2A6
        case 2:
            if(quant_sign){
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
            }else{}
            break;
        #endif
        #ifdef W6A6
        case 6:
            if (quant_sign) {
                ////// W6A6 int
                // cta<1,32,256> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(6, 6, true, 1, 32, 256, 8, 96, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 1, 32, 256, 8, 96, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 1, 32, 256, 8, 96, 128, 8, 8, 128, 4);
                // cta<4,32,256> warp<24,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(6, 6, true, 4, 32, 256, 24, 96, 128, 8, 8, 128, 2);
                // cta<8,32,256> warp<48,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(6, 6, true, 8, 32, 256, 48, 96, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 8, 32, 256, 48, 96, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 8, 32, 256, 48, 96, 128, 8, 8, 128, 4);
                TEST(6, 6, true, 8, 32, 256, 48, 96, 128, 8, 8, 128, 5);
                // cta<1,32,256> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 1, 32, 256, 8, 48, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 1, 32, 256, 8, 48, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 1, 32, 256, 8, 48, 128, 8, 8, 128, 4);
                // cta<1,48,256> warp<8,72,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 1, 48, 256, 8, 72, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 1, 48, 256, 8, 72, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 1, 48, 256, 8, 72, 128, 8, 8, 128, 4);
                // cta<1,64,256> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 1, 64, 256, 8, 96, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 1, 64, 256, 8, 96, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 1, 64, 256, 8, 96, 128, 8, 8, 128, 4);
                // cta<1,80,256> warp<8,120,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 1, 80, 256, 8, 120, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 1, 80, 256, 8, 120, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 1, 80, 256, 8, 120, 128, 8, 8, 128, 4);
                // cta<4,32,256> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 4, 32, 256, 24, 48, 128, 8, 8, 128, 2);
                // cta<4,48,256> warp<24,72,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 4, 48, 256, 24, 72, 128, 8, 8, 128, 2);
                // cta<4,64,256> warp<24,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 4, 64, 256, 24, 96, 128, 8, 8, 128, 2);
                // cta<4,80,256> warp<24,120,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 4, 80, 256, 24, 120, 128, 8, 8, 128, 2);
                // cta<8,32,256> warp<48,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 8, 32, 256, 48, 48, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 8, 32, 256, 48, 48, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 8, 32, 256, 48, 48, 128, 8, 8, 128, 4);
                TEST(6, 6, true, 8, 32, 256, 48, 48, 128, 8, 8, 128, 5);
                // cta<8,48,256> warp<48,72,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 8, 48, 256, 48, 72, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 8, 48, 256, 48, 72, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 8, 48, 256, 48, 72, 128, 8, 8, 128, 4);
                TEST(6, 6, true, 8, 48, 256, 48, 72, 128, 8, 8, 128, 5);
                // cta<8,64,256> warp<48,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 8, 64, 256, 48, 96, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 8, 64, 256, 48, 96, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 8, 64, 256, 48, 96, 128, 8, 8, 128, 4);
                TEST(6, 6, true, 8, 64, 256, 48, 96, 128, 8, 8, 128, 5);
                // cta<8,80,256> warp<48,120,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 8, 80, 256, 48, 120, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 8, 80, 256, 48, 120, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 8, 80, 256, 48, 120, 128, 8, 8, 128, 4);
                TEST(6, 6, true, 8, 80, 256, 48, 120, 128, 8, 8, 128, 5);
                // cta<8,32,256> warp<24,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(6, 6, true, 8, 32, 256, 24, 96, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 8, 32, 256, 24, 96, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 8, 32, 256, 24, 96, 128, 8, 8, 128, 4);
                TEST(6, 6, true, 8, 32, 256, 24, 96, 128, 8, 8, 128, 5);
                // cta<1,32,384> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(6, 6, true, 1, 32, 384, 8, 96, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 1, 32, 384, 8, 96, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 1, 32, 384, 8, 96, 128, 8, 8, 128, 4);
                // cta<4,32,384> warp<24,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(6, 6, true, 4, 32, 384, 24, 96, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 4, 32, 384, 24, 96, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 4, 32, 384, 24, 96, 128, 8, 8, 128, 4);
                // cta<8,32,384> warp<48,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(6, 6, true, 8, 32, 384, 48, 96, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 8, 32, 384, 48, 96, 128, 8, 8, 128, 3);
                // cta<1,32,384> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 1, 32, 384, 8, 48, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 1, 32, 384, 8, 48, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 1, 32, 384, 8, 48, 128, 8, 8, 128, 4);
                // cta<1,48,384> warp<8,72,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 1, 48, 384, 8, 72, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 1, 48, 384, 8, 72, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 1, 48, 384, 8, 72, 128, 8, 8, 128, 4);
                // cta<1,64,384> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 1, 64, 384, 8, 96, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 1, 64, 384, 8, 96, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 1, 64, 384, 8, 96, 128, 8, 8, 128, 4);
                // cta<1,80,384> warp<8,120,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 1, 80, 384, 8, 120, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 1, 80, 384, 8, 120, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 1, 80, 384, 8, 120, 128, 8, 8, 128, 4);
                // cta<4,32,384> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 4, 32, 384, 24, 48, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 4, 32, 384, 24, 48, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 4, 32, 384, 24, 48, 128, 8, 8, 128, 4);
                // cta<4,48,384> warp<24,72,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 4, 48, 384, 24, 72, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 4, 48, 384, 24, 72, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 4, 48, 384, 24, 72, 128, 8, 8, 128, 4);
                // cta<4,64,384> warp<24,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 4, 64, 384, 24, 96, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 4, 64, 384, 24, 96, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 4, 64, 384, 24, 96, 128, 8, 8, 128, 4);
                // cta<4,80,384> warp<24,120,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 4, 80, 384, 24, 120, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 4, 80, 384, 24, 120, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 4, 80, 384, 24, 120, 128, 8, 8, 128, 4);
                // cta<8,32,384> warp<48,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 8, 32, 384, 48, 48, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 8, 32, 384, 48, 48, 128, 8, 8, 128, 3);
                // cta<8,48,384> warp<48,72,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 8, 48, 384, 48, 72, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 8, 48, 384, 48, 72, 128, 8, 8, 128, 3);
                // cta<8,64,384> warp<48,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 8, 64, 384, 48, 96, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 8, 64, 384, 48, 96, 128, 8, 8, 128, 3);
                // cta<8,80,384> warp<48,120,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 8, 80, 384, 48, 120, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 8, 80, 384, 48, 120, 128, 8, 8, 128, 3);
                // cta<8,32,384> warp<24,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(6, 6, true, 8, 32, 384, 24, 96, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 8, 32, 384, 24, 96, 128, 8, 8, 128, 3);
                // cta<1,32,512> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(6, 6, true, 1, 32, 512, 8, 96, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 1, 32, 512, 8, 96, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 1, 32, 512, 8, 96, 128, 8, 8, 128, 4);
                // cta<4,32,512> warp<24,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(6, 6, true, 4, 32, 512, 24, 96, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 4, 32, 512, 24, 96, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 4, 32, 512, 24, 96, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<48,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(6, 6, true, 8, 32, 512, 48, 96, 128, 8, 8, 128, 2);
                // cta<1,32,512> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 1, 32, 512, 8, 48, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 1, 32, 512, 8, 48, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 1, 32, 512, 8, 48, 128, 8, 8, 128, 4);
                // cta<1,48,512> warp<8,72,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 1, 48, 512, 8, 72, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 1, 48, 512, 8, 72, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 1, 48, 512, 8, 72, 128, 8, 8, 128, 4);
                // cta<1,64,512> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 1, 64, 512, 8, 96, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 1, 64, 512, 8, 96, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 1, 64, 512, 8, 96, 128, 8, 8, 128, 4);
                // cta<1,80,512> warp<8,120,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 1, 80, 512, 8, 120, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 1, 80, 512, 8, 120, 128, 8, 8, 128, 3);
                // cta<4,32,512> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 4, 32, 512, 24, 48, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 4, 32, 512, 24, 48, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 4, 32, 512, 24, 48, 128, 8, 8, 128, 4);
                // cta<4,48,512> warp<24,72,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 4, 48, 512, 24, 72, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 4, 48, 512, 24, 72, 128, 8, 8, 128, 3);
                TEST(6, 6, true, 4, 48, 512, 24, 72, 128, 8, 8, 128, 4);
                // cta<4,64,512> warp<24,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 4, 64, 512, 24, 96, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 4, 64, 512, 24, 96, 128, 8, 8, 128, 3);
                // cta<4,80,512> warp<24,120,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 4, 80, 512, 24, 120, 128, 8, 8, 128, 2);
                TEST(6, 6, true, 4, 80, 512, 24, 120, 128, 8, 8, 128, 3);
                // cta<8,32,512> warp<48,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 8, 32, 512, 48, 48, 128, 8, 8, 128, 2);
                // cta<8,48,512> warp<48,72,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 8, 48, 512, 48, 72, 128, 8, 8, 128, 2);
                // cta<8,64,512> warp<48,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 8, 64, 512, 48, 96, 128, 8, 8, 128, 2);
                // cta<8,80,512> warp<48,120,128> mma<8,8,128>   WARPS[1x4]
                TEST(6, 6, true, 8, 80, 512, 48, 120, 128, 8, 8, 128, 2);
                // cta<8,32,512> warp<24,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(6, 6, true, 8, 32, 512, 24, 96, 128, 8, 8, 128, 2);
            } else {
            }
            break;
        #endif
        default:
            printf("unsupport w%da%d\n", w_bits, x_bits);
        }
        break;
    case 7:
        switch (w_bits) {
        #ifdef W7A7
        case 7:
            if (quant_sign) {
                ////// W7A7 int
                // cta<1,32,256> warp<8,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(7, 7, true, 1, 32, 256, 8, 112, 128, 8, 8, 128, 2);
                TEST(7, 7, true, 1, 32, 256, 8, 112, 128, 8, 8, 128, 3);
                TEST(7, 7, true, 1, 32, 256, 8, 112, 128, 8, 8, 128, 4);
                // cta<8,32,256> warp<56,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(7, 7, true, 8, 32, 256, 56, 112, 128, 8, 8, 128, 2);
                TEST(7, 7, true, 8, 32, 256, 56, 112, 128, 8, 8, 128, 3);
                TEST(7, 7, true, 8, 32, 256, 56, 112, 128, 8, 8, 128, 4);
                TEST(7, 7, true, 8, 32, 256, 56, 112, 128, 8, 8, 128, 5);
                // cta<1,32,256> warp<8,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(7, 7, true, 1, 32, 256, 8, 56, 128, 8, 8, 128, 2);
                TEST(7, 7, true, 1, 32, 256, 8, 56, 128, 8, 8, 128, 3);
                TEST(7, 7, true, 1, 32, 256, 8, 56, 128, 8, 8, 128, 4);
                // cta<1,64,256> warp<8,112,128> mma<8,8,128>   WARPS[1x4]
                TEST(7, 7, true, 1, 64, 256, 8, 112, 128, 8, 8, 128, 2);
                TEST(7, 7, true, 1, 64, 256, 8, 112, 128, 8, 8, 128, 3);
                TEST(7, 7, true, 1, 64, 256, 8, 112, 128, 8, 8, 128, 4);
                // cta<8,32,256> warp<56,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(7, 7, true, 8, 32, 256, 56, 56, 128, 8, 8, 128, 2);
                TEST(7, 7, true, 8, 32, 256, 56, 56, 128, 8, 8, 128, 3);
                TEST(7, 7, true, 8, 32, 256, 56, 56, 128, 8, 8, 128, 4);
                TEST(7, 7, true, 8, 32, 256, 56, 56, 128, 8, 8, 128, 5);
                // cta<1,32,384> warp<8,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(7, 7, true, 1, 32, 384, 8, 112, 128, 8, 8, 128, 2);
                TEST(7, 7, true, 1, 32, 384, 8, 112, 128, 8, 8, 128, 3);
                TEST(7, 7, true, 1, 32, 384, 8, 112, 128, 8, 8, 128, 4);
                // cta<8,32,384> warp<56,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(7, 7, true, 8, 32, 384, 56, 112, 128, 8, 8, 128, 2);
                TEST(7, 7, true, 8, 32, 384, 56, 112, 128, 8, 8, 128, 3);
                // cta<1,32,384> warp<8,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(7, 7, true, 1, 32, 384, 8, 56, 128, 8, 8, 128, 2);
                TEST(7, 7, true, 1, 32, 384, 8, 56, 128, 8, 8, 128, 3);
                TEST(7, 7, true, 1, 32, 384, 8, 56, 128, 8, 8, 128, 4);
                // cta<1,64,384> warp<8,112,128> mma<8,8,128>   WARPS[1x4]
                TEST(7, 7, true, 1, 64, 384, 8, 112, 128, 8, 8, 128, 2);
                TEST(7, 7, true, 1, 64, 384, 8, 112, 128, 8, 8, 128, 3);
                TEST(7, 7, true, 1, 64, 384, 8, 112, 128, 8, 8, 128, 4);
                // cta<8,32,384> warp<56,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(7, 7, true, 8, 32, 384, 56, 56, 128, 8, 8, 128, 2);
                TEST(7, 7, true, 8, 32, 384, 56, 56, 128, 8, 8, 128, 3);
                // cta<1,32,512> warp<8,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(7, 7, true, 1, 32, 512, 8, 112, 128, 8, 8, 128, 2);
                TEST(7, 7, true, 1, 32, 512, 8, 112, 128, 8, 8, 128, 3);
                TEST(7, 7, true, 1, 32, 512, 8, 112, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<56,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(7, 7, true, 8, 32, 512, 56, 112, 128, 8, 8, 128, 2);
                // cta<1,32,512> warp<8,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(7, 7, true, 1, 32, 512, 8, 56, 128, 8, 8, 128, 2);
                TEST(7, 7, true, 1, 32, 512, 8, 56, 128, 8, 8, 128, 3);
                TEST(7, 7, true, 1, 32, 512, 8, 56, 128, 8, 8, 128, 4);
                // cta<1,64,512> warp<8,112,128> mma<8,8,128>   WARPS[1x4]
                TEST(7, 7, true, 1, 64, 512, 8, 112, 128, 8, 8, 128, 2);
                TEST(7, 7, true, 1, 64, 512, 8, 112, 128, 8, 8, 128, 3);
                // cta<8,32,512> warp<56,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(7, 7, true, 8, 32, 512, 56, 56, 128, 8, 8, 128, 2);
            } else {
            }
            break;
        #endif
        default:
            printf("unsupport w%da%d\n", w_bits, x_bits);
        }
        break;
    case 8:
        switch (w_bits) {
        #ifdef W2A8
        case 2:
            if (quant_sign) {
                ////// W2A8 int
                // cta<1,32,256> warp<8,32,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 1, 32, 256, 8, 32, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 32, 256, 8, 32, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 32, 256, 8, 32, 128, 8, 8, 128, 4);
                // cta<1,48,256> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 1, 48, 256, 8, 48, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 48, 256, 8, 48, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 48, 256, 8, 48, 128, 8, 8, 128, 4);
                // cta<1,64,256> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 1, 64, 256, 8, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 64, 256, 8, 64, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 64, 256, 8, 64, 128, 8, 8, 128, 4);
                // cta<1,80,256> warp<8,80,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 1, 80, 256, 8, 80, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 80, 256, 8, 80, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 80, 256, 8, 80, 128, 8, 8, 128, 4);
                // cta<1,96,256> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 1, 96, 256, 8, 96, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 96, 256, 8, 96, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 96, 256, 8, 96, 128, 8, 8, 128, 4);
                // cta<1,112,256> warp<8,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 1, 112, 256, 8, 112, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 112, 256, 8, 112, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 112, 256, 8, 112, 128, 8, 8, 128, 4);
                // cta<1,128,256> warp<8,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 1, 128, 256, 8, 128, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 128, 256, 8, 128, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 128, 256, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,256> warp<32,32,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 4, 32, 256, 32, 32, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 32, 256, 32, 32, 128, 8, 8, 128, 3);
                // cta<4,48,256> warp<32,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 4, 48, 256, 32, 48, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 48, 256, 32, 48, 128, 8, 8, 128, 3);
                // cta<4,64,256> warp<32,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 4, 64, 256, 32, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 64, 256, 32, 64, 128, 8, 8, 128, 3);
                // cta<4,80,256> warp<32,80,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 4, 80, 256, 32, 80, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 80, 256, 32, 80, 128, 8, 8, 128, 3);
                // cta<4,96,256> warp<32,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 4, 96, 256, 32, 96, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 96, 256, 32, 96, 128, 8, 8, 128, 3);
                // cta<4,112,256> warp<32,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 4, 112, 256, 32, 112, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 112, 256, 32, 112, 128, 8, 8, 128, 3);
                // cta<4,128,256> warp<32,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 4, 128, 256, 32, 128, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 128, 256, 32, 128, 128, 8, 8, 128, 3);
                // cta<8,32,256> warp<64,32,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 8, 32, 256, 64, 32, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 32, 256, 64, 32, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 32, 256, 64, 32, 128, 8, 8, 128, 4);
                // cta<8,48,256> warp<64,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 8, 48, 256, 64, 48, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 48, 256, 64, 48, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 48, 256, 64, 48, 128, 8, 8, 128, 4);
                TEST(8, 2, true, 8, 48, 256, 64, 48, 128, 8, 8, 128, 5);
                // cta<8,64,256> warp<64,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 8, 64, 256, 64, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 64, 256, 64, 64, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 64, 256, 64, 64, 128, 8, 8, 128, 4);
                TEST(8, 2, true, 8, 64, 256, 64, 64, 128, 8, 8, 128, 5);
                // cta<8,80,256> warp<64,80,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 8, 80, 256, 64, 80, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 80, 256, 64, 80, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 80, 256, 64, 80, 128, 8, 8, 128, 4);
                TEST(8, 2, true, 8, 80, 256, 64, 80, 128, 8, 8, 128, 5);
                TEST(8, 2, true, 8, 80, 256, 64, 80, 128, 8, 8, 128, 6);
                // cta<8,96,256> warp<64,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 8, 96, 256, 64, 96, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 96, 256, 64, 96, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 96, 256, 64, 96, 128, 8, 8, 128, 4);
                TEST(8, 2, true, 8, 96, 256, 64, 96, 128, 8, 8, 128, 5);
                TEST(8, 2, true, 8, 96, 256, 64, 96, 128, 8, 8, 128, 6);
                // cta<8,112,256> warp<64,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 8, 112, 256, 64, 112, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 112, 256, 64, 112, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 112, 256, 64, 112, 128, 8, 8, 128, 4);
                TEST(8, 2, true, 8, 112, 256, 64, 112, 128, 8, 8, 128, 5);
                TEST(8, 2, true, 8, 112, 256, 64, 112, 128, 8, 8, 128, 6);
                // cta<1,32,256> warp<8,16,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 1, 32, 256, 8, 16, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 32, 256, 8, 16, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 32, 256, 8, 16, 128, 8, 8, 128, 4);
                // cta<1,48,256> warp<8,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 1, 48, 256, 8, 24, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 48, 256, 8, 24, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 48, 256, 8, 24, 128, 8, 8, 128, 4);
                // cta<1,64,256> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 1, 64, 256, 8, 32, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 64, 256, 8, 32, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 64, 256, 8, 32, 128, 8, 8, 128, 4);
                // cta<1,80,256> warp<8,40,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 1, 80, 256, 8, 40, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 80, 256, 8, 40, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 80, 256, 8, 40, 128, 8, 8, 128, 4);
                // cta<1,96,256> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 1, 96, 256, 8, 48, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 96, 256, 8, 48, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 96, 256, 8, 48, 128, 8, 8, 128, 4);
                // cta<1,112,256> warp<8,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 1, 112, 256, 8, 56, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 112, 256, 8, 56, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 112, 256, 8, 56, 128, 8, 8, 128, 4);
                // cta<1,128,256> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 1, 128, 256, 8, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 128, 256, 8, 64, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 128, 256, 8, 64, 128, 8, 8, 128, 4);
                // cta<1,256,256> warp<8,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 1, 256, 256, 8, 128, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 256, 256, 8, 128, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 256, 256, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,256> warp<32,16,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 4, 32, 256, 32, 16, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 32, 256, 32, 16, 128, 8, 8, 128, 3);
                // cta<4,48,256> warp<32,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 4, 48, 256, 32, 24, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 48, 256, 32, 24, 128, 8, 8, 128, 3);
                // cta<4,64,256> warp<32,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 4, 64, 256, 32, 32, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 64, 256, 32, 32, 128, 8, 8, 128, 3);
                // cta<4,80,256> warp<32,40,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 4, 80, 256, 32, 40, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 80, 256, 32, 40, 128, 8, 8, 128, 3);
                // cta<4,96,256> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 4, 96, 256, 32, 48, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 96, 256, 32, 48, 128, 8, 8, 128, 3);
                // cta<4,112,256> warp<32,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 4, 112, 256, 32, 56, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 112, 256, 32, 56, 128, 8, 8, 128, 3);
                // cta<4,128,256> warp<32,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 4, 128, 256, 32, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 128, 256, 32, 64, 128, 8, 8, 128, 3);
                // cta<4,256,256> warp<32,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 4, 256, 256, 32, 128, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 256, 256, 32, 128, 128, 8, 8, 128, 3);
                // cta<8,32,256> warp<64,16,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 8, 32, 256, 64, 16, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 32, 256, 64, 16, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 32, 256, 64, 16, 128, 8, 8, 128, 4);
                // cta<8,48,256> warp<64,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 8, 48, 256, 64, 24, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 48, 256, 64, 24, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 48, 256, 64, 24, 128, 8, 8, 128, 4);
                TEST(8, 2, true, 8, 48, 256, 64, 24, 128, 8, 8, 128, 5);
                // cta<8,64,256> warp<64,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 8, 64, 256, 64, 32, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 64, 256, 64, 32, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 64, 256, 64, 32, 128, 8, 8, 128, 4);
                TEST(8, 2, true, 8, 64, 256, 64, 32, 128, 8, 8, 128, 5);
                // cta<8,80,256> warp<64,40,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 8, 80, 256, 64, 40, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 80, 256, 64, 40, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 80, 256, 64, 40, 128, 8, 8, 128, 4);
                TEST(8, 2, true, 8, 80, 256, 64, 40, 128, 8, 8, 128, 5);
                TEST(8, 2, true, 8, 80, 256, 64, 40, 128, 8, 8, 128, 6);
                // cta<8,96,256> warp<64,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 8, 96, 256, 64, 48, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 96, 256, 64, 48, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 96, 256, 64, 48, 128, 8, 8, 128, 4);
                TEST(8, 2, true, 8, 96, 256, 64, 48, 128, 8, 8, 128, 5);
                TEST(8, 2, true, 8, 96, 256, 64, 48, 128, 8, 8, 128, 6);
                // cta<8,112,256> warp<64,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 8, 112, 256, 64, 56, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 112, 256, 64, 56, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 112, 256, 64, 56, 128, 8, 8, 128, 4);
                TEST(8, 2, true, 8, 112, 256, 64, 56, 128, 8, 8, 128, 5);
                TEST(8, 2, true, 8, 112, 256, 64, 56, 128, 8, 8, 128, 6);
                // cta<8,128,256> warp<64,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 8, 128, 256, 64, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 128, 256, 64, 64, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 128, 256, 64, 64, 128, 8, 8, 128, 4);
                TEST(8, 2, true, 8, 128, 256, 64, 64, 128, 8, 8, 128, 5);
                TEST(8, 2, true, 8, 128, 256, 64, 64, 128, 8, 8, 128, 6);
                // cta<4,32,256> warp<16,64,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 2, true, 4, 32, 256, 16, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 32, 256, 16, 64, 128, 8, 8, 128, 3);
                // cta<4,48,256> warp<16,96,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 2, true, 4, 48, 256, 16, 96, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 48, 256, 16, 96, 128, 8, 8, 128, 3);
                // cta<4,64,256> warp<16,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 2, true, 4, 64, 256, 16, 128, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 64, 256, 16, 128, 128, 8, 8, 128, 3);
                // cta<8,32,256> warp<32,64,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 2, true, 8, 32, 256, 32, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 32, 256, 32, 64, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 32, 256, 32, 64, 128, 8, 8, 128, 4);
                // cta<8,48,256> warp<32,96,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 2, true, 8, 48, 256, 32, 96, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 48, 256, 32, 96, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 48, 256, 32, 96, 128, 8, 8, 128, 4);
                TEST(8, 2, true, 8, 48, 256, 32, 96, 128, 8, 8, 128, 5);
                // cta<8,64,256> warp<32,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 2, true, 8, 64, 256, 32, 128, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 64, 256, 32, 128, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 64, 256, 32, 128, 128, 8, 8, 128, 4);
                TEST(8, 2, true, 8, 64, 256, 32, 128, 128, 8, 8, 128, 5);
                // cta<4,32,256> warp<16,32,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 4, 32, 256, 16, 32, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 32, 256, 16, 32, 128, 8, 8, 128, 3);
                // cta<4,48,256> warp<16,48,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 4, 48, 256, 16, 48, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 48, 256, 16, 48, 128, 8, 8, 128, 3);
                // cta<4,64,256> warp<16,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 4, 64, 256, 16, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 64, 256, 16, 64, 128, 8, 8, 128, 3);
                // cta<4,80,256> warp<16,80,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 4, 80, 256, 16, 80, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 80, 256, 16, 80, 128, 8, 8, 128, 3);
                // cta<4,96,256> warp<16,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 4, 96, 256, 16, 96, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 96, 256, 16, 96, 128, 8, 8, 128, 3);
                // cta<4,112,256> warp<16,112,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 4, 112, 256, 16, 112, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 112, 256, 16, 112, 128, 8, 8, 128, 3);
                // cta<4,128,256> warp<16,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 4, 128, 256, 16, 128, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 128, 256, 16, 128, 128, 8, 8, 128, 3);
                // cta<8,32,256> warp<32,32,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 8, 32, 256, 32, 32, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 32, 256, 32, 32, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 32, 256, 32, 32, 128, 8, 8, 128, 4);
                // cta<8,48,256> warp<32,48,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 8, 48, 256, 32, 48, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 48, 256, 32, 48, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 48, 256, 32, 48, 128, 8, 8, 128, 4);
                TEST(8, 2, true, 8, 48, 256, 32, 48, 128, 8, 8, 128, 5);
                // cta<8,64,256> warp<32,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 8, 64, 256, 32, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 64, 256, 32, 64, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 64, 256, 32, 64, 128, 8, 8, 128, 4);
                TEST(8, 2, true, 8, 64, 256, 32, 64, 128, 8, 8, 128, 5);
                // cta<8,80,256> warp<32,80,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 8, 80, 256, 32, 80, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 80, 256, 32, 80, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 80, 256, 32, 80, 128, 8, 8, 128, 4);
                TEST(8, 2, true, 8, 80, 256, 32, 80, 128, 8, 8, 128, 5);
                TEST(8, 2, true, 8, 80, 256, 32, 80, 128, 8, 8, 128, 6);
                // cta<8,96,256> warp<32,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 8, 96, 256, 32, 96, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 96, 256, 32, 96, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 96, 256, 32, 96, 128, 8, 8, 128, 4);
                TEST(8, 2, true, 8, 96, 256, 32, 96, 128, 8, 8, 128, 5);
                TEST(8, 2, true, 8, 96, 256, 32, 96, 128, 8, 8, 128, 6);
                // cta<8,112,256> warp<32,112,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 8, 112, 256, 32, 112, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 112, 256, 32, 112, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 112, 256, 32, 112, 128, 8, 8, 128, 4);
                TEST(8, 2, true, 8, 112, 256, 32, 112, 128, 8, 8, 128, 5);
                TEST(8, 2, true, 8, 112, 256, 32, 112, 128, 8, 8, 128, 6);
                // cta<8,128,256> warp<32,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 8, 128, 256, 32, 128, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 128, 256, 32, 128, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 128, 256, 32, 128, 128, 8, 8, 128, 4);
                TEST(8, 2, true, 8, 128, 256, 32, 128, 128, 8, 8, 128, 5);
                TEST(8, 2, true, 8, 128, 256, 32, 128, 128, 8, 8, 128, 6);
                // cta<4,32,256> warp<8,64,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 2, true, 4, 32, 256, 8, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 32, 256, 8, 64, 128, 8, 8, 128, 3);
                // cta<4,48,256> warp<8,96,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 2, true, 4, 48, 256, 8, 96, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 48, 256, 8, 96, 128, 8, 8, 128, 3);
                // cta<4,64,256> warp<8,128,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 2, true, 4, 64, 256, 8, 128, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 64, 256, 8, 128, 128, 8, 8, 128, 3);
                // cta<8,32,256> warp<16,64,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 2, true, 8, 32, 256, 16, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 32, 256, 16, 64, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 32, 256, 16, 64, 128, 8, 8, 128, 4);
                // cta<8,48,256> warp<16,96,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 2, true, 8, 48, 256, 16, 96, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 48, 256, 16, 96, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 48, 256, 16, 96, 128, 8, 8, 128, 4);
                TEST(8, 2, true, 8, 48, 256, 16, 96, 128, 8, 8, 128, 5);
                // cta<8,64,256> warp<16,128,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 2, true, 8, 64, 256, 16, 128, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 64, 256, 16, 128, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 64, 256, 16, 128, 128, 8, 8, 128, 4);
                TEST(8, 2, true, 8, 64, 256, 16, 128, 128, 8, 8, 128, 5);
                // cta<1,32,384> warp<8,32,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 1, 32, 384, 8, 32, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 32, 384, 8, 32, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 32, 384, 8, 32, 128, 8, 8, 128, 4);
                // cta<1,48,384> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 1, 48, 384, 8, 48, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 48, 384, 8, 48, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 48, 384, 8, 48, 128, 8, 8, 128, 4);
                // cta<1,64,384> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 1, 64, 384, 8, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 64, 384, 8, 64, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 64, 384, 8, 64, 128, 8, 8, 128, 4);
                // cta<1,80,384> warp<8,80,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 1, 80, 384, 8, 80, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 80, 384, 8, 80, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 80, 384, 8, 80, 128, 8, 8, 128, 4);
                // cta<1,96,384> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 1, 96, 384, 8, 96, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 96, 384, 8, 96, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 96, 384, 8, 96, 128, 8, 8, 128, 4);
                // cta<1,112,384> warp<8,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 1, 112, 384, 8, 112, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 112, 384, 8, 112, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 112, 384, 8, 112, 128, 8, 8, 128, 4);
                // cta<1,128,384> warp<8,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 1, 128, 384, 8, 128, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 128, 384, 8, 128, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 128, 384, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,384> warp<32,32,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 4, 32, 384, 32, 32, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 32, 384, 32, 32, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 32, 384, 32, 32, 128, 8, 8, 128, 4);
                // cta<4,48,384> warp<32,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 4, 48, 384, 32, 48, 128, 8, 8, 128, 2);
                // cta<4,64,384> warp<32,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 4, 64, 384, 32, 64, 128, 8, 8, 128, 2);
                // cta<4,80,384> warp<32,80,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 4, 80, 384, 32, 80, 128, 8, 8, 128, 2);
                // cta<4,96,384> warp<32,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 4, 96, 384, 32, 96, 128, 8, 8, 128, 2);
                // cta<4,112,384> warp<32,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 4, 112, 384, 32, 112, 128, 8, 8, 128, 2);
                // cta<4,128,384> warp<32,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 4, 128, 384, 32, 128, 128, 8, 8, 128, 2);
                // cta<8,32,384> warp<64,32,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 8, 32, 384, 64, 32, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 32, 384, 64, 32, 128, 8, 8, 128, 3);
                // cta<8,48,384> warp<64,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 8, 48, 384, 64, 48, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 48, 384, 64, 48, 128, 8, 8, 128, 3);
                // cta<8,64,384> warp<64,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 8, 64, 384, 64, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 64, 384, 64, 64, 128, 8, 8, 128, 3);
                // cta<8,80,384> warp<64,80,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 8, 80, 384, 64, 80, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 80, 384, 64, 80, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 80, 384, 64, 80, 128, 8, 8, 128, 4);
                // cta<8,96,384> warp<64,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 8, 96, 384, 64, 96, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 96, 384, 64, 96, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 96, 384, 64, 96, 128, 8, 8, 128, 4);
                // cta<8,112,384> warp<64,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 8, 112, 384, 64, 112, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 112, 384, 64, 112, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 112, 384, 64, 112, 128, 8, 8, 128, 4);
                // cta<1,32,384> warp<8,16,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 1, 32, 384, 8, 16, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 32, 384, 8, 16, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 32, 384, 8, 16, 128, 8, 8, 128, 4);
                // cta<1,48,384> warp<8,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 1, 48, 384, 8, 24, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 48, 384, 8, 24, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 48, 384, 8, 24, 128, 8, 8, 128, 4);
                // cta<1,64,384> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 1, 64, 384, 8, 32, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 64, 384, 8, 32, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 64, 384, 8, 32, 128, 8, 8, 128, 4);
                // cta<1,80,384> warp<8,40,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 1, 80, 384, 8, 40, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 80, 384, 8, 40, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 80, 384, 8, 40, 128, 8, 8, 128, 4);
                // cta<1,96,384> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 1, 96, 384, 8, 48, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 96, 384, 8, 48, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 96, 384, 8, 48, 128, 8, 8, 128, 4);
                // cta<1,112,384> warp<8,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 1, 112, 384, 8, 56, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 112, 384, 8, 56, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 112, 384, 8, 56, 128, 8, 8, 128, 4);
                // cta<1,128,384> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 1, 128, 384, 8, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 128, 384, 8, 64, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 128, 384, 8, 64, 128, 8, 8, 128, 4);
                // cta<1,256,384> warp<8,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 1, 256, 384, 8, 128, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 256, 384, 8, 128, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 256, 384, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,384> warp<32,16,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 4, 32, 384, 32, 16, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 32, 384, 32, 16, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 32, 384, 32, 16, 128, 8, 8, 128, 4);
                // cta<4,48,384> warp<32,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 4, 48, 384, 32, 24, 128, 8, 8, 128, 2);
                // cta<4,64,384> warp<32,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 4, 64, 384, 32, 32, 128, 8, 8, 128, 2);
                // cta<4,80,384> warp<32,40,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 4, 80, 384, 32, 40, 128, 8, 8, 128, 2);
                // cta<4,96,384> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 4, 96, 384, 32, 48, 128, 8, 8, 128, 2);
                // cta<4,112,384> warp<32,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 4, 112, 384, 32, 56, 128, 8, 8, 128, 2);
                // cta<4,128,384> warp<32,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 4, 128, 384, 32, 64, 128, 8, 8, 128, 2);
                // cta<4,256,384> warp<32,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 4, 256, 384, 32, 128, 128, 8, 8, 128, 2);
                // cta<8,32,384> warp<64,16,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 8, 32, 384, 64, 16, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 32, 384, 64, 16, 128, 8, 8, 128, 3);
                // cta<8,48,384> warp<64,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 8, 48, 384, 64, 24, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 48, 384, 64, 24, 128, 8, 8, 128, 3);
                // cta<8,64,384> warp<64,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 8, 64, 384, 64, 32, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 64, 384, 64, 32, 128, 8, 8, 128, 3);
                // cta<8,80,384> warp<64,40,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 8, 80, 384, 64, 40, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 80, 384, 64, 40, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 80, 384, 64, 40, 128, 8, 8, 128, 4);
                // cta<8,96,384> warp<64,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 8, 96, 384, 64, 48, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 96, 384, 64, 48, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 96, 384, 64, 48, 128, 8, 8, 128, 4);
                // cta<8,112,384> warp<64,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 8, 112, 384, 64, 56, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 112, 384, 64, 56, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 112, 384, 64, 56, 128, 8, 8, 128, 4);
                // cta<8,128,384> warp<64,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 8, 128, 384, 64, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 128, 384, 64, 64, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 128, 384, 64, 64, 128, 8, 8, 128, 4);
                // cta<4,32,384> warp<16,64,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 2, true, 4, 32, 384, 16, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 32, 384, 16, 64, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 32, 384, 16, 64, 128, 8, 8, 128, 4);
                // cta<4,48,384> warp<16,96,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 2, true, 4, 48, 384, 16, 96, 128, 8, 8, 128, 2);
                // cta<4,64,384> warp<16,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 2, true, 4, 64, 384, 16, 128, 128, 8, 8, 128, 2);
                // cta<8,32,384> warp<32,64,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 2, true, 8, 32, 384, 32, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 32, 384, 32, 64, 128, 8, 8, 128, 3);
                // cta<8,48,384> warp<32,96,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 2, true, 8, 48, 384, 32, 96, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 48, 384, 32, 96, 128, 8, 8, 128, 3);
                // cta<8,64,384> warp<32,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 2, true, 8, 64, 384, 32, 128, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 64, 384, 32, 128, 128, 8, 8, 128, 3);
                // cta<4,32,384> warp<16,32,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 4, 32, 384, 16, 32, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 32, 384, 16, 32, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 32, 384, 16, 32, 128, 8, 8, 128, 4);
                // cta<4,48,384> warp<16,48,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 4, 48, 384, 16, 48, 128, 8, 8, 128, 2);
                // cta<4,64,384> warp<16,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 4, 64, 384, 16, 64, 128, 8, 8, 128, 2);
                // cta<4,80,384> warp<16,80,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 4, 80, 384, 16, 80, 128, 8, 8, 128, 2);
                // cta<4,96,384> warp<16,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 4, 96, 384, 16, 96, 128, 8, 8, 128, 2);
                // cta<4,112,384> warp<16,112,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 4, 112, 384, 16, 112, 128, 8, 8, 128, 2);
                // cta<4,128,384> warp<16,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 4, 128, 384, 16, 128, 128, 8, 8, 128, 2);
                // cta<8,32,384> warp<32,32,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 8, 32, 384, 32, 32, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 32, 384, 32, 32, 128, 8, 8, 128, 3);
                // cta<8,48,384> warp<32,48,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 8, 48, 384, 32, 48, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 48, 384, 32, 48, 128, 8, 8, 128, 3);
                // cta<8,64,384> warp<32,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 8, 64, 384, 32, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 64, 384, 32, 64, 128, 8, 8, 128, 3);
                // cta<8,80,384> warp<32,80,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 8, 80, 384, 32, 80, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 80, 384, 32, 80, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 80, 384, 32, 80, 128, 8, 8, 128, 4);
                // cta<8,96,384> warp<32,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 8, 96, 384, 32, 96, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 96, 384, 32, 96, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 96, 384, 32, 96, 128, 8, 8, 128, 4);
                // cta<8,112,384> warp<32,112,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 8, 112, 384, 32, 112, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 112, 384, 32, 112, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 112, 384, 32, 112, 128, 8, 8, 128, 4);
                // cta<8,128,384> warp<32,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 8, 128, 384, 32, 128, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 128, 384, 32, 128, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 8, 128, 384, 32, 128, 128, 8, 8, 128, 4);
                // cta<4,32,384> warp<8,64,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 2, true, 4, 32, 384, 8, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 32, 384, 8, 64, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 32, 384, 8, 64, 128, 8, 8, 128, 4);
                // cta<4,48,384> warp<8,96,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 2, true, 4, 48, 384, 8, 96, 128, 8, 8, 128, 2);
                // cta<4,64,384> warp<8,128,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 2, true, 4, 64, 384, 8, 128, 128, 8, 8, 128, 2);
                // cta<8,32,384> warp<16,64,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 2, true, 8, 32, 384, 16, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 32, 384, 16, 64, 128, 8, 8, 128, 3);
                // cta<8,48,384> warp<16,96,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 2, true, 8, 48, 384, 16, 96, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 48, 384, 16, 96, 128, 8, 8, 128, 3);
                // cta<8,64,384> warp<16,128,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 2, true, 8, 64, 384, 16, 128, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 64, 384, 16, 128, 128, 8, 8, 128, 3);
                // cta<1,32,512> warp<8,32,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 1, 32, 512, 8, 32, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 32, 512, 8, 32, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 32, 512, 8, 32, 128, 8, 8, 128, 4);
                // cta<1,48,512> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 1, 48, 512, 8, 48, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 48, 512, 8, 48, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 48, 512, 8, 48, 128, 8, 8, 128, 4);
                // cta<1,64,512> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 1, 64, 512, 8, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 64, 512, 8, 64, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 64, 512, 8, 64, 128, 8, 8, 128, 4);
                // cta<1,80,512> warp<8,80,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 1, 80, 512, 8, 80, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 80, 512, 8, 80, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 80, 512, 8, 80, 128, 8, 8, 128, 4);
                // cta<1,96,512> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 1, 96, 512, 8, 96, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 96, 512, 8, 96, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 96, 512, 8, 96, 128, 8, 8, 128, 4);
                // cta<1,112,512> warp<8,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 1, 112, 512, 8, 112, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 112, 512, 8, 112, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 112, 512, 8, 112, 128, 8, 8, 128, 4);
                // cta<1,128,512> warp<8,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 1, 128, 512, 8, 128, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 128, 512, 8, 128, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 128, 512, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,512> warp<32,32,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 4, 32, 512, 32, 32, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 32, 512, 32, 32, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 32, 512, 32, 32, 128, 8, 8, 128, 4);
                // cta<4,48,512> warp<32,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 4, 48, 512, 32, 48, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 48, 512, 32, 48, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 48, 512, 32, 48, 128, 8, 8, 128, 4);
                // cta<4,64,512> warp<32,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 4, 64, 512, 32, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 64, 512, 32, 64, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 64, 512, 32, 64, 128, 8, 8, 128, 4);
                // cta<4,80,512> warp<32,80,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 4, 80, 512, 32, 80, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 80, 512, 32, 80, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 80, 512, 32, 80, 128, 8, 8, 128, 4);
                // cta<4,96,512> warp<32,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 4, 96, 512, 32, 96, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 96, 512, 32, 96, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 96, 512, 32, 96, 128, 8, 8, 128, 4);
                // cta<4,112,512> warp<32,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 4, 112, 512, 32, 112, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 112, 512, 32, 112, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 112, 512, 32, 112, 128, 8, 8, 128, 4);
                // cta<4,128,512> warp<32,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 4, 128, 512, 32, 128, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 128, 512, 32, 128, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 128, 512, 32, 128, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<64,32,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 8, 32, 512, 64, 32, 128, 8, 8, 128, 2);
                // cta<8,48,512> warp<64,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 8, 48, 512, 64, 48, 128, 8, 8, 128, 2);
                // cta<8,64,512> warp<64,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 8, 64, 512, 64, 64, 128, 8, 8, 128, 2);
                // cta<8,80,512> warp<64,80,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 8, 80, 512, 64, 80, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 80, 512, 64, 80, 128, 8, 8, 128, 3);
                // cta<8,96,512> warp<64,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 8, 96, 512, 64, 96, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 96, 512, 64, 96, 128, 8, 8, 128, 3);
                // cta<8,112,512> warp<64,112,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 2, true, 8, 112, 512, 64, 112, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 112, 512, 64, 112, 128, 8, 8, 128, 3);
                // cta<1,32,512> warp<8,16,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 1, 32, 512, 8, 16, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 32, 512, 8, 16, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 32, 512, 8, 16, 128, 8, 8, 128, 4);
                // cta<1,48,512> warp<8,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 1, 48, 512, 8, 24, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 48, 512, 8, 24, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 48, 512, 8, 24, 128, 8, 8, 128, 4);
                // cta<1,64,512> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 1, 64, 512, 8, 32, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 64, 512, 8, 32, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 64, 512, 8, 32, 128, 8, 8, 128, 4);
                // cta<1,80,512> warp<8,40,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 1, 80, 512, 8, 40, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 80, 512, 8, 40, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 80, 512, 8, 40, 128, 8, 8, 128, 4);
                // cta<1,96,512> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 1, 96, 512, 8, 48, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 96, 512, 8, 48, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 96, 512, 8, 48, 128, 8, 8, 128, 4);
                // cta<1,112,512> warp<8,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 1, 112, 512, 8, 56, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 112, 512, 8, 56, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 112, 512, 8, 56, 128, 8, 8, 128, 4);
                // cta<1,128,512> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 1, 128, 512, 8, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 128, 512, 8, 64, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 1, 128, 512, 8, 64, 128, 8, 8, 128, 4);
                // cta<1,256,512> warp<8,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 1, 256, 512, 8, 128, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 1, 256, 512, 8, 128, 128, 8, 8, 128, 3);
                // cta<4,32,512> warp<32,16,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 4, 32, 512, 32, 16, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 32, 512, 32, 16, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 32, 512, 32, 16, 128, 8, 8, 128, 4);
                // cta<4,48,512> warp<32,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 4, 48, 512, 32, 24, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 48, 512, 32, 24, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 48, 512, 32, 24, 128, 8, 8, 128, 4);
                // cta<4,64,512> warp<32,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 4, 64, 512, 32, 32, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 64, 512, 32, 32, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 64, 512, 32, 32, 128, 8, 8, 128, 4);
                // cta<4,80,512> warp<32,40,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 4, 80, 512, 32, 40, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 80, 512, 32, 40, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 80, 512, 32, 40, 128, 8, 8, 128, 4);
                // cta<4,96,512> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 4, 96, 512, 32, 48, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 96, 512, 32, 48, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 96, 512, 32, 48, 128, 8, 8, 128, 4);
                // cta<4,112,512> warp<32,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 4, 112, 512, 32, 56, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 112, 512, 32, 56, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 112, 512, 32, 56, 128, 8, 8, 128, 4);
                // cta<4,128,512> warp<32,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 4, 128, 512, 32, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 128, 512, 32, 64, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 128, 512, 32, 64, 128, 8, 8, 128, 4);
                // cta<4,256,512> warp<32,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 4, 256, 512, 32, 128, 128, 8, 8, 128, 2);
                // cta<8,32,512> warp<64,16,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 8, 32, 512, 64, 16, 128, 8, 8, 128, 2);
                // cta<8,48,512> warp<64,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 8, 48, 512, 64, 24, 128, 8, 8, 128, 2);
                // cta<8,64,512> warp<64,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 8, 64, 512, 64, 32, 128, 8, 8, 128, 2);
                // cta<8,80,512> warp<64,40,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 8, 80, 512, 64, 40, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 80, 512, 64, 40, 128, 8, 8, 128, 3);
                // cta<8,96,512> warp<64,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 8, 96, 512, 64, 48, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 96, 512, 64, 48, 128, 8, 8, 128, 3);
                // cta<8,112,512> warp<64,56,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 8, 112, 512, 64, 56, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 112, 512, 64, 56, 128, 8, 8, 128, 3);
                // cta<8,128,512> warp<64,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 2, true, 8, 128, 512, 64, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 128, 512, 64, 64, 128, 8, 8, 128, 3);
                // cta<4,32,512> warp<16,64,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 2, true, 4, 32, 512, 16, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 32, 512, 16, 64, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 32, 512, 16, 64, 128, 8, 8, 128, 4);
                // cta<4,48,512> warp<16,96,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 2, true, 4, 48, 512, 16, 96, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 48, 512, 16, 96, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 48, 512, 16, 96, 128, 8, 8, 128, 4);
                // cta<4,64,512> warp<16,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 2, true, 4, 64, 512, 16, 128, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 64, 512, 16, 128, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 64, 512, 16, 128, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<32,64,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 2, true, 8, 32, 512, 32, 64, 128, 8, 8, 128, 2);
                // cta<8,48,512> warp<32,96,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 2, true, 8, 48, 512, 32, 96, 128, 8, 8, 128, 2);
                // cta<8,64,512> warp<32,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 2, true, 8, 64, 512, 32, 128, 128, 8, 8, 128, 2);
                // cta<4,32,512> warp<16,32,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 4, 32, 512, 16, 32, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 32, 512, 16, 32, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 32, 512, 16, 32, 128, 8, 8, 128, 4);
                // cta<4,48,512> warp<16,48,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 4, 48, 512, 16, 48, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 48, 512, 16, 48, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 48, 512, 16, 48, 128, 8, 8, 128, 4);
                // cta<4,64,512> warp<16,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 4, 64, 512, 16, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 64, 512, 16, 64, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 64, 512, 16, 64, 128, 8, 8, 128, 4);
                // cta<4,80,512> warp<16,80,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 4, 80, 512, 16, 80, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 80, 512, 16, 80, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 80, 512, 16, 80, 128, 8, 8, 128, 4);
                // cta<4,96,512> warp<16,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 4, 96, 512, 16, 96, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 96, 512, 16, 96, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 96, 512, 16, 96, 128, 8, 8, 128, 4);
                // cta<4,112,512> warp<16,112,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 4, 112, 512, 16, 112, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 112, 512, 16, 112, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 112, 512, 16, 112, 128, 8, 8, 128, 4);
                // cta<4,128,512> warp<16,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 4, 128, 512, 16, 128, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 128, 512, 16, 128, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 128, 512, 16, 128, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<32,32,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 8, 32, 512, 32, 32, 128, 8, 8, 128, 2);
                // cta<8,48,512> warp<32,48,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 8, 48, 512, 32, 48, 128, 8, 8, 128, 2);
                // cta<8,64,512> warp<32,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 8, 64, 512, 32, 64, 128, 8, 8, 128, 2);
                // cta<8,80,512> warp<32,80,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 8, 80, 512, 32, 80, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 80, 512, 32, 80, 128, 8, 8, 128, 3);
                // cta<8,96,512> warp<32,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 8, 96, 512, 32, 96, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 96, 512, 32, 96, 128, 8, 8, 128, 3);
                // cta<8,112,512> warp<32,112,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 8, 112, 512, 32, 112, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 112, 512, 32, 112, 128, 8, 8, 128, 3);
                // cta<8,128,512> warp<32,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 2, true, 8, 128, 512, 32, 128, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 8, 128, 512, 32, 128, 128, 8, 8, 128, 3);
                // cta<4,32,512> warp<8,64,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 2, true, 4, 32, 512, 8, 64, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 32, 512, 8, 64, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 32, 512, 8, 64, 128, 8, 8, 128, 4);
                // cta<4,48,512> warp<8,96,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 2, true, 4, 48, 512, 8, 96, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 48, 512, 8, 96, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 48, 512, 8, 96, 128, 8, 8, 128, 4);
                // cta<4,64,512> warp<8,128,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 2, true, 4, 64, 512, 8, 128, 128, 8, 8, 128, 2);
                TEST(8, 2, true, 4, 64, 512, 8, 128, 128, 8, 8, 128, 3);
                TEST(8, 2, true, 4, 64, 512, 8, 128, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<16,64,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 2, true, 8, 32, 512, 16, 64, 128, 8, 8, 128, 2);
                // cta<8,48,512> warp<16,96,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 2, true, 8, 48, 512, 16, 96, 128, 8, 8, 128, 2);
                // cta<8,64,512> warp<16,128,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 2, true, 8, 64, 512, 16, 128, 128, 8, 8, 128, 2);
            } else {
            }
            break;
        #endif
        #ifdef W3A8
        case 3:
            if(quant_sign){
                ////// W3A8 int
                // cta<1,32,256> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 1, 32, 256, 8, 48, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 1, 32, 256, 8, 48, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 1, 32, 256, 8, 48, 128, 8, 8, 128, 4);
                // cta<1,48,256> warp<8,72,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 1, 48, 256, 8, 72, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 1, 48, 256, 8, 72, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 1, 48, 256, 8, 72, 128, 8, 8, 128, 4);
                // cta<1,64,256> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 1, 64, 256, 8, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 1, 64, 256, 8, 96, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 1, 64, 256, 8, 96, 128, 8, 8, 128, 4);
                // cta<1,80,256> warp<8,120,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 1, 80, 256, 8, 120, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 1, 80, 256, 8, 120, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 1, 80, 256, 8, 120, 128, 8, 8, 128, 4);
                // cta<4,32,256> warp<32,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 4, 32, 256, 32, 48, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 32, 256, 32, 48, 128, 8, 8, 128, 3);
                // cta<4,48,256> warp<32,72,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 4, 48, 256, 32, 72, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 48, 256, 32, 72, 128, 8, 8, 128, 3);
                // cta<4,64,256> warp<32,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 4, 64, 256, 32, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 64, 256, 32, 96, 128, 8, 8, 128, 3);
                // cta<4,80,256> warp<32,120,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 4, 80, 256, 32, 120, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 80, 256, 32, 120, 128, 8, 8, 128, 3);
                // cta<8,32,256> warp<64,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 8, 32, 256, 64, 48, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 32, 256, 64, 48, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 8, 32, 256, 64, 48, 128, 8, 8, 128, 4);
                TEST(8, 3, true, 8, 32, 256, 64, 48, 128, 8, 8, 128, 5);
                // cta<8,48,256> warp<64,72,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 8, 48, 256, 64, 72, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 48, 256, 64, 72, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 8, 48, 256, 64, 72, 128, 8, 8, 128, 4);
                TEST(8, 3, true, 8, 48, 256, 64, 72, 128, 8, 8, 128, 5);
                // cta<8,64,256> warp<64,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 8, 64, 256, 64, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 64, 256, 64, 96, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 8, 64, 256, 64, 96, 128, 8, 8, 128, 4);
                TEST(8, 3, true, 8, 64, 256, 64, 96, 128, 8, 8, 128, 5);
                TEST(8, 3, true, 8, 64, 256, 64, 96, 128, 8, 8, 128, 6);
                // cta<1,32,256> warp<8,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 1, 32, 256, 8, 24, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 1, 32, 256, 8, 24, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 1, 32, 256, 8, 24, 128, 8, 8, 128, 4);
                // cta<1,64,256> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 1, 64, 256, 8, 48, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 1, 64, 256, 8, 48, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 1, 64, 256, 8, 48, 128, 8, 8, 128, 4);
                // cta<1,96,256> warp<8,72,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 1, 96, 256, 8, 72, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 1, 96, 256, 8, 72, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 1, 96, 256, 8, 72, 128, 8, 8, 128, 4);
                // cta<1,128,256> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 1, 128, 256, 8, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 1, 128, 256, 8, 96, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 1, 128, 256, 8, 96, 128, 8, 8, 128, 4);
                // cta<4,32,256> warp<32,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 4, 32, 256, 32, 24, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 32, 256, 32, 24, 128, 8, 8, 128, 3);
                // cta<4,64,256> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 4, 64, 256, 32, 48, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 64, 256, 32, 48, 128, 8, 8, 128, 3);
                // cta<4,96,256> warp<32,72,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 4, 96, 256, 32, 72, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 96, 256, 32, 72, 128, 8, 8, 128, 3);
                // cta<4,128,256> warp<32,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 4, 128, 256, 32, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 128, 256, 32, 96, 128, 8, 8, 128, 3);
                // cta<8,32,256> warp<64,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 8, 32, 256, 64, 24, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 32, 256, 64, 24, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 8, 32, 256, 64, 24, 128, 8, 8, 128, 4);
                TEST(8, 3, true, 8, 32, 256, 64, 24, 128, 8, 8, 128, 5);
                // cta<8,64,256> warp<64,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 8, 64, 256, 64, 48, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 64, 256, 64, 48, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 8, 64, 256, 64, 48, 128, 8, 8, 128, 4);
                TEST(8, 3, true, 8, 64, 256, 64, 48, 128, 8, 8, 128, 5);
                TEST(8, 3, true, 8, 64, 256, 64, 48, 128, 8, 8, 128, 6);
                // cta<8,96,256> warp<64,72,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 8, 96, 256, 64, 72, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 96, 256, 64, 72, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 8, 96, 256, 64, 72, 128, 8, 8, 128, 4);
                TEST(8, 3, true, 8, 96, 256, 64, 72, 128, 8, 8, 128, 5);
                TEST(8, 3, true, 8, 96, 256, 64, 72, 128, 8, 8, 128, 6);
                // cta<8,128,256> warp<64,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 8, 128, 256, 64, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 128, 256, 64, 96, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 8, 128, 256, 64, 96, 128, 8, 8, 128, 4);
                TEST(8, 3, true, 8, 128, 256, 64, 96, 128, 8, 8, 128, 5);
                TEST(8, 3, true, 8, 128, 256, 64, 96, 128, 8, 8, 128, 6);
                TEST(8, 3, true, 8, 128, 256, 64, 96, 128, 8, 8, 128, 7);
                // cta<4,32,256> warp<16,96,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 3, true, 4, 32, 256, 16, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 32, 256, 16, 96, 128, 8, 8, 128, 3);
                // cta<8,32,256> warp<32,96,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 3, true, 8, 32, 256, 32, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 32, 256, 32, 96, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 8, 32, 256, 32, 96, 128, 8, 8, 128, 4);
                TEST(8, 3, true, 8, 32, 256, 32, 96, 128, 8, 8, 128, 5);
                // cta<4,32,256> warp<16,48,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 3, true, 4, 32, 256, 16, 48, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 32, 256, 16, 48, 128, 8, 8, 128, 3);
                // cta<4,48,256> warp<16,72,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 3, true, 4, 48, 256, 16, 72, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 48, 256, 16, 72, 128, 8, 8, 128, 3);
                // cta<4,64,256> warp<16,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 3, true, 4, 64, 256, 16, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 64, 256, 16, 96, 128, 8, 8, 128, 3);
                // cta<4,80,256> warp<16,120,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 3, true, 4, 80, 256, 16, 120, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 80, 256, 16, 120, 128, 8, 8, 128, 3);
                // cta<8,32,256> warp<32,48,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 3, true, 8, 32, 256, 32, 48, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 32, 256, 32, 48, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 8, 32, 256, 32, 48, 128, 8, 8, 128, 4);
                TEST(8, 3, true, 8, 32, 256, 32, 48, 128, 8, 8, 128, 5);
                // cta<8,48,256> warp<32,72,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 3, true, 8, 48, 256, 32, 72, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 48, 256, 32, 72, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 8, 48, 256, 32, 72, 128, 8, 8, 128, 4);
                TEST(8, 3, true, 8, 48, 256, 32, 72, 128, 8, 8, 128, 5);
                // cta<8,64,256> warp<32,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 3, true, 8, 64, 256, 32, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 64, 256, 32, 96, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 8, 64, 256, 32, 96, 128, 8, 8, 128, 4);
                TEST(8, 3, true, 8, 64, 256, 32, 96, 128, 8, 8, 128, 5);
                TEST(8, 3, true, 8, 64, 256, 32, 96, 128, 8, 8, 128, 6);
                // cta<8,80,256> warp<32,120,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 3, true, 8, 80, 256, 32, 120, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 80, 256, 32, 120, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 8, 80, 256, 32, 120, 128, 8, 8, 128, 4);
                TEST(8, 3, true, 8, 80, 256, 32, 120, 128, 8, 8, 128, 5);
                TEST(8, 3, true, 8, 80, 256, 32, 120, 128, 8, 8, 128, 6);
                // cta<4,32,256> warp<8,96,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 3, true, 4, 32, 256, 8, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 32, 256, 8, 96, 128, 8, 8, 128, 3);
                // cta<8,32,256> warp<16,96,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 3, true, 8, 32, 256, 16, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 32, 256, 16, 96, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 8, 32, 256, 16, 96, 128, 8, 8, 128, 4);
                TEST(8, 3, true, 8, 32, 256, 16, 96, 128, 8, 8, 128, 5);
                // cta<1,32,384> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 1, 32, 384, 8, 48, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 1, 32, 384, 8, 48, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 1, 32, 384, 8, 48, 128, 8, 8, 128, 4);
                // cta<1,48,384> warp<8,72,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 1, 48, 384, 8, 72, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 1, 48, 384, 8, 72, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 1, 48, 384, 8, 72, 128, 8, 8, 128, 4);
                // cta<1,64,384> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 1, 64, 384, 8, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 1, 64, 384, 8, 96, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 1, 64, 384, 8, 96, 128, 8, 8, 128, 4);
                // cta<1,80,384> warp<8,120,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 1, 80, 384, 8, 120, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 1, 80, 384, 8, 120, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 1, 80, 384, 8, 120, 128, 8, 8, 128, 4);
                // cta<4,32,384> warp<32,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 4, 32, 384, 32, 48, 128, 8, 8, 128, 2);
                // cta<4,48,384> warp<32,72,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 4, 48, 384, 32, 72, 128, 8, 8, 128, 2);
                // cta<4,64,384> warp<32,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 4, 64, 384, 32, 96, 128, 8, 8, 128, 2);
                // cta<4,80,384> warp<32,120,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 4, 80, 384, 32, 120, 128, 8, 8, 128, 2);
                // cta<8,32,384> warp<64,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 8, 32, 384, 64, 48, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 32, 384, 64, 48, 128, 8, 8, 128, 3);
                // cta<8,48,384> warp<64,72,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 8, 48, 384, 64, 72, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 48, 384, 64, 72, 128, 8, 8, 128, 3);
                // cta<8,64,384> warp<64,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 8, 64, 384, 64, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 64, 384, 64, 96, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 8, 64, 384, 64, 96, 128, 8, 8, 128, 4);
                // cta<1,32,384> warp<8,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 1, 32, 384, 8, 24, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 1, 32, 384, 8, 24, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 1, 32, 384, 8, 24, 128, 8, 8, 128, 4);
                // cta<1,64,384> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 1, 64, 384, 8, 48, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 1, 64, 384, 8, 48, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 1, 64, 384, 8, 48, 128, 8, 8, 128, 4);
                // cta<1,96,384> warp<8,72,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 1, 96, 384, 8, 72, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 1, 96, 384, 8, 72, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 1, 96, 384, 8, 72, 128, 8, 8, 128, 4);
                // cta<1,128,384> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 1, 128, 384, 8, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 1, 128, 384, 8, 96, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 1, 128, 384, 8, 96, 128, 8, 8, 128, 4);
                // cta<4,32,384> warp<32,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 4, 32, 384, 32, 24, 128, 8, 8, 128, 2);
                // cta<4,64,384> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 4, 64, 384, 32, 48, 128, 8, 8, 128, 2);
                // cta<4,96,384> warp<32,72,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 4, 96, 384, 32, 72, 128, 8, 8, 128, 2);
                // cta<4,128,384> warp<32,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 4, 128, 384, 32, 96, 128, 8, 8, 128, 2);
                // cta<8,32,384> warp<64,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 8, 32, 384, 64, 24, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 32, 384, 64, 24, 128, 8, 8, 128, 3);
                // cta<8,64,384> warp<64,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 8, 64, 384, 64, 48, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 64, 384, 64, 48, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 8, 64, 384, 64, 48, 128, 8, 8, 128, 4);
                // cta<8,96,384> warp<64,72,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 8, 96, 384, 64, 72, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 96, 384, 64, 72, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 8, 96, 384, 64, 72, 128, 8, 8, 128, 4);
                // cta<8,128,384> warp<64,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 8, 128, 384, 64, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 128, 384, 64, 96, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 8, 128, 384, 64, 96, 128, 8, 8, 128, 4);
                // cta<4,32,384> warp<16,96,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 3, true, 4, 32, 384, 16, 96, 128, 8, 8, 128, 2);
                // cta<8,32,384> warp<32,96,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 3, true, 8, 32, 384, 32, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 32, 384, 32, 96, 128, 8, 8, 128, 3);
                // cta<4,32,384> warp<16,48,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 3, true, 4, 32, 384, 16, 48, 128, 8, 8, 128, 2);
                // cta<4,48,384> warp<16,72,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 3, true, 4, 48, 384, 16, 72, 128, 8, 8, 128, 2);
                // cta<4,64,384> warp<16,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 3, true, 4, 64, 384, 16, 96, 128, 8, 8, 128, 2);
                // cta<4,80,384> warp<16,120,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 3, true, 4, 80, 384, 16, 120, 128, 8, 8, 128, 2);
                // cta<8,32,384> warp<32,48,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 3, true, 8, 32, 384, 32, 48, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 32, 384, 32, 48, 128, 8, 8, 128, 3);
                // cta<8,48,384> warp<32,72,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 3, true, 8, 48, 384, 32, 72, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 48, 384, 32, 72, 128, 8, 8, 128, 3);
                // cta<8,64,384> warp<32,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 3, true, 8, 64, 384, 32, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 64, 384, 32, 96, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 8, 64, 384, 32, 96, 128, 8, 8, 128, 4);
                // cta<8,80,384> warp<32,120,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 3, true, 8, 80, 384, 32, 120, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 80, 384, 32, 120, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 8, 80, 384, 32, 120, 128, 8, 8, 128, 4);
                // cta<4,32,384> warp<8,96,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 3, true, 4, 32, 384, 8, 96, 128, 8, 8, 128, 2);
                // cta<8,32,384> warp<16,96,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 3, true, 8, 32, 384, 16, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 32, 384, 16, 96, 128, 8, 8, 128, 3);
                // cta<1,32,512> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 1, 32, 512, 8, 48, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 1, 32, 512, 8, 48, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 1, 32, 512, 8, 48, 128, 8, 8, 128, 4);
                // cta<1,48,512> warp<8,72,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 1, 48, 512, 8, 72, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 1, 48, 512, 8, 72, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 1, 48, 512, 8, 72, 128, 8, 8, 128, 4);
                // cta<1,64,512> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 1, 64, 512, 8, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 1, 64, 512, 8, 96, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 1, 64, 512, 8, 96, 128, 8, 8, 128, 4);
                // cta<1,80,512> warp<8,120,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 1, 80, 512, 8, 120, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 1, 80, 512, 8, 120, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 1, 80, 512, 8, 120, 128, 8, 8, 128, 4);
                // cta<4,32,512> warp<32,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 4, 32, 512, 32, 48, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 32, 512, 32, 48, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 4, 32, 512, 32, 48, 128, 8, 8, 128, 4);
                // cta<4,48,512> warp<32,72,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 4, 48, 512, 32, 72, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 48, 512, 32, 72, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 4, 48, 512, 32, 72, 128, 8, 8, 128, 4);
                // cta<4,64,512> warp<32,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 4, 64, 512, 32, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 64, 512, 32, 96, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 4, 64, 512, 32, 96, 128, 8, 8, 128, 4);
                // cta<4,80,512> warp<32,120,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 4, 80, 512, 32, 120, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 80, 512, 32, 120, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 4, 80, 512, 32, 120, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<64,48,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 8, 32, 512, 64, 48, 128, 8, 8, 128, 2);
                // cta<8,48,512> warp<64,72,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 8, 48, 512, 64, 72, 128, 8, 8, 128, 2);
                // cta<8,64,512> warp<64,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 3, true, 8, 64, 512, 64, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 64, 512, 64, 96, 128, 8, 8, 128, 3);
                // cta<1,32,512> warp<8,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 1, 32, 512, 8, 24, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 1, 32, 512, 8, 24, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 1, 32, 512, 8, 24, 128, 8, 8, 128, 4);
                // cta<1,64,512> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 1, 64, 512, 8, 48, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 1, 64, 512, 8, 48, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 1, 64, 512, 8, 48, 128, 8, 8, 128, 4);
                // cta<1,96,512> warp<8,72,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 1, 96, 512, 8, 72, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 1, 96, 512, 8, 72, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 1, 96, 512, 8, 72, 128, 8, 8, 128, 4);
                // cta<1,128,512> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 1, 128, 512, 8, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 1, 128, 512, 8, 96, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 1, 128, 512, 8, 96, 128, 8, 8, 128, 4);
                // cta<4,32,512> warp<32,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 4, 32, 512, 32, 24, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 32, 512, 32, 24, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 4, 32, 512, 32, 24, 128, 8, 8, 128, 4);
                // cta<4,64,512> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 4, 64, 512, 32, 48, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 64, 512, 32, 48, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 4, 64, 512, 32, 48, 128, 8, 8, 128, 4);
                // cta<4,96,512> warp<32,72,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 4, 96, 512, 32, 72, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 96, 512, 32, 72, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 4, 96, 512, 32, 72, 128, 8, 8, 128, 4);
                // cta<4,128,512> warp<32,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 4, 128, 512, 32, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 128, 512, 32, 96, 128, 8, 8, 128, 3);
                // cta<8,32,512> warp<64,24,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 8, 32, 512, 64, 24, 128, 8, 8, 128, 2);
                // cta<8,64,512> warp<64,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 8, 64, 512, 64, 48, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 64, 512, 64, 48, 128, 8, 8, 128, 3);
                // cta<8,96,512> warp<64,72,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 8, 96, 512, 64, 72, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 96, 512, 64, 72, 128, 8, 8, 128, 3);
                // cta<8,128,512> warp<64,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 3, true, 8, 128, 512, 64, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 128, 512, 64, 96, 128, 8, 8, 128, 3);
                // cta<4,32,512> warp<16,96,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 3, true, 4, 32, 512, 16, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 32, 512, 16, 96, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 4, 32, 512, 16, 96, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<32,96,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 3, true, 8, 32, 512, 32, 96, 128, 8, 8, 128, 2);
                // cta<4,32,512> warp<16,48,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 3, true, 4, 32, 512, 16, 48, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 32, 512, 16, 48, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 4, 32, 512, 16, 48, 128, 8, 8, 128, 4);
                // cta<4,48,512> warp<16,72,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 3, true, 4, 48, 512, 16, 72, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 48, 512, 16, 72, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 4, 48, 512, 16, 72, 128, 8, 8, 128, 4);
                // cta<4,64,512> warp<16,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 3, true, 4, 64, 512, 16, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 64, 512, 16, 96, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 4, 64, 512, 16, 96, 128, 8, 8, 128, 4);
                // cta<4,80,512> warp<16,120,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 3, true, 4, 80, 512, 16, 120, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 80, 512, 16, 120, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 4, 80, 512, 16, 120, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<32,48,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 3, true, 8, 32, 512, 32, 48, 128, 8, 8, 128, 2);
                // cta<8,48,512> warp<32,72,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 3, true, 8, 48, 512, 32, 72, 128, 8, 8, 128, 2);
                // cta<8,64,512> warp<32,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 3, true, 8, 64, 512, 32, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 64, 512, 32, 96, 128, 8, 8, 128, 3);
                // cta<8,80,512> warp<32,120,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 3, true, 8, 80, 512, 32, 120, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 8, 80, 512, 32, 120, 128, 8, 8, 128, 3);
                // cta<4,32,512> warp<8,96,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 3, true, 4, 32, 512, 8, 96, 128, 8, 8, 128, 2);
                TEST(8, 3, true, 4, 32, 512, 8, 96, 128, 8, 8, 128, 3);
                TEST(8, 3, true, 4, 32, 512, 8, 96, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<16,96,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 3, true, 8, 32, 512, 16, 96, 128, 8, 8, 128, 2);
            }
            break;
        #endif
        #ifdef W4A8
        case 4:
            if (quant_sign) {
                ////// W4A8 int
                // cta<1,32,256> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 4, true, 1, 32, 256, 8, 64, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 32, 256, 8, 64, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 32, 256, 8, 64, 128, 8, 8, 128, 4);
                // cta<1,48,256> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 4, true, 1, 48, 256, 8, 96, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 48, 256, 8, 96, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 48, 256, 8, 96, 128, 8, 8, 128, 4);
                // cta<1,64,256> warp<8,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 4, true, 1, 64, 256, 8, 128, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 64, 256, 8, 128, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 64, 256, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,256> warp<32,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 4, true, 4, 32, 256, 32, 64, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 32, 256, 32, 64, 128, 8, 8, 128, 3);
                // cta<4,48,256> warp<32,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 4, true, 4, 48, 256, 32, 96, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 48, 256, 32, 96, 128, 8, 8, 128, 3);
                // cta<4,64,256> warp<32,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 4, true, 4, 64, 256, 32, 128, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 64, 256, 32, 128, 128, 8, 8, 128, 3);
                // cta<8,32,256> warp<64,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 4, true, 8, 32, 256, 64, 64, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 32, 256, 64, 64, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 8, 32, 256, 64, 64, 128, 8, 8, 128, 4);
                TEST(8, 4, true, 8, 32, 256, 64, 64, 128, 8, 8, 128, 5);
                // cta<8,48,256> warp<64,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 4, true, 8, 48, 256, 64, 96, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 48, 256, 64, 96, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 8, 48, 256, 64, 96, 128, 8, 8, 128, 4);
                TEST(8, 4, true, 8, 48, 256, 64, 96, 128, 8, 8, 128, 5);
                TEST(8, 4, true, 8, 48, 256, 64, 96, 128, 8, 8, 128, 6);
                // cta<1,32,256> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 1, 32, 256, 8, 32, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 32, 256, 8, 32, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 32, 256, 8, 32, 128, 8, 8, 128, 4);
                // cta<1,48,256> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 1, 48, 256, 8, 48, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 48, 256, 8, 48, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 48, 256, 8, 48, 128, 8, 8, 128, 4);
                // cta<1,64,256> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 1, 64, 256, 8, 64, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 64, 256, 8, 64, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 64, 256, 8, 64, 128, 8, 8, 128, 4);
                // cta<1,80,256> warp<8,80,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 1, 80, 256, 8, 80, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 80, 256, 8, 80, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 80, 256, 8, 80, 128, 8, 8, 128, 4);
                // cta<1,96,256> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 1, 96, 256, 8, 96, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 96, 256, 8, 96, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 96, 256, 8, 96, 128, 8, 8, 128, 4);
                // cta<1,112,256> warp<8,112,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 1, 112, 256, 8, 112, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 112, 256, 8, 112, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 112, 256, 8, 112, 128, 8, 8, 128, 4);
                // cta<1,128,256> warp<8,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 1, 128, 256, 8, 128, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 128, 256, 8, 128, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 128, 256, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,256> warp<32,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 4, 32, 256, 32, 32, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 32, 256, 32, 32, 128, 8, 8, 128, 3);
                // cta<4,48,256> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 4, 48, 256, 32, 48, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 48, 256, 32, 48, 128, 8, 8, 128, 3);
                // cta<4,64,256> warp<32,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 4, 64, 256, 32, 64, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 64, 256, 32, 64, 128, 8, 8, 128, 3);
                // cta<4,80,256> warp<32,80,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 4, 80, 256, 32, 80, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 80, 256, 32, 80, 128, 8, 8, 128, 3);
                // cta<4,96,256> warp<32,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 4, 96, 256, 32, 96, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 96, 256, 32, 96, 128, 8, 8, 128, 3);
                // cta<4,112,256> warp<32,112,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 4, 112, 256, 32, 112, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 112, 256, 32, 112, 128, 8, 8, 128, 3);
                // cta<4,128,256> warp<32,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 4, 128, 256, 32, 128, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 128, 256, 32, 128, 128, 8, 8, 128, 3);
                // cta<8,32,256> warp<64,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 8, 32, 256, 64, 32, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 32, 256, 64, 32, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 8, 32, 256, 64, 32, 128, 8, 8, 128, 4);
                TEST(8, 4, true, 8, 32, 256, 64, 32, 128, 8, 8, 128, 5);
                // cta<8,48,256> warp<64,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 8, 48, 256, 64, 48, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 48, 256, 64, 48, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 8, 48, 256, 64, 48, 128, 8, 8, 128, 4);
                TEST(8, 4, true, 8, 48, 256, 64, 48, 128, 8, 8, 128, 5);
                TEST(8, 4, true, 8, 48, 256, 64, 48, 128, 8, 8, 128, 6);
                // cta<8,64,256> warp<64,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 8, 64, 256, 64, 64, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 64, 256, 64, 64, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 8, 64, 256, 64, 64, 128, 8, 8, 128, 4);
                TEST(8, 4, true, 8, 64, 256, 64, 64, 128, 8, 8, 128, 5);
                TEST(8, 4, true, 8, 64, 256, 64, 64, 128, 8, 8, 128, 6);
                // cta<8,80,256> warp<64,80,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 8, 80, 256, 64, 80, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 80, 256, 64, 80, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 8, 80, 256, 64, 80, 128, 8, 8, 128, 4);
                TEST(8, 4, true, 8, 80, 256, 64, 80, 128, 8, 8, 128, 5);
                TEST(8, 4, true, 8, 80, 256, 64, 80, 128, 8, 8, 128, 6);
                // cta<8,96,256> warp<64,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 8, 96, 256, 64, 96, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 96, 256, 64, 96, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 8, 96, 256, 64, 96, 128, 8, 8, 128, 4);
                TEST(8, 4, true, 8, 96, 256, 64, 96, 128, 8, 8, 128, 5);
                TEST(8, 4, true, 8, 96, 256, 64, 96, 128, 8, 8, 128, 6);
                TEST(8, 4, true, 8, 96, 256, 64, 96, 128, 8, 8, 128, 7);
                // cta<4,32,256> warp<16,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 4, true, 4, 32, 256, 16, 128, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 32, 256, 16, 128, 128, 8, 8, 128, 3);
                // cta<8,32,256> warp<32,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 4, true, 8, 32, 256, 32, 128, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 32, 256, 32, 128, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 8, 32, 256, 32, 128, 128, 8, 8, 128, 4);
                TEST(8, 4, true, 8, 32, 256, 32, 128, 128, 8, 8, 128, 5);
                // cta<4,32,256> warp<16,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 4, true, 4, 32, 256, 16, 64, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 32, 256, 16, 64, 128, 8, 8, 128, 3);
                // cta<4,48,256> warp<16,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 4, true, 4, 48, 256, 16, 96, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 48, 256, 16, 96, 128, 8, 8, 128, 3);
                // cta<4,64,256> warp<16,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 4, true, 4, 64, 256, 16, 128, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 64, 256, 16, 128, 128, 8, 8, 128, 3);
                // cta<8,32,256> warp<32,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 4, true, 8, 32, 256, 32, 64, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 32, 256, 32, 64, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 8, 32, 256, 32, 64, 128, 8, 8, 128, 4);
                TEST(8, 4, true, 8, 32, 256, 32, 64, 128, 8, 8, 128, 5);
                // cta<8,48,256> warp<32,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 4, true, 8, 48, 256, 32, 96, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 48, 256, 32, 96, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 8, 48, 256, 32, 96, 128, 8, 8, 128, 4);
                TEST(8, 4, true, 8, 48, 256, 32, 96, 128, 8, 8, 128, 5);
                TEST(8, 4, true, 8, 48, 256, 32, 96, 128, 8, 8, 128, 6);
                // cta<8,64,256> warp<32,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 4, true, 8, 64, 256, 32, 128, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 64, 256, 32, 128, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 8, 64, 256, 32, 128, 128, 8, 8, 128, 4);
                TEST(8, 4, true, 8, 64, 256, 32, 128, 128, 8, 8, 128, 5);
                TEST(8, 4, true, 8, 64, 256, 32, 128, 128, 8, 8, 128, 6);
                // cta<4,32,256> warp<8,128,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 4, true, 4, 32, 256, 8, 128, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 32, 256, 8, 128, 128, 8, 8, 128, 3);
                // cta<8,32,256> warp<16,128,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 4, true, 8, 32, 256, 16, 128, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 32, 256, 16, 128, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 8, 32, 256, 16, 128, 128, 8, 8, 128, 4);
                TEST(8, 4, true, 8, 32, 256, 16, 128, 128, 8, 8, 128, 5);
                // cta<1,32,384> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 4, true, 1, 32, 384, 8, 64, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 32, 384, 8, 64, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 32, 384, 8, 64, 128, 8, 8, 128, 4);
                // cta<1,48,384> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 4, true, 1, 48, 384, 8, 96, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 48, 384, 8, 96, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 48, 384, 8, 96, 128, 8, 8, 128, 4);
                // cta<1,64,384> warp<8,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 4, true, 1, 64, 384, 8, 128, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 64, 384, 8, 128, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 64, 384, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,384> warp<32,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 4, true, 4, 32, 384, 32, 64, 128, 8, 8, 128, 2);
                // cta<4,48,384> warp<32,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 4, true, 4, 48, 384, 32, 96, 128, 8, 8, 128, 2);
                // cta<4,64,384> warp<32,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 4, true, 4, 64, 384, 32, 128, 128, 8, 8, 128, 2);
                // cta<8,32,384> warp<64,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 4, true, 8, 32, 384, 64, 64, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 32, 384, 64, 64, 128, 8, 8, 128, 3);
                // cta<8,48,384> warp<64,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 4, true, 8, 48, 384, 64, 96, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 48, 384, 64, 96, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 8, 48, 384, 64, 96, 128, 8, 8, 128, 4);
                // cta<1,32,384> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 1, 32, 384, 8, 32, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 32, 384, 8, 32, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 32, 384, 8, 32, 128, 8, 8, 128, 4);
                // cta<1,48,384> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 1, 48, 384, 8, 48, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 48, 384, 8, 48, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 48, 384, 8, 48, 128, 8, 8, 128, 4);
                // cta<1,64,384> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 1, 64, 384, 8, 64, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 64, 384, 8, 64, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 64, 384, 8, 64, 128, 8, 8, 128, 4);
                // cta<1,80,384> warp<8,80,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 1, 80, 384, 8, 80, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 80, 384, 8, 80, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 80, 384, 8, 80, 128, 8, 8, 128, 4);
                // cta<1,96,384> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 1, 96, 384, 8, 96, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 96, 384, 8, 96, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 96, 384, 8, 96, 128, 8, 8, 128, 4);
                // cta<1,112,384> warp<8,112,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 1, 112, 384, 8, 112, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 112, 384, 8, 112, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 112, 384, 8, 112, 128, 8, 8, 128, 4);
                // cta<1,128,384> warp<8,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 1, 128, 384, 8, 128, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 128, 384, 8, 128, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 128, 384, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,384> warp<32,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 4, 32, 384, 32, 32, 128, 8, 8, 128, 2);
                // cta<4,48,384> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 4, 48, 384, 32, 48, 128, 8, 8, 128, 2);
                // cta<4,64,384> warp<32,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 4, 64, 384, 32, 64, 128, 8, 8, 128, 2);
                // cta<4,80,384> warp<32,80,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 4, 80, 384, 32, 80, 128, 8, 8, 128, 2);
                // cta<4,96,384> warp<32,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 4, 96, 384, 32, 96, 128, 8, 8, 128, 2);
                // cta<4,112,384> warp<32,112,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 4, 112, 384, 32, 112, 128, 8, 8, 128, 2);
                // cta<4,128,384> warp<32,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 4, 128, 384, 32, 128, 128, 8, 8, 128, 2);
                // cta<8,32,384> warp<64,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 8, 32, 384, 64, 32, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 32, 384, 64, 32, 128, 8, 8, 128, 3);
                // cta<8,48,384> warp<64,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 8, 48, 384, 64, 48, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 48, 384, 64, 48, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 8, 48, 384, 64, 48, 128, 8, 8, 128, 4);
                // cta<8,64,384> warp<64,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 8, 64, 384, 64, 64, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 64, 384, 64, 64, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 8, 64, 384, 64, 64, 128, 8, 8, 128, 4);
                // cta<8,80,384> warp<64,80,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 8, 80, 384, 64, 80, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 80, 384, 64, 80, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 8, 80, 384, 64, 80, 128, 8, 8, 128, 4);
                // cta<8,96,384> warp<64,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 8, 96, 384, 64, 96, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 96, 384, 64, 96, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 8, 96, 384, 64, 96, 128, 8, 8, 128, 4);
                // cta<4,32,384> warp<16,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 4, true, 4, 32, 384, 16, 128, 128, 8, 8, 128, 2);
                // cta<8,32,384> warp<32,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 4, true, 8, 32, 384, 32, 128, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 32, 384, 32, 128, 128, 8, 8, 128, 3);
                // cta<4,32,384> warp<16,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 4, true, 4, 32, 384, 16, 64, 128, 8, 8, 128, 2);
                // cta<4,48,384> warp<16,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 4, true, 4, 48, 384, 16, 96, 128, 8, 8, 128, 2);
                // cta<4,64,384> warp<16,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 4, true, 4, 64, 384, 16, 128, 128, 8, 8, 128, 2);
                // cta<8,32,384> warp<32,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 4, true, 8, 32, 384, 32, 64, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 32, 384, 32, 64, 128, 8, 8, 128, 3);
                // cta<8,48,384> warp<32,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 4, true, 8, 48, 384, 32, 96, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 48, 384, 32, 96, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 8, 48, 384, 32, 96, 128, 8, 8, 128, 4);
                // cta<8,64,384> warp<32,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 4, true, 8, 64, 384, 32, 128, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 64, 384, 32, 128, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 8, 64, 384, 32, 128, 128, 8, 8, 128, 4);
                // cta<4,32,384> warp<8,128,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 4, true, 4, 32, 384, 8, 128, 128, 8, 8, 128, 2);
                // cta<8,32,384> warp<16,128,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 4, true, 8, 32, 384, 16, 128, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 32, 384, 16, 128, 128, 8, 8, 128, 3);
                // cta<1,32,512> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 4, true, 1, 32, 512, 8, 64, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 32, 512, 8, 64, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 32, 512, 8, 64, 128, 8, 8, 128, 4);
                // cta<1,48,512> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 4, true, 1, 48, 512, 8, 96, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 48, 512, 8, 96, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 48, 512, 8, 96, 128, 8, 8, 128, 4);
                // cta<1,64,512> warp<8,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 4, true, 1, 64, 512, 8, 128, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 64, 512, 8, 128, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 64, 512, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,512> warp<32,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 4, true, 4, 32, 512, 32, 64, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 32, 512, 32, 64, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 4, 32, 512, 32, 64, 128, 8, 8, 128, 4);
                // cta<4,48,512> warp<32,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 4, true, 4, 48, 512, 32, 96, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 48, 512, 32, 96, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 4, 48, 512, 32, 96, 128, 8, 8, 128, 4);
                // cta<4,64,512> warp<32,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 4, true, 4, 64, 512, 32, 128, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 64, 512, 32, 128, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 4, 64, 512, 32, 128, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<64,64,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 4, true, 8, 32, 512, 64, 64, 128, 8, 8, 128, 2);
                // cta<8,48,512> warp<64,96,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 4, true, 8, 48, 512, 64, 96, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 48, 512, 64, 96, 128, 8, 8, 128, 3);
                // cta<1,32,512> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 1, 32, 512, 8, 32, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 32, 512, 8, 32, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 32, 512, 8, 32, 128, 8, 8, 128, 4);
                // cta<1,48,512> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 1, 48, 512, 8, 48, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 48, 512, 8, 48, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 48, 512, 8, 48, 128, 8, 8, 128, 4);
                // cta<1,64,512> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 1, 64, 512, 8, 64, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 64, 512, 8, 64, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 64, 512, 8, 64, 128, 8, 8, 128, 4);
                // cta<1,80,512> warp<8,80,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 1, 80, 512, 8, 80, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 80, 512, 8, 80, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 80, 512, 8, 80, 128, 8, 8, 128, 4);
                // cta<1,96,512> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 1, 96, 512, 8, 96, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 96, 512, 8, 96, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 1, 96, 512, 8, 96, 128, 8, 8, 128, 4);
                // cta<1,112,512> warp<8,112,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 1, 112, 512, 8, 112, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 112, 512, 8, 112, 128, 8, 8, 128, 3);
                // cta<1,128,512> warp<8,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 1, 128, 512, 8, 128, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 1, 128, 512, 8, 128, 128, 8, 8, 128, 3);
                // cta<4,32,512> warp<32,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 4, 32, 512, 32, 32, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 32, 512, 32, 32, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 4, 32, 512, 32, 32, 128, 8, 8, 128, 4);
                // cta<4,48,512> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 4, 48, 512, 32, 48, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 48, 512, 32, 48, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 4, 48, 512, 32, 48, 128, 8, 8, 128, 4);
                // cta<4,64,512> warp<32,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 4, 64, 512, 32, 64, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 64, 512, 32, 64, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 4, 64, 512, 32, 64, 128, 8, 8, 128, 4);
                // cta<4,80,512> warp<32,80,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 4, 80, 512, 32, 80, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 80, 512, 32, 80, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 4, 80, 512, 32, 80, 128, 8, 8, 128, 4);
                // cta<4,96,512> warp<32,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 4, 96, 512, 32, 96, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 96, 512, 32, 96, 128, 8, 8, 128, 3);
                // cta<4,112,512> warp<32,112,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 4, 112, 512, 32, 112, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 112, 512, 32, 112, 128, 8, 8, 128, 3);
                // cta<4,128,512> warp<32,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 4, 128, 512, 32, 128, 128, 8, 8, 128, 2);
                // cta<8,32,512> warp<64,32,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 8, 32, 512, 64, 32, 128, 8, 8, 128, 2);
                // cta<8,48,512> warp<64,48,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 8, 48, 512, 64, 48, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 48, 512, 64, 48, 128, 8, 8, 128, 3);
                // cta<8,64,512> warp<64,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 8, 64, 512, 64, 64, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 64, 512, 64, 64, 128, 8, 8, 128, 3);
                // cta<8,80,512> warp<64,80,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 8, 80, 512, 64, 80, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 80, 512, 64, 80, 128, 8, 8, 128, 3);
                // cta<8,96,512> warp<64,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 4, true, 8, 96, 512, 64, 96, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 96, 512, 64, 96, 128, 8, 8, 128, 3);
                // cta<4,32,512> warp<16,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 4, true, 4, 32, 512, 16, 128, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 32, 512, 16, 128, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 4, 32, 512, 16, 128, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<32,128,128> mma<8,8,128>   WARPS[2x1]
                TEST(8, 4, true, 8, 32, 512, 32, 128, 128, 8, 8, 128, 2);
                // cta<4,32,512> warp<16,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 4, true, 4, 32, 512, 16, 64, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 32, 512, 16, 64, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 4, 32, 512, 16, 64, 128, 8, 8, 128, 4);
                // cta<4,48,512> warp<16,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 4, true, 4, 48, 512, 16, 96, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 48, 512, 16, 96, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 4, 48, 512, 16, 96, 128, 8, 8, 128, 4);
                // cta<4,64,512> warp<16,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 4, true, 4, 64, 512, 16, 128, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 64, 512, 16, 128, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 4, 64, 512, 16, 128, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<32,64,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 4, true, 8, 32, 512, 32, 64, 128, 8, 8, 128, 2);
                // cta<8,48,512> warp<32,96,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 4, true, 8, 48, 512, 32, 96, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 48, 512, 32, 96, 128, 8, 8, 128, 3);
                // cta<8,64,512> warp<32,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 4, true, 8, 64, 512, 32, 128, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 8, 64, 512, 32, 128, 128, 8, 8, 128, 3);
                // cta<4,32,512> warp<8,128,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 4, true, 4, 32, 512, 8, 128, 128, 8, 8, 128, 2);
                TEST(8, 4, true, 4, 32, 512, 8, 128, 128, 8, 8, 128, 3);
                TEST(8, 4, true, 4, 32, 512, 8, 128, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<16,128,128> mma<8,8,128>   WARPS[4x1]
                TEST(8, 4, true, 8, 32, 512, 16, 128, 128, 8, 8, 128, 2);
            } else {
            }
            break;
        #endif
        #ifdef W8A8
        case 8:
            if (quant_sign) {
                ////// W8A8 int
                // cta<1,32,256> warp<8,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 8, true, 1, 32, 256, 8, 128, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 1, 32, 256, 8, 128, 128, 8, 8, 128, 3);
                TEST(8, 8, true, 1, 32, 256, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,256> warp<32,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 8, true, 4, 32, 256, 32, 128, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 4, 32, 256, 32, 128, 128, 8, 8, 128, 3);
                // cta<1,32,256> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 8, true, 1, 32, 256, 8, 64, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 1, 32, 256, 8, 64, 128, 8, 8, 128, 3);
                TEST(8, 8, true, 1, 32, 256, 8, 64, 128, 8, 8, 128, 4);
                // cta<1,48,256> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 8, true, 1, 48, 256, 8, 96, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 1, 48, 256, 8, 96, 128, 8, 8, 128, 3);
                TEST(8, 8, true, 1, 48, 256, 8, 96, 128, 8, 8, 128, 4);
                // cta<1,64,256> warp<8,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 8, true, 1, 64, 256, 8, 128, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 1, 64, 256, 8, 128, 128, 8, 8, 128, 3);
                TEST(8, 8, true, 1, 64, 256, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,256> warp<32,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 8, true, 4, 32, 256, 32, 64, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 4, 32, 256, 32, 64, 128, 8, 8, 128, 3);
                // cta<4,48,256> warp<32,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 8, true, 4, 48, 256, 32, 96, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 4, 48, 256, 32, 96, 128, 8, 8, 128, 3);
                // cta<4,64,256> warp<32,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 8, true, 4, 64, 256, 32, 128, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 4, 64, 256, 32, 128, 128, 8, 8, 128, 3);
                // cta<8,32,256> warp<64,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 8, true, 8, 32, 256, 64, 64, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 8, 32, 256, 64, 64, 128, 8, 8, 128, 3);
                TEST(8, 8, true, 8, 32, 256, 64, 64, 128, 8, 8, 128, 4);
                TEST(8, 8, true, 8, 32, 256, 64, 64, 128, 8, 8, 128, 5);
                TEST(8, 8, true, 8, 32, 256, 64, 64, 128, 8, 8, 128, 6);
                // cta<8,48,256> warp<64,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 8, true, 8, 48, 256, 64, 96, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 8, 48, 256, 64, 96, 128, 8, 8, 128, 3);
                TEST(8, 8, true, 8, 48, 256, 64, 96, 128, 8, 8, 128, 4);
                TEST(8, 8, true, 8, 48, 256, 64, 96, 128, 8, 8, 128, 5);
                TEST(8, 8, true, 8, 48, 256, 64, 96, 128, 8, 8, 128, 6);
                TEST(8, 8, true, 8, 48, 256, 64, 96, 128, 8, 8, 128, 7);
                // cta<4,32,256> warp<16,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 8, true, 4, 32, 256, 16, 128, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 4, 32, 256, 16, 128, 128, 8, 8, 128, 3);
                // cta<8,32,256> warp<32,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 8, true, 8, 32, 256, 32, 128, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 8, 32, 256, 32, 128, 128, 8, 8, 128, 3);
                TEST(8, 8, true, 8, 32, 256, 32, 128, 128, 8, 8, 128, 4);
                TEST(8, 8, true, 8, 32, 256, 32, 128, 128, 8, 8, 128, 5);
                TEST(8, 8, true, 8, 32, 256, 32, 128, 128, 8, 8, 128, 6);
                // cta<1,32,384> warp<8,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 8, true, 1, 32, 384, 8, 128, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 1, 32, 384, 8, 128, 128, 8, 8, 128, 3);
                TEST(8, 8, true, 1, 32, 384, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,384> warp<32,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 8, true, 4, 32, 384, 32, 128, 128, 8, 8, 128, 2);
                // cta<1,32,384> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 8, true, 1, 32, 384, 8, 64, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 1, 32, 384, 8, 64, 128, 8, 8, 128, 3);
                TEST(8, 8, true, 1, 32, 384, 8, 64, 128, 8, 8, 128, 4);
                // cta<1,48,384> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 8, true, 1, 48, 384, 8, 96, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 1, 48, 384, 8, 96, 128, 8, 8, 128, 3);
                TEST(8, 8, true, 1, 48, 384, 8, 96, 128, 8, 8, 128, 4);
                // cta<1,64,384> warp<8,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 8, true, 1, 64, 384, 8, 128, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 1, 64, 384, 8, 128, 128, 8, 8, 128, 3);
                TEST(8, 8, true, 1, 64, 384, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,384> warp<32,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 8, true, 4, 32, 384, 32, 64, 128, 8, 8, 128, 2);
                // cta<4,48,384> warp<32,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 8, true, 4, 48, 384, 32, 96, 128, 8, 8, 128, 2);
                // cta<4,64,384> warp<32,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 8, true, 4, 64, 384, 32, 128, 128, 8, 8, 128, 2);
                // cta<8,32,384> warp<64,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 8, true, 8, 32, 384, 64, 64, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 8, 32, 384, 64, 64, 128, 8, 8, 128, 3);
                TEST(8, 8, true, 8, 32, 384, 64, 64, 128, 8, 8, 128, 4);
                // cta<8,48,384> warp<64,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 8, true, 8, 48, 384, 64, 96, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 8, 48, 384, 64, 96, 128, 8, 8, 128, 3);
                TEST(8, 8, true, 8, 48, 384, 64, 96, 128, 8, 8, 128, 4);
                // cta<4,32,384> warp<16,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 8, true, 4, 32, 384, 16, 128, 128, 8, 8, 128, 2);
                // cta<8,32,384> warp<32,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 8, true, 8, 32, 384, 32, 128, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 8, 32, 384, 32, 128, 128, 8, 8, 128, 3);
                TEST(8, 8, true, 8, 32, 384, 32, 128, 128, 8, 8, 128, 4);
                // cta<1,32,512> warp<8,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 8, true, 1, 32, 512, 8, 128, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 1, 32, 512, 8, 128, 128, 8, 8, 128, 3);
                TEST(8, 8, true, 1, 32, 512, 8, 128, 128, 8, 8, 128, 4);
                // cta<4,32,512> warp<32,128,128> mma<8,8,128>   WARPS[1x2]
                TEST(8, 8, true, 4, 32, 512, 32, 128, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 4, 32, 512, 32, 128, 128, 8, 8, 128, 3);
                TEST(8, 8, true, 4, 32, 512, 32, 128, 128, 8, 8, 128, 4);
                // cta<1,32,512> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 8, true, 1, 32, 512, 8, 64, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 1, 32, 512, 8, 64, 128, 8, 8, 128, 3);
                TEST(8, 8, true, 1, 32, 512, 8, 64, 128, 8, 8, 128, 4);
                // cta<1,48,512> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 8, true, 1, 48, 512, 8, 96, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 1, 48, 512, 8, 96, 128, 8, 8, 128, 3);
                TEST(8, 8, true, 1, 48, 512, 8, 96, 128, 8, 8, 128, 4);
                // cta<1,64,512> warp<8,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 8, true, 1, 64, 512, 8, 128, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 1, 64, 512, 8, 128, 128, 8, 8, 128, 3);
                // cta<4,32,512> warp<32,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 8, true, 4, 32, 512, 32, 64, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 4, 32, 512, 32, 64, 128, 8, 8, 128, 3);
                TEST(8, 8, true, 4, 32, 512, 32, 64, 128, 8, 8, 128, 4);
                // cta<4,48,512> warp<32,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 8, true, 4, 48, 512, 32, 96, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 4, 48, 512, 32, 96, 128, 8, 8, 128, 3);
                // cta<4,64,512> warp<32,128,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 8, true, 4, 64, 512, 32, 128, 128, 8, 8, 128, 2);
                // cta<8,32,512> warp<64,64,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 8, true, 8, 32, 512, 64, 64, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 8, 32, 512, 64, 64, 128, 8, 8, 128, 3);
                // cta<8,48,512> warp<64,96,128> mma<8,8,128>   WARPS[1x4]
                TEST(8, 8, true, 8, 48, 512, 64, 96, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 8, 48, 512, 64, 96, 128, 8, 8, 128, 3);
                // cta<4,32,512> warp<16,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 8, true, 4, 32, 512, 16, 128, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 4, 32, 512, 16, 128, 128, 8, 8, 128, 3);
                TEST(8, 8, true, 4, 32, 512, 16, 128, 128, 8, 8, 128, 4);
                // cta<8,32,512> warp<32,128,128> mma<8,8,128>   WARPS[2x2]
                TEST(8, 8, true, 8, 32, 512, 32, 128, 128, 8, 8, 128, 2);
                TEST(8, 8, true, 8, 32, 512, 32, 128, 128, 8, 8, 128, 3);
            } else {
            }
            break;
        #endif
        default:
            printf("unsupport w%da%d\n", w_bits, x_bits);
        }
        break;
    default:
        printf("unsupport w%da%d\n", w_bits, x_bits);
    }
    printf("The best kernel config is %s with %f TOPS\n", best_config.str().c_str(), max_gflop);
    free(h_x);
    free(h_w);
    free(h_x_pack);
    free(h_w_pack);
    free(h_out);
    free(h_ref_out);
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_x_pack);
    cudaFree(d_w_pack);
    cudaFree(d_out);

    cudaStreamDestroy(stream);
    return 0;
}