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

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <cfloat>
#include "common/base.h"
#include "common/pack.h"
#include "common/timer.h"
#include "mma_any/aq_bmma_library.h"
#include "mma_any/aq_bmma_op.h"

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

template <typename InitFuncType, typename ExecFuncType, typename OpStateType>
inline int benchmark(InitFuncType init_fn, ExecFuncType exec_fn, int X_BITS, int W_BITS, int *X,
                     int *W, int *X_PACKED, int *W_PACKED, int M, int N, int K, int *D, half *C,
                     int *H_OUT, const int *H_REF_OUT, bool bias, bool SIGNED, float &exec_dur,
                     float &pack_dur, cudaStream_t stream = NULL, int warmup = 10, int repeat = 100)
{
    auto w_pack_func = [&]() {
        if (W_BITS <= 32) {
            cudaError_t err = launch_pack(W, W_PACKED, N, K, W_BITS, stream);
            if (err != cudaSuccess) {
                printf("Line %d: 'weight launch_pack' failed: %s\n", __LINE__,
                       cudaGetErrorString(err));
            }
        } else {
            printf("unsupport W_BITS %d: for launch_pack func \n", W_BITS);
            return -1;
        }
        return 0;
    };

    auto x_pack_func = [&]() {
        if (X_BITS <= 32) {
            cudaError_t err = launch_pack(X, X_PACKED, M, K, X_BITS, stream);
            if (err != cudaSuccess) {
                printf("Line %d: 'activation launch_pack' failed: %s\n", __LINE__,
                       cudaGetErrorString(err));
            }
        } else {
            printf("unsupport X_BITS %d: for launch_pack func \n", X_BITS);
            return -1;
        }
        return 0;
    };

    
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "return due to previous error. ";
        return -1;
    }
    w_pack_func();
    x_pack_func();
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "return due to previous error. ";
        return -1;
    }
    OpStateType state = (*init_fn)(X_PACKED, W_PACKED, M, N, K, D, nullptr, false);
    if (!state.initSuccess) {
        std::cerr << "return due to unsuccessful initialization. " << std::endl;
        return -1;
    }
    (*exec_fn)(state, stream);
    cudaDeviceSynchronize();
    if (!isCudaSuccess(cudaGetLastError())) {
        std::cerr << "kernel failed." << std::endl;
        return -1;
    }

    // profling exec func
    CudaTimer exec_timer(stream);
    for (int i = 0; i < warmup + repeat; i++) {
        if (i == warmup)
            exec_timer.start();
        (*exec_fn)(state, stream);
    }
    exec_timer.stop();
    if (!isCudaSuccess(cudaGetLastError())) {
        std::cerr << "exec kernel failed." << std::endl;
        return -1;
    }
    exec_dur = exec_timer.elapsed_msecs() / repeat;

    // profling packing func
    CudaTimer packing_timer(stream);
    for (int i = 0; i < warmup + repeat; i++) {
        if (i == warmup)
            packing_timer.start();
        x_pack_func();
    }
    packing_timer.stop();
    if (!isCudaSuccess(cudaGetLastError())) {
        std::cerr << "packing kernel failed." << std::endl;
        return -1;
    }
    pack_dur = packing_timer.elapsed_msecs() / repeat;

    // accuracy comparison
    cudaMemcpy(H_OUT, D, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    if (!check(H_REF_OUT, H_OUT, M, N)) {
        return -2;
    }
    return 0;
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
        int ret = benchmark<AQ_INIT_FUN(AqBMMA), AQ_EXEC_FUN(AqBMMA), AQ_OP_STATE(AqBMMA)>( \
            AQ_NAME_FUN(AqBMMA, Init, X_BITS, W_BITS, SIGNED, BM, BN, BK, WM, WN, WK, MMA_M,  \
                        MMA_N, MMA_K, NSTAGE),                                                 \
            AQ_NAME_FUN(AqBMMA, Exec, X_BITS, W_BITS, SIGNED, BM, BN, BK, WM, WN, WK, MMA_M,  \
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
    int repeat = 10;
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
#ifdef _WIN32
    float min_latency = FLT_MAX;
#elif defined(__linux__)
    float min_latency = FLT_MAX;
#endif
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
    case 4:
        switch (w_bits) {
        case 4:
            TEST(4, 4, true, 8, 8, 128, 8, 32, 128, 8, 8, 128, 2);
            TEST(4, 4, true, 8, 8, 128, 8, 32, 128, 8, 8, 128, 3);
            TEST(4, 4, true, 8, 8, 128, 8, 32, 128, 8, 8, 128, 4);
            TEST(4, 4, true, 8, 8, 128, 16, 16, 128, 8, 8, 128, 2);
            TEST(4, 4, true, 8, 8, 128, 16, 16, 128, 8, 8, 128, 3);
            TEST(4, 4, true, 8, 8, 128, 16, 16, 128, 8, 8, 128, 4);
            TEST(4, 4, true, 8, 8, 128, 16, 32, 128, 8, 8, 128, 2);
            TEST(4, 4, true, 8, 8, 128, 16, 32, 128, 8, 8, 128, 3);
            TEST(4, 4, true, 8, 8, 128, 16, 32, 128, 8, 8, 128, 4);
            TEST(4, 4, true, 8, 8, 128, 32, 32, 128, 8, 8, 128, 2);
            TEST(4, 4, true, 8, 8, 128, 32, 32, 128, 8, 8, 128, 3);
            TEST(4, 4, true, 8, 8, 128, 32, 32, 128, 8, 8, 128, 4);
            TEST(4, 4, true, 8, 8, 256, 16, 16, 128, 8, 8, 128, 2);
            TEST(4, 4, true, 8, 8, 256, 16, 16, 128, 8, 8, 128, 3);
            TEST(4, 4, true, 8, 8, 256, 16, 16, 128, 8, 8, 128, 4);
            break;
        default:
            std::cout << "unsupport w_bits:" << w_bits << std::endl;
        }
        break;
    default:
        std::cout << "unsupport x_bits:" << x_bits << std::endl;
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