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
#include <string>
#include <cuda_runtime.h>
#include "common/pack.h"
#include "common/timer.h"

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

inline bool isCudaSuccess(cudaError_t status)
{
    cudaError_t error = status;
    if (error != cudaSuccess) {
        std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    return true;
}

inline bool check(const int *ref_out, const int *out, int m, int n)
{
    for (int i = 0; i < m * n; ++i) {
        if (ref_out[i] != out[i]) {
            return false;
        }
    }
    return true;
}

/// benchmark func for bmma
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

void test_mma_w2a2(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream);
void test_mma_w2a4(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream);
void test_mma_w2a6(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream);
void test_mma_w2a8(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream);
void test_mma_w3a3(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream);
void test_mma_w3a8(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream);
void test_mma_w4a4(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream);
void test_mma_w4a8(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream);
void test_mma_w5a5(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream);
void test_mma_w6a6(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream);
void test_mma_w7a7(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream);
void test_mma_w8a8(int x_bits, int w_bits, int *d_x, int *d_w, int *d_x_pack, int *d_w_pack, int m,
                    int n, int k, int *d_out, int *h_out, int *h_ref_out, int warmup, int repeat,
                    bool quant_sign, cudaStream_t stream);