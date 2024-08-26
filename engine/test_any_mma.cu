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


#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include "common/base.h"
#include "common/pack.h"
#include "common/timer.h"
#include "test/test_mma/test_mma.h"

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

int main(int argc, char **argv)
{
    if (argc < 7) {
        printf("Usage: ./test_any_mma M N K X_BITS W_BITS SIGNED\n");
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

    cudaStream_t stream;
    cudaStreamCreate(&stream);

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
        case 2:
            test_mma_w2a2(x_bits, w_bits, d_x, d_w, d_x_pack, d_w_pack, m, n, k, d_out, h_out,
                           h_ref_out, warmup, repeat, quant_sign, stream);
            break;
        default:
            printf("unsupport w%da%d\n", w_bits, x_bits);
        }
        break;
    case 3:
        switch (w_bits) {
        case 3:
            test_mma_w3a3(x_bits, w_bits, d_x, d_w, d_x_pack, d_w_pack, m, n, k, d_out, h_out,
                           h_ref_out, warmup, repeat, quant_sign, stream);
            break;
        default:
            printf("unsupport w%da%d\n", w_bits, x_bits);
        }
        break;
    case 4:
        switch (w_bits) {
        case 2:
            test_mma_w2a4(x_bits, w_bits, d_x, d_w, d_x_pack, d_w_pack, m, n, k, d_out, h_out,
                           h_ref_out, warmup, repeat, quant_sign, stream);
            break;
        case 4:
            test_mma_w4a4(x_bits, w_bits, d_x, d_w, d_x_pack, d_w_pack, m, n, k, d_out, h_out,
                           h_ref_out, warmup, repeat, quant_sign, stream);
            break;
        default:
            printf("unsupport w%da%d\n", w_bits, x_bits);
        }
        break;
    case 5:
        switch (w_bits) {
        case 5:
            test_mma_w5a5(x_bits, w_bits, d_x, d_w, d_x_pack, d_w_pack, m, n, k, d_out, h_out,
                           h_ref_out, warmup, repeat, quant_sign, stream);
            break;
        default:
            printf("unsupport w%da%d\n", w_bits, x_bits);
        }
        break;
    case 6:
        switch (w_bits) {
        case 2:
            test_mma_w2a6(x_bits, w_bits, d_x, d_w, d_x_pack, d_w_pack, m, n, k, d_out, h_out,
                           h_ref_out, warmup, repeat, quant_sign, stream);
            break;
        case 6:
            test_mma_w6a6(x_bits, w_bits, d_x, d_w, d_x_pack, d_w_pack, m, n, k, d_out, h_out,
                           h_ref_out, warmup, repeat, quant_sign, stream);
            break;
        default:
            printf("unsupport w%da%d\n", w_bits, x_bits);
        }
        break;
    case 7:
        switch (w_bits) {

        case 7:
            test_mma_w7a7(x_bits, w_bits, d_x, d_w, d_x_pack, d_w_pack, m, n, k, d_out, h_out,
                           h_ref_out, warmup, repeat, quant_sign, stream);
            break;

        default:
            printf("unsupport w%da%d\n", w_bits, x_bits);
        }
        break;
    case 8:
        switch (w_bits) {

        case 2:
            test_mma_w2a8(x_bits, w_bits, d_x, d_w, d_x_pack, d_w_pack, m, n, k, d_out, h_out,
                           h_ref_out, warmup, repeat, quant_sign, stream);
            break;
        case 3:
            test_mma_w3a8(x_bits, w_bits, d_x, d_w, d_x_pack, d_w_pack, m, n, k, d_out, h_out,
                           h_ref_out, warmup, repeat, quant_sign, stream);
            break;
        case 4:
            test_mma_w4a8(x_bits, w_bits, d_x, d_w, d_x_pack, d_w_pack, m, n, k, d_out, h_out,
                           h_ref_out, warmup, repeat, quant_sign, stream);
            break;
        case 8:
            test_mma_w8a8(x_bits, w_bits, d_x, d_w, d_x_pack, d_w_pack, m, n, k, d_out, h_out,
                           h_ref_out, warmup, repeat, quant_sign, stream);
            break;

        default:
            printf("unsupport w%da%d\n", w_bits, x_bits);
        }
        break;
    default:
        printf("unsupport w%da%d\n", w_bits, x_bits);
    }

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