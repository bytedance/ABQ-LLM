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

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

class ABQGEMMWrapper {
private:
    int           x_bits_;
    int           w_bits_;
    bool          signed_;

public:
    ABQGEMMWrapper(int X_BITS, int W_BITS, bool SIGNED);
    ~ABQGEMMWrapper();

    void pack(const int* in_data, int* packed_data, int M, int K, int BIT, cudaStream_t stream);
    void pack(const half* in_data, const float* scale, int* packed_data, int M, int K, int BIT, cudaStream_t stream);
    void gemm(const int    M,
              const int    N,
              const int    K,
              const half*   A,
              const int*   B,
              const half*  C,
              half*        D,
              const float* scale,
              const float* scale_inter,
              const float* scale_out,
              bool         bias,
              char*        abq_gemm_workspace,
              size_t       abq_gemm_ws_bytes,
              cudaStream_t stream);
};
