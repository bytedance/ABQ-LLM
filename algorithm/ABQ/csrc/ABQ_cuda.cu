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

#include <torch/all.h>
#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "common/base.h"
#include "mma_any/aq_wmma_library.h"
#include "mma_any/aq_wmma_op.h"

void mul_w2a8(
    const torch::Tensor &A,
    const torch::Tensor &B,
    torch::Tensor &C,
    int prob_m,
    int prob_n,
    int prob_k,
    int sms = -1,
    int max_par = 8)
{

  int dev = A.get_device();
  if (prob_k < 128)
  {
    AT_ERROR("Line %d: 'abqgemm_cuda' failed: prob_k < 128!\n", __LINE__);
    return;
  }
  int *X = (int*)A.data_ptr();
  int *W = (int*)B.data_ptr();
  int* D = (int*)C.data_ptr();
  AqBWMMAOpState state = AqBWMMA_8x2xtrue_2x64x256_16x64x128_8x8x128_4_InitFn(X, W, prob_m, prob_n, prob_k, D, nullptr, false);
  if (!state.initSuccess) {
    AT_ERROR("Line %d: 'abqgemm_cuda' failed due to unsuccessful initialization\n", __LINE__);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("mul_w2a8", &mul_w2a8, "W2A8 matmul.");
}