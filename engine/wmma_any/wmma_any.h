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

#pragma once
#include "common/base.h"

struct AnyQuantParams {
    // Input and output buffer
    const int *X;
    const int *W;
    const half *C;
    int *D;
    bool bias = false;
    // actual matrix shape
    const int M;
    const int N;
    const int K;
};

template <
    // quant info
    typename QuantType,
    // tiling shapes
    typename ThreadBlockShape, typename WarpShape, typename MmaShape,
    // threadblock level pipeline stage
    int kThreadBlockStage>
cudaError_t launchWmmaAnyKernel(const AnyQuantParams &params, const cudaStream_t &stream);
