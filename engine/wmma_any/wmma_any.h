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
