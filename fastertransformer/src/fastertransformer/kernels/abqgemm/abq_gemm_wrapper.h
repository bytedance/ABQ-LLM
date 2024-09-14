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
