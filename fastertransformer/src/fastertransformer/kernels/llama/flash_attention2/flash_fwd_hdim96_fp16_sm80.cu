//
// Created by ByteDance on 2023/11/29.
//
#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_<cutlass::half_t, 96>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim96<cutlass::half_t>(params, stream);
}