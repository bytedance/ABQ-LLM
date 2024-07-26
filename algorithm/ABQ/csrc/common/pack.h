#pragma once
#include "common/base.h"

// pack 4 bit
cudaError_t launch_pack4(const int *in_data, int *pack_data, int m, int k, int BIT,
                         cudaStream_t stream = 0);

cudaError_t launch_pack(const int *in_data, int *pack_data, int m, int k, int BIT,
                        cudaStream_t stream = 0);