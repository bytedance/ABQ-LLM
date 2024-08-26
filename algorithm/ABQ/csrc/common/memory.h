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
#include "common/base.h"

// device function to convert shared memory address into unsigned format
DEVICE_INLINE unsigned getSmemPtr(const void *ptr)
{
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

DEVICE_INLINE
void copyAndSync(unsigned *dst, const unsigned *src, int size)
{
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        dst[i] = src[i];
    }
    __syncthreads();
}

template <int SizeInBytes>
DEVICE_INLINE void cpAsyncPredZfill(void *smem_ptr, void const *gmem_ptr,
                                    const bool pred_guard = true, const bool zfill = false)
{
    unsigned smem_int_ptr = getSmemPtr(smem_ptr);
    int src_in_bytes = (zfill ? 0 : SizeInBytes);
    ASSEMBLY("{\n"
             "  .reg .pred p;\n"
             "  setp.ne.b32 p, %0, 0;\n"
             "  @p cp.async.cg.shared.global [%1], [%2], %3, %4;\n"
             "}\n" ::"r"((int)pred_guard),
             "r"(smem_int_ptr), "l"(gmem_ptr), "n"(SizeInBytes), "r"(src_in_bytes));
}