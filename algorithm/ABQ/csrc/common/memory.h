

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

// template<int TileSize, int Nthreads> DEVICE_INLINE
// void cpAsyncTile(int* smem_ptr, int* gmem_ptr, const int valid_size) {
//     // assume zfill_size % 4 == 0;
//     for (int i = threadIdx.x; i < ROUND_UP(TileSize, Nthreads); i += Nthreads) {
//         bool valid = i < TileSize;
//         int src_in_bytes = (i < valid_size) ? 4 : 0;
//         unsigned dst = get_smem_ptr(smem_ptr + i);
//         const void *src = gmem_ptr + i;
//         ASSEMBLY (
//             "{\n"
//             "  .reg .pred p;\n"
//             "  setp.ne.b32 p, %0, 0;\n"
//             "  @p cp.async.ca.shared.global [%1], [%2], %3, %4;\n"
//             "}\n" ::"r"((int)valid),
//             "r"(dst), "l"(src), "n"(4), "r"(src_in_bytes)
//         );
//     }
// }