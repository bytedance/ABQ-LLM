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
#include "common/memory.h"

namespace aq_bmma
{

#if GPU_ARCH >= 75

template <typename Shape> struct fragment_a_rowmajor;
template <typename Shape> struct fragment_b_colmajor;
template <typename Shape, typename Accumulator> struct fragment_c;
template <bool trans, int num_reg, int nbit>
DEVICE_INLINE void ldmatrix(uint32_t *dst, const void *src);

// *** BMMA: 8x8x128 int32.b1 ***
template <> struct fragment_a_rowmajor<ShapeBase<8, 8, 128>> {
    uint32_t x;
};
template <> struct fragment_b_colmajor<ShapeBase<8, 8, 128>> {
    uint32_t x;
};
template <> struct fragment_c<ShapeBase<8, 8, 128>, int32_t> {
    int32_t x[2] = { 0 };
};
template <class F>
DEVICE_INLINE void loadMatrixSync(fragment_a_rowmajor<ShapeBase<8, 8, 128>> &a, const int *base,
                                  const int offset, const int ldm);
DEVICE_INLINE
void loadMatrixSync(fragment_a_rowmajor<ShapeBase<8, 8, 128>> &a, const int *base, const int offset,
                    const int ldm);
template <class F>
DEVICE_INLINE void loadMatrixSync(fragment_b_colmajor<ShapeBase<8, 8, 128>> &b, const int *base,
                                  const int offset, const int ldm);
DEVICE_INLINE
void loadMatrixSync(fragment_b_colmajor<ShapeBase<8, 8, 128>> &b, const int *base, const int offset,
                    const int ldm);
DEVICE_INLINE
void bmmaSync(fragment_c<ShapeBase<8, 8, 128>, int32_t> &d,
              const fragment_a_rowmajor<ShapeBase<8, 8, 128>> &a,
              const fragment_b_colmajor<ShapeBase<8, 8, 128>> &b,
              const fragment_c<ShapeBase<8, 8, 128>, int32_t> &c);
template <class F>
DEVICE_INLINE void storeMatrixSync(const fragment_c<ShapeBase<8, 8, 128>, int32_t> &c, int *base,
                                   const int offset, const int ldm);
DEVICE_INLINE
void storeMatrixSync(const fragment_c<ShapeBase<8, 8, 128>, int32_t> &c, int *base,
                     const int offset, const int ldm);

// *** implementation ***

// ldmatrix
// Unfortunately ldmatrix currently does not support the b1 data type
template <bool trans, int num_reg, int nbit>
DEVICE_INLINE void ldmatrix(uint32_t *dst, const void *src)
{
    // no f32 transpose is supported in current cuda
    // static_assert((!trans) || nbit==16);
    unsigned smem_ptr = getSmemPtr(src);
    uint32_t *x = dst;
    if (!trans) {
        if (num_reg == 4) {
            ASSEMBLY("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
                     : "r"(smem_ptr));
        } else if (num_reg == 2) {
            ASSEMBLY("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                     : "=r"(x[0]), "=r"(x[1])
                     : "r"(smem_ptr));
        } else if (num_reg == 1) {
            ASSEMBLY("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                     : "=r"(x[0])
                     : "r"(smem_ptr));
        } else
            assert(0);
    } else { // trans
        if (num_reg == 4) {
            ASSEMBLY("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
                     : "r"(smem_ptr));
        } else if (num_reg == 2) {
            ASSEMBLY("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                     : "=r"(x[0]), "=r"(x[1])
                     : "r"(smem_ptr));
        } else if (num_reg == 1) {
            ASSEMBLY("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                     : "=r"(x[0])
                     : "r"(smem_ptr));
        } else
            assert(0);
    }
}

// load a matrix [8, 128] rowmajor
// ldm counts with integer pointers
template <class F>
DEVICE_INLINE void loadMatrixSync(fragment_a_rowmajor<ShapeBase<8, 8, 128>> &a, const int *base,
                                  const int offset, const int ldm)
{
    int lane = threadIdx.x & 31;
    int row = lane >> 2;
    int col = lane % 4;
    F f;
    const int *src = base + f(offset + row * ldm + col);
    a.x = *(uint32_t *)src;
}
// ldm counts with integer pointers
DEVICE_INLINE
void loadMatrixSync(fragment_a_rowmajor<ShapeBase<8, 8, 128>> &a, const int *base, const int offset,
                    const int ldm)
{
    int lane = threadIdx.x & 31;
    int row = lane >> 2;
    int col = lane % 4; // 32 b1 = 1 int
    const int *src = base + offset + row * ldm + col;
    a.x = *(uint32_t *)src;
}

// load b matrix [128, 8] colmajor = [8, 128] rowmajor
// ldm counts with integer pointers
// base data [mma_N, mma_K] rowmajor = [mma_K, mma_N] colmajor
// So just follow the normal rowmajor thread allocation to read.
template <class F>
DEVICE_INLINE void loadMatrixSync(fragment_b_colmajor<ShapeBase<8, 8, 128>> &b, const int *base,
                                  const int offset, const int ldm)
{
    int lane = threadIdx.x & 31;
    int row = lane >> 2;
    int col = lane % 4;
    F f;
    const int *src = base + f(offset + row * ldm + col);
    b.x = *(uint32_t *)src;
}
// ldm counts with integer pointers
DEVICE_INLINE
void loadMatrixSync(fragment_b_colmajor<ShapeBase<8, 8, 128>> &b, const int *base, const int offset,
                    const int ldm)
{
    int lane = threadIdx.x & 31;
    int row = lane >> 2;
    int col = lane % 4; // 32 b1 = 1 int
    const int *src = base + offset + row * ldm + col;
    b.x = *(uint32_t *)src;
}

// a matrix [8, 128] rowmajor *  b matrix [128, 8] colmajor |  b matrix [8, 128] rowmajor
DEVICE_INLINE void bmmaSync(fragment_c<ShapeBase<8, 8, 128>, int32_t> &d,
                            const fragment_a_rowmajor<ShapeBase<8, 8, 128>> &a,
                            const fragment_b_colmajor<ShapeBase<8, 8, 128>> &b,
                            const fragment_c<ShapeBase<8, 8, 128>, int32_t> &c)
{
    ASSEMBLY("mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc"
             "{%0, %1}, {%2}, {%3}, {%4,%5};\n"
             : "=r"(d.x[0]), "=r"(d.x[1])
             : "r"(a.x), "r"(b.x), "r"(c.x[0]), "r"(c.x[1]));
}

template <class F>
DEVICE_INLINE void storeMatrixSync(const fragment_c<ShapeBase<8, 8, 128>, int32_t> &c, int *base,
                                   const int offset, const int ldm)
{
    int lane = threadIdx.x & 31;
    int row = lane >> 2;
    int col = (lane % 4) * 2; // Each thread holds two s32 type data
    int offset_ = offset + row * ldm + col;
    F f;
    *(base + f(offset_)) = c.x[0];
    *(base + f(offset_ + 1)) = c.x[1];
}

DEVICE_INLINE
void storeMatrixSync(const fragment_c<ShapeBase<8, 8, 128>, int32_t> &c, int *base,
                     const int offset, const int ldm)
{
    int lane = threadIdx.x & 31;
    int row = lane >> 2;
    int col = (lane % 4) * 2; // // Each thread holds two s32 type data
    int offset_ = offset + row * ldm + col;
    *(base + offset_) = c.x[0];
    *(base + offset_ + 1) = c.x[1];
}

// #else

//     assert(false && "bmma is not supported on this architecture( >= 75)\n");

#endif

} // namespace aq_bmma