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
#include <cassert>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>

#if defined(_MSC_VER)
#define ASSEMBLY asm volatile
#else
#define ASSEMBLY asm volatile
#endif

#define CEIL(x, y) (((x) + (y) - 1) / (y))
#define ROUND_UP(x, y) ((CEIL((x), (y))) * (y))

// macro to declare a device-side function
#define DEVICE_INLINE __device__ __forceinline__

template <int M_, int N_, int K_> struct ShapeBase {
    static constexpr int M = M_, N = N_, K = K_;
};

template <int X_BITS_, int W_BITS_, bool SIGNED_> struct QuantType {
    static constexpr int X_BITS = X_BITS_, W_BITS = W_BITS_;
    static constexpr bool SIGNED = SIGNED_;
};

struct SwizzleIdentity {
    DEVICE_INLINE
    int operator()(int offset)
    {
        return offset;
    }
};

struct Swizzle8BWiseXor {
    DEVICE_INLINE
    int operator()(int offset)
    {
        return (offset ^ ((offset & (7 << 6)) >> 3));
    }
};

template <int NStage, bool UseMinSync> struct Pipeline;

template <int NStage> struct Pipeline<NStage, false> {
    DEVICE_INLINE
    void acquireWriter()
    {
    }
    DEVICE_INLINE
    void commitStage()
    {
        asm volatile("cp.async.commit_group;\n" ::);
    }
    DEVICE_INLINE
    void acquireReader()
    {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(NStage - 1));
        __syncthreads();
    }
    DEVICE_INLINE
    void releaseReader()
    {
        __syncthreads();
    }
};

template <int NStage> struct Pipeline<NStage, true> {
    int ahead_stage = 0;
    DEVICE_INLINE
    void acquireWriter()
    {
        if (ahead_stage == NStage - 1) {
            asm volatile("cp.async.wait_group %0;\n" ::"n"(NStage - 2));
            __syncthreads();
        }
    }
    DEVICE_INLINE
    void commitStage()
    {
        asm volatile("cp.async.commit_group;\n" ::);
        ahead_stage++;
    }
    DEVICE_INLINE
    void acquireReader()
    {
    }
    DEVICE_INLINE
    void releaseReader()
    {
        ahead_stage--;
    }
};

template <int N> struct HalfVector;

template <> struct HalfVector<8> {
    half x[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    DEVICE_INLINE
    void ld(const half *src)
    {
        *(int4 *)x = *(int4 *)src;
    }
    DEVICE_INLINE
    void st(half *dst)
    {
        *(int4 *)dst = *(int4 *)x;
    }
};

template <int N> struct IntVector;

template <> struct IntVector<4> {
    int x[4] = { 0, 0, 0, 0 };
    DEVICE_INLINE
    void ld(const int *src)
    {
        *(int4 *)x = *(int4 *)src;
    }
    DEVICE_INLINE
    void st(int *dst)
    {
        *(int4 *)dst = *(int4 *)x;
    }
    DEVICE_INLINE
    void reset()
    {
        x[0] = 0;
        x[1] = 0;
        x[2] = 0;
        x[3] = 0;
    }
};

////////
#define AQ_INIT_FUN(type) type##InitFn_t

#define AQ_EXEC_FUN(type) type##ExecFn_t

#define AQ_OP_STATE(type) type##OpState

#define AQ_NAME_FUN(type, fn, X_BITS, W_BITS, SIGNED, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, \
                    WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE)                                         \
    type##_##X_BITS##x##W_BITS##x##SIGNED##_##BLOCK_M##x##BLOCK_N##x##BLOCK_K##_##WARP_M##x##WARP_N##x##WARP_K##_##MMA_M##x##MMA_N##x##MMA_K##_##NSTAGE##_##fn##Fn

#define AQ_INSTANTIATE_FUN(type, X_BITS, W_BITS, SIGNED, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M,      \
                           WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE)                          \
    type##InitFn_t AQ_NAME_FUN(type, Init, X_BITS, W_BITS, SIGNED, BLOCK_M, BLOCK_N, BLOCK_K,    \
                               WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE) =            \
        type##InitFn<QuantType<X_BITS, W_BITS, SIGNED>, ShapeBase<BLOCK_M, BLOCK_N, BLOCK_K>,    \
                     ShapeBase<WARP_M, WARP_N, WARP_K>, ShapeBase<MMA_M, MMA_N, MMA_K>, NSTAGE>; \
    type##ExecFn_t AQ_NAME_FUN(type, Exec, X_BITS, W_BITS, SIGNED, BLOCK_M, BLOCK_N, BLOCK_K,    \
                               WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE) =            \
        type##ExecFn<QuantType<X_BITS, W_BITS, SIGNED>, ShapeBase<BLOCK_M, BLOCK_N, BLOCK_K>,    \
                     ShapeBase<WARP_M, WARP_N, WARP_K>, ShapeBase<MMA_M, MMA_N, MMA_K>, NSTAGE>;

#define AQ_DECL_FUN(type, X_BITS, W_BITS, SIGNED, BLOCK_M, BLOCK_N, BLOCK_K, WARP_M, WARP_N, \
                    WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE)                                     \
    extern type##InitFn_t AQ_NAME_FUN(type, Init, X_BITS, W_BITS, SIGNED, BLOCK_M, BLOCK_N,  \
                                      BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K,  \
                                      NSTAGE);                                               \
    extern type##ExecFn_t AQ_NAME_FUN(type, Exec, X_BITS, W_BITS, SIGNED, BLOCK_M, BLOCK_N,  \
                                      BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K,  \
                                      NSTAGE);
