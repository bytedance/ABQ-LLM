#pragma once
#include "common/base.h"
#include "mma_any/aq_bmma_op.h"


////// W1A1 int
// cta<8,32,128> warp<8,16,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 1, 1, true, 8, 32, 128, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 1, 1, true, 8, 32, 128, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 1, 1, true, 8, 32, 128, 8, 16, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<8,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 1, 1, true, 8, 64, 128, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 1, 1, true, 8, 64, 128, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 1, 1, true, 8, 64, 128, 8, 32, 128, 8, 8, 128, 4);
// cta<8,128,128> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 1, 1, true, 8, 128, 128, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 1, 1, true, 8, 128, 128, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 1, 1, true, 8, 128, 128, 8, 32, 128, 8, 8, 128, 4);

////// W1A4 int
// cta<8,64,128> warp<32,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 4, 1, true, 8, 64, 128, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 4, 1, true, 8, 64, 128, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 4, 1, true, 8, 64, 128, 32, 32, 128, 8, 8, 128, 4);
// cta<8,128,128> warp<32,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 4, 1, true, 8, 128, 128, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 4, 1, true, 8, 128, 128, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 4, 1, true, 8, 128, 128, 32, 32, 128, 8, 8, 128, 4);

////// W1A8 int
// cta<8,64,128> warp<32,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 8, 1, true, 8, 64, 128, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 1, true, 8, 64, 128, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 1, true, 8, 64, 128, 32, 32, 128, 8, 8, 128, 4);
// cta<8,128,128> warp<64,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 1, true, 8, 128, 128, 64, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 1, true, 8, 128, 128, 64, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 1, true, 8, 128, 128, 64, 32, 128, 8, 8, 128, 4);

////// W1A16 int
// cta<8,32,128> warp<32,32,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBMMA, 16, 1, true, 8, 32, 128, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 16, 1, true, 8, 32, 128, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 16, 1, true, 8, 32, 128, 32, 32, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<32,32,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBMMA, 16, 1, true, 8, 32, 256, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 16, 1, true, 8, 32, 256, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 16, 1, true, 8, 32, 256, 32, 32, 128, 8, 8, 128, 4);

////// W2A2 int
// cta<8,64,128> warp<16,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 2, 2, true, 8, 64, 128, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 2, 2, true, 8, 64, 128, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 2, 2, true, 8, 64, 128, 16, 32, 128, 8, 8, 128, 4);

////// W2A4 int
// cta<8,64,128> warp<32,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 4, 2, true, 8, 64, 128, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 4, 2, true, 8, 64, 128, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 4, 2, true, 8, 64, 128, 32, 32, 128, 8, 8, 128, 4);

////// W2A6 int
// cta<8,64,128> warp<48,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 6, 2, true, 8, 64, 128, 48, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 6, 2, true, 8, 64, 128, 48, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 6, 2, true, 8, 64, 128, 48, 32, 128, 8, 8, 128, 4);

////// W2A8 int
// cta<8,32,128> warp<32,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 8, 2, true, 8, 32, 128, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 2, true, 8, 32, 128, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 2, true, 8, 32, 128, 32, 32, 128, 8, 8, 128, 4);

////// W2A16 int
// cta<8,32,128> warp<64,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 16, 2, true, 8, 32, 128, 64, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 16, 2, true, 8, 32, 128, 64, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 16, 2, true, 8, 32, 128, 64, 32, 128, 8, 8, 128, 4);

////// W4A4 int
// cta<8,8,128> warp<8,32,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBMMA, 4, 4, true, 8, 8, 128, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 4, 4, true, 8, 8, 128, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 4, 4, true, 8, 8, 128, 8, 32, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<16,16,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 4, 4, true, 8, 8, 128, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 4, 4, true, 8, 8, 128, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 4, 4, true, 8, 8, 128, 16, 16, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<16,32,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBMMA, 4, 4, true, 8, 8, 128, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 4, 4, true, 8, 8, 128, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 4, 4, true, 8, 8, 128, 16, 32, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<32,32,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBMMA, 4, 4, true, 8, 8, 128, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 4, 4, true, 8, 8, 128, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 4, 4, true, 8, 8, 128, 32, 32, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<16,16,128> mma<8,8,128>   WARPS[2x2] split_k = 2
AQ_DECL_FUN(AqBMMA, 4, 4, true, 8, 8, 256, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 4, 4, true, 8, 8, 256, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 4, 4, true, 8, 8, 256, 16, 16, 128, 8, 8, 128, 4);

// cta<16,16,128> warp<16,32,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBMMA, 4, 4, true, 16, 16, 128, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 4, 4, true, 16, 16, 128, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 4, 4, true, 16, 16, 128, 16, 32, 128, 8, 8, 128, 4);
// cta<16,16,128> warp<16,64,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBMMA, 4, 4, true, 16, 16, 128, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 4, 4, true, 16, 16, 128, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 4, 4, true, 16, 16, 128, 16, 64, 128, 8, 8, 128, 4);
// cta<16,16,128> warp<32,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 4, 4, true, 16, 16, 128, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 4, 4, true, 16, 16, 128, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 4, 4, true, 16, 16, 128, 32, 32, 128, 8, 8, 128, 4);
// cta<16,16,256> warp<16,32,128> mma<8,8,128>   WARPS[4x2] split_k = 2
AQ_DECL_FUN(AqBMMA, 4, 4, true, 16, 16, 256, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 4, 4, true, 16, 16, 256, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 4, 4, true, 16, 16, 256, 16, 32, 128, 8, 8, 128, 4);

// W4A4 uint
// cta<8,8,128> warp<8,32,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBMMA, 4, 4, false, 8, 8, 128, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 4, 4, false, 8, 8, 128, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 4, 4, false, 8, 8, 128, 8, 32, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<16,16,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 4, 4, false, 8, 8, 128, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 4, 4, false, 8, 8, 128, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 4, 4, false, 8, 8, 128, 16, 16, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<16,32,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBMMA, 4, 4, false, 8, 8, 128, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 4, 4, false, 8, 8, 128, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 4, 4, false, 8, 8, 128, 16, 32, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<32,32,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBMMA, 4, 4, false, 8, 8, 128, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 4, 4, false, 8, 8, 128, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 4, 4, false, 8, 8, 128, 32, 32, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<16,16,128> mma<8,8,128>   WARPS[2x2] split_k = 2
AQ_DECL_FUN(AqBMMA, 4, 4, false, 8, 8, 256, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 4, 4, false, 8, 8, 256, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 4, 4, false, 8, 8, 256, 16, 16, 128, 8, 8, 128, 4);

// cta<16,16,128> warp<16,32,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBMMA, 4, 4, false, 16, 16, 128, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 4, 4, false, 16, 16, 128, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 4, 4, false, 16, 16, 128, 16, 32, 128, 8, 8, 128, 4);
// cta<16,16,128> warp<16,64,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBMMA, 4, 4, false, 16, 16, 128, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 4, 4, false, 16, 16, 128, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 4, 4, false, 16, 16, 128, 16, 64, 128, 8, 8, 128, 4);
// cta<16,16,128> warp<32,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 4, 4, false, 16, 16, 128, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 4, 4, false, 16, 16, 128, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 4, 4, false, 16, 16, 128, 32, 32, 128, 8, 8, 128, 4);
// cta<16,16,256> warp<16,32,128> mma<8,8,128>   WARPS[4x2] split_k = 2
AQ_DECL_FUN(AqBMMA, 4, 4, false, 16, 16, 256, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 4, 4, false, 16, 16, 256, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 4, 4, false, 16, 16, 256, 16, 32, 128, 8, 8, 128, 4);

// W4A8 int TODO:@liusongwei or xieyusheng
