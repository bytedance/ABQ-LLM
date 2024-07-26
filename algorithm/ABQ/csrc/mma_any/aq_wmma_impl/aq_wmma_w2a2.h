#pragma once
#include "common/base.h"
#include "mma_any/aq_wmma_op.h"

#ifdef W2A2
////// W2A2 int
// cta<4,32,128> warp<8,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 128, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 128, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 128, 8, 32, 128, 8, 8, 128, 4);
// cta<4,64,128> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 128, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 128, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 128, 8, 64, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<16,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 128, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 128, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 128, 16, 32, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<16,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 128, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 128, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 128, 16, 64, 128, 8, 8, 128, 4);
// cta<12,32,128> warp<24,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 128, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 128, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 128, 24, 32, 128, 8, 8, 128, 4);
// cta<12,64,128> warp<24,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 128, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 128, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 128, 24, 64, 128, 8, 8, 128, 4);
// cta<4,32,128> warp<8,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 128, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 128, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 128, 8, 16, 128, 8, 8, 128, 4);
// cta<4,64,128> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 128, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 128, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 128, 8, 32, 128, 8, 8, 128, 4);
// cta<4,96,128> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 96, 128, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 96, 128, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 96, 128, 8, 48, 128, 8, 8, 128, 4);
// cta<4,128,128> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 128, 128, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 128, 128, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 128, 128, 8, 64, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<16,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 128, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 128, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 128, 16, 16, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<16,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 128, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 128, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 128, 16, 32, 128, 8, 8, 128, 4);
// cta<8,96,128> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 128, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 128, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 128, 16, 48, 128, 8, 8, 128, 4);
// cta<8,128,128> warp<16,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 128, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 128, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 128, 16, 64, 128, 8, 8, 128, 4);
// cta<12,32,128> warp<24,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 128, 24, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 128, 24, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 128, 24, 16, 128, 8, 8, 128, 4);
// cta<12,64,128> warp<24,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 128, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 128, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 128, 24, 32, 128, 8, 8, 128, 4);
// cta<12,96,128> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 96, 128, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 96, 128, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 96, 128, 24, 48, 128, 8, 8, 128, 4);
// cta<12,128,128> warp<24,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 128, 128, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 128, 128, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 128, 128, 24, 64, 128, 8, 8, 128, 4);
// cta<4,32,128> warp<8,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 128, 8, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 128, 8, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 128, 8, 8, 128, 8, 8, 128, 4);
// cta<4,64,128> warp<8,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 128, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 128, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 128, 8, 16, 128, 8, 8, 128, 4);
// cta<4,96,128> warp<8,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 96, 128, 8, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 96, 128, 8, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 96, 128, 8, 24, 128, 8, 8, 128, 4);
// cta<4,128,128> warp<8,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 128, 128, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 128, 128, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 128, 128, 8, 32, 128, 8, 8, 128, 4);
// cta<4,256,128> warp<8,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 256, 128, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 256, 128, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 256, 128, 8, 64, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<16,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 128, 16, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 128, 16, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 128, 16, 8, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<16,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 128, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 128, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 128, 16, 16, 128, 8, 8, 128, 4);
// cta<8,96,128> warp<16,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 128, 16, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 128, 16, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 128, 16, 24, 128, 8, 8, 128, 4);
// cta<8,128,128> warp<16,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 128, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 128, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 128, 16, 32, 128, 8, 8, 128, 4);
// cta<8,256,128> warp<16,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 256, 128, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 256, 128, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 256, 128, 16, 64, 128, 8, 8, 128, 4);
// cta<12,32,128> warp<24,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 128, 24, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 128, 24, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 128, 24, 8, 128, 8, 8, 128, 4);
// cta<12,64,128> warp<24,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 128, 24, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 128, 24, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 128, 24, 16, 128, 8, 8, 128, 4);
// cta<12,96,128> warp<24,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 96, 128, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 96, 128, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 96, 128, 24, 24, 128, 8, 8, 128, 4);
// cta<12,128,128> warp<24,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 128, 128, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 128, 128, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 128, 128, 24, 32, 128, 8, 8, 128, 4);
// cta<12,256,128> warp<24,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 256, 128, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 256, 128, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 256, 128, 24, 64, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<8,64,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 128, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 128, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 128, 8, 64, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<8,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 128, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 128, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 128, 8, 32, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<8,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 128, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 128, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 128, 8, 64, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<8,16,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 128, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 128, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 128, 8, 16, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<8,32,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 128, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 128, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 128, 8, 32, 128, 8, 8, 128, 4);
// cta<8,96,128> warp<8,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 128, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 128, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 128, 8, 48, 128, 8, 8, 128, 4);
// cta<8,128,128> warp<8,64,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 128, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 128, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 128, 8, 64, 128, 8, 8, 128, 4);
// cta<4,32,256> warp<8,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 256, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 256, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 256, 8, 32, 128, 8, 8, 128, 4);
// cta<4,64,256> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 256, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 256, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 256, 8, 64, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<16,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 256, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 256, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 256, 16, 32, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<16,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 256, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 256, 16, 64, 128, 8, 8, 128, 4);
// cta<12,32,256> warp<24,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 256, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 256, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 256, 24, 32, 128, 8, 8, 128, 4);
// cta<12,64,256> warp<24,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 256, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 256, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 256, 24, 64, 128, 8, 8, 128, 4);
// cta<4,32,256> warp<8,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 256, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 256, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 256, 8, 16, 128, 8, 8, 128, 4);
// cta<4,64,256> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 256, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 256, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 256, 8, 32, 128, 8, 8, 128, 4);
// cta<4,96,256> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 96, 256, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 96, 256, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 96, 256, 8, 48, 128, 8, 8, 128, 4);
// cta<4,128,256> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 128, 256, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 128, 256, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 128, 256, 8, 64, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<16,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 256, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 256, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 256, 16, 16, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<16,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 256, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 256, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 256, 16, 32, 128, 8, 8, 128, 4);
// cta<8,96,256> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 256, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 256, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 256, 16, 48, 128, 8, 8, 128, 4);
// cta<8,128,256> warp<16,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 256, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 256, 16, 64, 128, 8, 8, 128, 4);
// cta<12,32,256> warp<24,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 256, 24, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 256, 24, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 256, 24, 16, 128, 8, 8, 128, 4);
// cta<12,64,256> warp<24,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 256, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 256, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 256, 24, 32, 128, 8, 8, 128, 4);
// cta<12,96,256> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 96, 256, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 96, 256, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 96, 256, 24, 48, 128, 8, 8, 128, 4);
// cta<12,128,256> warp<24,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 128, 256, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 128, 256, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 128, 256, 24, 64, 128, 8, 8, 128, 4);
// cta<4,32,256> warp<8,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 256, 8, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 256, 8, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 256, 8, 8, 128, 8, 8, 128, 4);
// cta<4,64,256> warp<8,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 256, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 256, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 256, 8, 16, 128, 8, 8, 128, 4);
// cta<4,96,256> warp<8,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 96, 256, 8, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 96, 256, 8, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 96, 256, 8, 24, 128, 8, 8, 128, 4);
// cta<4,128,256> warp<8,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 128, 256, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 128, 256, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 128, 256, 8, 32, 128, 8, 8, 128, 4);
// cta<4,256,256> warp<8,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 256, 256, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 256, 256, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 256, 256, 8, 64, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<16,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 256, 16, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 256, 16, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 256, 16, 8, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<16,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 256, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 256, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 256, 16, 16, 128, 8, 8, 128, 4);
// cta<8,96,256> warp<16,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 256, 16, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 256, 16, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 256, 16, 24, 128, 8, 8, 128, 4);
// cta<8,128,256> warp<16,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 256, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 256, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 256, 16, 32, 128, 8, 8, 128, 4);
// cta<8,256,256> warp<16,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 256, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 256, 256, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 256, 256, 16, 64, 128, 8, 8, 128, 4);
// cta<12,32,256> warp<24,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 256, 24, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 256, 24, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 256, 24, 8, 128, 8, 8, 128, 4);
// cta<12,64,256> warp<24,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 256, 24, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 256, 24, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 256, 24, 16, 128, 8, 8, 128, 4);
// cta<12,96,256> warp<24,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 96, 256, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 96, 256, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 96, 256, 24, 24, 128, 8, 8, 128, 4);
// cta<12,128,256> warp<24,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 128, 256, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 128, 256, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 128, 256, 24, 32, 128, 8, 8, 128, 4);
// cta<12,256,256> warp<24,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 256, 256, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 256, 256, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 256, 256, 24, 64, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<8,64,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 256, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 256, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 256, 8, 64, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<8,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 256, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 256, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 256, 8, 32, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<8,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 256, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 256, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 256, 8, 64, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<8,16,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 256, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 256, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 256, 8, 16, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<8,32,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 256, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 256, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 256, 8, 32, 128, 8, 8, 128, 4);
// cta<8,96,256> warp<8,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 256, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 256, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 256, 8, 48, 128, 8, 8, 128, 4);
// cta<8,128,256> warp<8,64,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 256, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 256, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 256, 8, 64, 128, 8, 8, 128, 4);
// cta<4,32,384> warp<8,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 384, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 384, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 384, 8, 32, 128, 8, 8, 128, 4);
// cta<4,64,384> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 384, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 384, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 384, 8, 64, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<16,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 384, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 384, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 384, 16, 32, 128, 8, 8, 128, 4);
// cta<8,64,384> warp<16,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 384, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 384, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 384, 16, 64, 128, 8, 8, 128, 4);
// cta<12,32,384> warp<24,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 384, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 384, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 384, 24, 32, 128, 8, 8, 128, 4);
// cta<12,64,384> warp<24,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 384, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 384, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 384, 24, 64, 128, 8, 8, 128, 4);
// cta<4,32,384> warp<8,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 384, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 384, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 384, 8, 16, 128, 8, 8, 128, 4);
// cta<4,64,384> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 384, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 384, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 384, 8, 32, 128, 8, 8, 128, 4);
// cta<4,96,384> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 96, 384, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 96, 384, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 96, 384, 8, 48, 128, 8, 8, 128, 4);
// cta<4,128,384> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 128, 384, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 128, 384, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 128, 384, 8, 64, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<16,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 384, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 384, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 384, 16, 16, 128, 8, 8, 128, 4);
// cta<8,64,384> warp<16,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 384, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 384, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 384, 16, 32, 128, 8, 8, 128, 4);
// cta<8,96,384> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 384, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 384, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 384, 16, 48, 128, 8, 8, 128, 4);
// cta<8,128,384> warp<16,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 384, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 384, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 384, 16, 64, 128, 8, 8, 128, 4);
// cta<12,32,384> warp<24,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 384, 24, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 384, 24, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 384, 24, 16, 128, 8, 8, 128, 4);
// cta<12,64,384> warp<24,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 384, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 384, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 384, 24, 32, 128, 8, 8, 128, 4);
// cta<12,96,384> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 96, 384, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 96, 384, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 96, 384, 24, 48, 128, 8, 8, 128, 4);
// cta<12,128,384> warp<24,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 128, 384, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 128, 384, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 128, 384, 24, 64, 128, 8, 8, 128, 4);
// cta<4,32,384> warp<8,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 384, 8, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 384, 8, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 384, 8, 8, 128, 8, 8, 128, 4);
// cta<4,64,384> warp<8,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 384, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 384, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 384, 8, 16, 128, 8, 8, 128, 4);
// cta<4,96,384> warp<8,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 96, 384, 8, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 96, 384, 8, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 96, 384, 8, 24, 128, 8, 8, 128, 4);
// cta<4,128,384> warp<8,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 128, 384, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 128, 384, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 128, 384, 8, 32, 128, 8, 8, 128, 4);
// cta<4,256,384> warp<8,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 256, 384, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 256, 384, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 256, 384, 8, 64, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<16,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 384, 16, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 384, 16, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 384, 16, 8, 128, 8, 8, 128, 4);
// cta<8,64,384> warp<16,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 384, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 384, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 384, 16, 16, 128, 8, 8, 128, 4);
// cta<8,96,384> warp<16,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 384, 16, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 384, 16, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 384, 16, 24, 128, 8, 8, 128, 4);
// cta<8,128,384> warp<16,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 384, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 384, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 384, 16, 32, 128, 8, 8, 128, 4);
// cta<8,256,384> warp<16,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 256, 384, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 256, 384, 16, 64, 128, 8, 8, 128, 3);
// cta<12,32,384> warp<24,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 384, 24, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 384, 24, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 384, 24, 8, 128, 8, 8, 128, 4);
// cta<12,64,384> warp<24,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 384, 24, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 384, 24, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 384, 24, 16, 128, 8, 8, 128, 4);
// cta<12,96,384> warp<24,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 96, 384, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 96, 384, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 96, 384, 24, 24, 128, 8, 8, 128, 4);
// cta<12,128,384> warp<24,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 128, 384, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 128, 384, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 128, 384, 24, 32, 128, 8, 8, 128, 4);
// cta<12,256,384> warp<24,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 256, 384, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 256, 384, 24, 64, 128, 8, 8, 128, 3);
// cta<8,32,384> warp<8,64,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 384, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 384, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 384, 8, 64, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<8,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 384, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 384, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 384, 8, 32, 128, 8, 8, 128, 4);
// cta<8,64,384> warp<8,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 384, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 384, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 384, 8, 64, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<8,16,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 384, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 384, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 384, 8, 16, 128, 8, 8, 128, 4);
// cta<8,64,384> warp<8,32,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 384, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 384, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 384, 8, 32, 128, 8, 8, 128, 4);
// cta<8,96,384> warp<8,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 384, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 384, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 384, 8, 48, 128, 8, 8, 128, 4);
// cta<8,128,384> warp<8,64,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 384, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 384, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 384, 8, 64, 128, 8, 8, 128, 4);
// cta<4,32,512> warp<8,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 512, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 512, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 512, 8, 32, 128, 8, 8, 128, 4);
// cta<4,64,512> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 512, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 512, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 512, 8, 64, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<16,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 512, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 512, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 512, 16, 32, 128, 8, 8, 128, 4);
// cta<8,64,512> warp<16,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 512, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 512, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 512, 16, 64, 128, 8, 8, 128, 4);
// cta<12,32,512> warp<24,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 512, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 512, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 512, 24, 32, 128, 8, 8, 128, 4);
// cta<12,64,512> warp<24,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 512, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 512, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 512, 24, 64, 128, 8, 8, 128, 4);
// cta<4,32,512> warp<8,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 512, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 512, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 512, 8, 16, 128, 8, 8, 128, 4);
// cta<4,64,512> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 512, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 512, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 512, 8, 32, 128, 8, 8, 128, 4);
// cta<4,96,512> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 96, 512, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 96, 512, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 96, 512, 8, 48, 128, 8, 8, 128, 4);
// cta<4,128,512> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 128, 512, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 128, 512, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 128, 512, 8, 64, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<16,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 512, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 512, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 512, 16, 16, 128, 8, 8, 128, 4);
// cta<8,64,512> warp<16,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 512, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 512, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 512, 16, 32, 128, 8, 8, 128, 4);
// cta<8,96,512> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 512, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 512, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 512, 16, 48, 128, 8, 8, 128, 4);
// cta<8,128,512> warp<16,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 512, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 512, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 512, 16, 64, 128, 8, 8, 128, 4);
// cta<12,32,512> warp<24,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 512, 24, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 512, 24, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 512, 24, 16, 128, 8, 8, 128, 4);
// cta<12,64,512> warp<24,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 512, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 512, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 512, 24, 32, 128, 8, 8, 128, 4);
// cta<12,96,512> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 96, 512, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 96, 512, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 96, 512, 24, 48, 128, 8, 8, 128, 4);
// cta<12,128,512> warp<24,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 128, 512, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 128, 512, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 128, 512, 24, 64, 128, 8, 8, 128, 4);
// cta<4,32,512> warp<8,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 512, 8, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 512, 8, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 32, 512, 8, 8, 128, 8, 8, 128, 4);
// cta<4,64,512> warp<8,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 512, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 512, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 64, 512, 8, 16, 128, 8, 8, 128, 4);
// cta<4,96,512> warp<8,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 96, 512, 8, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 96, 512, 8, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 96, 512, 8, 24, 128, 8, 8, 128, 4);
// cta<4,128,512> warp<8,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 128, 512, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 128, 512, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 128, 512, 8, 32, 128, 8, 8, 128, 4);
// cta<4,256,512> warp<8,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 256, 512, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 4, 256, 512, 8, 64, 128, 8, 8, 128, 3);
// cta<8,32,512> warp<16,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 512, 16, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 512, 16, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 512, 16, 8, 128, 8, 8, 128, 4);
// cta<8,64,512> warp<16,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 512, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 512, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 512, 16, 16, 128, 8, 8, 128, 4);
// cta<8,96,512> warp<16,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 512, 16, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 512, 16, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 512, 16, 24, 128, 8, 8, 128, 4);
// cta<8,128,512> warp<16,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 512, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 512, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 512, 16, 32, 128, 8, 8, 128, 4);
// cta<8,256,512> warp<16,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 256, 512, 16, 64, 128, 8, 8, 128, 2);
// cta<12,32,512> warp<24,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 512, 24, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 512, 24, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 32, 512, 24, 8, 128, 8, 8, 128, 4);
// cta<12,64,512> warp<24,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 512, 24, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 512, 24, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 64, 512, 24, 16, 128, 8, 8, 128, 4);
// cta<12,96,512> warp<24,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 96, 512, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 96, 512, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 96, 512, 24, 24, 128, 8, 8, 128, 4);
// cta<12,128,512> warp<24,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 128, 512, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 128, 512, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 128, 512, 24, 32, 128, 8, 8, 128, 4);
// cta<12,256,512> warp<24,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 12, 256, 512, 24, 64, 128, 8, 8, 128, 2);
// cta<8,32,512> warp<8,64,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 512, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 512, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 512, 8, 64, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<8,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 512, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 512, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 512, 8, 32, 128, 8, 8, 128, 4);
// cta<8,64,512> warp<8,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 512, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 512, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 512, 8, 64, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<8,16,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 512, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 512, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 32, 512, 8, 16, 128, 8, 8, 128, 4);
// cta<8,64,512> warp<8,32,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 512, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 512, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 64, 512, 8, 32, 128, 8, 8, 128, 4);
// cta<8,96,512> warp<8,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 512, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 512, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 96, 512, 8, 48, 128, 8, 8, 128, 4);
// cta<8,128,512> warp<8,64,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 512, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 512, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 2, 2, true, 8, 128, 512, 8, 64, 128, 8, 8, 128, 4);
#endif