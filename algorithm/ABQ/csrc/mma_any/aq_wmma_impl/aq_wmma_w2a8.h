#pragma once
#include "common/base.h"
#include "mma_any/aq_wmma_op.h"

AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 64, 256, 16, 64, 128, 8, 8, 128, 4);

#ifdef W2A8
////// W2A8 int
// cta<2,32,256> warp<16,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 32, 256, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 32, 256, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 32, 256, 16, 32, 128, 8, 8, 128, 4);
// cta<2,64,256> warp<16,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 64, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 64, 256, 16, 64, 128, 8, 8, 128, 3);
// cta<4,32,256> warp<32,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 256, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 256, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 256, 32, 32, 128, 8, 8, 128, 4);
// cta<4,64,256> warp<32,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 256, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 256, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 256, 32, 64, 128, 8, 8, 128, 4);
// cta<6,32,256> warp<48,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 32, 256, 48, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 32, 256, 48, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 32, 256, 48, 32, 128, 8, 8, 128, 4);
// cta<6,64,256> warp<48,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 64, 256, 48, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 64, 256, 48, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 64, 256, 48, 64, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<64,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 256, 64, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 256, 64, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 256, 64, 32, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<64,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 256, 64, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 256, 64, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 256, 64, 64, 128, 8, 8, 128, 4);
// cta<2,32,256> warp<16,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 32, 256, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 32, 256, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 32, 256, 16, 16, 128, 8, 8, 128, 4);
// cta<2,64,256> warp<16,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 64, 256, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 64, 256, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 64, 256, 16, 32, 128, 8, 8, 128, 4);
// cta<2,96,256> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 96, 256, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 96, 256, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 96, 256, 16, 48, 128, 8, 8, 128, 4);
// cta<2,128,256> warp<16,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 128, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 128, 256, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 128, 256, 16, 64, 128, 8, 8, 128, 4);
// cta<4,32,256> warp<32,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 256, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 256, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 256, 32, 16, 128, 8, 8, 128, 4);
// cta<4,64,256> warp<32,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 256, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 256, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 256, 32, 32, 128, 8, 8, 128, 4);
// cta<4,96,256> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 96, 256, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 96, 256, 32, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 96, 256, 32, 48, 128, 8, 8, 128, 4);
// cta<4,128,256> warp<32,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 128, 256, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 128, 256, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 128, 256, 32, 64, 128, 8, 8, 128, 4);
// cta<6,32,256> warp<48,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 32, 256, 48, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 32, 256, 48, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 32, 256, 48, 16, 128, 8, 8, 128, 4);
// cta<6,64,256> warp<48,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 64, 256, 48, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 64, 256, 48, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 64, 256, 48, 32, 128, 8, 8, 128, 4);
// cta<6,96,256> warp<48,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 96, 256, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 96, 256, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 96, 256, 48, 48, 128, 8, 8, 128, 4);
// cta<6,128,256> warp<48,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 128, 256, 48, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 128, 256, 48, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 128, 256, 48, 64, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<64,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 256, 64, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 256, 64, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 256, 64, 16, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<64,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 256, 64, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 256, 64, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 256, 64, 32, 128, 8, 8, 128, 4);
// cta<8,96,256> warp<64,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 96, 256, 64, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 96, 256, 64, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 96, 256, 64, 48, 128, 8, 8, 128, 4);
// cta<8,128,256> warp<64,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 128, 256, 64, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 128, 256, 64, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 128, 256, 64, 64, 128, 8, 8, 128, 4);
// cta<2,64,256> warp<16,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 64, 256, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 64, 256, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 64, 256, 16, 16, 128, 8, 8, 128, 4);
// cta<2,128,256> warp<16,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 128, 256, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 128, 256, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 128, 256, 16, 32, 128, 8, 8, 128, 4);
// cta<2,256,256> warp<16,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 256, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 256, 256, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 256, 256, 16, 64, 128, 8, 8, 128, 4);
// cta<4,64,256> warp<32,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 256, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 256, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 256, 32, 16, 128, 8, 8, 128, 4);
// cta<4,128,256> warp<32,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 128, 256, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 128, 256, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 128, 256, 32, 32, 128, 8, 8, 128, 4);
// cta<4,256,256> warp<32,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 256, 256, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 256, 256, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 256, 256, 32, 64, 128, 8, 8, 128, 4);
// cta<6,64,256> warp<48,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 64, 256, 48, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 64, 256, 48, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 64, 256, 48, 16, 128, 8, 8, 128, 4);
// cta<6,128,256> warp<48,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 128, 256, 48, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 128, 256, 48, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 128, 256, 48, 32, 128, 8, 8, 128, 4);
// cta<6,256,256> warp<48,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 256, 256, 48, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 256, 256, 48, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 256, 256, 48, 64, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<64,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 256, 64, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 256, 64, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 256, 64, 16, 128, 8, 8, 128, 4);
// cta<8,128,256> warp<64,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 128, 256, 64, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 128, 256, 64, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 128, 256, 64, 32, 128, 8, 8, 128, 4);
// cta<4,32,256> warp<16,64,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 256, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 256, 16, 64, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<32,64,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 256, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 256, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 256, 32, 64, 128, 8, 8, 128, 4);
// cta<12,32,256> warp<48,64,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 32, 256, 48, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 32, 256, 48, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 32, 256, 48, 64, 128, 8, 8, 128, 4);
// cta<4,32,256> warp<16,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 256, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 256, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 256, 16, 32, 128, 8, 8, 128, 4);
// cta<4,64,256> warp<16,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 256, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 256, 16, 64, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<32,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 256, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 256, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 256, 32, 32, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<32,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 256, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 256, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 256, 32, 64, 128, 8, 8, 128, 4);
// cta<12,32,256> warp<48,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 32, 256, 48, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 32, 256, 48, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 32, 256, 48, 32, 128, 8, 8, 128, 4);
// cta<12,64,256> warp<48,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 64, 256, 48, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 64, 256, 48, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 64, 256, 48, 64, 128, 8, 8, 128, 4);
// cta<4,32,256> warp<16,16,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 256, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 256, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 256, 16, 16, 128, 8, 8, 128, 4);
// cta<4,64,256> warp<16,32,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 256, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 256, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 256, 16, 32, 128, 8, 8, 128, 4);
// cta<4,96,256> warp<16,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 96, 256, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 96, 256, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 96, 256, 16, 48, 128, 8, 8, 128, 4);
// cta<4,128,256> warp<16,64,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 128, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 128, 256, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 128, 256, 16, 64, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<32,16,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 256, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 256, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 256, 32, 16, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<32,32,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 256, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 256, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 256, 32, 32, 128, 8, 8, 128, 4);
// cta<8,96,256> warp<32,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 96, 256, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 96, 256, 32, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 96, 256, 32, 48, 128, 8, 8, 128, 4);
// cta<8,128,256> warp<32,64,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 128, 256, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 128, 256, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 128, 256, 32, 64, 128, 8, 8, 128, 4);
// cta<12,32,256> warp<48,16,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 32, 256, 48, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 32, 256, 48, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 32, 256, 48, 16, 128, 8, 8, 128, 4);
// cta<12,64,256> warp<48,32,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 64, 256, 48, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 64, 256, 48, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 64, 256, 48, 32, 128, 8, 8, 128, 4);
// cta<12,96,256> warp<48,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 96, 256, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 96, 256, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 96, 256, 48, 48, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<16,64,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 256, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 256, 16, 64, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<16,32,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 256, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 256, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 256, 16, 32, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<16,64,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 256, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 256, 16, 64, 128, 8, 8, 128, 4);
// cta<2,32,384> warp<16,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 32, 384, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 32, 384, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 32, 384, 16, 32, 128, 8, 8, 128, 4);
// cta<2,64,384> warp<16,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 64, 384, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 64, 384, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 64, 384, 16, 64, 128, 8, 8, 128, 4);
// cta<4,32,384> warp<32,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 384, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 384, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 384, 32, 32, 128, 8, 8, 128, 4);
// cta<4,64,384> warp<32,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 384, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 384, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 384, 32, 64, 128, 8, 8, 128, 4);
// cta<6,32,384> warp<48,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 32, 384, 48, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 32, 384, 48, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 32, 384, 48, 32, 128, 8, 8, 128, 4);
// cta<6,64,384> warp<48,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 64, 384, 48, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 64, 384, 48, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 64, 384, 48, 64, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<64,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 384, 64, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 384, 64, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 384, 64, 32, 128, 8, 8, 128, 4);
// cta<8,64,384> warp<64,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 384, 64, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 384, 64, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 384, 64, 64, 128, 8, 8, 128, 4);
// cta<2,32,384> warp<16,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 32, 384, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 32, 384, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 32, 384, 16, 16, 128, 8, 8, 128, 4);
// cta<2,64,384> warp<16,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 64, 384, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 64, 384, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 64, 384, 16, 32, 128, 8, 8, 128, 4);
// cta<2,96,384> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 96, 384, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 96, 384, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 96, 384, 16, 48, 128, 8, 8, 128, 4);
// cta<2,128,384> warp<16,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 128, 384, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 128, 384, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 128, 384, 16, 64, 128, 8, 8, 128, 4);
// cta<4,32,384> warp<32,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 384, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 384, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 384, 32, 16, 128, 8, 8, 128, 4);
// cta<4,64,384> warp<32,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 384, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 384, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 384, 32, 32, 128, 8, 8, 128, 4);
// cta<4,96,384> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 96, 384, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 96, 384, 32, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 96, 384, 32, 48, 128, 8, 8, 128, 4);
// cta<4,128,384> warp<32,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 128, 384, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 128, 384, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 128, 384, 32, 64, 128, 8, 8, 128, 4);
// cta<6,32,384> warp<48,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 32, 384, 48, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 32, 384, 48, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 32, 384, 48, 16, 128, 8, 8, 128, 4);
// cta<6,64,384> warp<48,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 64, 384, 48, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 64, 384, 48, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 64, 384, 48, 32, 128, 8, 8, 128, 4);
// cta<6,96,384> warp<48,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 96, 384, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 96, 384, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 96, 384, 48, 48, 128, 8, 8, 128, 4);
// cta<6,128,384> warp<48,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 128, 384, 48, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 128, 384, 48, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 128, 384, 48, 64, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<64,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 384, 64, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 384, 64, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 384, 64, 16, 128, 8, 8, 128, 4);
// cta<8,64,384> warp<64,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 384, 64, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 384, 64, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 384, 64, 32, 128, 8, 8, 128, 4);
// cta<8,96,384> warp<64,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 96, 384, 64, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 96, 384, 64, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 96, 384, 64, 48, 128, 8, 8, 128, 4);
// cta<8,128,384> warp<64,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 128, 384, 64, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 128, 384, 64, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 128, 384, 64, 64, 128, 8, 8, 128, 4);
// cta<2,64,384> warp<16,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 64, 384, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 64, 384, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 64, 384, 16, 16, 128, 8, 8, 128, 4);
// cta<2,128,384> warp<16,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 128, 384, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 128, 384, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 128, 384, 16, 32, 128, 8, 8, 128, 4);
// cta<2,256,384> warp<16,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 256, 384, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 2, 256, 384, 16, 64, 128, 8, 8, 128, 3);
// cta<4,64,384> warp<32,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 384, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 384, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 384, 32, 16, 128, 8, 8, 128, 4);
// cta<4,128,384> warp<32,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 128, 384, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 128, 384, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 128, 384, 32, 32, 128, 8, 8, 128, 4);
// cta<4,256,384> warp<32,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 256, 384, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 256, 384, 32, 64, 128, 8, 8, 128, 3);
// cta<6,64,384> warp<48,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 64, 384, 48, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 64, 384, 48, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 64, 384, 48, 16, 128, 8, 8, 128, 4);
// cta<6,128,384> warp<48,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 128, 384, 48, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 128, 384, 48, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 128, 384, 48, 32, 128, 8, 8, 128, 4);
// cta<6,256,384> warp<48,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 256, 384, 48, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 6, 256, 384, 48, 64, 128, 8, 8, 128, 3);
// cta<8,64,384> warp<64,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 384, 64, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 384, 64, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 384, 64, 16, 128, 8, 8, 128, 4);
// cta<8,128,384> warp<64,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 128, 384, 64, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 128, 384, 64, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 128, 384, 64, 32, 128, 8, 8, 128, 4);
// cta<4,32,384> warp<16,64,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 384, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 384, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 384, 16, 64, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<32,64,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 384, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 384, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 384, 32, 64, 128, 8, 8, 128, 4);
// cta<12,32,384> warp<48,64,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 32, 384, 48, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 32, 384, 48, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 32, 384, 48, 64, 128, 8, 8, 128, 4);
// cta<4,32,384> warp<16,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 384, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 384, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 384, 16, 32, 128, 8, 8, 128, 4);
// cta<4,64,384> warp<16,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 384, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 384, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 384, 16, 64, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<32,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 384, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 384, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 384, 32, 32, 128, 8, 8, 128, 4);
// cta<8,64,384> warp<32,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 384, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 384, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 384, 32, 64, 128, 8, 8, 128, 4);
// cta<12,32,384> warp<48,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 32, 384, 48, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 32, 384, 48, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 32, 384, 48, 32, 128, 8, 8, 128, 4);
// cta<12,64,384> warp<48,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 64, 384, 48, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 64, 384, 48, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 64, 384, 48, 64, 128, 8, 8, 128, 4);
// cta<4,32,384> warp<16,16,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 384, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 384, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 32, 384, 16, 16, 128, 8, 8, 128, 4);
// cta<4,64,384> warp<16,32,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 384, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 384, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 64, 384, 16, 32, 128, 8, 8, 128, 4);
// cta<4,96,384> warp<16,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 96, 384, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 96, 384, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 96, 384, 16, 48, 128, 8, 8, 128, 4);
// cta<4,128,384> warp<16,64,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 128, 384, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 128, 384, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 4, 128, 384, 16, 64, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<32,16,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 384, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 384, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 384, 32, 16, 128, 8, 8, 128, 4);
// cta<8,64,384> warp<32,32,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 384, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 384, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 384, 32, 32, 128, 8, 8, 128, 4);
// cta<8,96,384> warp<32,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 96, 384, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 96, 384, 32, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 96, 384, 32, 48, 128, 8, 8, 128, 4);
// cta<8,128,384> warp<32,64,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 128, 384, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 128, 384, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 128, 384, 32, 64, 128, 8, 8, 128, 4);
// cta<12,32,384> warp<48,16,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 32, 384, 48, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 32, 384, 48, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 32, 384, 48, 16, 128, 8, 8, 128, 4);
// cta<12,64,384> warp<48,32,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 64, 384, 48, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 64, 384, 48, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 64, 384, 48, 32, 128, 8, 8, 128, 4);
// cta<12,96,384> warp<48,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 96, 384, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 96, 384, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 12, 96, 384, 48, 48, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<16,64,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 384, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 384, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 384, 16, 64, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<16,32,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 384, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 384, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 32, 384, 16, 32, 128, 8, 8, 128, 4);
// cta<8,64,384> warp<16,64,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 384, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 384, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 2, true, 8, 64, 384, 16, 64, 128, 8, 8, 128, 4);
#endif