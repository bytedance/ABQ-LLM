#pragma once
#include "common/base.h"
#include "mma_any/aq_wmma_op.h"

#ifdef W4A8
////// W4A8 int
// cta<8,8,128> warp<64,32,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 128, 64, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 128, 64, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 128, 64, 32, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<64,64,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 64, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 64, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 64, 64, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<64,16,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 128, 64, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 128, 64, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 128, 64, 16, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<64,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 64, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 64, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 64, 32, 128, 8, 8, 128, 4);
// cta<8,24,128> warp<64,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 24, 128, 64, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 24, 128, 64, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 24, 128, 64, 48, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<64,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 64, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 64, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 64, 64, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<64,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 64, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 64, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 64, 16, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<64,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 64, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 64, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 64, 32, 128, 8, 8, 128, 4);
// cta<8,48,128> warp<64,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 48, 128, 64, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 48, 128, 64, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 48, 128, 64, 48, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<64,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 128, 64, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 128, 64, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 128, 64, 64, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<64,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 64, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 64, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 64, 16, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<64,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 128, 64, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 128, 64, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 128, 64, 32, 128, 8, 8, 128, 4);
// cta<8,96,128> warp<64,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 96, 128, 64, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 96, 128, 64, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 96, 128, 64, 48, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<64,16,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 128, 64, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 128, 64, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 128, 64, 16, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<32,32,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 128, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 128, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 128, 32, 32, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<32,64,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 32, 64, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<32,16,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 128, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 128, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 128, 32, 16, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<32,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 32, 32, 128, 8, 8, 128, 4);
// cta<8,24,128> warp<32,48,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 24, 128, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 24, 128, 32, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 24, 128, 32, 48, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<32,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 32, 64, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<32,16,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 32, 16, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<32,32,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 32, 32, 128, 8, 8, 128, 4);
// cta<8,48,128> warp<32,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 48, 128, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 48, 128, 32, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 48, 128, 32, 48, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<32,64,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 128, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 128, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 128, 32, 64, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<32,16,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 32, 16, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<32,32,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 128, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 128, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 128, 32, 32, 128, 8, 8, 128, 4);
// cta<8,96,128> warp<32,48,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 96, 128, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 96, 128, 32, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 96, 128, 32, 48, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<32,16,128> mma<8,8,128>   WARPS[2x16]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 128, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 128, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 128, 32, 16, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<16,32,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 128, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 128, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 128, 16, 32, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<16,64,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 16, 64, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<16,16,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 128, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 128, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 128, 16, 16, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<16,32,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 16, 32, 128, 8, 8, 128, 4);
// cta<8,24,128> warp<16,48,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 24, 128, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 24, 128, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 24, 128, 16, 48, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<16,64,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 16, 64, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<16,16,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 128, 16, 16, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<16,32,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 16, 32, 128, 8, 8, 128, 4);
// cta<8,48,128> warp<16,48,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 48, 128, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 48, 128, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 48, 128, 16, 48, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<16,64,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 128, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 128, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 128, 16, 64, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<16,16,128> mma<8,8,128>   WARPS[4x8]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 128, 16, 16, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<16,32,128> mma<8,8,128>   WARPS[4x8]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 128, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 128, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 128, 16, 32, 128, 8, 8, 128, 4);
// cta<8,96,128> warp<16,48,128> mma<8,8,128>   WARPS[4x8]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 96, 128, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 96, 128, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 96, 128, 16, 48, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<64,32,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 256, 64, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 256, 64, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 256, 64, 32, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<64,64,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 64, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 64, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 64, 64, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<64,16,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 256, 64, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 256, 64, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 256, 64, 16, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<64,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 64, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 64, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 64, 32, 128, 8, 8, 128, 4);
// cta<8,24,256> warp<64,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 24, 256, 64, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 24, 256, 64, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 24, 256, 64, 48, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<64,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 64, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 64, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 64, 64, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<64,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 64, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 64, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 64, 16, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<64,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 64, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 64, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 64, 32, 128, 8, 8, 128, 4);
// cta<8,48,256> warp<64,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 48, 256, 64, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 48, 256, 64, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 48, 256, 64, 48, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<64,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 256, 64, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 256, 64, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 256, 64, 64, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<64,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 64, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 64, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 64, 16, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<64,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 256, 64, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 256, 64, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 256, 64, 32, 128, 8, 8, 128, 4);
// cta<8,96,256> warp<64,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 96, 256, 64, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 96, 256, 64, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 96, 256, 64, 48, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<64,16,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 256, 64, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 256, 64, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 256, 64, 16, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<32,32,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 256, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 256, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 256, 32, 32, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<32,64,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 32, 64, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<32,16,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 256, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 256, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 256, 32, 16, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<32,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 32, 32, 128, 8, 8, 128, 4);
// cta<8,24,256> warp<32,48,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 24, 256, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 24, 256, 32, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 24, 256, 32, 48, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<32,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 32, 64, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<32,16,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 32, 16, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<32,32,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 32, 32, 128, 8, 8, 128, 4);
// cta<8,48,256> warp<32,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 48, 256, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 48, 256, 32, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 48, 256, 32, 48, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<32,64,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 256, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 256, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 256, 32, 64, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<32,16,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 32, 16, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<32,32,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 256, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 256, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 256, 32, 32, 128, 8, 8, 128, 4);
// cta<8,96,256> warp<32,48,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 96, 256, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 96, 256, 32, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 96, 256, 32, 48, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<32,16,128> mma<8,8,128>   WARPS[2x16]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 256, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 256, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 256, 32, 16, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<16,32,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 256, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 256, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 256, 16, 32, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<16,64,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 16, 64, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<16,16,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 256, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 256, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 8, 256, 16, 16, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<16,32,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 16, 32, 128, 8, 8, 128, 4);
// cta<8,24,256> warp<16,48,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 24, 256, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 24, 256, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 24, 256, 16, 48, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<16,64,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 16, 64, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<16,16,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 16, 256, 16, 16, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<16,32,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 16, 32, 128, 8, 8, 128, 4);
// cta<8,48,256> warp<16,48,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 48, 256, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 48, 256, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 48, 256, 16, 48, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<16,64,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 256, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 256, 16, 64, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<16,16,128> mma<8,8,128>   WARPS[4x8]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 32, 256, 16, 16, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<16,32,128> mma<8,8,128>   WARPS[4x8]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 256, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 256, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 64, 256, 16, 32, 128, 8, 8, 128, 4);
// cta<8,96,256> warp<16,48,128> mma<8,8,128>   WARPS[4x8]
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 96, 256, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 96, 256, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 4, true, 8, 96, 256, 16, 48, 128, 8, 8, 128, 4);
#endif