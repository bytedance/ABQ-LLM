#pragma once
#include "common/base.h"
#include "mma_any/aq_wmma_op.h"

#ifdef W5A5
////// W5A5 int
// cta<8,8,128> warp<40,40,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 8, 128, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 8, 128, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 8, 128, 40, 40, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<40,40,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 16, 128, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 16, 128, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 16, 128, 40, 40, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<40,40,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 32, 128, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 32, 128, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 32, 128, 40, 40, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<40,40,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 64, 128, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 64, 128, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 64, 128, 40, 40, 128, 8, 8, 128, 4);
// cta<16,8,128> warp<40,40,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 8, 128, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 8, 128, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 8, 128, 40, 40, 128, 8, 8, 128, 4);
// cta<16,16,128> warp<40,40,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 16, 128, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 16, 128, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 16, 128, 40, 40, 128, 8, 8, 128, 4);
// cta<16,32,128> warp<40,40,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 32, 128, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 32, 128, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 32, 128, 40, 40, 128, 8, 8, 128, 4);
// cta<32,8,128> warp<40,40,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 8, 128, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 8, 128, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 8, 128, 40, 40, 128, 8, 8, 128, 4);
// cta<32,16,128> warp<40,40,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 16, 128, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 16, 128, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 16, 128, 40, 40, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<40,40,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 8, 256, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 8, 256, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 8, 256, 40, 40, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<40,40,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 16, 256, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 16, 256, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 16, 256, 40, 40, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<40,40,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 32, 256, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 32, 256, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 32, 256, 40, 40, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<40,40,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 64, 256, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 64, 256, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 64, 256, 40, 40, 128, 8, 8, 128, 4);
// cta<16,8,256> warp<40,40,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 8, 256, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 8, 256, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 8, 256, 40, 40, 128, 8, 8, 128, 4);
// cta<16,16,256> warp<40,40,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 16, 256, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 16, 256, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 16, 256, 40, 40, 128, 8, 8, 128, 4);
// cta<16,32,256> warp<40,40,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 32, 256, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 32, 256, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 32, 256, 40, 40, 128, 8, 8, 128, 4);
// cta<32,8,256> warp<40,40,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 8, 256, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 8, 256, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 8, 256, 40, 40, 128, 8, 8, 128, 4);
// cta<32,16,256> warp<40,40,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 16, 256, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 16, 256, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 16, 256, 40, 40, 128, 8, 8, 128, 4);
// cta<8,8,384> warp<40,40,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 8, 384, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 8, 384, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 8, 384, 40, 40, 128, 8, 8, 128, 4);
// cta<8,16,384> warp<40,40,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 16, 384, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 16, 384, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 16, 384, 40, 40, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<40,40,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 32, 384, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 32, 384, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 32, 384, 40, 40, 128, 8, 8, 128, 4);
// cta<8,64,384> warp<40,40,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 64, 384, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 64, 384, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 64, 384, 40, 40, 128, 8, 8, 128, 4);
// cta<16,8,384> warp<40,40,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 8, 384, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 8, 384, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 8, 384, 40, 40, 128, 8, 8, 128, 4);
// cta<16,16,384> warp<40,40,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 16, 384, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 16, 384, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 16, 384, 40, 40, 128, 8, 8, 128, 4);
// cta<16,32,384> warp<40,40,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 32, 384, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 32, 384, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 32, 384, 40, 40, 128, 8, 8, 128, 4);
// cta<32,8,384> warp<40,40,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 8, 384, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 8, 384, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 8, 384, 40, 40, 128, 8, 8, 128, 4);
// cta<32,16,384> warp<40,40,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 16, 384, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 16, 384, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 16, 384, 40, 40, 128, 8, 8, 128, 4);
// cta<8,8,512> warp<40,40,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 8, 512, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 8, 512, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 8, 512, 40, 40, 128, 8, 8, 128, 4);
// cta<8,16,512> warp<40,40,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 16, 512, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 16, 512, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 16, 512, 40, 40, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<40,40,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 32, 512, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 32, 512, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 32, 512, 40, 40, 128, 8, 8, 128, 4);
// cta<8,64,512> warp<40,40,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 64, 512, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 64, 512, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 64, 512, 40, 40, 128, 8, 8, 128, 4);
// cta<16,8,512> warp<40,40,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 8, 512, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 8, 512, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 8, 512, 40, 40, 128, 8, 8, 128, 4);
// cta<16,16,512> warp<40,40,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 16, 512, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 16, 512, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 16, 512, 40, 40, 128, 8, 8, 128, 4);
// cta<16,32,512> warp<40,40,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 32, 512, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 32, 512, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 32, 512, 40, 40, 128, 8, 8, 128, 4);
// cta<32,8,512> warp<40,40,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 8, 512, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 8, 512, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 8, 512, 40, 40, 128, 8, 8, 128, 4);
// cta<32,16,512> warp<40,40,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 16, 512, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 16, 512, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 16, 512, 40, 40, 128, 8, 8, 128, 4);
// cta<8,8,640> warp<40,40,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 8, 640, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 8, 640, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 8, 640, 40, 40, 128, 8, 8, 128, 4);
// cta<8,16,640> warp<40,40,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 16, 640, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 16, 640, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 16, 640, 40, 40, 128, 8, 8, 128, 4);
// cta<8,32,640> warp<40,40,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 32, 640, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 32, 640, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 32, 640, 40, 40, 128, 8, 8, 128, 4);
// cta<8,64,640> warp<40,40,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 64, 640, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 64, 640, 40, 40, 128, 8, 8, 128, 3);
// cta<16,8,640> warp<40,40,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 8, 640, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 8, 640, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 8, 640, 40, 40, 128, 8, 8, 128, 4);
// cta<16,16,640> warp<40,40,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 16, 640, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 16, 640, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 16, 640, 40, 40, 128, 8, 8, 128, 4);
// cta<16,32,640> warp<40,40,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 32, 640, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 32, 640, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 32, 640, 40, 40, 128, 8, 8, 128, 4);
// cta<32,8,640> warp<40,40,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 8, 640, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 8, 640, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 8, 640, 40, 40, 128, 8, 8, 128, 4);
// cta<32,16,640> warp<40,40,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 16, 640, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 16, 640, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 16, 640, 40, 40, 128, 8, 8, 128, 4);
// cta<8,8,768> warp<40,40,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 8, 768, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 8, 768, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 8, 768, 40, 40, 128, 8, 8, 128, 4);
// cta<8,16,768> warp<40,40,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 16, 768, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 16, 768, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 16, 768, 40, 40, 128, 8, 8, 128, 4);
// cta<8,32,768> warp<40,40,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 32, 768, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 32, 768, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 32, 768, 40, 40, 128, 8, 8, 128, 4);
// cta<8,64,768> warp<40,40,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 64, 768, 40, 40, 128, 8, 8, 128, 2);
// cta<16,8,768> warp<40,40,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 8, 768, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 8, 768, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 8, 768, 40, 40, 128, 8, 8, 128, 4);
// cta<16,16,768> warp<40,40,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 16, 768, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 16, 768, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 16, 768, 40, 40, 128, 8, 8, 128, 4);
// cta<16,32,768> warp<40,40,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 32, 768, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 32, 768, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 32, 768, 40, 40, 128, 8, 8, 128, 4);
// cta<32,8,768> warp<40,40,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 8, 768, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 8, 768, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 8, 768, 40, 40, 128, 8, 8, 128, 4);
// cta<32,16,768> warp<40,40,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 16, 768, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 16, 768, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 16, 768, 40, 40, 128, 8, 8, 128, 4);
// cta<8,8,896> warp<40,40,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 8, 896, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 8, 896, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 8, 896, 40, 40, 128, 8, 8, 128, 4);
// cta<8,16,896> warp<40,40,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 16, 896, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 16, 896, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 16, 896, 40, 40, 128, 8, 8, 128, 4);
// cta<8,32,896> warp<40,40,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 32, 896, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 32, 896, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 32, 896, 40, 40, 128, 8, 8, 128, 4);
// cta<8,64,896> warp<40,40,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 8, 64, 896, 40, 40, 128, 8, 8, 128, 2);
// cta<16,8,896> warp<40,40,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 8, 896, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 8, 896, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 8, 896, 40, 40, 128, 8, 8, 128, 4);
// cta<16,16,896> warp<40,40,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 16, 896, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 16, 896, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 16, 896, 40, 40, 128, 8, 8, 128, 4);
// cta<16,32,896> warp<40,40,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 32, 896, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 16, 32, 896, 40, 40, 128, 8, 8, 128, 3);
// cta<32,8,896> warp<40,40,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 8, 896, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 8, 896, 40, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 8, 896, 40, 40, 128, 8, 8, 128, 4);
// cta<32,16,896> warp<40,40,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 16, 896, 40, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 5, 5, true, 32, 16, 896, 40, 40, 128, 8, 8, 128, 3);
#endif