#pragma once
#include "common/base.h"
#include "mma_any/aq_bmma_op.h"

#ifdef W7A7
////// W7A7 int
// cta<1,32,256> warp<8,112,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 7, 7, true, 1, 32, 256, 8, 112, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 7, 7, true, 1, 32, 256, 8, 112, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 7, 7, true, 1, 32, 256, 8, 112, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<56,112,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 7, 7, true, 8, 32, 256, 56, 112, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 7, 7, true, 8, 32, 256, 56, 112, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 7, 7, true, 8, 32, 256, 56, 112, 128, 8, 8, 128, 4);
AQ_DECL_FUN(AqBMMA, 7, 7, true, 8, 32, 256, 56, 112, 128, 8, 8, 128, 5);
// cta<1,32,256> warp<8,56,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 7, 7, true, 1, 32, 256, 8, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 7, 7, true, 1, 32, 256, 8, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 7, 7, true, 1, 32, 256, 8, 56, 128, 8, 8, 128, 4);
// cta<1,64,256> warp<8,112,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 7, 7, true, 1, 64, 256, 8, 112, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 7, 7, true, 1, 64, 256, 8, 112, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 7, 7, true, 1, 64, 256, 8, 112, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<56,56,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 7, 7, true, 8, 32, 256, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 7, 7, true, 8, 32, 256, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 7, 7, true, 8, 32, 256, 56, 56, 128, 8, 8, 128, 4);
AQ_DECL_FUN(AqBMMA, 7, 7, true, 8, 32, 256, 56, 56, 128, 8, 8, 128, 5);
// cta<1,32,384> warp<8,112,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 7, 7, true, 1, 32, 384, 8, 112, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 7, 7, true, 1, 32, 384, 8, 112, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 7, 7, true, 1, 32, 384, 8, 112, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<56,112,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 7, 7, true, 8, 32, 384, 56, 112, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 7, 7, true, 8, 32, 384, 56, 112, 128, 8, 8, 128, 3);
// cta<1,32,384> warp<8,56,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 7, 7, true, 1, 32, 384, 8, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 7, 7, true, 1, 32, 384, 8, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 7, 7, true, 1, 32, 384, 8, 56, 128, 8, 8, 128, 4);
// cta<1,64,384> warp<8,112,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 7, 7, true, 1, 64, 384, 8, 112, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 7, 7, true, 1, 64, 384, 8, 112, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 7, 7, true, 1, 64, 384, 8, 112, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<56,56,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 7, 7, true, 8, 32, 384, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 7, 7, true, 8, 32, 384, 56, 56, 128, 8, 8, 128, 3);
// cta<1,32,512> warp<8,112,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 7, 7, true, 1, 32, 512, 8, 112, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 7, 7, true, 1, 32, 512, 8, 112, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 7, 7, true, 1, 32, 512, 8, 112, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<56,112,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 7, 7, true, 8, 32, 512, 56, 112, 128, 8, 8, 128, 2);
// cta<1,32,512> warp<8,56,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 7, 7, true, 1, 32, 512, 8, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 7, 7, true, 1, 32, 512, 8, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 7, 7, true, 1, 32, 512, 8, 56, 128, 8, 8, 128, 4);
// cta<1,64,512> warp<8,112,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 7, 7, true, 1, 64, 512, 8, 112, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 7, 7, true, 1, 64, 512, 8, 112, 128, 8, 8, 128, 3);
// cta<8,32,512> warp<56,56,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 7, 7, true, 8, 32, 512, 56, 56, 128, 8, 8, 128, 2);
#endif