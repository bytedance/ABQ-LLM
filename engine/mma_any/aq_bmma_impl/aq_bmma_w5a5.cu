#include "common/base.h"
#include "mma_any/aq_bmma_op.h"

#ifdef W5A5
////// W5A5 int
// cta<1,32,256> warp<8,80,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 32, 256, 8, 80, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 32, 256, 8, 80, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 32, 256, 8, 80, 128, 8, 8, 128, 4);
// cta<1,48,256> warp<8,120,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 48, 256, 8, 120, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 48, 256, 8, 120, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 48, 256, 8, 120, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<40,80,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 32, 256, 40, 80, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 32, 256, 40, 80, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 32, 256, 40, 80, 128, 8, 8, 128, 4);
// cta<8,48,256> warp<40,120,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 48, 256, 40, 120, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 48, 256, 40, 120, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 48, 256, 40, 120, 128, 8, 8, 128, 4);
// cta<1,32,256> warp<8,40,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 32, 256, 8, 40, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 32, 256, 8, 40, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 32, 256, 8, 40, 128, 8, 8, 128, 4);
// cta<1,64,256> warp<8,80,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 64, 256, 8, 80, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 64, 256, 8, 80, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 64, 256, 8, 80, 128, 8, 8, 128, 4);
// cta<1,96,256> warp<8,120,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 96, 256, 8, 120, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 96, 256, 8, 120, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 96, 256, 8, 120, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<40,40,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 32, 256, 40, 40, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 32, 256, 40, 40, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 32, 256, 40, 40, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<40,80,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 64, 256, 40, 80, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 64, 256, 40, 80, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 64, 256, 40, 80, 128, 8, 8, 128, 4);
// cta<8,96,256> warp<40,120,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 96, 256, 40, 120, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 96, 256, 40, 120, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 96, 256, 40, 120, 128, 8, 8, 128, 4);
// cta<1,32,384> warp<8,80,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 32, 384, 8, 80, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 32, 384, 8, 80, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 32, 384, 8, 80, 128, 8, 8, 128, 4);
// cta<1,48,384> warp<8,120,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 48, 384, 8, 120, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 48, 384, 8, 120, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 48, 384, 8, 120, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<40,80,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 32, 384, 40, 80, 128, 8, 8, 128, 2);
// cta<8,48,384> warp<40,120,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 48, 384, 40, 120, 128, 8, 8, 128, 2);
// cta<1,32,384> warp<8,40,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 32, 384, 8, 40, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 32, 384, 8, 40, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 32, 384, 8, 40, 128, 8, 8, 128, 4);
// cta<1,64,384> warp<8,80,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 64, 384, 8, 80, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 64, 384, 8, 80, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 64, 384, 8, 80, 128, 8, 8, 128, 4);
// cta<1,96,384> warp<8,120,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 96, 384, 8, 120, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 96, 384, 8, 120, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 96, 384, 8, 120, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<40,40,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 32, 384, 40, 40, 128, 8, 8, 128, 2);
// cta<8,64,384> warp<40,80,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 64, 384, 40, 80, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 64, 384, 40, 80, 128, 8, 8, 128, 3);
// cta<8,96,384> warp<40,120,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 96, 384, 40, 120, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 96, 384, 40, 120, 128, 8, 8, 128, 3);
// cta<1,32,512> warp<8,80,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 32, 512, 8, 80, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 32, 512, 8, 80, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 32, 512, 8, 80, 128, 8, 8, 128, 4);
// cta<1,48,512> warp<8,120,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 48, 512, 8, 120, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 48, 512, 8, 120, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 48, 512, 8, 120, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<40,80,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 32, 512, 40, 80, 128, 8, 8, 128, 2);
// cta<8,48,512> warp<40,120,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 48, 512, 40, 120, 128, 8, 8, 128, 2);
// cta<1,32,512> warp<8,40,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 32, 512, 8, 40, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 32, 512, 8, 40, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 32, 512, 8, 40, 128, 8, 8, 128, 4);
// cta<1,64,512> warp<8,80,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 64, 512, 8, 80, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 64, 512, 8, 80, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 64, 512, 8, 80, 128, 8, 8, 128, 4);
// cta<1,96,512> warp<8,120,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 96, 512, 8, 120, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 1, 96, 512, 8, 120, 128, 8, 8, 128, 3);
// cta<8,32,512> warp<40,40,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 32, 512, 40, 40, 128, 8, 8, 128, 2);
// cta<8,64,512> warp<40,80,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 64, 512, 40, 80, 128, 8, 8, 128, 2);
// cta<8,96,512> warp<40,120,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 5, 5, true, 8, 96, 512, 40, 120, 128, 8, 8, 128, 2);
#endif