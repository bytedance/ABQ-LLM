
#include "common/base.h"
#include "mma_any/aq_wmma_op.h"

////// W1A1 int
// cta<8,32,128> warp<8,16,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBWMMA, 1, 1, true, 8, 32, 128, 8, 16, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 1, 1, true, 8, 32, 128, 8, 16, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 1, 1, true, 8, 32, 128, 8, 16, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<8,32,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBWMMA, 1, 1, true, 8, 64, 128, 8, 32, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 1, 1, true, 8, 64, 128, 8, 32, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 1, 1, true, 8, 64, 128, 8, 32, 128, 8, 8, 128, 4);
// cta<8,128,128> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBWMMA, 1, 1, true, 8, 128, 128, 8, 32, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 1, 1, true, 8, 128, 128, 8, 32, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 1, 1, true, 8, 128, 128, 8, 32, 128, 8, 8, 128, 4);

////// W1A4 int
// cta<8,64,128> warp<32,32,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBWMMA, 4, 1, true, 8, 64, 128, 32, 32, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 4, 1, true, 8, 64, 128, 32, 32, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 4, 1, true, 8, 64, 128, 32, 32, 128, 8, 8, 128, 4);
// cta<8,128,128> warp<32,32,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBWMMA, 4, 1, true, 8, 128, 128, 32, 32, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 4, 1, true, 8, 128, 128, 32, 32, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 4, 1, true, 8, 128, 128, 32, 32, 128, 8, 8, 128, 4);

////// W1A8 int
// cta<8,64,128> warp<32,32,128> mma<8,8,128>   WARPS[2x2]
AQ_INSTANTIATE_FUN(AqBWMMA, 8, 1, true, 8, 64, 128, 32, 32, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 8, 1, true, 8, 64, 128, 32, 32, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 8, 1, true, 8, 64, 128, 32, 32, 128, 8, 8, 128, 4);
// cta<8,128,128> warp<64,32,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBWMMA, 8, 1, true, 8, 128, 128, 64, 32, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 8, 1, true, 8, 128, 128, 64, 32, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 8, 1, true, 8, 128, 128, 64, 32, 128, 8, 8, 128, 4);

////// W1A16 int
// cta<8,32,128> warp<32,32,128> mma<8,8,128>   WARPS[4x1]
AQ_INSTANTIATE_FUN(AqBWMMA, 16, 1, true, 8, 32, 128, 32, 32, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 16, 1, true, 8, 32, 128, 32, 32, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 16, 1, true, 8, 32, 128, 32, 32, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<32,32,128> mma<8,8,128>   WARPS[4x1]
AQ_INSTANTIATE_FUN(AqBWMMA, 16, 1, true, 8, 32, 256, 32, 32, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 16, 1, true, 8, 32, 256, 32, 32, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 16, 1, true, 8, 32, 256, 32, 32, 128, 8, 8, 128, 4);

////// W2A16 int
// cta<8,32,128> warp<64,32,128> mma<8,8,128>   WARPS[2x2]
AQ_INSTANTIATE_FUN(AqBWMMA, 16, 2, true, 8, 32, 128, 64, 32, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 16, 2, true, 8, 32, 128, 64, 32, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 16, 2, true, 8, 32, 128, 64, 32, 128, 8, 8, 128, 4);