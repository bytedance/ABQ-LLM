#include "src/fastertransformer/kernels/abqgemm/common/base.h"
#include "src/fastertransformer/kernels/abqgemm/mma_any/aq_bmma_op.h"

// cta<1,48,512> warp<8,24,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 8, 2, true, 1, 48, 512, 8, 24, 128, 8, 8, 128, 4);
AQ_INSTANTIATE_FUN(AqBMMA, 8, 2, false, 1, 48, 512, 8, 24, 128, 8, 8, 128, 4);
// cta<8,48,256> warp<32,48,128> mma<8,8,128>   WARPS[2x2]
AQ_INSTANTIATE_FUN(AqBMMA, 8, 2, true, 8, 48, 256, 32, 48, 128, 8, 8, 128, 5);
AQ_INSTANTIATE_FUN(AqBMMA, 8, 2, false, 8, 48, 256, 32, 48, 128, 8, 8, 128, 5);