#include "src/fastertransformer/kernels/abqgemm/common/base.h"
#include "src/fastertransformer/kernels/abqgemm/mma_any/aq_wmma_op.h"

// cta<2,64,256> warp<16,64,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBWMMA, 8, 2, true, 2, 64, 256, 16, 64, 128, 8, 8, 128, 4);
AQ_INSTANTIATE_FUN(AqBWMMA, 8, 2, false, 2, 64, 256, 16, 64, 128, 8, 8, 128, 4);