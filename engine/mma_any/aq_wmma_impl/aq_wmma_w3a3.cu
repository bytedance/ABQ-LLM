// Copyright (C) ABQ-LLM (xieyusheng.12@bytedance.com)
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//          http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "common/base.h"
#include "mma_any/aq_wmma_op.h"

#ifdef W3A3
////// W3A3 int
// cta<8,16,128> warp<24,24,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 16, 128, 24, 24, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 16, 128, 24, 24, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 16, 128, 24, 24, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<24,48,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 32, 128, 24, 48, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 32, 128, 24, 48, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 32, 128, 24, 48, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<24,24,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 32, 128, 24, 24, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 32, 128, 24, 24, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 32, 128, 24, 24, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 64, 128, 24, 48, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 64, 128, 24, 48, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 64, 128, 24, 48, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<24,24,128> mma<8,8,128>   WARPS[1x8]
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 64, 128, 24, 24, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 64, 128, 24, 24, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 64, 128, 24, 24, 128, 8, 8, 128, 4);
// cta<8,128,128> warp<24,48,128> mma<8,8,128>   WARPS[1x8]
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 128, 128, 24, 48, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 128, 128, 24, 48, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 128, 128, 24, 48, 128, 8, 8, 128, 4);
// cta<8,128,128> warp<24,24,128> mma<8,8,128>   WARPS[1x16]
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 128, 128, 24, 24, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 128, 128, 24, 24, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 128, 128, 24, 24, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<24,24,128> mma<8,8,128>   WARPS[1x1]
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 8, 256, 24, 24, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 8, 256, 24, 24, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 8, 256, 24, 24, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<24,48,128> mma<8,8,128>   WARPS[1x1]
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 16, 256, 24, 48, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 16, 256, 24, 48, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 16, 256, 24, 48, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<24,24,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 16, 256, 24, 24, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 16, 256, 24, 24, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 16, 256, 24, 24, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<24,48,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 32, 256, 24, 48, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 32, 256, 24, 48, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 32, 256, 24, 48, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<24,24,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 32, 256, 24, 24, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 32, 256, 24, 24, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 32, 256, 24, 24, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 64, 256, 24, 48, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 64, 256, 24, 48, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 64, 256, 24, 48, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<24,24,128> mma<8,8,128>   WARPS[1x8]
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 64, 256, 24, 24, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 64, 256, 24, 24, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 64, 256, 24, 24, 128, 8, 8, 128, 4);
// cta<8,128,256> warp<24,48,128> mma<8,8,128>   WARPS[1x8]
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 128, 256, 24, 48, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 128, 256, 24, 48, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 128, 256, 24, 48, 128, 8, 8, 128, 4);
// cta<8,128,256> warp<24,24,128> mma<8,8,128>   WARPS[1x16]
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 128, 256, 24, 24, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 128, 256, 24, 24, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 128, 256, 24, 24, 128, 8, 8, 128, 4);

// cta<8,128,384> warp<24,48,128> mma<8,8,128>   WARPS[1x8]
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 128, 384, 24, 48, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 128, 384, 24, 48, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 128, 384, 24, 48, 128, 8, 8, 128, 4);

// cta<8,128,512> warp<24,48,128> mma<8,8,128>   WARPS[1x8]
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 128, 512, 24, 48, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 128, 512, 24, 48, 128, 8, 8, 128, 3);

// cta<8,128,640> warp<24,48,128> mma<8,8,128>   WARPS[1x8]
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 128, 640, 24, 48, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 128, 640, 24, 48, 128, 8, 8, 128, 3);

// cta<8,128,768> warp<24,48,128> mma<8,8,128>   WARPS[1x8]
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 128, 768, 24, 48, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 128, 768, 24, 48, 128, 8, 8, 128, 3);

// cta<8,128,896> warp<24,48,128> mma<8,8,128>   WARPS[1x8]
AQ_INSTANTIATE_FUN(AqBWMMA, 3, 3, true, 8, 128, 896, 24, 48, 128, 8, 8, 128, 2);
#endif