// Copyright (C) 2024 ByteDance and/or its affiliates
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

#ifdef W7A7
////// W7A7 int
// cta<1,32,256> warp<8,112,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 1, 32, 256, 8, 112, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 1, 32, 256, 8, 112, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 1, 32, 256, 8, 112, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<56,112,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 8, 32, 256, 56, 112, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 8, 32, 256, 56, 112, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 8, 32, 256, 56, 112, 128, 8, 8, 128, 4);
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 8, 32, 256, 56, 112, 128, 8, 8, 128, 5);
// cta<1,32,256> warp<8,56,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 1, 32, 256, 8, 56, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 1, 32, 256, 8, 56, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 1, 32, 256, 8, 56, 128, 8, 8, 128, 4);
// cta<1,64,256> warp<8,112,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 1, 64, 256, 8, 112, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 1, 64, 256, 8, 112, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 1, 64, 256, 8, 112, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<56,56,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 8, 32, 256, 56, 56, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 8, 32, 256, 56, 56, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 8, 32, 256, 56, 56, 128, 8, 8, 128, 4);
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 8, 32, 256, 56, 56, 128, 8, 8, 128, 5);
// cta<1,32,384> warp<8,112,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 1, 32, 384, 8, 112, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 1, 32, 384, 8, 112, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 1, 32, 384, 8, 112, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<56,112,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 8, 32, 384, 56, 112, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 8, 32, 384, 56, 112, 128, 8, 8, 128, 3);
// cta<1,32,384> warp<8,56,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 1, 32, 384, 8, 56, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 1, 32, 384, 8, 56, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 1, 32, 384, 8, 56, 128, 8, 8, 128, 4);
// cta<1,64,384> warp<8,112,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 1, 64, 384, 8, 112, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 1, 64, 384, 8, 112, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 1, 64, 384, 8, 112, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<56,56,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 8, 32, 384, 56, 56, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 8, 32, 384, 56, 56, 128, 8, 8, 128, 3);
// cta<1,32,512> warp<8,112,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 1, 32, 512, 8, 112, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 1, 32, 512, 8, 112, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 1, 32, 512, 8, 112, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<56,112,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 8, 32, 512, 56, 112, 128, 8, 8, 128, 2);
// cta<1,32,512> warp<8,56,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 1, 32, 512, 8, 56, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 1, 32, 512, 8, 56, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 1, 32, 512, 8, 56, 128, 8, 8, 128, 4);
// cta<1,64,512> warp<8,112,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 1, 64, 512, 8, 112, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 1, 64, 512, 8, 112, 128, 8, 8, 128, 3);
// cta<8,32,512> warp<56,56,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBWMMA, 7, 7, true, 8, 32, 512, 56, 56, 128, 8, 8, 128, 2);
#endif