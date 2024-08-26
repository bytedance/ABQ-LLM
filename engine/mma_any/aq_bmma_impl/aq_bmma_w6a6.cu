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
#include "mma_any/aq_bmma_op.h"

#ifdef W6A6
////// W6A6 int
// cta<1,32,256> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 32, 256, 8, 96, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 32, 256, 8, 96, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 32, 256, 8, 96, 128, 8, 8, 128, 4);
// cta<4,32,256> warp<24,96,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 32, 256, 24, 96, 128, 8, 8, 128, 2);
// cta<8,32,256> warp<48,96,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 32, 256, 48, 96, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 32, 256, 48, 96, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 32, 256, 48, 96, 128, 8, 8, 128, 4);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 32, 256, 48, 96, 128, 8, 8, 128, 5);
// cta<1,32,256> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 32, 256, 8, 48, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 32, 256, 8, 48, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 32, 256, 8, 48, 128, 8, 8, 128, 4);
// cta<1,48,256> warp<8,72,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 48, 256, 8, 72, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 48, 256, 8, 72, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 48, 256, 8, 72, 128, 8, 8, 128, 4);
// cta<1,64,256> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 64, 256, 8, 96, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 64, 256, 8, 96, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 64, 256, 8, 96, 128, 8, 8, 128, 4);
// cta<1,80,256> warp<8,120,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 80, 256, 8, 120, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 80, 256, 8, 120, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 80, 256, 8, 120, 128, 8, 8, 128, 4);
// cta<4,32,256> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 32, 256, 24, 48, 128, 8, 8, 128, 2);
// cta<4,48,256> warp<24,72,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 48, 256, 24, 72, 128, 8, 8, 128, 2);
// cta<4,64,256> warp<24,96,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 64, 256, 24, 96, 128, 8, 8, 128, 2);
// cta<4,80,256> warp<24,120,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 80, 256, 24, 120, 128, 8, 8, 128, 2);
// cta<8,32,256> warp<48,48,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 32, 256, 48, 48, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 32, 256, 48, 48, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 32, 256, 48, 48, 128, 8, 8, 128, 4);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 32, 256, 48, 48, 128, 8, 8, 128, 5);
// cta<8,48,256> warp<48,72,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 48, 256, 48, 72, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 48, 256, 48, 72, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 48, 256, 48, 72, 128, 8, 8, 128, 4);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 48, 256, 48, 72, 128, 8, 8, 128, 5);
// cta<8,64,256> warp<48,96,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 64, 256, 48, 96, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 64, 256, 48, 96, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 64, 256, 48, 96, 128, 8, 8, 128, 4);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 64, 256, 48, 96, 128, 8, 8, 128, 5);
// cta<8,80,256> warp<48,120,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 80, 256, 48, 120, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 80, 256, 48, 120, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 80, 256, 48, 120, 128, 8, 8, 128, 4);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 80, 256, 48, 120, 128, 8, 8, 128, 5);
// cta<8,32,256> warp<24,96,128> mma<8,8,128>   WARPS[2x2]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 32, 256, 24, 96, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 32, 256, 24, 96, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 32, 256, 24, 96, 128, 8, 8, 128, 4);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 32, 256, 24, 96, 128, 8, 8, 128, 5);
// cta<1,32,384> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 32, 384, 8, 96, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 32, 384, 8, 96, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 32, 384, 8, 96, 128, 8, 8, 128, 4);
// cta<4,32,384> warp<24,96,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 32, 384, 24, 96, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 32, 384, 24, 96, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 32, 384, 24, 96, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<48,96,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 32, 384, 48, 96, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 32, 384, 48, 96, 128, 8, 8, 128, 3);
// cta<1,32,384> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 32, 384, 8, 48, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 32, 384, 8, 48, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 32, 384, 8, 48, 128, 8, 8, 128, 4);
// cta<1,48,384> warp<8,72,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 48, 384, 8, 72, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 48, 384, 8, 72, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 48, 384, 8, 72, 128, 8, 8, 128, 4);
// cta<1,64,384> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 64, 384, 8, 96, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 64, 384, 8, 96, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 64, 384, 8, 96, 128, 8, 8, 128, 4);
// cta<1,80,384> warp<8,120,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 80, 384, 8, 120, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 80, 384, 8, 120, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 80, 384, 8, 120, 128, 8, 8, 128, 4);
// cta<4,32,384> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 32, 384, 24, 48, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 32, 384, 24, 48, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 32, 384, 24, 48, 128, 8, 8, 128, 4);
// cta<4,48,384> warp<24,72,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 48, 384, 24, 72, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 48, 384, 24, 72, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 48, 384, 24, 72, 128, 8, 8, 128, 4);
// cta<4,64,384> warp<24,96,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 64, 384, 24, 96, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 64, 384, 24, 96, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 64, 384, 24, 96, 128, 8, 8, 128, 4);
// cta<4,80,384> warp<24,120,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 80, 384, 24, 120, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 80, 384, 24, 120, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 80, 384, 24, 120, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<48,48,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 32, 384, 48, 48, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 32, 384, 48, 48, 128, 8, 8, 128, 3);
// cta<8,48,384> warp<48,72,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 48, 384, 48, 72, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 48, 384, 48, 72, 128, 8, 8, 128, 3);
// cta<8,64,384> warp<48,96,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 64, 384, 48, 96, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 64, 384, 48, 96, 128, 8, 8, 128, 3);
// cta<8,80,384> warp<48,120,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 80, 384, 48, 120, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 80, 384, 48, 120, 128, 8, 8, 128, 3);
// cta<8,32,384> warp<24,96,128> mma<8,8,128>   WARPS[2x2]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 32, 384, 24, 96, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 32, 384, 24, 96, 128, 8, 8, 128, 3);
// cta<1,32,512> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 32, 512, 8, 96, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 32, 512, 8, 96, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 32, 512, 8, 96, 128, 8, 8, 128, 4);
// cta<4,32,512> warp<24,96,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 32, 512, 24, 96, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 32, 512, 24, 96, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 32, 512, 24, 96, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<48,96,128> mma<8,8,128>   WARPS[1x2]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 32, 512, 48, 96, 128, 8, 8, 128, 2);
// cta<1,32,512> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 32, 512, 8, 48, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 32, 512, 8, 48, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 32, 512, 8, 48, 128, 8, 8, 128, 4);
// cta<1,48,512> warp<8,72,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 48, 512, 8, 72, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 48, 512, 8, 72, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 48, 512, 8, 72, 128, 8, 8, 128, 4);
// cta<1,64,512> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 64, 512, 8, 96, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 64, 512, 8, 96, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 64, 512, 8, 96, 128, 8, 8, 128, 4);
// cta<1,80,512> warp<8,120,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 80, 512, 8, 120, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 1, 80, 512, 8, 120, 128, 8, 8, 128, 3);
// cta<4,32,512> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 32, 512, 24, 48, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 32, 512, 24, 48, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 32, 512, 24, 48, 128, 8, 8, 128, 4);
// cta<4,48,512> warp<24,72,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 48, 512, 24, 72, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 48, 512, 24, 72, 128, 8, 8, 128, 3);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 48, 512, 24, 72, 128, 8, 8, 128, 4);
// cta<4,64,512> warp<24,96,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 64, 512, 24, 96, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 64, 512, 24, 96, 128, 8, 8, 128, 3);
// cta<4,80,512> warp<24,120,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 80, 512, 24, 120, 128, 8, 8, 128, 2);
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 4, 80, 512, 24, 120, 128, 8, 8, 128, 3);
// cta<8,32,512> warp<48,48,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 32, 512, 48, 48, 128, 8, 8, 128, 2);
// cta<8,48,512> warp<48,72,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 48, 512, 48, 72, 128, 8, 8, 128, 2);
// cta<8,64,512> warp<48,96,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 64, 512, 48, 96, 128, 8, 8, 128, 2);
// cta<8,80,512> warp<48,120,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 80, 512, 48, 120, 128, 8, 8, 128, 2);
// cta<8,32,512> warp<24,96,128> mma<8,8,128>   WARPS[2x2]
AQ_INSTANTIATE_FUN(AqBMMA, 6, 6, true, 8, 32, 512, 24, 96, 128, 8, 8, 128, 2);
#endif