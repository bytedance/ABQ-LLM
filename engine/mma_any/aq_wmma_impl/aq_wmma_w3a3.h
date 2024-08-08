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

#pragma once
#include "common/base.h"
#include "mma_any/aq_wmma_op.h"

#ifdef W3A3
////// W3A3 int
// cta<2,32,256> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 32, 256, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 32, 256, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 32, 256, 8, 48, 128, 8, 8, 128, 4);
// cta<2,48,256> warp<8,72,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 48, 256, 8, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 48, 256, 8, 72, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 48, 256, 8, 72, 128, 8, 8, 128, 4);
// cta<2,64,256> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 64, 256, 8, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 64, 256, 8, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 64, 256, 8, 96, 128, 8, 8, 128, 4);
// cta<2,80,256> warp<8,120,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 80, 256, 8, 120, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 80, 256, 8, 120, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 80, 256, 8, 120, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<24,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 32, 256, 24, 48, 128, 8, 8, 128, 2);
// cta<8,48,256> warp<24,72,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 48, 256, 24, 72, 128, 8, 8, 128, 2);
// cta<8,64,256> warp<24,96,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 64, 256, 24, 96, 128, 8, 8, 128, 2);
// cta<8,80,256> warp<24,120,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 80, 256, 24, 120, 128, 8, 8, 128, 2);
// cta<2,32,256> warp<8,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 32, 256, 8, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 32, 256, 8, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 32, 256, 8, 24, 128, 8, 8, 128, 4);
// cta<2,64,256> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 64, 256, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 64, 256, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 64, 256, 8, 48, 128, 8, 8, 128, 4);
// cta<2,96,256> warp<8,72,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 96, 256, 8, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 96, 256, 8, 72, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 96, 256, 8, 72, 128, 8, 8, 128, 4);
// cta<2,128,256> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 128, 256, 8, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 128, 256, 8, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 128, 256, 8, 96, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<24,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 32, 256, 24, 24, 128, 8, 8, 128, 2);
// cta<8,64,256> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 64, 256, 24, 48, 128, 8, 8, 128, 2);
// cta<8,96,256> warp<24,72,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 96, 256, 24, 72, 128, 8, 8, 128, 2);
// cta<8,128,256> warp<24,96,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 128, 256, 24, 96, 128, 8, 8, 128, 2);
// cta<2,32,384> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 32, 384, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 32, 384, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 32, 384, 8, 48, 128, 8, 8, 128, 4);
// cta<2,48,384> warp<8,72,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 48, 384, 8, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 48, 384, 8, 72, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 48, 384, 8, 72, 128, 8, 8, 128, 4);
// cta<2,64,384> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 64, 384, 8, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 64, 384, 8, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 64, 384, 8, 96, 128, 8, 8, 128, 4);
// cta<2,80,384> warp<8,120,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 80, 384, 8, 120, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 80, 384, 8, 120, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 80, 384, 8, 120, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<24,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 32, 384, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 32, 384, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 32, 384, 24, 48, 128, 8, 8, 128, 4);
// cta<8,48,384> warp<24,72,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 48, 384, 24, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 48, 384, 24, 72, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 48, 384, 24, 72, 128, 8, 8, 128, 4);
// cta<8,64,384> warp<24,96,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 64, 384, 24, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 64, 384, 24, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 64, 384, 24, 96, 128, 8, 8, 128, 4);
// cta<8,80,384> warp<24,120,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 80, 384, 24, 120, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 80, 384, 24, 120, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 80, 384, 24, 120, 128, 8, 8, 128, 4);
// cta<2,32,384> warp<8,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 32, 384, 8, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 32, 384, 8, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 32, 384, 8, 24, 128, 8, 8, 128, 4);
// cta<2,64,384> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 64, 384, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 64, 384, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 64, 384, 8, 48, 128, 8, 8, 128, 4);
// cta<2,96,384> warp<8,72,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 96, 384, 8, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 96, 384, 8, 72, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 96, 384, 8, 72, 128, 8, 8, 128, 4);
// cta<2,128,384> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 128, 384, 8, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 128, 384, 8, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 128, 384, 8, 96, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<24,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 32, 384, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 32, 384, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 32, 384, 24, 24, 128, 8, 8, 128, 4);
// cta<8,64,384> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 64, 384, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 64, 384, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 64, 384, 24, 48, 128, 8, 8, 128, 4);
// cta<8,96,384> warp<24,72,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 96, 384, 24, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 96, 384, 24, 72, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 96, 384, 24, 72, 128, 8, 8, 128, 4);
// cta<8,128,384> warp<24,96,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 128, 384, 24, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 128, 384, 24, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 128, 384, 24, 96, 128, 8, 8, 128, 4);
// cta<2,32,512> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 32, 512, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 32, 512, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 32, 512, 8, 48, 128, 8, 8, 128, 4);
// cta<2,48,512> warp<8,72,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 48, 512, 8, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 48, 512, 8, 72, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 48, 512, 8, 72, 128, 8, 8, 128, 4);
// cta<2,64,512> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 64, 512, 8, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 64, 512, 8, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 64, 512, 8, 96, 128, 8, 8, 128, 4);
// cta<2,80,512> warp<8,120,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 80, 512, 8, 120, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 80, 512, 8, 120, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 80, 512, 8, 120, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<24,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 32, 512, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 32, 512, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 32, 512, 24, 48, 128, 8, 8, 128, 4);
// cta<8,48,512> warp<24,72,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 48, 512, 24, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 48, 512, 24, 72, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 48, 512, 24, 72, 128, 8, 8, 128, 4);
// cta<8,64,512> warp<24,96,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 64, 512, 24, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 64, 512, 24, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 64, 512, 24, 96, 128, 8, 8, 128, 4);
// cta<8,80,512> warp<24,120,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 80, 512, 24, 120, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 80, 512, 24, 120, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 80, 512, 24, 120, 128, 8, 8, 128, 4);
// cta<2,32,512> warp<8,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 32, 512, 8, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 32, 512, 8, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 32, 512, 8, 24, 128, 8, 8, 128, 4);
// cta<2,64,512> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 64, 512, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 64, 512, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 64, 512, 8, 48, 128, 8, 8, 128, 4);
// cta<2,96,512> warp<8,72,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 96, 512, 8, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 96, 512, 8, 72, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 96, 512, 8, 72, 128, 8, 8, 128, 4);
// cta<2,128,512> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 128, 512, 8, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 128, 512, 8, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 2, 128, 512, 8, 96, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<24,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 32, 512, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 32, 512, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 32, 512, 24, 24, 128, 8, 8, 128, 4);
// cta<8,64,512> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 64, 512, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 64, 512, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 64, 512, 24, 48, 128, 8, 8, 128, 4);
// cta<8,96,512> warp<24,72,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 96, 512, 24, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 96, 512, 24, 72, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 96, 512, 24, 72, 128, 8, 8, 128, 4);
// cta<8,128,512> warp<24,96,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 128, 512, 24, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 3, 3, true, 8, 128, 512, 24, 96, 128, 8, 8, 128, 3);
#endif