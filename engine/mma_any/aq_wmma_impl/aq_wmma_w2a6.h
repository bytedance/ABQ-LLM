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

#ifdef W2A6
////// W2A6 int
// cta<4,8,128> warp<24,8,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 8, 128, 24, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 8, 128, 24, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 8, 128, 24, 8, 128, 8, 8, 128, 4);
// cta<4,16,128> warp<24,16,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 16, 128, 24, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 16, 128, 24, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 16, 128, 24, 16, 128, 8, 8, 128, 4);
// cta<4,24,128> warp<24,24,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 24, 128, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 24, 128, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 24, 128, 24, 24, 128, 8, 8, 128, 4);
// cta<4,32,128> warp<24,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 128, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 128, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 128, 24, 32, 128, 8, 8, 128, 4);
// cta<4,40,128> warp<24,40,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 40, 128, 24, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 40, 128, 24, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 40, 128, 24, 40, 128, 8, 8, 128, 4);
// cta<4,48,128> warp<24,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 48, 128, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 48, 128, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 48, 128, 24, 48, 128, 8, 8, 128, 4);
// cta<4,56,128> warp<24,56,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 56, 128, 24, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 56, 128, 24, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 56, 128, 24, 56, 128, 8, 8, 128, 4);
// cta<4,64,128> warp<24,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 128, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 128, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 128, 24, 64, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<48,8,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 128, 48, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 128, 48, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 128, 48, 8, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<48,16,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 128, 48, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 128, 48, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 128, 48, 16, 128, 8, 8, 128, 4);
// cta<8,24,128> warp<48,24,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 128, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 128, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 128, 48, 24, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<48,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 128, 48, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 128, 48, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 128, 48, 32, 128, 8, 8, 128, 4);
// cta<8,40,128> warp<48,40,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 40, 128, 48, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 40, 128, 48, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 40, 128, 48, 40, 128, 8, 8, 128, 4);
// cta<8,48,128> warp<48,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 128, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 128, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 128, 48, 48, 128, 8, 8, 128, 4);
// cta<8,56,128> warp<48,56,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 56, 128, 48, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 56, 128, 48, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 56, 128, 48, 56, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<48,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 128, 48, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 128, 48, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 128, 48, 64, 128, 8, 8, 128, 4);
// cta<4,16,128> warp<24,8,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 16, 128, 24, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 16, 128, 24, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 16, 128, 24, 8, 128, 8, 8, 128, 4);
// cta<4,32,128> warp<24,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 128, 24, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 128, 24, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 128, 24, 16, 128, 8, 8, 128, 4);
// cta<4,48,128> warp<24,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 48, 128, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 48, 128, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 48, 128, 24, 24, 128, 8, 8, 128, 4);
// cta<4,64,128> warp<24,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 128, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 128, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 128, 24, 32, 128, 8, 8, 128, 4);
// cta<4,80,128> warp<24,40,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 80, 128, 24, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 80, 128, 24, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 80, 128, 24, 40, 128, 8, 8, 128, 4);
// cta<4,96,128> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 96, 128, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 96, 128, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 96, 128, 24, 48, 128, 8, 8, 128, 4);
// cta<4,112,128> warp<24,56,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 112, 128, 24, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 112, 128, 24, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 112, 128, 24, 56, 128, 8, 8, 128, 4);
// cta<4,128,128> warp<24,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 128, 128, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 128, 128, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 128, 128, 24, 64, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<48,8,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 128, 48, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 128, 48, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 128, 48, 8, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<48,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 128, 48, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 128, 48, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 128, 48, 16, 128, 8, 8, 128, 4);
// cta<8,48,128> warp<48,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 128, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 128, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 128, 48, 24, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<48,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 128, 48, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 128, 48, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 128, 48, 32, 128, 8, 8, 128, 4);
// cta<8,80,128> warp<48,40,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 80, 128, 48, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 80, 128, 48, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 80, 128, 48, 40, 128, 8, 8, 128, 4);
// cta<8,96,128> warp<48,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 128, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 128, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 128, 48, 48, 128, 8, 8, 128, 4);
// cta<8,112,128> warp<48,56,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 112, 128, 48, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 112, 128, 48, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 112, 128, 48, 56, 128, 8, 8, 128, 4);
// cta<8,128,128> warp<48,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 128, 48, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 128, 48, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 128, 48, 64, 128, 8, 8, 128, 4);
// cta<4,32,128> warp<24,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 128, 24, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 128, 24, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 128, 24, 8, 128, 8, 8, 128, 4);
// cta<4,64,128> warp<24,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 128, 24, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 128, 24, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 128, 24, 16, 128, 8, 8, 128, 4);
// cta<4,96,128> warp<24,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 96, 128, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 96, 128, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 96, 128, 24, 24, 128, 8, 8, 128, 4);
// cta<4,128,128> warp<24,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 128, 128, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 128, 128, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 128, 128, 24, 32, 128, 8, 8, 128, 4);
// cta<4,256,128> warp<24,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 256, 128, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 256, 128, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 256, 128, 24, 64, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<48,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 128, 48, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 128, 48, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 128, 48, 8, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<48,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 128, 48, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 128, 48, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 128, 48, 16, 128, 8, 8, 128, 4);
// cta<8,96,128> warp<48,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 128, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 128, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 128, 48, 24, 128, 8, 8, 128, 4);
// cta<8,128,128> warp<48,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 128, 48, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 128, 48, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 128, 48, 32, 128, 8, 8, 128, 4);
// cta<8,256,128> warp<48,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 256, 128, 48, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 256, 128, 48, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 256, 128, 48, 64, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<24,16,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 128, 24, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 128, 24, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 128, 24, 16, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<24,32,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 128, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 128, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 128, 24, 32, 128, 8, 8, 128, 4);
// cta<8,24,128> warp<24,48,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 128, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 128, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 128, 24, 48, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<24,64,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 128, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 128, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 128, 24, 64, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<24,8,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 128, 24, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 128, 24, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 128, 24, 8, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<24,16,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 128, 24, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 128, 24, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 128, 24, 16, 128, 8, 8, 128, 4);
// cta<8,24,128> warp<24,24,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 128, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 128, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 128, 24, 24, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<24,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 128, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 128, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 128, 24, 32, 128, 8, 8, 128, 4);
// cta<8,40,128> warp<24,40,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 40, 128, 24, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 40, 128, 24, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 40, 128, 24, 40, 128, 8, 8, 128, 4);
// cta<8,48,128> warp<24,48,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 128, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 128, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 128, 24, 48, 128, 8, 8, 128, 4);
// cta<8,56,128> warp<24,56,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 56, 128, 24, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 56, 128, 24, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 56, 128, 24, 56, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<24,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 128, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 128, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 128, 24, 64, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<24,8,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 128, 24, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 128, 24, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 128, 24, 8, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<24,16,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 128, 24, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 128, 24, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 128, 24, 16, 128, 8, 8, 128, 4);
// cta<8,48,128> warp<24,24,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 128, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 128, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 128, 24, 24, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<24,32,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 128, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 128, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 128, 24, 32, 128, 8, 8, 128, 4);
// cta<8,80,128> warp<24,40,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 80, 128, 24, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 80, 128, 24, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 80, 128, 24, 40, 128, 8, 8, 128, 4);
// cta<8,96,128> warp<24,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 128, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 128, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 128, 24, 48, 128, 8, 8, 128, 4);
// cta<8,112,128> warp<24,56,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 112, 128, 24, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 112, 128, 24, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 112, 128, 24, 56, 128, 8, 8, 128, 4);
// cta<8,128,128> warp<24,64,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 128, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 128, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 128, 24, 64, 128, 8, 8, 128, 4);
// cta<4,8,256> warp<24,8,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 8, 256, 24, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 8, 256, 24, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 8, 256, 24, 8, 128, 8, 8, 128, 4);
// cta<4,16,256> warp<24,16,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 16, 256, 24, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 16, 256, 24, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 16, 256, 24, 16, 128, 8, 8, 128, 4);
// cta<4,24,256> warp<24,24,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 24, 256, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 24, 256, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 24, 256, 24, 24, 128, 8, 8, 128, 4);
// cta<4,32,256> warp<24,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 256, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 256, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 256, 24, 32, 128, 8, 8, 128, 4);
// cta<4,40,256> warp<24,40,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 40, 256, 24, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 40, 256, 24, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 40, 256, 24, 40, 128, 8, 8, 128, 4);
// cta<4,48,256> warp<24,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 48, 256, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 48, 256, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 48, 256, 24, 48, 128, 8, 8, 128, 4);
// cta<4,56,256> warp<24,56,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 56, 256, 24, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 56, 256, 24, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 56, 256, 24, 56, 128, 8, 8, 128, 4);
// cta<4,64,256> warp<24,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 256, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 256, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 256, 24, 64, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<48,8,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 256, 48, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 256, 48, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 256, 48, 8, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<48,16,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 256, 48, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 256, 48, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 256, 48, 16, 128, 8, 8, 128, 4);
// cta<8,24,256> warp<48,24,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 256, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 256, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 256, 48, 24, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<48,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 256, 48, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 256, 48, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 256, 48, 32, 128, 8, 8, 128, 4);
// cta<8,40,256> warp<48,40,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 40, 256, 48, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 40, 256, 48, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 40, 256, 48, 40, 128, 8, 8, 128, 4);
// cta<8,48,256> warp<48,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 256, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 256, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 256, 48, 48, 128, 8, 8, 128, 4);
// cta<8,56,256> warp<48,56,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 56, 256, 48, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 56, 256, 48, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 56, 256, 48, 56, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<48,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 256, 48, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 256, 48, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 256, 48, 64, 128, 8, 8, 128, 4);
// cta<4,16,256> warp<24,8,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 16, 256, 24, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 16, 256, 24, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 16, 256, 24, 8, 128, 8, 8, 128, 4);
// cta<4,32,256> warp<24,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 256, 24, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 256, 24, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 256, 24, 16, 128, 8, 8, 128, 4);
// cta<4,48,256> warp<24,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 48, 256, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 48, 256, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 48, 256, 24, 24, 128, 8, 8, 128, 4);
// cta<4,64,256> warp<24,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 256, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 256, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 256, 24, 32, 128, 8, 8, 128, 4);
// cta<4,80,256> warp<24,40,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 80, 256, 24, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 80, 256, 24, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 80, 256, 24, 40, 128, 8, 8, 128, 4);
// cta<4,96,256> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 96, 256, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 96, 256, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 96, 256, 24, 48, 128, 8, 8, 128, 4);
// cta<4,112,256> warp<24,56,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 112, 256, 24, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 112, 256, 24, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 112, 256, 24, 56, 128, 8, 8, 128, 4);
// cta<4,128,256> warp<24,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 128, 256, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 128, 256, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 128, 256, 24, 64, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<48,8,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 256, 48, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 256, 48, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 256, 48, 8, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<48,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 256, 48, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 256, 48, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 256, 48, 16, 128, 8, 8, 128, 4);
// cta<8,48,256> warp<48,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 256, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 256, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 256, 48, 24, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<48,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 256, 48, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 256, 48, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 256, 48, 32, 128, 8, 8, 128, 4);
// cta<8,80,256> warp<48,40,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 80, 256, 48, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 80, 256, 48, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 80, 256, 48, 40, 128, 8, 8, 128, 4);
// cta<8,96,256> warp<48,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 256, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 256, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 256, 48, 48, 128, 8, 8, 128, 4);
// cta<8,112,256> warp<48,56,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 112, 256, 48, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 112, 256, 48, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 112, 256, 48, 56, 128, 8, 8, 128, 4);
// cta<8,128,256> warp<48,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 256, 48, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 256, 48, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 256, 48, 64, 128, 8, 8, 128, 4);
// cta<4,32,256> warp<24,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 256, 24, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 256, 24, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 256, 24, 8, 128, 8, 8, 128, 4);
// cta<4,64,256> warp<24,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 256, 24, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 256, 24, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 256, 24, 16, 128, 8, 8, 128, 4);
// cta<4,96,256> warp<24,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 96, 256, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 96, 256, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 96, 256, 24, 24, 128, 8, 8, 128, 4);
// cta<4,128,256> warp<24,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 128, 256, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 128, 256, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 128, 256, 24, 32, 128, 8, 8, 128, 4);
// cta<4,256,256> warp<24,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 256, 256, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 256, 256, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 256, 256, 24, 64, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<48,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 256, 48, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 256, 48, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 256, 48, 8, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<48,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 256, 48, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 256, 48, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 256, 48, 16, 128, 8, 8, 128, 4);
// cta<8,96,256> warp<48,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 256, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 256, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 256, 48, 24, 128, 8, 8, 128, 4);
// cta<8,128,256> warp<48,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 256, 48, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 256, 48, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 256, 48, 32, 128, 8, 8, 128, 4);
// cta<8,256,256> warp<48,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 256, 256, 48, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 256, 256, 48, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 256, 256, 48, 64, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<24,16,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 256, 24, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 256, 24, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 256, 24, 16, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<24,32,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 256, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 256, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 256, 24, 32, 128, 8, 8, 128, 4);
// cta<8,24,256> warp<24,48,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 256, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 256, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 256, 24, 48, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<24,64,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 256, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 256, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 256, 24, 64, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<24,8,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 256, 24, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 256, 24, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 256, 24, 8, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<24,16,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 256, 24, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 256, 24, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 256, 24, 16, 128, 8, 8, 128, 4);
// cta<8,24,256> warp<24,24,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 256, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 256, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 256, 24, 24, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<24,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 256, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 256, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 256, 24, 32, 128, 8, 8, 128, 4);
// cta<8,40,256> warp<24,40,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 40, 256, 24, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 40, 256, 24, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 40, 256, 24, 40, 128, 8, 8, 128, 4);
// cta<8,48,256> warp<24,48,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 256, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 256, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 256, 24, 48, 128, 8, 8, 128, 4);
// cta<8,56,256> warp<24,56,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 56, 256, 24, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 56, 256, 24, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 56, 256, 24, 56, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<24,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 256, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 256, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 256, 24, 64, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<24,8,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 256, 24, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 256, 24, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 256, 24, 8, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<24,16,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 256, 24, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 256, 24, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 256, 24, 16, 128, 8, 8, 128, 4);
// cta<8,48,256> warp<24,24,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 256, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 256, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 256, 24, 24, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<24,32,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 256, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 256, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 256, 24, 32, 128, 8, 8, 128, 4);
// cta<8,80,256> warp<24,40,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 80, 256, 24, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 80, 256, 24, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 80, 256, 24, 40, 128, 8, 8, 128, 4);
// cta<8,96,256> warp<24,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 256, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 256, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 256, 24, 48, 128, 8, 8, 128, 4);
// cta<8,112,256> warp<24,56,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 112, 256, 24, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 112, 256, 24, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 112, 256, 24, 56, 128, 8, 8, 128, 4);
// cta<8,128,256> warp<24,64,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 256, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 256, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 256, 24, 64, 128, 8, 8, 128, 4);
// cta<4,8,512> warp<24,8,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 8, 512, 24, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 8, 512, 24, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 8, 512, 24, 8, 128, 8, 8, 128, 4);
// cta<4,16,512> warp<24,16,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 16, 512, 24, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 16, 512, 24, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 16, 512, 24, 16, 128, 8, 8, 128, 4);
// cta<4,24,512> warp<24,24,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 24, 512, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 24, 512, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 24, 512, 24, 24, 128, 8, 8, 128, 4);
// cta<4,32,512> warp<24,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 512, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 512, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 512, 24, 32, 128, 8, 8, 128, 4);
// cta<4,40,512> warp<24,40,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 40, 512, 24, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 40, 512, 24, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 40, 512, 24, 40, 128, 8, 8, 128, 4);
// cta<4,48,512> warp<24,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 48, 512, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 48, 512, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 48, 512, 24, 48, 128, 8, 8, 128, 4);
// cta<4,56,512> warp<24,56,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 56, 512, 24, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 56, 512, 24, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 56, 512, 24, 56, 128, 8, 8, 128, 4);
// cta<4,64,512> warp<24,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 512, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 512, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 512, 24, 64, 128, 8, 8, 128, 4);
// cta<8,8,512> warp<48,8,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 512, 48, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 512, 48, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 512, 48, 8, 128, 8, 8, 128, 4);
// cta<8,16,512> warp<48,16,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 512, 48, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 512, 48, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 512, 48, 16, 128, 8, 8, 128, 4);
// cta<8,24,512> warp<48,24,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 512, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 512, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 512, 48, 24, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<48,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 512, 48, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 512, 48, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 512, 48, 32, 128, 8, 8, 128, 4);
// cta<8,40,512> warp<48,40,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 40, 512, 48, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 40, 512, 48, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 40, 512, 48, 40, 128, 8, 8, 128, 4);
// cta<8,48,512> warp<48,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 512, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 512, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 512, 48, 48, 128, 8, 8, 128, 4);
// cta<8,56,512> warp<48,56,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 56, 512, 48, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 56, 512, 48, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 56, 512, 48, 56, 128, 8, 8, 128, 4);
// cta<8,64,512> warp<48,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 512, 48, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 512, 48, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 512, 48, 64, 128, 8, 8, 128, 4);
// cta<4,16,512> warp<24,8,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 16, 512, 24, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 16, 512, 24, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 16, 512, 24, 8, 128, 8, 8, 128, 4);
// cta<4,32,512> warp<24,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 512, 24, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 512, 24, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 512, 24, 16, 128, 8, 8, 128, 4);
// cta<4,48,512> warp<24,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 48, 512, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 48, 512, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 48, 512, 24, 24, 128, 8, 8, 128, 4);
// cta<4,64,512> warp<24,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 512, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 512, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 512, 24, 32, 128, 8, 8, 128, 4);
// cta<4,80,512> warp<24,40,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 80, 512, 24, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 80, 512, 24, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 80, 512, 24, 40, 128, 8, 8, 128, 4);
// cta<4,96,512> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 96, 512, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 96, 512, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 96, 512, 24, 48, 128, 8, 8, 128, 4);
// cta<4,112,512> warp<24,56,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 112, 512, 24, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 112, 512, 24, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 112, 512, 24, 56, 128, 8, 8, 128, 4);
// cta<4,128,512> warp<24,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 128, 512, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 128, 512, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 128, 512, 24, 64, 128, 8, 8, 128, 4);
// cta<8,16,512> warp<48,8,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 512, 48, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 512, 48, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 512, 48, 8, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<48,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 512, 48, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 512, 48, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 512, 48, 16, 128, 8, 8, 128, 4);
// cta<8,48,512> warp<48,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 512, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 512, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 512, 48, 24, 128, 8, 8, 128, 4);
// cta<8,64,512> warp<48,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 512, 48, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 512, 48, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 512, 48, 32, 128, 8, 8, 128, 4);
// cta<8,80,512> warp<48,40,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 80, 512, 48, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 80, 512, 48, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 80, 512, 48, 40, 128, 8, 8, 128, 4);
// cta<8,96,512> warp<48,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 512, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 512, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 512, 48, 48, 128, 8, 8, 128, 4);
// cta<8,112,512> warp<48,56,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 112, 512, 48, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 112, 512, 48, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 112, 512, 48, 56, 128, 8, 8, 128, 4);
// cta<8,128,512> warp<48,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 512, 48, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 512, 48, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 512, 48, 64, 128, 8, 8, 128, 4);
// cta<4,32,512> warp<24,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 512, 24, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 512, 24, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 32, 512, 24, 8, 128, 8, 8, 128, 4);
// cta<4,64,512> warp<24,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 512, 24, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 512, 24, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 64, 512, 24, 16, 128, 8, 8, 128, 4);
// cta<4,96,512> warp<24,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 96, 512, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 96, 512, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 96, 512, 24, 24, 128, 8, 8, 128, 4);
// cta<4,128,512> warp<24,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 128, 512, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 128, 512, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 128, 512, 24, 32, 128, 8, 8, 128, 4);
// cta<4,256,512> warp<24,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 4, 256, 512, 24, 64, 128, 8, 8, 128, 2);
// cta<8,32,512> warp<48,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 512, 48, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 512, 48, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 512, 48, 8, 128, 8, 8, 128, 4);
// cta<8,64,512> warp<48,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 512, 48, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 512, 48, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 512, 48, 16, 128, 8, 8, 128, 4);
// cta<8,96,512> warp<48,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 512, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 512, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 512, 48, 24, 128, 8, 8, 128, 4);
// cta<8,128,512> warp<48,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 512, 48, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 512, 48, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 512, 48, 32, 128, 8, 8, 128, 4);
// cta<8,256,512> warp<48,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 256, 512, 48, 64, 128, 8, 8, 128, 2);
// cta<8,8,512> warp<24,16,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 512, 24, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 512, 24, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 512, 24, 16, 128, 8, 8, 128, 4);
// cta<8,16,512> warp<24,32,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 512, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 512, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 512, 24, 32, 128, 8, 8, 128, 4);
// cta<8,24,512> warp<24,48,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 512, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 512, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 512, 24, 48, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<24,64,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 512, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 512, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 512, 24, 64, 128, 8, 8, 128, 4);
// cta<8,8,512> warp<24,8,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 512, 24, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 512, 24, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 8, 512, 24, 8, 128, 8, 8, 128, 4);
// cta<8,16,512> warp<24,16,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 512, 24, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 512, 24, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 512, 24, 16, 128, 8, 8, 128, 4);
// cta<8,24,512> warp<24,24,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 512, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 512, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 24, 512, 24, 24, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<24,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 512, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 512, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 512, 24, 32, 128, 8, 8, 128, 4);
// cta<8,40,512> warp<24,40,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 40, 512, 24, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 40, 512, 24, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 40, 512, 24, 40, 128, 8, 8, 128, 4);
// cta<8,48,512> warp<24,48,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 512, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 512, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 512, 24, 48, 128, 8, 8, 128, 4);
// cta<8,56,512> warp<24,56,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 56, 512, 24, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 56, 512, 24, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 56, 512, 24, 56, 128, 8, 8, 128, 4);
// cta<8,64,512> warp<24,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 512, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 512, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 512, 24, 64, 128, 8, 8, 128, 4);
// cta<8,16,512> warp<24,8,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 512, 24, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 512, 24, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 16, 512, 24, 8, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<24,16,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 512, 24, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 512, 24, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 32, 512, 24, 16, 128, 8, 8, 128, 4);
// cta<8,48,512> warp<24,24,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 512, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 512, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 48, 512, 24, 24, 128, 8, 8, 128, 4);
// cta<8,64,512> warp<24,32,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 512, 24, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 512, 24, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 64, 512, 24, 32, 128, 8, 8, 128, 4);
// cta<8,80,512> warp<24,40,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 80, 512, 24, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 80, 512, 24, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 80, 512, 24, 40, 128, 8, 8, 128, 4);
// cta<8,96,512> warp<24,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 512, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 512, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 96, 512, 24, 48, 128, 8, 8, 128, 4);
// cta<8,112,512> warp<24,56,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 112, 512, 24, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 112, 512, 24, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 112, 512, 24, 56, 128, 8, 8, 128, 4);
// cta<8,128,512> warp<24,64,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 512, 24, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 512, 24, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 2, true, 8, 128, 512, 24, 64, 128, 8, 8, 128, 4);
#endif