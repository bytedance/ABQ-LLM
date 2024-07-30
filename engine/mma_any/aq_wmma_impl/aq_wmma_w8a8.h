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

#ifdef W8A8
////// W8A8 int
// cta<1,8,128> warp<8,64,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 128, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 128, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 128, 8, 64, 128, 8, 8, 128, 4);
// cta<2,8,128> warp<16,64,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 128, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 128, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 128, 16, 64, 128, 8, 8, 128, 4);
// cta<4,8,128> warp<32,64,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 32, 64, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<64,64,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 64, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 64, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 64, 64, 128, 8, 8, 128, 4);
// cta<1,8,128> warp<8,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 128, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 128, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 128, 8, 32, 128, 8, 8, 128, 4);
// cta<1,16,128> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 128, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 128, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 128, 8, 64, 128, 8, 8, 128, 4);
// cta<2,8,128> warp<16,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 128, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 128, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 128, 16, 32, 128, 8, 8, 128, 4);
// cta<2,16,128> warp<16,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 128, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 128, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 128, 16, 64, 128, 8, 8, 128, 4);
// cta<4,8,128> warp<32,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 32, 32, 128, 8, 8, 128, 4);
// cta<4,16,128> warp<32,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 32, 64, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<64,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 64, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 64, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 64, 32, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<64,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 64, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 64, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 64, 64, 128, 8, 8, 128, 4);
// cta<1,8,128> warp<8,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 128, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 128, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 128, 8, 16, 128, 8, 8, 128, 4);
// cta<1,16,128> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 128, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 128, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 128, 8, 32, 128, 8, 8, 128, 4);
// cta<1,24,128> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 24, 128, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 24, 128, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 24, 128, 8, 48, 128, 8, 8, 128, 4);
// cta<1,32,128> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 128, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 128, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 128, 8, 64, 128, 8, 8, 128, 4);
// cta<2,8,128> warp<16,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 128, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 128, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 128, 16, 16, 128, 8, 8, 128, 4);
// cta<2,16,128> warp<16,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 128, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 128, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 128, 16, 32, 128, 8, 8, 128, 4);
// cta<2,24,128> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 128, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 128, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 128, 16, 48, 128, 8, 8, 128, 4);
// cta<2,32,128> warp<16,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 128, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 128, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 128, 16, 64, 128, 8, 8, 128, 4);
// cta<4,8,128> warp<32,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 32, 16, 128, 8, 8, 128, 4);
// cta<4,16,128> warp<32,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 32, 32, 128, 8, 8, 128, 4);
// cta<4,24,128> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 128, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 128, 32, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 128, 32, 48, 128, 8, 8, 128, 4);
// cta<4,32,128> warp<32,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 128, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 128, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 128, 32, 64, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<64,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 64, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 64, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 64, 16, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<64,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 64, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 64, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 64, 32, 128, 8, 8, 128, 4);
// cta<8,24,128> warp<64,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 128, 64, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 128, 64, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 128, 64, 48, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<64,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 128, 64, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 128, 64, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 128, 64, 64, 128, 8, 8, 128, 4);
// cta<1,8,128> warp<8,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 128, 8, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 128, 8, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 128, 8, 8, 128, 8, 8, 128, 4);
// cta<1,16,128> warp<8,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 128, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 128, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 128, 8, 16, 128, 8, 8, 128, 4);
// cta<1,24,128> warp<8,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 24, 128, 8, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 24, 128, 8, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 24, 128, 8, 24, 128, 8, 8, 128, 4);
// cta<1,32,128> warp<8,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 128, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 128, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 128, 8, 32, 128, 8, 8, 128, 4);
// cta<1,40,128> warp<8,40,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 40, 128, 8, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 40, 128, 8, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 40, 128, 8, 40, 128, 8, 8, 128, 4);
// cta<1,48,128> warp<8,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 48, 128, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 48, 128, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 48, 128, 8, 48, 128, 8, 8, 128, 4);
// cta<1,56,128> warp<8,56,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 56, 128, 8, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 56, 128, 8, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 56, 128, 8, 56, 128, 8, 8, 128, 4);
// cta<1,64,128> warp<8,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 64, 128, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 64, 128, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 64, 128, 8, 64, 128, 8, 8, 128, 4);
// cta<2,8,128> warp<16,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 128, 16, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 128, 16, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 128, 16, 8, 128, 8, 8, 128, 4);
// cta<2,16,128> warp<16,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 128, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 128, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 128, 16, 16, 128, 8, 8, 128, 4);
// cta<2,24,128> warp<16,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 128, 16, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 128, 16, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 128, 16, 24, 128, 8, 8, 128, 4);
// cta<2,32,128> warp<16,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 128, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 128, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 128, 16, 32, 128, 8, 8, 128, 4);
// cta<2,40,128> warp<16,40,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 40, 128, 16, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 40, 128, 16, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 40, 128, 16, 40, 128, 8, 8, 128, 4);
// cta<2,48,128> warp<16,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 48, 128, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 48, 128, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 48, 128, 16, 48, 128, 8, 8, 128, 4);
// cta<2,56,128> warp<16,56,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 56, 128, 16, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 56, 128, 16, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 56, 128, 16, 56, 128, 8, 8, 128, 4);
// cta<2,64,128> warp<16,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 64, 128, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 64, 128, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 64, 128, 16, 64, 128, 8, 8, 128, 4);
// cta<4,8,128> warp<32,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 32, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 32, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 32, 8, 128, 8, 8, 128, 4);
// cta<4,16,128> warp<32,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 32, 16, 128, 8, 8, 128, 4);
// cta<4,24,128> warp<32,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 128, 32, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 128, 32, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 128, 32, 24, 128, 8, 8, 128, 4);
// cta<4,32,128> warp<32,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 128, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 128, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 128, 32, 32, 128, 8, 8, 128, 4);
// cta<4,40,128> warp<32,40,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 40, 128, 32, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 40, 128, 32, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 40, 128, 32, 40, 128, 8, 8, 128, 4);
// cta<4,48,128> warp<32,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 48, 128, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 48, 128, 32, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 48, 128, 32, 48, 128, 8, 8, 128, 4);
// cta<4,56,128> warp<32,56,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 56, 128, 32, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 56, 128, 32, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 56, 128, 32, 56, 128, 8, 8, 128, 4);
// cta<4,64,128> warp<32,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 64, 128, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 64, 128, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 64, 128, 32, 64, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<64,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 64, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 64, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 64, 8, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<64,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 64, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 64, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 64, 16, 128, 8, 8, 128, 4);
// cta<8,24,128> warp<64,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 128, 64, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 128, 64, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 128, 64, 24, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<64,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 128, 64, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 128, 64, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 128, 64, 32, 128, 8, 8, 128, 4);
// cta<8,40,128> warp<64,40,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 40, 128, 64, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 40, 128, 64, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 40, 128, 64, 40, 128, 8, 8, 128, 4);
// cta<8,48,128> warp<64,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 48, 128, 64, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 48, 128, 64, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 48, 128, 64, 48, 128, 8, 8, 128, 4);
// cta<1,16,128> warp<8,8,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 128, 8, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 128, 8, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 128, 8, 8, 128, 8, 8, 128, 4);
// cta<1,32,128> warp<8,16,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 128, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 128, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 128, 8, 16, 128, 8, 8, 128, 4);
// cta<1,48,128> warp<8,24,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 48, 128, 8, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 48, 128, 8, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 48, 128, 8, 24, 128, 8, 8, 128, 4);
// cta<1,64,128> warp<8,32,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 64, 128, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 64, 128, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 64, 128, 8, 32, 128, 8, 8, 128, 4);
// cta<1,80,128> warp<8,40,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 80, 128, 8, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 80, 128, 8, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 80, 128, 8, 40, 128, 8, 8, 128, 4);
// cta<1,96,128> warp<8,48,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 96, 128, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 96, 128, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 96, 128, 8, 48, 128, 8, 8, 128, 4);
// cta<1,112,128> warp<8,56,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 112, 128, 8, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 112, 128, 8, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 112, 128, 8, 56, 128, 8, 8, 128, 4);
// cta<1,128,128> warp<8,64,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 128, 128, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 128, 128, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 128, 128, 8, 64, 128, 8, 8, 128, 4);
// cta<2,16,128> warp<16,8,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 128, 16, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 128, 16, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 128, 16, 8, 128, 8, 8, 128, 4);
// cta<2,32,128> warp<16,16,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 128, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 128, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 128, 16, 16, 128, 8, 8, 128, 4);
// cta<2,48,128> warp<16,24,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 48, 128, 16, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 48, 128, 16, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 48, 128, 16, 24, 128, 8, 8, 128, 4);
// cta<2,64,128> warp<16,32,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 64, 128, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 64, 128, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 64, 128, 16, 32, 128, 8, 8, 128, 4);
// cta<2,80,128> warp<16,40,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 80, 128, 16, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 80, 128, 16, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 80, 128, 16, 40, 128, 8, 8, 128, 4);
// cta<2,96,128> warp<16,48,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 96, 128, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 96, 128, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 96, 128, 16, 48, 128, 8, 8, 128, 4);
// cta<2,112,128> warp<16,56,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 112, 128, 16, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 112, 128, 16, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 112, 128, 16, 56, 128, 8, 8, 128, 4);
// cta<2,128,128> warp<16,64,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 128, 128, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 128, 128, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 128, 128, 16, 64, 128, 8, 8, 128, 4);
// cta<4,16,128> warp<32,8,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 32, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 32, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 32, 8, 128, 8, 8, 128, 4);
// cta<4,32,128> warp<32,16,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 128, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 128, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 128, 32, 16, 128, 8, 8, 128, 4);
// cta<4,48,128> warp<32,24,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 48, 128, 32, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 48, 128, 32, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 48, 128, 32, 24, 128, 8, 8, 128, 4);
// cta<4,64,128> warp<32,32,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 64, 128, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 64, 128, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 64, 128, 32, 32, 128, 8, 8, 128, 4);
// cta<4,80,128> warp<32,40,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 80, 128, 32, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 80, 128, 32, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 80, 128, 32, 40, 128, 8, 8, 128, 4);
// cta<4,96,128> warp<32,48,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 96, 128, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 96, 128, 32, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 96, 128, 32, 48, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<64,8,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 64, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 64, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 64, 8, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<64,16,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 128, 64, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 128, 64, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 128, 64, 16, 128, 8, 8, 128, 4);
// cta<8,48,128> warp<64,24,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 48, 128, 64, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 48, 128, 64, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 48, 128, 64, 24, 128, 8, 8, 128, 4);
// cta<2,8,128> warp<8,64,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 128, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 128, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 128, 8, 64, 128, 8, 8, 128, 4);
// cta<4,8,128> warp<16,64,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 16, 64, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<32,64,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 32, 64, 128, 8, 8, 128, 4);
// cta<2,8,128> warp<8,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 128, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 128, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 128, 8, 32, 128, 8, 8, 128, 4);
// cta<2,16,128> warp<8,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 128, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 128, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 128, 8, 64, 128, 8, 8, 128, 4);
// cta<4,8,128> warp<16,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 16, 32, 128, 8, 8, 128, 4);
// cta<4,16,128> warp<16,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 16, 64, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<32,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 32, 32, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<32,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 32, 64, 128, 8, 8, 128, 4);
// cta<2,8,128> warp<8,16,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 128, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 128, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 128, 8, 16, 128, 8, 8, 128, 4);
// cta<2,16,128> warp<8,32,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 128, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 128, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 128, 8, 32, 128, 8, 8, 128, 4);
// cta<2,24,128> warp<8,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 128, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 128, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 128, 8, 48, 128, 8, 8, 128, 4);
// cta<2,32,128> warp<8,64,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 128, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 128, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 128, 8, 64, 128, 8, 8, 128, 4);
// cta<4,8,128> warp<16,16,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 16, 16, 128, 8, 8, 128, 4);
// cta<4,16,128> warp<16,32,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 16, 32, 128, 8, 8, 128, 4);
// cta<4,24,128> warp<16,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 128, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 128, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 128, 16, 48, 128, 8, 8, 128, 4);
// cta<4,32,128> warp<16,64,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 128, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 128, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 128, 16, 64, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<32,16,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 32, 16, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<32,32,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 32, 32, 128, 8, 8, 128, 4);
// cta<8,24,128> warp<32,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 128, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 128, 32, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 128, 32, 48, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<32,64,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 128, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 128, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 128, 32, 64, 128, 8, 8, 128, 4);
// cta<2,8,128> warp<8,8,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 128, 8, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 128, 8, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 128, 8, 8, 128, 8, 8, 128, 4);
// cta<2,16,128> warp<8,16,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 128, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 128, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 128, 8, 16, 128, 8, 8, 128, 4);
// cta<2,24,128> warp<8,24,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 128, 8, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 128, 8, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 128, 8, 24, 128, 8, 8, 128, 4);
// cta<2,32,128> warp<8,32,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 128, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 128, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 128, 8, 32, 128, 8, 8, 128, 4);
// cta<2,40,128> warp<8,40,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 40, 128, 8, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 40, 128, 8, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 40, 128, 8, 40, 128, 8, 8, 128, 4);
// cta<2,48,128> warp<8,48,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 48, 128, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 48, 128, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 48, 128, 8, 48, 128, 8, 8, 128, 4);
// cta<2,56,128> warp<8,56,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 56, 128, 8, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 56, 128, 8, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 56, 128, 8, 56, 128, 8, 8, 128, 4);
// cta<2,64,128> warp<8,64,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 64, 128, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 64, 128, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 64, 128, 8, 64, 128, 8, 8, 128, 4);
// cta<4,8,128> warp<16,8,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 16, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 16, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 16, 8, 128, 8, 8, 128, 4);
// cta<4,16,128> warp<16,16,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 16, 16, 128, 8, 8, 128, 4);
// cta<4,24,128> warp<16,24,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 128, 16, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 128, 16, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 128, 16, 24, 128, 8, 8, 128, 4);
// cta<4,32,128> warp<16,32,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 128, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 128, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 128, 16, 32, 128, 8, 8, 128, 4);
// cta<4,40,128> warp<16,40,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 40, 128, 16, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 40, 128, 16, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 40, 128, 16, 40, 128, 8, 8, 128, 4);
// cta<4,48,128> warp<16,48,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 48, 128, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 48, 128, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 48, 128, 16, 48, 128, 8, 8, 128, 4);
// cta<4,56,128> warp<16,56,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 56, 128, 16, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 56, 128, 16, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 56, 128, 16, 56, 128, 8, 8, 128, 4);
// cta<4,64,128> warp<16,64,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 64, 128, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 64, 128, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 64, 128, 16, 64, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<32,8,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 32, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 32, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 32, 8, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<32,16,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 32, 16, 128, 8, 8, 128, 4);
// cta<8,24,128> warp<32,24,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 128, 32, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 128, 32, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 128, 32, 24, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<32,32,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 128, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 128, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 128, 32, 32, 128, 8, 8, 128, 4);
// cta<8,40,128> warp<32,40,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 40, 128, 32, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 40, 128, 32, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 40, 128, 32, 40, 128, 8, 8, 128, 4);
// cta<8,48,128> warp<32,48,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 48, 128, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 48, 128, 32, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 48, 128, 32, 48, 128, 8, 8, 128, 4);
// cta<4,8,128> warp<8,64,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 8, 64, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<16,64,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 16, 64, 128, 8, 8, 128, 4);
// cta<4,8,128> warp<8,32,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 8, 32, 128, 8, 8, 128, 4);
// cta<4,16,128> warp<8,64,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 8, 64, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<16,32,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 16, 32, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<16,64,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 16, 64, 128, 8, 8, 128, 4);
// cta<4,8,128> warp<8,16,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 128, 8, 16, 128, 8, 8, 128, 4);
// cta<4,16,128> warp<8,32,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 128, 8, 32, 128, 8, 8, 128, 4);
// cta<4,24,128> warp<8,48,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 128, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 128, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 128, 8, 48, 128, 8, 8, 128, 4);
// cta<4,32,128> warp<8,64,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 128, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 128, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 128, 8, 64, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<16,16,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 16, 16, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<16,32,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 16, 32, 128, 8, 8, 128, 4);
// cta<8,24,128> warp<16,48,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 128, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 128, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 128, 16, 48, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<16,64,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 128, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 128, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 128, 16, 64, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<8,64,128> mma<8,8,128>   WARPS[8x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 8, 64, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<8,32,128> mma<8,8,128>   WARPS[8x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 128, 8, 32, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<8,64,128> mma<8,8,128>   WARPS[8x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 128, 8, 64, 128, 8, 8, 128, 4);
// cta<1,8,256> warp<8,64,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 256, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 256, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 256, 8, 64, 128, 8, 8, 128, 4);
// cta<2,8,256> warp<16,64,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 256, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 256, 16, 64, 128, 8, 8, 128, 4);
// cta<4,8,256> warp<32,64,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 32, 64, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<64,64,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 64, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 64, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 64, 64, 128, 8, 8, 128, 4);
// cta<1,8,256> warp<8,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 256, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 256, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 256, 8, 32, 128, 8, 8, 128, 4);
// cta<1,16,256> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 256, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 256, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 256, 8, 64, 128, 8, 8, 128, 4);
// cta<2,8,256> warp<16,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 256, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 256, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 256, 16, 32, 128, 8, 8, 128, 4);
// cta<2,16,256> warp<16,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 256, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 256, 16, 64, 128, 8, 8, 128, 4);
// cta<4,8,256> warp<32,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 32, 32, 128, 8, 8, 128, 4);
// cta<4,16,256> warp<32,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 32, 64, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<64,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 64, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 64, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 64, 32, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<64,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 64, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 64, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 64, 64, 128, 8, 8, 128, 4);
// cta<1,8,256> warp<8,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 256, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 256, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 256, 8, 16, 128, 8, 8, 128, 4);
// cta<1,16,256> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 256, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 256, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 256, 8, 32, 128, 8, 8, 128, 4);
// cta<1,24,256> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 24, 256, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 24, 256, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 24, 256, 8, 48, 128, 8, 8, 128, 4);
// cta<1,32,256> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 256, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 256, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 256, 8, 64, 128, 8, 8, 128, 4);
// cta<2,8,256> warp<16,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 256, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 256, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 256, 16, 16, 128, 8, 8, 128, 4);
// cta<2,16,256> warp<16,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 256, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 256, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 256, 16, 32, 128, 8, 8, 128, 4);
// cta<2,24,256> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 256, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 256, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 256, 16, 48, 128, 8, 8, 128, 4);
// cta<2,32,256> warp<16,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 256, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 256, 16, 64, 128, 8, 8, 128, 4);
// cta<4,8,256> warp<32,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 32, 16, 128, 8, 8, 128, 4);
// cta<4,16,256> warp<32,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 32, 32, 128, 8, 8, 128, 4);
// cta<4,24,256> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 256, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 256, 32, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 256, 32, 48, 128, 8, 8, 128, 4);
// cta<4,32,256> warp<32,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 256, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 256, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 256, 32, 64, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<64,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 64, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 64, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 64, 16, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<64,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 64, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 64, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 64, 32, 128, 8, 8, 128, 4);
// cta<8,24,256> warp<64,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 256, 64, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 256, 64, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 256, 64, 48, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<64,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 256, 64, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 256, 64, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 256, 64, 64, 128, 8, 8, 128, 4);
// cta<1,8,256> warp<8,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 256, 8, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 256, 8, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 256, 8, 8, 128, 8, 8, 128, 4);
// cta<1,16,256> warp<8,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 256, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 256, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 256, 8, 16, 128, 8, 8, 128, 4);
// cta<1,24,256> warp<8,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 24, 256, 8, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 24, 256, 8, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 24, 256, 8, 24, 128, 8, 8, 128, 4);
// cta<1,32,256> warp<8,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 256, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 256, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 256, 8, 32, 128, 8, 8, 128, 4);
// cta<1,40,256> warp<8,40,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 40, 256, 8, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 40, 256, 8, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 40, 256, 8, 40, 128, 8, 8, 128, 4);
// cta<1,48,256> warp<8,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 48, 256, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 48, 256, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 48, 256, 8, 48, 128, 8, 8, 128, 4);
// cta<1,56,256> warp<8,56,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 56, 256, 8, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 56, 256, 8, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 56, 256, 8, 56, 128, 8, 8, 128, 4);
// cta<1,64,256> warp<8,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 64, 256, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 64, 256, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 64, 256, 8, 64, 128, 8, 8, 128, 4);
// cta<2,8,256> warp<16,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 256, 16, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 256, 16, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 256, 16, 8, 128, 8, 8, 128, 4);
// cta<2,16,256> warp<16,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 256, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 256, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 256, 16, 16, 128, 8, 8, 128, 4);
// cta<2,24,256> warp<16,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 256, 16, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 256, 16, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 256, 16, 24, 128, 8, 8, 128, 4);
// cta<2,32,256> warp<16,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 256, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 256, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 256, 16, 32, 128, 8, 8, 128, 4);
// cta<2,40,256> warp<16,40,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 40, 256, 16, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 40, 256, 16, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 40, 256, 16, 40, 128, 8, 8, 128, 4);
// cta<2,48,256> warp<16,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 48, 256, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 48, 256, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 48, 256, 16, 48, 128, 8, 8, 128, 4);
// cta<2,56,256> warp<16,56,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 56, 256, 16, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 56, 256, 16, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 56, 256, 16, 56, 128, 8, 8, 128, 4);
// cta<2,64,256> warp<16,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 64, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 64, 256, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 64, 256, 16, 64, 128, 8, 8, 128, 4);
// cta<4,8,256> warp<32,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 32, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 32, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 32, 8, 128, 8, 8, 128, 4);
// cta<4,16,256> warp<32,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 32, 16, 128, 8, 8, 128, 4);
// cta<4,24,256> warp<32,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 256, 32, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 256, 32, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 256, 32, 24, 128, 8, 8, 128, 4);
// cta<4,32,256> warp<32,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 256, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 256, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 256, 32, 32, 128, 8, 8, 128, 4);
// cta<4,40,256> warp<32,40,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 40, 256, 32, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 40, 256, 32, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 40, 256, 32, 40, 128, 8, 8, 128, 4);
// cta<4,48,256> warp<32,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 48, 256, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 48, 256, 32, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 48, 256, 32, 48, 128, 8, 8, 128, 4);
// cta<4,56,256> warp<32,56,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 56, 256, 32, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 56, 256, 32, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 56, 256, 32, 56, 128, 8, 8, 128, 4);
// cta<4,64,256> warp<32,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 64, 256, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 64, 256, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 64, 256, 32, 64, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<64,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 64, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 64, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 64, 8, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<64,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 64, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 64, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 64, 16, 128, 8, 8, 128, 4);
// cta<8,24,256> warp<64,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 256, 64, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 256, 64, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 256, 64, 24, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<64,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 256, 64, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 256, 64, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 256, 64, 32, 128, 8, 8, 128, 4);
// cta<8,40,256> warp<64,40,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 40, 256, 64, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 40, 256, 64, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 40, 256, 64, 40, 128, 8, 8, 128, 4);
// cta<8,48,256> warp<64,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 48, 256, 64, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 48, 256, 64, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 48, 256, 64, 48, 128, 8, 8, 128, 4);
// cta<1,16,256> warp<8,8,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 256, 8, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 256, 8, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 256, 8, 8, 128, 8, 8, 128, 4);
// cta<1,32,256> warp<8,16,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 256, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 256, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 256, 8, 16, 128, 8, 8, 128, 4);
// cta<1,48,256> warp<8,24,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 48, 256, 8, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 48, 256, 8, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 48, 256, 8, 24, 128, 8, 8, 128, 4);
// cta<1,64,256> warp<8,32,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 64, 256, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 64, 256, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 64, 256, 8, 32, 128, 8, 8, 128, 4);
// cta<1,80,256> warp<8,40,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 80, 256, 8, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 80, 256, 8, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 80, 256, 8, 40, 128, 8, 8, 128, 4);
// cta<1,96,256> warp<8,48,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 96, 256, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 96, 256, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 96, 256, 8, 48, 128, 8, 8, 128, 4);
// cta<1,112,256> warp<8,56,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 112, 256, 8, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 112, 256, 8, 56, 128, 8, 8, 128, 3);
// cta<1,128,256> warp<8,64,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 128, 256, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 128, 256, 8, 64, 128, 8, 8, 128, 3);
// cta<2,16,256> warp<16,8,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 256, 16, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 256, 16, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 256, 16, 8, 128, 8, 8, 128, 4);
// cta<2,32,256> warp<16,16,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 256, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 256, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 256, 16, 16, 128, 8, 8, 128, 4);
// cta<2,48,256> warp<16,24,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 48, 256, 16, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 48, 256, 16, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 48, 256, 16, 24, 128, 8, 8, 128, 4);
// cta<2,64,256> warp<16,32,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 64, 256, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 64, 256, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 64, 256, 16, 32, 128, 8, 8, 128, 4);
// cta<2,80,256> warp<16,40,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 80, 256, 16, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 80, 256, 16, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 80, 256, 16, 40, 128, 8, 8, 128, 4);
// cta<2,96,256> warp<16,48,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 96, 256, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 96, 256, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 96, 256, 16, 48, 128, 8, 8, 128, 4);
// cta<2,112,256> warp<16,56,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 112, 256, 16, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 112, 256, 16, 56, 128, 8, 8, 128, 3);
// cta<2,128,256> warp<16,64,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 128, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 128, 256, 16, 64, 128, 8, 8, 128, 3);
// cta<4,16,256> warp<32,8,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 32, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 32, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 32, 8, 128, 8, 8, 128, 4);
// cta<4,32,256> warp<32,16,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 256, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 256, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 256, 32, 16, 128, 8, 8, 128, 4);
// cta<4,48,256> warp<32,24,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 48, 256, 32, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 48, 256, 32, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 48, 256, 32, 24, 128, 8, 8, 128, 4);
// cta<4,64,256> warp<32,32,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 64, 256, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 64, 256, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 64, 256, 32, 32, 128, 8, 8, 128, 4);
// cta<4,80,256> warp<32,40,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 80, 256, 32, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 80, 256, 32, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 80, 256, 32, 40, 128, 8, 8, 128, 4);
// cta<4,96,256> warp<32,48,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 96, 256, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 96, 256, 32, 48, 128, 8, 8, 128, 3);
// cta<8,16,256> warp<64,8,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 64, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 64, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 64, 8, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<64,16,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 256, 64, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 256, 64, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 256, 64, 16, 128, 8, 8, 128, 4);
// cta<8,48,256> warp<64,24,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 48, 256, 64, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 48, 256, 64, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 48, 256, 64, 24, 128, 8, 8, 128, 4);
// cta<2,8,256> warp<8,64,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 256, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 256, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 256, 8, 64, 128, 8, 8, 128, 4);
// cta<4,8,256> warp<16,64,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 16, 64, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<32,64,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 32, 64, 128, 8, 8, 128, 4);
// cta<2,8,256> warp<8,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 256, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 256, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 256, 8, 32, 128, 8, 8, 128, 4);
// cta<2,16,256> warp<8,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 256, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 256, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 256, 8, 64, 128, 8, 8, 128, 4);
// cta<4,8,256> warp<16,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 16, 32, 128, 8, 8, 128, 4);
// cta<4,16,256> warp<16,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 16, 64, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<32,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 32, 32, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<32,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 32, 64, 128, 8, 8, 128, 4);
// cta<2,8,256> warp<8,16,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 256, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 256, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 256, 8, 16, 128, 8, 8, 128, 4);
// cta<2,16,256> warp<8,32,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 256, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 256, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 256, 8, 32, 128, 8, 8, 128, 4);
// cta<2,24,256> warp<8,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 256, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 256, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 256, 8, 48, 128, 8, 8, 128, 4);
// cta<2,32,256> warp<8,64,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 256, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 256, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 256, 8, 64, 128, 8, 8, 128, 4);
// cta<4,8,256> warp<16,16,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 16, 16, 128, 8, 8, 128, 4);
// cta<4,16,256> warp<16,32,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 16, 32, 128, 8, 8, 128, 4);
// cta<4,24,256> warp<16,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 256, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 256, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 256, 16, 48, 128, 8, 8, 128, 4);
// cta<4,32,256> warp<16,64,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 256, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 256, 16, 64, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<32,16,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 32, 16, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<32,32,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 32, 32, 128, 8, 8, 128, 4);
// cta<8,24,256> warp<32,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 256, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 256, 32, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 256, 32, 48, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<32,64,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 256, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 256, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 256, 32, 64, 128, 8, 8, 128, 4);
// cta<2,8,256> warp<8,8,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 256, 8, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 256, 8, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 256, 8, 8, 128, 8, 8, 128, 4);
// cta<2,16,256> warp<8,16,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 256, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 256, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 256, 8, 16, 128, 8, 8, 128, 4);
// cta<2,24,256> warp<8,24,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 256, 8, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 256, 8, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 256, 8, 24, 128, 8, 8, 128, 4);
// cta<2,32,256> warp<8,32,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 256, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 256, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 256, 8, 32, 128, 8, 8, 128, 4);
// cta<2,40,256> warp<8,40,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 40, 256, 8, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 40, 256, 8, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 40, 256, 8, 40, 128, 8, 8, 128, 4);
// cta<2,48,256> warp<8,48,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 48, 256, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 48, 256, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 48, 256, 8, 48, 128, 8, 8, 128, 4);
// cta<2,56,256> warp<8,56,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 56, 256, 8, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 56, 256, 8, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 56, 256, 8, 56, 128, 8, 8, 128, 4);
// cta<2,64,256> warp<8,64,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 64, 256, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 64, 256, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 64, 256, 8, 64, 128, 8, 8, 128, 4);
// cta<4,8,256> warp<16,8,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 16, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 16, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 16, 8, 128, 8, 8, 128, 4);
// cta<4,16,256> warp<16,16,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 16, 16, 128, 8, 8, 128, 4);
// cta<4,24,256> warp<16,24,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 256, 16, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 256, 16, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 256, 16, 24, 128, 8, 8, 128, 4);
// cta<4,32,256> warp<16,32,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 256, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 256, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 256, 16, 32, 128, 8, 8, 128, 4);
// cta<4,40,256> warp<16,40,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 40, 256, 16, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 40, 256, 16, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 40, 256, 16, 40, 128, 8, 8, 128, 4);
// cta<4,48,256> warp<16,48,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 48, 256, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 48, 256, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 48, 256, 16, 48, 128, 8, 8, 128, 4);
// cta<4,56,256> warp<16,56,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 56, 256, 16, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 56, 256, 16, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 56, 256, 16, 56, 128, 8, 8, 128, 4);
// cta<4,64,256> warp<16,64,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 64, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 64, 256, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 64, 256, 16, 64, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<32,8,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 32, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 32, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 32, 8, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<32,16,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 32, 16, 128, 8, 8, 128, 4);
// cta<8,24,256> warp<32,24,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 256, 32, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 256, 32, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 256, 32, 24, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<32,32,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 256, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 256, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 256, 32, 32, 128, 8, 8, 128, 4);
// cta<8,40,256> warp<32,40,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 40, 256, 32, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 40, 256, 32, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 40, 256, 32, 40, 128, 8, 8, 128, 4);
// cta<8,48,256> warp<32,48,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 48, 256, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 48, 256, 32, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 48, 256, 32, 48, 128, 8, 8, 128, 4);
// cta<4,8,256> warp<8,64,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 8, 64, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<16,64,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 16, 64, 128, 8, 8, 128, 4);
// cta<4,8,256> warp<8,32,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 8, 32, 128, 8, 8, 128, 4);
// cta<4,16,256> warp<8,64,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 8, 64, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<16,32,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 16, 32, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<16,64,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 16, 64, 128, 8, 8, 128, 4);
// cta<4,8,256> warp<8,16,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 256, 8, 16, 128, 8, 8, 128, 4);
// cta<4,16,256> warp<8,32,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 256, 8, 32, 128, 8, 8, 128, 4);
// cta<4,24,256> warp<8,48,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 256, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 256, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 256, 8, 48, 128, 8, 8, 128, 4);
// cta<4,32,256> warp<8,64,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 256, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 256, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 256, 8, 64, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<16,16,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 16, 16, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<16,32,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 16, 32, 128, 8, 8, 128, 4);
// cta<8,24,256> warp<16,48,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 256, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 256, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 256, 16, 48, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<16,64,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 256, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 256, 16, 64, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<8,64,128> mma<8,8,128>   WARPS[8x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 8, 64, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<8,32,128> mma<8,8,128>   WARPS[8x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 256, 8, 32, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<8,64,128> mma<8,8,128>   WARPS[8x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 256, 8, 64, 128, 8, 8, 128, 4);
// cta<1,8,512> warp<8,64,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 512, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 512, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 512, 8, 64, 128, 8, 8, 128, 4);
// cta<2,8,512> warp<16,64,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 512, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 512, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 512, 16, 64, 128, 8, 8, 128, 4);
// cta<4,8,512> warp<32,64,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 32, 64, 128, 8, 8, 128, 4);
// cta<8,8,512> warp<64,64,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 64, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 64, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 64, 64, 128, 8, 8, 128, 4);
// cta<1,8,512> warp<8,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 512, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 512, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 512, 8, 32, 128, 8, 8, 128, 4);
// cta<1,16,512> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 512, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 512, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 512, 8, 64, 128, 8, 8, 128, 4);
// cta<2,8,512> warp<16,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 512, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 512, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 512, 16, 32, 128, 8, 8, 128, 4);
// cta<2,16,512> warp<16,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 512, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 512, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 512, 16, 64, 128, 8, 8, 128, 4);
// cta<4,8,512> warp<32,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 32, 32, 128, 8, 8, 128, 4);
// cta<4,16,512> warp<32,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 32, 64, 128, 8, 8, 128, 4);
// cta<8,8,512> warp<64,32,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 64, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 64, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 64, 32, 128, 8, 8, 128, 4);
// cta<8,16,512> warp<64,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 64, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 64, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 64, 64, 128, 8, 8, 128, 4);
// cta<1,8,512> warp<8,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 512, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 512, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 512, 8, 16, 128, 8, 8, 128, 4);
// cta<1,16,512> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 512, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 512, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 512, 8, 32, 128, 8, 8, 128, 4);
// cta<1,24,512> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 24, 512, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 24, 512, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 24, 512, 8, 48, 128, 8, 8, 128, 4);
// cta<1,32,512> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 512, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 512, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 512, 8, 64, 128, 8, 8, 128, 4);
// cta<2,8,512> warp<16,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 512, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 512, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 512, 16, 16, 128, 8, 8, 128, 4);
// cta<2,16,512> warp<16,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 512, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 512, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 512, 16, 32, 128, 8, 8, 128, 4);
// cta<2,24,512> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 512, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 512, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 512, 16, 48, 128, 8, 8, 128, 4);
// cta<2,32,512> warp<16,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 512, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 512, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 512, 16, 64, 128, 8, 8, 128, 4);
// cta<4,8,512> warp<32,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 32, 16, 128, 8, 8, 128, 4);
// cta<4,16,512> warp<32,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 32, 32, 128, 8, 8, 128, 4);
// cta<4,24,512> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 512, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 512, 32, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 512, 32, 48, 128, 8, 8, 128, 4);
// cta<4,32,512> warp<32,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 512, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 512, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 512, 32, 64, 128, 8, 8, 128, 4);
// cta<8,8,512> warp<64,16,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 64, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 64, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 64, 16, 128, 8, 8, 128, 4);
// cta<8,16,512> warp<64,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 64, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 64, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 64, 32, 128, 8, 8, 128, 4);
// cta<8,24,512> warp<64,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 512, 64, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 512, 64, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 512, 64, 48, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<64,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 512, 64, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 512, 64, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 512, 64, 64, 128, 8, 8, 128, 4);
// cta<1,8,512> warp<8,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 512, 8, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 512, 8, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 8, 512, 8, 8, 128, 8, 8, 128, 4);
// cta<1,16,512> warp<8,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 512, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 512, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 512, 8, 16, 128, 8, 8, 128, 4);
// cta<1,24,512> warp<8,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 24, 512, 8, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 24, 512, 8, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 24, 512, 8, 24, 128, 8, 8, 128, 4);
// cta<1,32,512> warp<8,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 512, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 512, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 512, 8, 32, 128, 8, 8, 128, 4);
// cta<1,40,512> warp<8,40,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 40, 512, 8, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 40, 512, 8, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 40, 512, 8, 40, 128, 8, 8, 128, 4);
// cta<1,48,512> warp<8,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 48, 512, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 48, 512, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 48, 512, 8, 48, 128, 8, 8, 128, 4);
// cta<1,56,512> warp<8,56,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 56, 512, 8, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 56, 512, 8, 56, 128, 8, 8, 128, 3);
// cta<1,64,512> warp<8,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 64, 512, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 64, 512, 8, 64, 128, 8, 8, 128, 3);
// cta<2,8,512> warp<16,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 512, 16, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 512, 16, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 512, 16, 8, 128, 8, 8, 128, 4);
// cta<2,16,512> warp<16,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 512, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 512, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 512, 16, 16, 128, 8, 8, 128, 4);
// cta<2,24,512> warp<16,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 512, 16, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 512, 16, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 512, 16, 24, 128, 8, 8, 128, 4);
// cta<2,32,512> warp<16,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 512, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 512, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 512, 16, 32, 128, 8, 8, 128, 4);
// cta<2,40,512> warp<16,40,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 40, 512, 16, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 40, 512, 16, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 40, 512, 16, 40, 128, 8, 8, 128, 4);
// cta<2,48,512> warp<16,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 48, 512, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 48, 512, 16, 48, 128, 8, 8, 128, 3);
// cta<2,56,512> warp<16,56,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 56, 512, 16, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 56, 512, 16, 56, 128, 8, 8, 128, 3);
// cta<2,64,512> warp<16,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 64, 512, 16, 64, 128, 8, 8, 128, 2);
// cta<4,8,512> warp<32,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 32, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 32, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 32, 8, 128, 8, 8, 128, 4);
// cta<4,16,512> warp<32,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 32, 16, 128, 8, 8, 128, 4);
// cta<4,24,512> warp<32,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 512, 32, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 512, 32, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 512, 32, 24, 128, 8, 8, 128, 4);
// cta<4,32,512> warp<32,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 512, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 512, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 512, 32, 32, 128, 8, 8, 128, 4);
// cta<4,40,512> warp<32,40,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 40, 512, 32, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 40, 512, 32, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 40, 512, 32, 40, 128, 8, 8, 128, 4);
// cta<4,48,512> warp<32,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 48, 512, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 48, 512, 32, 48, 128, 8, 8, 128, 3);
// cta<4,56,512> warp<32,56,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 56, 512, 32, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 56, 512, 32, 56, 128, 8, 8, 128, 3);
// cta<4,64,512> warp<32,64,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 64, 512, 32, 64, 128, 8, 8, 128, 2);
// cta<8,8,512> warp<64,8,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 64, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 64, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 64, 8, 128, 8, 8, 128, 4);
// cta<8,16,512> warp<64,16,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 64, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 64, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 64, 16, 128, 8, 8, 128, 4);
// cta<8,24,512> warp<64,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 512, 64, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 512, 64, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 512, 64, 24, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<64,32,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 512, 64, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 512, 64, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 512, 64, 32, 128, 8, 8, 128, 4);
// cta<8,40,512> warp<64,40,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 40, 512, 64, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 40, 512, 64, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 40, 512, 64, 40, 128, 8, 8, 128, 4);
// cta<8,48,512> warp<64,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 48, 512, 64, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 48, 512, 64, 48, 128, 8, 8, 128, 3);
// cta<1,16,512> warp<8,8,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 512, 8, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 512, 8, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 16, 512, 8, 8, 128, 8, 8, 128, 4);
// cta<1,32,512> warp<8,16,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 512, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 512, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 32, 512, 8, 16, 128, 8, 8, 128, 4);
// cta<1,48,512> warp<8,24,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 48, 512, 8, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 48, 512, 8, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 48, 512, 8, 24, 128, 8, 8, 128, 4);
// cta<1,64,512> warp<8,32,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 64, 512, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 64, 512, 8, 32, 128, 8, 8, 128, 3);
// cta<1,80,512> warp<8,40,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 80, 512, 8, 40, 128, 8, 8, 128, 2);
// cta<1,96,512> warp<8,48,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 1, 96, 512, 8, 48, 128, 8, 8, 128, 2);
// cta<2,16,512> warp<16,8,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 512, 16, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 512, 16, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 512, 16, 8, 128, 8, 8, 128, 4);
// cta<2,32,512> warp<16,16,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 512, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 512, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 512, 16, 16, 128, 8, 8, 128, 4);
// cta<2,48,512> warp<16,24,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 48, 512, 16, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 48, 512, 16, 24, 128, 8, 8, 128, 3);
// cta<2,64,512> warp<16,32,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 64, 512, 16, 32, 128, 8, 8, 128, 2);
// cta<2,80,512> warp<16,40,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 80, 512, 16, 40, 128, 8, 8, 128, 2);
// cta<2,96,512> warp<16,48,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 96, 512, 16, 48, 128, 8, 8, 128, 2);
// cta<4,16,512> warp<32,8,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 32, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 32, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 32, 8, 128, 8, 8, 128, 4);
// cta<4,32,512> warp<32,16,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 512, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 512, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 512, 32, 16, 128, 8, 8, 128, 4);
// cta<4,48,512> warp<32,24,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 48, 512, 32, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 48, 512, 32, 24, 128, 8, 8, 128, 3);
// cta<4,64,512> warp<32,32,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 64, 512, 32, 32, 128, 8, 8, 128, 2);
// cta<4,80,512> warp<32,40,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 80, 512, 32, 40, 128, 8, 8, 128, 2);
// cta<8,16,512> warp<64,8,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 64, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 64, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 64, 8, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<64,16,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 512, 64, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 512, 64, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 512, 64, 16, 128, 8, 8, 128, 4);
// cta<8,48,512> warp<64,24,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 48, 512, 64, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 48, 512, 64, 24, 128, 8, 8, 128, 3);
// cta<2,8,512> warp<8,64,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 512, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 512, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 512, 8, 64, 128, 8, 8, 128, 4);
// cta<4,8,512> warp<16,64,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 16, 64, 128, 8, 8, 128, 4);
// cta<8,8,512> warp<32,64,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 32, 64, 128, 8, 8, 128, 4);
// cta<2,8,512> warp<8,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 512, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 512, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 512, 8, 32, 128, 8, 8, 128, 4);
// cta<2,16,512> warp<8,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 512, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 512, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 512, 8, 64, 128, 8, 8, 128, 4);
// cta<4,8,512> warp<16,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 16, 32, 128, 8, 8, 128, 4);
// cta<4,16,512> warp<16,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 16, 64, 128, 8, 8, 128, 4);
// cta<8,8,512> warp<32,32,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 32, 32, 128, 8, 8, 128, 4);
// cta<8,16,512> warp<32,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 32, 64, 128, 8, 8, 128, 4);
// cta<2,8,512> warp<8,16,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 512, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 512, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 512, 8, 16, 128, 8, 8, 128, 4);
// cta<2,16,512> warp<8,32,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 512, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 512, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 512, 8, 32, 128, 8, 8, 128, 4);
// cta<2,24,512> warp<8,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 512, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 512, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 512, 8, 48, 128, 8, 8, 128, 4);
// cta<2,32,512> warp<8,64,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 512, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 512, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 512, 8, 64, 128, 8, 8, 128, 4);
// cta<4,8,512> warp<16,16,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 16, 16, 128, 8, 8, 128, 4);
// cta<4,16,512> warp<16,32,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 16, 32, 128, 8, 8, 128, 4);
// cta<4,24,512> warp<16,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 512, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 512, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 512, 16, 48, 128, 8, 8, 128, 4);
// cta<4,32,512> warp<16,64,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 512, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 512, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 512, 16, 64, 128, 8, 8, 128, 4);
// cta<8,8,512> warp<32,16,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 32, 16, 128, 8, 8, 128, 4);
// cta<8,16,512> warp<32,32,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 32, 32, 128, 8, 8, 128, 4);
// cta<8,24,512> warp<32,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 512, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 512, 32, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 512, 32, 48, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<32,64,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 512, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 512, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 512, 32, 64, 128, 8, 8, 128, 4);
// cta<2,8,512> warp<8,8,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 512, 8, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 512, 8, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 8, 512, 8, 8, 128, 8, 8, 128, 4);
// cta<2,16,512> warp<8,16,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 512, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 512, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 16, 512, 8, 16, 128, 8, 8, 128, 4);
// cta<2,24,512> warp<8,24,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 512, 8, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 512, 8, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 24, 512, 8, 24, 128, 8, 8, 128, 4);
// cta<2,32,512> warp<8,32,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 512, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 512, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 32, 512, 8, 32, 128, 8, 8, 128, 4);
// cta<2,40,512> warp<8,40,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 40, 512, 8, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 40, 512, 8, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 40, 512, 8, 40, 128, 8, 8, 128, 4);
// cta<2,48,512> warp<8,48,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 48, 512, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 48, 512, 8, 48, 128, 8, 8, 128, 3);
// cta<2,56,512> warp<8,56,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 56, 512, 8, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 56, 512, 8, 56, 128, 8, 8, 128, 3);
// cta<2,64,512> warp<8,64,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 2, 64, 512, 8, 64, 128, 8, 8, 128, 2);
// cta<4,8,512> warp<16,8,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 16, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 16, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 16, 8, 128, 8, 8, 128, 4);
// cta<4,16,512> warp<16,16,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 16, 16, 128, 8, 8, 128, 4);
// cta<4,24,512> warp<16,24,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 512, 16, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 512, 16, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 512, 16, 24, 128, 8, 8, 128, 4);
// cta<4,32,512> warp<16,32,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 512, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 512, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 512, 16, 32, 128, 8, 8, 128, 4);
// cta<4,40,512> warp<16,40,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 40, 512, 16, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 40, 512, 16, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 40, 512, 16, 40, 128, 8, 8, 128, 4);
// cta<4,48,512> warp<16,48,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 48, 512, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 48, 512, 16, 48, 128, 8, 8, 128, 3);
// cta<4,56,512> warp<16,56,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 56, 512, 16, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 56, 512, 16, 56, 128, 8, 8, 128, 3);
// cta<4,64,512> warp<16,64,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 64, 512, 16, 64, 128, 8, 8, 128, 2);
// cta<8,8,512> warp<32,8,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 32, 8, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 32, 8, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 32, 8, 128, 8, 8, 128, 4);
// cta<8,16,512> warp<32,16,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 32, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 32, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 32, 16, 128, 8, 8, 128, 4);
// cta<8,24,512> warp<32,24,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 512, 32, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 512, 32, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 512, 32, 24, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<32,32,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 512, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 512, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 512, 32, 32, 128, 8, 8, 128, 4);
// cta<8,40,512> warp<32,40,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 40, 512, 32, 40, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 40, 512, 32, 40, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 40, 512, 32, 40, 128, 8, 8, 128, 4);
// cta<8,48,512> warp<32,48,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 48, 512, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 48, 512, 32, 48, 128, 8, 8, 128, 3);
// cta<4,8,512> warp<8,64,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 8, 64, 128, 8, 8, 128, 4);
// cta<8,8,512> warp<16,64,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 16, 64, 128, 8, 8, 128, 4);
// cta<4,8,512> warp<8,32,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 8, 32, 128, 8, 8, 128, 4);
// cta<4,16,512> warp<8,64,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 8, 64, 128, 8, 8, 128, 4);
// cta<8,8,512> warp<16,32,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 16, 32, 128, 8, 8, 128, 4);
// cta<8,16,512> warp<16,64,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 16, 64, 128, 8, 8, 128, 4);
// cta<4,8,512> warp<8,16,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 8, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 8, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 8, 512, 8, 16, 128, 8, 8, 128, 4);
// cta<4,16,512> warp<8,32,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 16, 512, 8, 32, 128, 8, 8, 128, 4);
// cta<4,24,512> warp<8,48,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 512, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 512, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 24, 512, 8, 48, 128, 8, 8, 128, 4);
// cta<4,32,512> warp<8,64,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 512, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 512, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 4, 32, 512, 8, 64, 128, 8, 8, 128, 4);
// cta<8,8,512> warp<16,16,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 16, 16, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 16, 16, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 16, 16, 128, 8, 8, 128, 4);
// cta<8,16,512> warp<16,32,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 16, 32, 128, 8, 8, 128, 4);
// cta<8,24,512> warp<16,48,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 512, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 512, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 24, 512, 16, 48, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<16,64,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 512, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 512, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 32, 512, 16, 64, 128, 8, 8, 128, 4);
// cta<8,8,512> warp<8,64,128> mma<8,8,128>   WARPS[8x1]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 8, 64, 128, 8, 8, 128, 4);
// cta<8,8,512> warp<8,32,128> mma<8,8,128>   WARPS[8x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 8, 512, 8, 32, 128, 8, 8, 128, 4);
// cta<8,16,512> warp<8,64,128> mma<8,8,128>   WARPS[8x2]
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 8, 8, true, 8, 16, 512, 8, 64, 128, 8, 8, 128, 4);
#endif