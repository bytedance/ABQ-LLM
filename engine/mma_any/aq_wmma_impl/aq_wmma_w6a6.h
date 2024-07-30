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

#ifdef W6A6
////// W6A6 int
// cta<4,8,128> warp<24,48,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 128, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 128, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 128, 24, 48, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<48,48,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 128, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 128, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 128, 48, 48, 128, 8, 8, 128, 4);
// cta<4,8,128> warp<24,24,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 128, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 128, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 128, 24, 24, 128, 8, 8, 128, 4);
// cta<4,16,128> warp<24,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 128, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 128, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 128, 24, 48, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<48,24,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 128, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 128, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 128, 48, 24, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<48,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 128, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 128, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 128, 48, 48, 128, 8, 8, 128, 4);
// cta<4,16,128> warp<24,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 128, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 128, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 128, 24, 24, 128, 8, 8, 128, 4);
// cta<4,32,128> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 128, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 128, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 128, 24, 48, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<48,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 128, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 128, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 128, 48, 24, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<48,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 128, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 128, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 128, 48, 48, 128, 8, 8, 128, 4);
// cta<4,32,128> warp<24,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 128, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 128, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 128, 24, 24, 128, 8, 8, 128, 4);
// cta<4,64,128> warp<24,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 128, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 128, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 128, 24, 48, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<48,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 128, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 128, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 128, 48, 24, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<48,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 128, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 128, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 128, 48, 48, 128, 8, 8, 128, 4);
// cta<4,64,128> warp<24,24,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 128, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 128, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 128, 24, 24, 128, 8, 8, 128, 4);
// cta<4,128,128> warp<24,48,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 128, 128, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 128, 128, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 128, 128, 24, 48, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<48,24,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 128, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 128, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 128, 48, 24, 128, 8, 8, 128, 4);
// cta<4,128,128> warp<24,24,128> mma<8,8,128>   WARPS[1x32]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 128, 128, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 128, 128, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 128, 128, 24, 24, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<24,48,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 128, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 128, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 128, 24, 48, 128, 8, 8, 128, 4);
// cta<16,8,128> warp<48,48,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 128, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 128, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 128, 48, 48, 128, 8, 8, 128, 4);
// cta<8,8,128> warp<24,24,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 128, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 128, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 128, 24, 24, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<24,48,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 128, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 128, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 128, 24, 48, 128, 8, 8, 128, 4);
// cta<16,8,128> warp<48,24,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 128, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 128, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 128, 48, 24, 128, 8, 8, 128, 4);
// cta<16,16,128> warp<48,48,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 128, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 128, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 128, 48, 48, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<24,24,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 128, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 128, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 128, 24, 24, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<24,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 128, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 128, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 128, 24, 48, 128, 8, 8, 128, 4);
// cta<16,16,128> warp<48,24,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 128, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 128, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 128, 48, 24, 128, 8, 8, 128, 4);
// cta<16,32,128> warp<48,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 128, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 128, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 128, 48, 48, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<24,24,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 128, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 128, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 128, 24, 24, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<24,48,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 128, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 128, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 128, 24, 48, 128, 8, 8, 128, 4);
// cta<16,32,128> warp<48,24,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 128, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 128, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 128, 48, 24, 128, 8, 8, 128, 4);
// cta<8,64,128> warp<24,24,128> mma<8,8,128>   WARPS[2x16]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 128, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 128, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 128, 24, 24, 128, 8, 8, 128, 4);
// cta<16,8,128> warp<24,48,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 128, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 128, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 128, 24, 48, 128, 8, 8, 128, 4);
// cta<32,8,128> warp<48,48,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 128, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 128, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 128, 48, 48, 128, 8, 8, 128, 4);
// cta<16,8,128> warp<24,24,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 128, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 128, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 128, 24, 24, 128, 8, 8, 128, 4);
// cta<16,16,128> warp<24,48,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 128, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 128, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 128, 24, 48, 128, 8, 8, 128, 4);
// cta<32,8,128> warp<48,24,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 128, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 128, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 128, 48, 24, 128, 8, 8, 128, 4);
// cta<32,16,128> warp<48,48,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 128, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 128, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 128, 48, 48, 128, 8, 8, 128, 4);
// cta<16,16,128> warp<24,24,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 128, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 128, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 128, 24, 24, 128, 8, 8, 128, 4);
// cta<16,32,128> warp<24,48,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 128, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 128, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 128, 24, 48, 128, 8, 8, 128, 4);
// cta<32,16,128> warp<48,24,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 128, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 128, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 128, 48, 24, 128, 8, 8, 128, 4);
// cta<16,32,128> warp<24,24,128> mma<8,8,128>   WARPS[4x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 128, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 128, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 128, 24, 24, 128, 8, 8, 128, 4);
// cta<32,8,128> warp<24,48,128> mma<8,8,128>   WARPS[8x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 128, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 128, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 128, 24, 48, 128, 8, 8, 128, 4);
// cta<32,8,128> warp<24,24,128> mma<8,8,128>   WARPS[8x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 128, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 128, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 128, 24, 24, 128, 8, 8, 128, 4);
// cta<32,16,128> warp<24,48,128> mma<8,8,128>   WARPS[8x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 128, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 128, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 128, 24, 48, 128, 8, 8, 128, 4);
// cta<32,16,128> warp<24,24,128> mma<8,8,128>   WARPS[8x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 128, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 128, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 128, 24, 24, 128, 8, 8, 128, 4);
// cta<4,8,256> warp<24,48,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 256, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 256, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 256, 24, 48, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<48,48,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 256, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 256, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 256, 48, 48, 128, 8, 8, 128, 4);
// cta<4,8,256> warp<24,24,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 256, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 256, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 256, 24, 24, 128, 8, 8, 128, 4);
// cta<4,16,256> warp<24,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 256, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 256, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 256, 24, 48, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<48,24,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 256, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 256, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 256, 48, 24, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<48,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 256, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 256, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 256, 48, 48, 128, 8, 8, 128, 4);
// cta<4,16,256> warp<24,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 256, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 256, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 256, 24, 24, 128, 8, 8, 128, 4);
// cta<4,32,256> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 256, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 256, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 256, 24, 48, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<48,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 256, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 256, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 256, 48, 24, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<48,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 256, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 256, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 256, 48, 48, 128, 8, 8, 128, 4);
// cta<4,32,256> warp<24,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 256, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 256, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 256, 24, 24, 128, 8, 8, 128, 4);
// cta<4,64,256> warp<24,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 256, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 256, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 256, 24, 48, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<48,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 256, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 256, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 256, 48, 24, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<48,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 256, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 256, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 256, 48, 48, 128, 8, 8, 128, 4);
// cta<4,64,256> warp<24,24,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 256, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 256, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 256, 24, 24, 128, 8, 8, 128, 4);
// cta<4,128,256> warp<24,48,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 128, 256, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 128, 256, 24, 48, 128, 8, 8, 128, 3);
// cta<8,64,256> warp<48,24,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 256, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 256, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 256, 48, 24, 128, 8, 8, 128, 4);
// cta<4,128,256> warp<24,24,128> mma<8,8,128>   WARPS[1x32]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 128, 256, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 128, 256, 24, 24, 128, 8, 8, 128, 3);
// cta<8,8,256> warp<24,48,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 256, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 256, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 256, 24, 48, 128, 8, 8, 128, 4);
// cta<16,8,256> warp<48,48,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 256, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 256, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 256, 48, 48, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<24,24,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 256, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 256, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 256, 24, 24, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<24,48,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 256, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 256, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 256, 24, 48, 128, 8, 8, 128, 4);
// cta<16,8,256> warp<48,24,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 256, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 256, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 256, 48, 24, 128, 8, 8, 128, 4);
// cta<16,16,256> warp<48,48,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 256, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 256, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 256, 48, 48, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<24,24,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 256, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 256, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 256, 24, 24, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<24,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 256, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 256, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 256, 24, 48, 128, 8, 8, 128, 4);
// cta<16,16,256> warp<48,24,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 256, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 256, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 256, 48, 24, 128, 8, 8, 128, 4);
// cta<16,32,256> warp<48,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 256, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 256, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 256, 48, 48, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<24,24,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 256, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 256, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 256, 24, 24, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<24,48,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 256, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 256, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 256, 24, 48, 128, 8, 8, 128, 4);
// cta<16,32,256> warp<48,24,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 256, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 256, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 256, 48, 24, 128, 8, 8, 128, 4);
// cta<8,64,256> warp<24,24,128> mma<8,8,128>   WARPS[2x16]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 256, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 256, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 256, 24, 24, 128, 8, 8, 128, 4);
// cta<16,8,256> warp<24,48,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 256, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 256, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 256, 24, 48, 128, 8, 8, 128, 4);
// cta<32,8,256> warp<48,48,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 256, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 256, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 256, 48, 48, 128, 8, 8, 128, 4);
// cta<16,8,256> warp<24,24,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 256, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 256, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 256, 24, 24, 128, 8, 8, 128, 4);
// cta<16,16,256> warp<24,48,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 256, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 256, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 256, 24, 48, 128, 8, 8, 128, 4);
// cta<32,8,256> warp<48,24,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 256, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 256, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 256, 48, 24, 128, 8, 8, 128, 4);
// cta<32,16,256> warp<48,48,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 256, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 256, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 256, 48, 48, 128, 8, 8, 128, 4);
// cta<16,16,256> warp<24,24,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 256, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 256, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 256, 24, 24, 128, 8, 8, 128, 4);
// cta<16,32,256> warp<24,48,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 256, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 256, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 256, 24, 48, 128, 8, 8, 128, 4);
// cta<32,16,256> warp<48,24,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 256, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 256, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 256, 48, 24, 128, 8, 8, 128, 4);
// cta<16,32,256> warp<24,24,128> mma<8,8,128>   WARPS[4x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 256, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 256, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 256, 24, 24, 128, 8, 8, 128, 4);
// cta<32,8,256> warp<24,48,128> mma<8,8,128>   WARPS[8x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 256, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 256, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 256, 24, 48, 128, 8, 8, 128, 4);
// cta<32,8,256> warp<24,24,128> mma<8,8,128>   WARPS[8x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 256, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 256, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 256, 24, 24, 128, 8, 8, 128, 4);
// cta<32,16,256> warp<24,48,128> mma<8,8,128>   WARPS[8x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 256, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 256, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 256, 24, 48, 128, 8, 8, 128, 4);
// cta<32,16,256> warp<24,24,128> mma<8,8,128>   WARPS[8x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 256, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 256, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 256, 24, 24, 128, 8, 8, 128, 4);
// cta<4,8,384> warp<24,48,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 384, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 384, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 384, 24, 48, 128, 8, 8, 128, 4);
// cta<8,8,384> warp<48,48,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 384, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 384, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 384, 48, 48, 128, 8, 8, 128, 4);
// cta<4,8,384> warp<24,24,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 384, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 384, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 384, 24, 24, 128, 8, 8, 128, 4);
// cta<4,16,384> warp<24,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 384, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 384, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 384, 24, 48, 128, 8, 8, 128, 4);
// cta<8,8,384> warp<48,24,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 384, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 384, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 384, 48, 24, 128, 8, 8, 128, 4);
// cta<8,16,384> warp<48,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 384, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 384, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 384, 48, 48, 128, 8, 8, 128, 4);
// cta<4,16,384> warp<24,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 384, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 384, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 384, 24, 24, 128, 8, 8, 128, 4);
// cta<4,32,384> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 384, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 384, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 384, 24, 48, 128, 8, 8, 128, 4);
// cta<8,16,384> warp<48,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 384, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 384, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 384, 48, 24, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<48,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 384, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 384, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 384, 48, 48, 128, 8, 8, 128, 4);
// cta<4,32,384> warp<24,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 384, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 384, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 384, 24, 24, 128, 8, 8, 128, 4);
// cta<4,64,384> warp<24,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 384, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 384, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 384, 24, 48, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<48,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 384, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 384, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 384, 48, 24, 128, 8, 8, 128, 4);
// cta<8,64,384> warp<48,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 384, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 384, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 384, 48, 48, 128, 8, 8, 128, 4);
// cta<4,64,384> warp<24,24,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 384, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 384, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 384, 24, 24, 128, 8, 8, 128, 4);
// cta<4,128,384> warp<24,48,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 128, 384, 24, 48, 128, 8, 8, 128, 2);
// cta<8,64,384> warp<48,24,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 384, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 384, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 384, 48, 24, 128, 8, 8, 128, 4);
// cta<4,128,384> warp<24,24,128> mma<8,8,128>   WARPS[1x32]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 128, 384, 24, 24, 128, 8, 8, 128, 2);
// cta<8,8,384> warp<24,48,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 384, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 384, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 384, 24, 48, 128, 8, 8, 128, 4);
// cta<16,8,384> warp<48,48,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 384, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 384, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 384, 48, 48, 128, 8, 8, 128, 4);
// cta<8,8,384> warp<24,24,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 384, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 384, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 384, 24, 24, 128, 8, 8, 128, 4);
// cta<8,16,384> warp<24,48,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 384, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 384, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 384, 24, 48, 128, 8, 8, 128, 4);
// cta<16,8,384> warp<48,24,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 384, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 384, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 384, 48, 24, 128, 8, 8, 128, 4);
// cta<16,16,384> warp<48,48,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 384, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 384, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 384, 48, 48, 128, 8, 8, 128, 4);
// cta<8,16,384> warp<24,24,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 384, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 384, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 384, 24, 24, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<24,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 384, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 384, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 384, 24, 48, 128, 8, 8, 128, 4);
// cta<16,16,384> warp<48,24,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 384, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 384, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 384, 48, 24, 128, 8, 8, 128, 4);
// cta<16,32,384> warp<48,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 384, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 384, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 384, 48, 48, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<24,24,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 384, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 384, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 384, 24, 24, 128, 8, 8, 128, 4);
// cta<8,64,384> warp<24,48,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 384, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 384, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 384, 24, 48, 128, 8, 8, 128, 4);
// cta<16,32,384> warp<48,24,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 384, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 384, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 384, 48, 24, 128, 8, 8, 128, 4);
// cta<8,64,384> warp<24,24,128> mma<8,8,128>   WARPS[2x16]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 384, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 384, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 384, 24, 24, 128, 8, 8, 128, 4);
// cta<16,8,384> warp<24,48,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 384, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 384, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 384, 24, 48, 128, 8, 8, 128, 4);
// cta<32,8,384> warp<48,48,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 384, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 384, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 384, 48, 48, 128, 8, 8, 128, 4);
// cta<16,8,384> warp<24,24,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 384, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 384, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 384, 24, 24, 128, 8, 8, 128, 4);
// cta<16,16,384> warp<24,48,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 384, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 384, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 384, 24, 48, 128, 8, 8, 128, 4);
// cta<32,8,384> warp<48,24,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 384, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 384, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 384, 48, 24, 128, 8, 8, 128, 4);
// cta<32,16,384> warp<48,48,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 384, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 384, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 384, 48, 48, 128, 8, 8, 128, 4);
// cta<16,16,384> warp<24,24,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 384, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 384, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 384, 24, 24, 128, 8, 8, 128, 4);
// cta<16,32,384> warp<24,48,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 384, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 384, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 384, 24, 48, 128, 8, 8, 128, 4);
// cta<32,16,384> warp<48,24,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 384, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 384, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 384, 48, 24, 128, 8, 8, 128, 4);
// cta<16,32,384> warp<24,24,128> mma<8,8,128>   WARPS[4x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 384, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 384, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 384, 24, 24, 128, 8, 8, 128, 4);
// cta<32,8,384> warp<24,48,128> mma<8,8,128>   WARPS[8x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 384, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 384, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 384, 24, 48, 128, 8, 8, 128, 4);
// cta<32,8,384> warp<24,24,128> mma<8,8,128>   WARPS[8x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 384, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 384, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 384, 24, 24, 128, 8, 8, 128, 4);
// cta<32,16,384> warp<24,48,128> mma<8,8,128>   WARPS[8x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 384, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 384, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 384, 24, 48, 128, 8, 8, 128, 4);
// cta<32,16,384> warp<24,24,128> mma<8,8,128>   WARPS[8x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 384, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 384, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 384, 24, 24, 128, 8, 8, 128, 4);
// cta<4,8,512> warp<24,48,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 512, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 512, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 512, 24, 48, 128, 8, 8, 128, 4);
// cta<8,8,512> warp<48,48,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 512, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 512, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 512, 48, 48, 128, 8, 8, 128, 4);
// cta<4,8,512> warp<24,24,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 512, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 512, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 512, 24, 24, 128, 8, 8, 128, 4);
// cta<4,16,512> warp<24,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 512, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 512, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 512, 24, 48, 128, 8, 8, 128, 4);
// cta<8,8,512> warp<48,24,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 512, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 512, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 512, 48, 24, 128, 8, 8, 128, 4);
// cta<8,16,512> warp<48,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 512, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 512, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 512, 48, 48, 128, 8, 8, 128, 4);
// cta<4,16,512> warp<24,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 512, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 512, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 512, 24, 24, 128, 8, 8, 128, 4);
// cta<4,32,512> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 512, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 512, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 512, 24, 48, 128, 8, 8, 128, 4);
// cta<8,16,512> warp<48,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 512, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 512, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 512, 48, 24, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<48,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 512, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 512, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 512, 48, 48, 128, 8, 8, 128, 4);
// cta<4,32,512> warp<24,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 512, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 512, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 512, 24, 24, 128, 8, 8, 128, 4);
// cta<4,64,512> warp<24,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 512, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 512, 24, 48, 128, 8, 8, 128, 3);
// cta<8,32,512> warp<48,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 512, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 512, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 512, 48, 24, 128, 8, 8, 128, 4);
// cta<8,64,512> warp<48,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 512, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 512, 48, 48, 128, 8, 8, 128, 3);
// cta<4,64,512> warp<24,24,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 512, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 512, 24, 24, 128, 8, 8, 128, 3);
// cta<8,64,512> warp<48,24,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 512, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 512, 48, 24, 128, 8, 8, 128, 3);
// cta<8,8,512> warp<24,48,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 512, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 512, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 512, 24, 48, 128, 8, 8, 128, 4);
// cta<16,8,512> warp<48,48,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 512, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 512, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 512, 48, 48, 128, 8, 8, 128, 4);
// cta<8,8,512> warp<24,24,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 512, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 512, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 512, 24, 24, 128, 8, 8, 128, 4);
// cta<8,16,512> warp<24,48,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 512, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 512, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 512, 24, 48, 128, 8, 8, 128, 4);
// cta<16,8,512> warp<48,24,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 512, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 512, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 512, 48, 24, 128, 8, 8, 128, 4);
// cta<16,16,512> warp<48,48,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 512, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 512, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 512, 48, 48, 128, 8, 8, 128, 4);
// cta<8,16,512> warp<24,24,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 512, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 512, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 512, 24, 24, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<24,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 512, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 512, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 512, 24, 48, 128, 8, 8, 128, 4);
// cta<16,16,512> warp<48,24,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 512, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 512, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 512, 48, 24, 128, 8, 8, 128, 4);
// cta<16,32,512> warp<48,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 512, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 512, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 512, 48, 48, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<24,24,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 512, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 512, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 512, 24, 24, 128, 8, 8, 128, 4);
// cta<8,64,512> warp<24,48,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 512, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 512, 24, 48, 128, 8, 8, 128, 3);
// cta<16,32,512> warp<48,24,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 512, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 512, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 512, 48, 24, 128, 8, 8, 128, 4);
// cta<8,64,512> warp<24,24,128> mma<8,8,128>   WARPS[2x16]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 512, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 512, 24, 24, 128, 8, 8, 128, 3);
// cta<16,8,512> warp<24,48,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 512, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 512, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 512, 24, 48, 128, 8, 8, 128, 4);
// cta<32,8,512> warp<48,48,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 512, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 512, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 512, 48, 48, 128, 8, 8, 128, 4);
// cta<16,8,512> warp<24,24,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 512, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 512, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 512, 24, 24, 128, 8, 8, 128, 4);
// cta<16,16,512> warp<24,48,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 512, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 512, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 512, 24, 48, 128, 8, 8, 128, 4);
// cta<32,8,512> warp<48,24,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 512, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 512, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 512, 48, 24, 128, 8, 8, 128, 4);
// cta<32,16,512> warp<48,48,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 512, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 512, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 512, 48, 48, 128, 8, 8, 128, 4);
// cta<16,16,512> warp<24,24,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 512, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 512, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 512, 24, 24, 128, 8, 8, 128, 4);
// cta<16,32,512> warp<24,48,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 512, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 512, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 512, 24, 48, 128, 8, 8, 128, 4);
// cta<32,16,512> warp<48,24,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 512, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 512, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 512, 48, 24, 128, 8, 8, 128, 4);
// cta<16,32,512> warp<24,24,128> mma<8,8,128>   WARPS[4x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 512, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 512, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 512, 24, 24, 128, 8, 8, 128, 4);
// cta<32,8,512> warp<24,48,128> mma<8,8,128>   WARPS[8x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 512, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 512, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 512, 24, 48, 128, 8, 8, 128, 4);
// cta<32,8,512> warp<24,24,128> mma<8,8,128>   WARPS[8x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 512, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 512, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 512, 24, 24, 128, 8, 8, 128, 4);
// cta<32,16,512> warp<24,48,128> mma<8,8,128>   WARPS[8x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 512, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 512, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 512, 24, 48, 128, 8, 8, 128, 4);
// cta<32,16,512> warp<24,24,128> mma<8,8,128>   WARPS[8x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 512, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 512, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 512, 24, 24, 128, 8, 8, 128, 4);
// cta<4,8,640> warp<24,48,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 640, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 640, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 640, 24, 48, 128, 8, 8, 128, 4);
// cta<8,8,640> warp<48,48,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 640, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 640, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 640, 48, 48, 128, 8, 8, 128, 4);
// cta<4,8,640> warp<24,24,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 640, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 640, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 640, 24, 24, 128, 8, 8, 128, 4);
// cta<4,16,640> warp<24,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 640, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 640, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 640, 24, 48, 128, 8, 8, 128, 4);
// cta<8,8,640> warp<48,24,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 640, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 640, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 640, 48, 24, 128, 8, 8, 128, 4);
// cta<8,16,640> warp<48,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 640, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 640, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 640, 48, 48, 128, 8, 8, 128, 4);
// cta<4,16,640> warp<24,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 640, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 640, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 640, 24, 24, 128, 8, 8, 128, 4);
// cta<4,32,640> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 640, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 640, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 640, 24, 48, 128, 8, 8, 128, 4);
// cta<8,16,640> warp<48,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 640, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 640, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 640, 48, 24, 128, 8, 8, 128, 4);
// cta<8,32,640> warp<48,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 640, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 640, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 640, 48, 48, 128, 8, 8, 128, 4);
// cta<4,32,640> warp<24,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 640, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 640, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 640, 24, 24, 128, 8, 8, 128, 4);
// cta<4,64,640> warp<24,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 640, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 640, 24, 48, 128, 8, 8, 128, 3);
// cta<8,32,640> warp<48,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 640, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 640, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 640, 48, 24, 128, 8, 8, 128, 4);
// cta<8,64,640> warp<48,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 640, 48, 48, 128, 8, 8, 128, 2);
// cta<4,64,640> warp<24,24,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 640, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 640, 24, 24, 128, 8, 8, 128, 3);
// cta<8,64,640> warp<48,24,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 640, 48, 24, 128, 8, 8, 128, 2);
// cta<8,8,640> warp<24,48,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 640, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 640, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 640, 24, 48, 128, 8, 8, 128, 4);
// cta<16,8,640> warp<48,48,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 640, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 640, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 640, 48, 48, 128, 8, 8, 128, 4);
// cta<8,8,640> warp<24,24,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 640, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 640, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 640, 24, 24, 128, 8, 8, 128, 4);
// cta<8,16,640> warp<24,48,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 640, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 640, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 640, 24, 48, 128, 8, 8, 128, 4);
// cta<16,8,640> warp<48,24,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 640, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 640, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 640, 48, 24, 128, 8, 8, 128, 4);
// cta<16,16,640> warp<48,48,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 640, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 640, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 640, 48, 48, 128, 8, 8, 128, 4);
// cta<8,16,640> warp<24,24,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 640, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 640, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 640, 24, 24, 128, 8, 8, 128, 4);
// cta<8,32,640> warp<24,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 640, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 640, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 640, 24, 48, 128, 8, 8, 128, 4);
// cta<16,16,640> warp<48,24,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 640, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 640, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 640, 48, 24, 128, 8, 8, 128, 4);
// cta<16,32,640> warp<48,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 640, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 640, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 640, 48, 48, 128, 8, 8, 128, 4);
// cta<8,32,640> warp<24,24,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 640, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 640, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 640, 24, 24, 128, 8, 8, 128, 4);
// cta<8,64,640> warp<24,48,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 640, 24, 48, 128, 8, 8, 128, 2);
// cta<16,32,640> warp<48,24,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 640, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 640, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 640, 48, 24, 128, 8, 8, 128, 4);
// cta<8,64,640> warp<24,24,128> mma<8,8,128>   WARPS[2x16]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 640, 24, 24, 128, 8, 8, 128, 2);
// cta<16,8,640> warp<24,48,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 640, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 640, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 640, 24, 48, 128, 8, 8, 128, 4);
// cta<32,8,640> warp<48,48,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 640, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 640, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 640, 48, 48, 128, 8, 8, 128, 4);
// cta<16,8,640> warp<24,24,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 640, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 640, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 640, 24, 24, 128, 8, 8, 128, 4);
// cta<16,16,640> warp<24,48,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 640, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 640, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 640, 24, 48, 128, 8, 8, 128, 4);
// cta<32,8,640> warp<48,24,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 640, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 640, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 640, 48, 24, 128, 8, 8, 128, 4);
// cta<32,16,640> warp<48,48,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 640, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 640, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 640, 48, 48, 128, 8, 8, 128, 4);
// cta<16,16,640> warp<24,24,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 640, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 640, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 640, 24, 24, 128, 8, 8, 128, 4);
// cta<16,32,640> warp<24,48,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 640, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 640, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 640, 24, 48, 128, 8, 8, 128, 4);
// cta<32,16,640> warp<48,24,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 640, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 640, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 640, 48, 24, 128, 8, 8, 128, 4);
// cta<16,32,640> warp<24,24,128> mma<8,8,128>   WARPS[4x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 640, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 640, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 640, 24, 24, 128, 8, 8, 128, 4);
// cta<32,8,640> warp<24,48,128> mma<8,8,128>   WARPS[8x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 640, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 640, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 640, 24, 48, 128, 8, 8, 128, 4);
// cta<32,8,640> warp<24,24,128> mma<8,8,128>   WARPS[8x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 640, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 640, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 640, 24, 24, 128, 8, 8, 128, 4);
// cta<32,16,640> warp<24,48,128> mma<8,8,128>   WARPS[8x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 640, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 640, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 640, 24, 48, 128, 8, 8, 128, 4);
// cta<32,16,640> warp<24,24,128> mma<8,8,128>   WARPS[8x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 640, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 640, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 640, 24, 24, 128, 8, 8, 128, 4);
// cta<4,8,768> warp<24,48,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 768, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 768, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 768, 24, 48, 128, 8, 8, 128, 4);
// cta<8,8,768> warp<48,48,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 768, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 768, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 768, 48, 48, 128, 8, 8, 128, 4);
// cta<4,8,768> warp<24,24,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 768, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 768, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 768, 24, 24, 128, 8, 8, 128, 4);
// cta<4,16,768> warp<24,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 768, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 768, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 768, 24, 48, 128, 8, 8, 128, 4);
// cta<8,8,768> warp<48,24,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 768, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 768, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 768, 48, 24, 128, 8, 8, 128, 4);
// cta<8,16,768> warp<48,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 768, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 768, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 768, 48, 48, 128, 8, 8, 128, 4);
// cta<4,16,768> warp<24,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 768, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 768, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 768, 24, 24, 128, 8, 8, 128, 4);
// cta<4,32,768> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 768, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 768, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 768, 24, 48, 128, 8, 8, 128, 4);
// cta<8,16,768> warp<48,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 768, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 768, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 768, 48, 24, 128, 8, 8, 128, 4);
// cta<8,32,768> warp<48,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 768, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 768, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 768, 48, 48, 128, 8, 8, 128, 4);
// cta<4,32,768> warp<24,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 768, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 768, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 768, 24, 24, 128, 8, 8, 128, 4);
// cta<4,64,768> warp<24,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 768, 24, 48, 128, 8, 8, 128, 2);
// cta<8,32,768> warp<48,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 768, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 768, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 768, 48, 24, 128, 8, 8, 128, 4);
// cta<8,64,768> warp<48,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 768, 48, 48, 128, 8, 8, 128, 2);
// cta<4,64,768> warp<24,24,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 768, 24, 24, 128, 8, 8, 128, 2);
// cta<8,64,768> warp<48,24,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 768, 48, 24, 128, 8, 8, 128, 2);
// cta<8,8,768> warp<24,48,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 768, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 768, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 768, 24, 48, 128, 8, 8, 128, 4);
// cta<16,8,768> warp<48,48,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 768, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 768, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 768, 48, 48, 128, 8, 8, 128, 4);
// cta<8,8,768> warp<24,24,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 768, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 768, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 768, 24, 24, 128, 8, 8, 128, 4);
// cta<8,16,768> warp<24,48,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 768, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 768, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 768, 24, 48, 128, 8, 8, 128, 4);
// cta<16,8,768> warp<48,24,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 768, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 768, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 768, 48, 24, 128, 8, 8, 128, 4);
// cta<16,16,768> warp<48,48,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 768, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 768, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 768, 48, 48, 128, 8, 8, 128, 4);
// cta<8,16,768> warp<24,24,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 768, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 768, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 768, 24, 24, 128, 8, 8, 128, 4);
// cta<8,32,768> warp<24,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 768, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 768, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 768, 24, 48, 128, 8, 8, 128, 4);
// cta<16,16,768> warp<48,24,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 768, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 768, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 768, 48, 24, 128, 8, 8, 128, 4);
// cta<16,32,768> warp<48,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 768, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 768, 48, 48, 128, 8, 8, 128, 3);
// cta<8,32,768> warp<24,24,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 768, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 768, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 768, 24, 24, 128, 8, 8, 128, 4);
// cta<8,64,768> warp<24,48,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 768, 24, 48, 128, 8, 8, 128, 2);
// cta<16,32,768> warp<48,24,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 768, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 768, 48, 24, 128, 8, 8, 128, 3);
// cta<8,64,768> warp<24,24,128> mma<8,8,128>   WARPS[2x16]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 768, 24, 24, 128, 8, 8, 128, 2);
// cta<16,8,768> warp<24,48,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 768, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 768, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 768, 24, 48, 128, 8, 8, 128, 4);
// cta<32,8,768> warp<48,48,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 768, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 768, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 768, 48, 48, 128, 8, 8, 128, 4);
// cta<16,8,768> warp<24,24,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 768, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 768, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 768, 24, 24, 128, 8, 8, 128, 4);
// cta<16,16,768> warp<24,48,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 768, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 768, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 768, 24, 48, 128, 8, 8, 128, 4);
// cta<32,8,768> warp<48,24,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 768, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 768, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 768, 48, 24, 128, 8, 8, 128, 4);
// cta<32,16,768> warp<48,48,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 768, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 768, 48, 48, 128, 8, 8, 128, 3);
// cta<16,16,768> warp<24,24,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 768, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 768, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 768, 24, 24, 128, 8, 8, 128, 4);
// cta<16,32,768> warp<24,48,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 768, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 768, 24, 48, 128, 8, 8, 128, 3);
// cta<32,16,768> warp<48,24,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 768, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 768, 48, 24, 128, 8, 8, 128, 3);
// cta<16,32,768> warp<24,24,128> mma<8,8,128>   WARPS[4x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 768, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 768, 24, 24, 128, 8, 8, 128, 3);
// cta<32,8,768> warp<24,48,128> mma<8,8,128>   WARPS[8x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 768, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 768, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 768, 24, 48, 128, 8, 8, 128, 4);
// cta<32,8,768> warp<24,24,128> mma<8,8,128>   WARPS[8x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 768, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 768, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 768, 24, 24, 128, 8, 8, 128, 4);
// cta<32,16,768> warp<24,48,128> mma<8,8,128>   WARPS[8x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 768, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 768, 24, 48, 128, 8, 8, 128, 3);
// cta<32,16,768> warp<24,24,128> mma<8,8,128>   WARPS[8x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 768, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 768, 24, 24, 128, 8, 8, 128, 3);
// cta<4,8,896> warp<24,48,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 896, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 896, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 896, 24, 48, 128, 8, 8, 128, 4);
// cta<8,8,896> warp<48,48,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 896, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 896, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 896, 48, 48, 128, 8, 8, 128, 4);
// cta<4,8,896> warp<24,24,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 896, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 896, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 8, 896, 24, 24, 128, 8, 8, 128, 4);
// cta<4,16,896> warp<24,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 896, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 896, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 896, 24, 48, 128, 8, 8, 128, 4);
// cta<8,8,896> warp<48,24,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 896, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 896, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 896, 48, 24, 128, 8, 8, 128, 4);
// cta<8,16,896> warp<48,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 896, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 896, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 896, 48, 48, 128, 8, 8, 128, 4);
// cta<4,16,896> warp<24,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 896, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 896, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 16, 896, 24, 24, 128, 8, 8, 128, 4);
// cta<4,32,896> warp<24,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 896, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 896, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 896, 24, 48, 128, 8, 8, 128, 4);
// cta<8,16,896> warp<48,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 896, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 896, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 896, 48, 24, 128, 8, 8, 128, 4);
// cta<8,32,896> warp<48,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 896, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 896, 48, 48, 128, 8, 8, 128, 3);
// cta<4,32,896> warp<24,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 896, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 896, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 32, 896, 24, 24, 128, 8, 8, 128, 4);
// cta<4,64,896> warp<24,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 896, 24, 48, 128, 8, 8, 128, 2);
// cta<8,32,896> warp<48,24,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 896, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 896, 48, 24, 128, 8, 8, 128, 3);
// cta<8,64,896> warp<48,48,128> mma<8,8,128>   WARPS[1x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 896, 48, 48, 128, 8, 8, 128, 2);
// cta<4,64,896> warp<24,24,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 4, 64, 896, 24, 24, 128, 8, 8, 128, 2);
// cta<8,64,896> warp<48,24,128> mma<8,8,128>   WARPS[1x16]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 896, 48, 24, 128, 8, 8, 128, 2);
// cta<8,8,896> warp<24,48,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 896, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 896, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 896, 24, 48, 128, 8, 8, 128, 4);
// cta<16,8,896> warp<48,48,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 896, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 896, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 896, 48, 48, 128, 8, 8, 128, 4);
// cta<8,8,896> warp<24,24,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 896, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 896, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 8, 896, 24, 24, 128, 8, 8, 128, 4);
// cta<8,16,896> warp<24,48,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 896, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 896, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 896, 24, 48, 128, 8, 8, 128, 4);
// cta<16,8,896> warp<48,24,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 896, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 896, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 896, 48, 24, 128, 8, 8, 128, 4);
// cta<16,16,896> warp<48,48,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 896, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 896, 48, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 896, 48, 48, 128, 8, 8, 128, 4);
// cta<8,16,896> warp<24,24,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 896, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 896, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 16, 896, 24, 24, 128, 8, 8, 128, 4);
// cta<8,32,896> warp<24,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 896, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 896, 24, 48, 128, 8, 8, 128, 3);
// cta<16,16,896> warp<48,24,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 896, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 896, 48, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 896, 48, 24, 128, 8, 8, 128, 4);
// cta<16,32,896> warp<48,48,128> mma<8,8,128>   WARPS[2x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 896, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 896, 48, 48, 128, 8, 8, 128, 3);
// cta<8,32,896> warp<24,24,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 896, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 32, 896, 24, 24, 128, 8, 8, 128, 3);
// cta<8,64,896> warp<24,48,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 896, 24, 48, 128, 8, 8, 128, 2);
// cta<16,32,896> warp<48,24,128> mma<8,8,128>   WARPS[2x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 896, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 896, 48, 24, 128, 8, 8, 128, 3);
// cta<8,64,896> warp<24,24,128> mma<8,8,128>   WARPS[2x16]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 8, 64, 896, 24, 24, 128, 8, 8, 128, 2);
// cta<16,8,896> warp<24,48,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 896, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 896, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 896, 24, 48, 128, 8, 8, 128, 4);
// cta<32,8,896> warp<48,48,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 896, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 896, 48, 48, 128, 8, 8, 128, 3);
// cta<16,8,896> warp<24,24,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 896, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 896, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 8, 896, 24, 24, 128, 8, 8, 128, 4);
// cta<16,16,896> warp<24,48,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 896, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 896, 24, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 896, 24, 48, 128, 8, 8, 128, 4);
// cta<32,8,896> warp<48,24,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 896, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 896, 48, 24, 128, 8, 8, 128, 3);
// cta<32,16,896> warp<48,48,128> mma<8,8,128>   WARPS[4x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 896, 48, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 896, 48, 48, 128, 8, 8, 128, 3);
// cta<16,16,896> warp<24,24,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 896, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 896, 24, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 16, 896, 24, 24, 128, 8, 8, 128, 4);
// cta<16,32,896> warp<24,48,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 896, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 896, 24, 48, 128, 8, 8, 128, 3);
// cta<32,16,896> warp<48,24,128> mma<8,8,128>   WARPS[4x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 896, 48, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 896, 48, 24, 128, 8, 8, 128, 3);
// cta<16,32,896> warp<24,24,128> mma<8,8,128>   WARPS[4x8]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 896, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 16, 32, 896, 24, 24, 128, 8, 8, 128, 3);
// cta<32,8,896> warp<24,48,128> mma<8,8,128>   WARPS[8x1]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 896, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 896, 24, 48, 128, 8, 8, 128, 3);
// cta<32,8,896> warp<24,24,128> mma<8,8,128>   WARPS[8x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 896, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 8, 896, 24, 24, 128, 8, 8, 128, 3);
// cta<32,16,896> warp<24,48,128> mma<8,8,128>   WARPS[8x2]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 896, 24, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 896, 24, 48, 128, 8, 8, 128, 3);
// cta<32,16,896> warp<24,24,128> mma<8,8,128>   WARPS[8x4]
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 896, 24, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 6, 6, true, 32, 16, 896, 24, 24, 128, 8, 8, 128, 3);
#endif