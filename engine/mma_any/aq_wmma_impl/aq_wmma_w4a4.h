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

#ifdef W4A4
////// W4A4 int
// cta<2,32,256> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 32, 256, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 32, 256, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 32, 256, 8, 64, 128, 8, 8, 128, 4);
// cta<2,48,256> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 48, 256, 8, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 48, 256, 8, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 48, 256, 8, 96, 128, 8, 8, 128, 4);
// cta<2,64,256> warp<8,128,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 64, 256, 8, 128, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 64, 256, 8, 128, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 64, 256, 8, 128, 128, 8, 8, 128, 4);
// cta<4,32,256> warp<16,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 256, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 256, 16, 64, 128, 8, 8, 128, 4);
// cta<4,48,256> warp<16,96,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 256, 16, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 256, 16, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 256, 16, 96, 128, 8, 8, 128, 4);
// cta<4,64,256> warp<16,128,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 256, 16, 128, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 256, 16, 128, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 256, 16, 128, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<32,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 256, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 256, 32, 64, 128, 8, 8, 128, 3);
// cta<8,48,256> warp<32,96,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 48, 256, 32, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 48, 256, 32, 96, 128, 8, 8, 128, 3);
// cta<8,64,256> warp<32,128,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 64, 256, 32, 128, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 64, 256, 32, 128, 128, 8, 8, 128, 3);
// cta<2,32,256> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 32, 256, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 32, 256, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 32, 256, 8, 32, 128, 8, 8, 128, 4);
// cta<2,48,256> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 48, 256, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 48, 256, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 48, 256, 8, 48, 128, 8, 8, 128, 4);
// cta<2,64,256> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 64, 256, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 64, 256, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 64, 256, 8, 64, 128, 8, 8, 128, 4);
// cta<2,80,256> warp<8,80,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 80, 256, 8, 80, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 80, 256, 8, 80, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 80, 256, 8, 80, 128, 8, 8, 128, 4);
// cta<2,96,256> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 96, 256, 8, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 96, 256, 8, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 96, 256, 8, 96, 128, 8, 8, 128, 4);
// cta<2,112,256> warp<8,112,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 112, 256, 8, 112, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 112, 256, 8, 112, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 112, 256, 8, 112, 128, 8, 8, 128, 4);
// cta<2,128,256> warp<8,128,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 128, 256, 8, 128, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 128, 256, 8, 128, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 128, 256, 8, 128, 128, 8, 8, 128, 4);
// cta<4,32,256> warp<16,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 256, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 256, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 256, 16, 32, 128, 8, 8, 128, 4);
// cta<4,48,256> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 256, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 256, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 256, 16, 48, 128, 8, 8, 128, 4);
// cta<4,64,256> warp<16,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 256, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 256, 16, 64, 128, 8, 8, 128, 4);
// cta<4,80,256> warp<16,80,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 80, 256, 16, 80, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 80, 256, 16, 80, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 80, 256, 16, 80, 128, 8, 8, 128, 4);
// cta<4,96,256> warp<16,96,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 96, 256, 16, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 96, 256, 16, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 96, 256, 16, 96, 128, 8, 8, 128, 4);
// cta<4,112,256> warp<16,112,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 112, 256, 16, 112, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 112, 256, 16, 112, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 112, 256, 16, 112, 128, 8, 8, 128, 4);
// cta<4,128,256> warp<16,128,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 128, 256, 16, 128, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 128, 256, 16, 128, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 128, 256, 16, 128, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<32,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 256, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 256, 32, 32, 128, 8, 8, 128, 3);
// cta<8,48,256> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 48, 256, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 48, 256, 32, 48, 128, 8, 8, 128, 3);
// cta<8,64,256> warp<32,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 64, 256, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 64, 256, 32, 64, 128, 8, 8, 128, 3);
// cta<8,80,256> warp<32,80,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 80, 256, 32, 80, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 80, 256, 32, 80, 128, 8, 8, 128, 3);
// cta<8,96,256> warp<32,96,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 96, 256, 32, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 96, 256, 32, 96, 128, 8, 8, 128, 3);
// cta<8,112,256> warp<32,112,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 112, 256, 32, 112, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 112, 256, 32, 112, 128, 8, 8, 128, 3);
// cta<8,128,256> warp<32,128,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 128, 256, 32, 128, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 128, 256, 32, 128, 128, 8, 8, 128, 3);
// cta<4,32,256> warp<8,128,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 256, 8, 128, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 256, 8, 128, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 256, 8, 128, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<16,128,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 256, 16, 128, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 256, 16, 128, 128, 8, 8, 128, 3);
// cta<4,32,256> warp<8,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 256, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 256, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 256, 8, 64, 128, 8, 8, 128, 4);
// cta<4,48,256> warp<8,96,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 256, 8, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 256, 8, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 256, 8, 96, 128, 8, 8, 128, 4);
// cta<4,64,256> warp<8,128,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 256, 8, 128, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 256, 8, 128, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 256, 8, 128, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<16,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 256, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 256, 16, 64, 128, 8, 8, 128, 3);
// cta<8,48,256> warp<16,96,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 48, 256, 16, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 48, 256, 16, 96, 128, 8, 8, 128, 3);
// cta<8,64,256> warp<16,128,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 64, 256, 16, 128, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 64, 256, 16, 128, 128, 8, 8, 128, 3);
// cta<8,32,256> warp<8,128,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 256, 8, 128, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 256, 8, 128, 128, 8, 8, 128, 3);
// cta<2,32,384> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 32, 384, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 32, 384, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 32, 384, 8, 64, 128, 8, 8, 128, 4);
// cta<2,48,384> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 48, 384, 8, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 48, 384, 8, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 48, 384, 8, 96, 128, 8, 8, 128, 4);
// cta<2,64,384> warp<8,128,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 64, 384, 8, 128, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 64, 384, 8, 128, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 64, 384, 8, 128, 128, 8, 8, 128, 4);
// cta<4,32,384> warp<16,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 384, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 384, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 384, 16, 64, 128, 8, 8, 128, 4);
// cta<4,48,384> warp<16,96,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 384, 16, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 384, 16, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 384, 16, 96, 128, 8, 8, 128, 4);
// cta<4,64,384> warp<16,128,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 384, 16, 128, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 384, 16, 128, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 384, 16, 128, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<32,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 384, 32, 64, 128, 8, 8, 128, 2);
// cta<8,48,384> warp<32,96,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 48, 384, 32, 96, 128, 8, 8, 128, 2);
// cta<8,64,384> warp<32,128,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 64, 384, 32, 128, 128, 8, 8, 128, 2);
// cta<2,32,384> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 32, 384, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 32, 384, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 32, 384, 8, 32, 128, 8, 8, 128, 4);
// cta<2,48,384> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 48, 384, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 48, 384, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 48, 384, 8, 48, 128, 8, 8, 128, 4);
// cta<2,64,384> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 64, 384, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 64, 384, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 64, 384, 8, 64, 128, 8, 8, 128, 4);
// cta<2,80,384> warp<8,80,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 80, 384, 8, 80, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 80, 384, 8, 80, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 80, 384, 8, 80, 128, 8, 8, 128, 4);
// cta<2,96,384> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 96, 384, 8, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 96, 384, 8, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 96, 384, 8, 96, 128, 8, 8, 128, 4);
// cta<2,112,384> warp<8,112,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 112, 384, 8, 112, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 112, 384, 8, 112, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 112, 384, 8, 112, 128, 8, 8, 128, 4);
// cta<2,128,384> warp<8,128,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 128, 384, 8, 128, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 128, 384, 8, 128, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 128, 384, 8, 128, 128, 8, 8, 128, 4);
// cta<4,32,384> warp<16,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 384, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 384, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 384, 16, 32, 128, 8, 8, 128, 4);
// cta<4,48,384> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 384, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 384, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 384, 16, 48, 128, 8, 8, 128, 4);
// cta<4,64,384> warp<16,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 384, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 384, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 384, 16, 64, 128, 8, 8, 128, 4);
// cta<4,80,384> warp<16,80,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 80, 384, 16, 80, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 80, 384, 16, 80, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 80, 384, 16, 80, 128, 8, 8, 128, 4);
// cta<4,96,384> warp<16,96,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 96, 384, 16, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 96, 384, 16, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 96, 384, 16, 96, 128, 8, 8, 128, 4);
// cta<4,112,384> warp<16,112,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 112, 384, 16, 112, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 112, 384, 16, 112, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 112, 384, 16, 112, 128, 8, 8, 128, 4);
// cta<4,128,384> warp<16,128,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 128, 384, 16, 128, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 128, 384, 16, 128, 128, 8, 8, 128, 3);
// cta<8,32,384> warp<32,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 384, 32, 32, 128, 8, 8, 128, 2);
// cta<8,48,384> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 48, 384, 32, 48, 128, 8, 8, 128, 2);
// cta<8,64,384> warp<32,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 64, 384, 32, 64, 128, 8, 8, 128, 2);
// cta<8,80,384> warp<32,80,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 80, 384, 32, 80, 128, 8, 8, 128, 2);
// cta<8,96,384> warp<32,96,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 96, 384, 32, 96, 128, 8, 8, 128, 2);
// cta<8,112,384> warp<32,112,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 112, 384, 32, 112, 128, 8, 8, 128, 2);
// cta<8,128,384> warp<32,128,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 128, 384, 32, 128, 128, 8, 8, 128, 2);
// cta<4,32,384> warp<8,128,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 384, 8, 128, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 384, 8, 128, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 384, 8, 128, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<16,128,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 384, 16, 128, 128, 8, 8, 128, 2);
// cta<4,32,384> warp<8,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 384, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 384, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 384, 8, 64, 128, 8, 8, 128, 4);
// cta<4,48,384> warp<8,96,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 384, 8, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 384, 8, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 384, 8, 96, 128, 8, 8, 128, 4);
// cta<4,64,384> warp<8,128,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 384, 8, 128, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 384, 8, 128, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 384, 8, 128, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<16,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 384, 16, 64, 128, 8, 8, 128, 2);
// cta<8,48,384> warp<16,96,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 48, 384, 16, 96, 128, 8, 8, 128, 2);
// cta<8,64,384> warp<16,128,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 64, 384, 16, 128, 128, 8, 8, 128, 2);
// cta<8,32,384> warp<8,128,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 384, 8, 128, 128, 8, 8, 128, 2);
// cta<2,32,512> warp<8,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 32, 512, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 32, 512, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 32, 512, 8, 64, 128, 8, 8, 128, 4);
// cta<2,48,512> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 48, 512, 8, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 48, 512, 8, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 48, 512, 8, 96, 128, 8, 8, 128, 4);
// cta<2,64,512> warp<8,128,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 64, 512, 8, 128, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 64, 512, 8, 128, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 64, 512, 8, 128, 128, 8, 8, 128, 4);
// cta<4,32,512> warp<16,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 512, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 512, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 512, 16, 64, 128, 8, 8, 128, 4);
// cta<4,48,512> warp<16,96,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 512, 16, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 512, 16, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 512, 16, 96, 128, 8, 8, 128, 4);
// cta<4,64,512> warp<16,128,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 512, 16, 128, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 512, 16, 128, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 512, 16, 128, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<32,64,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 512, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 512, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 512, 32, 64, 128, 8, 8, 128, 4);
// cta<8,48,512> warp<32,96,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 48, 512, 32, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 48, 512, 32, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 48, 512, 32, 96, 128, 8, 8, 128, 4);
// cta<8,64,512> warp<32,128,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 64, 512, 32, 128, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 64, 512, 32, 128, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 64, 512, 32, 128, 128, 8, 8, 128, 4);
// cta<2,32,512> warp<8,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 32, 512, 8, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 32, 512, 8, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 32, 512, 8, 32, 128, 8, 8, 128, 4);
// cta<2,48,512> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 48, 512, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 48, 512, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 48, 512, 8, 48, 128, 8, 8, 128, 4);
// cta<2,64,512> warp<8,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 64, 512, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 64, 512, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 64, 512, 8, 64, 128, 8, 8, 128, 4);
// cta<2,80,512> warp<8,80,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 80, 512, 8, 80, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 80, 512, 8, 80, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 80, 512, 8, 80, 128, 8, 8, 128, 4);
// cta<2,96,512> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 96, 512, 8, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 96, 512, 8, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 96, 512, 8, 96, 128, 8, 8, 128, 4);
// cta<2,112,512> warp<8,112,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 112, 512, 8, 112, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 112, 512, 8, 112, 128, 8, 8, 128, 3);
// cta<2,128,512> warp<8,128,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 128, 512, 8, 128, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 2, 128, 512, 8, 128, 128, 8, 8, 128, 3);
// cta<4,32,512> warp<16,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 512, 16, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 512, 16, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 512, 16, 32, 128, 8, 8, 128, 4);
// cta<4,48,512> warp<16,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 512, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 512, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 512, 16, 48, 128, 8, 8, 128, 4);
// cta<4,64,512> warp<16,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 512, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 512, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 512, 16, 64, 128, 8, 8, 128, 4);
// cta<4,80,512> warp<16,80,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 80, 512, 16, 80, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 80, 512, 16, 80, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 80, 512, 16, 80, 128, 8, 8, 128, 4);
// cta<4,96,512> warp<16,96,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 96, 512, 16, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 96, 512, 16, 96, 128, 8, 8, 128, 3);
// cta<4,112,512> warp<16,112,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 112, 512, 16, 112, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 112, 512, 16, 112, 128, 8, 8, 128, 3);
// cta<4,128,512> warp<16,128,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 128, 512, 16, 128, 128, 8, 8, 128, 2);
// cta<8,32,512> warp<32,32,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 512, 32, 32, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 512, 32, 32, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 512, 32, 32, 128, 8, 8, 128, 4);
// cta<8,48,512> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 48, 512, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 48, 512, 32, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 48, 512, 32, 48, 128, 8, 8, 128, 4);
// cta<8,64,512> warp<32,64,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 64, 512, 32, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 64, 512, 32, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 64, 512, 32, 64, 128, 8, 8, 128, 4);
// cta<8,80,512> warp<32,80,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 80, 512, 32, 80, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 80, 512, 32, 80, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 80, 512, 32, 80, 128, 8, 8, 128, 4);
// cta<8,96,512> warp<32,96,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 96, 512, 32, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 96, 512, 32, 96, 128, 8, 8, 128, 3);
// cta<8,112,512> warp<32,112,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 112, 512, 32, 112, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 112, 512, 32, 112, 128, 8, 8, 128, 3);
// cta<8,128,512> warp<32,128,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 128, 512, 32, 128, 128, 8, 8, 128, 2);
// cta<4,32,512> warp<8,128,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 512, 8, 128, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 512, 8, 128, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 512, 8, 128, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<16,128,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 512, 16, 128, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 512, 16, 128, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 512, 16, 128, 128, 8, 8, 128, 4);
// cta<4,32,512> warp<8,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 512, 8, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 512, 8, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 32, 512, 8, 64, 128, 8, 8, 128, 4);
// cta<4,48,512> warp<8,96,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 512, 8, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 512, 8, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 48, 512, 8, 96, 128, 8, 8, 128, 4);
// cta<4,64,512> warp<8,128,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 512, 8, 128, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 512, 8, 128, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 4, 64, 512, 8, 128, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<16,64,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 512, 16, 64, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 512, 16, 64, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 512, 16, 64, 128, 8, 8, 128, 4);
// cta<8,48,512> warp<16,96,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 48, 512, 16, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 48, 512, 16, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 48, 512, 16, 96, 128, 8, 8, 128, 4);
// cta<8,64,512> warp<16,128,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 64, 512, 16, 128, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 64, 512, 16, 128, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 64, 512, 16, 128, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<8,128,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 512, 8, 128, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 512, 8, 128, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 4, 4, true, 8, 32, 512, 8, 128, 128, 8, 8, 128, 4);
#endif