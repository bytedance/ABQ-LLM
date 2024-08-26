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

#pragma once
#include "common/base.h"
#include "mma_any/aq_bmma_op.h"

#ifdef W3A8
////// W3A8 int
// cta<1,32,256> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 32, 256, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 32, 256, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 32, 256, 8, 48, 128, 8, 8, 128, 4);
// cta<1,48,256> warp<8,72,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 48, 256, 8, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 48, 256, 8, 72, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 48, 256, 8, 72, 128, 8, 8, 128, 4);
// cta<1,64,256> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 64, 256, 8, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 64, 256, 8, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 64, 256, 8, 96, 128, 8, 8, 128, 4);
// cta<1,80,256> warp<8,120,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 80, 256, 8, 120, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 80, 256, 8, 120, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 80, 256, 8, 120, 128, 8, 8, 128, 4);
// cta<4,32,256> warp<32,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 256, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 256, 32, 48, 128, 8, 8, 128, 3);
// cta<4,48,256> warp<32,72,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 48, 256, 32, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 48, 256, 32, 72, 128, 8, 8, 128, 3);
// cta<4,64,256> warp<32,96,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 64, 256, 32, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 64, 256, 32, 96, 128, 8, 8, 128, 3);
// cta<4,80,256> warp<32,120,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 80, 256, 32, 120, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 80, 256, 32, 120, 128, 8, 8, 128, 3);
// cta<8,32,256> warp<64,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 256, 64, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 256, 64, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 256, 64, 48, 128, 8, 8, 128, 4);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 256, 64, 48, 128, 8, 8, 128, 5);
// cta<8,48,256> warp<64,72,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 48, 256, 64, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 48, 256, 64, 72, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 48, 256, 64, 72, 128, 8, 8, 128, 4);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 48, 256, 64, 72, 128, 8, 8, 128, 5);
// cta<8,64,256> warp<64,96,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 256, 64, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 256, 64, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 256, 64, 96, 128, 8, 8, 128, 4);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 256, 64, 96, 128, 8, 8, 128, 5);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 256, 64, 96, 128, 8, 8, 128, 6);
// cta<1,32,256> warp<8,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 32, 256, 8, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 32, 256, 8, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 32, 256, 8, 24, 128, 8, 8, 128, 4);
// cta<1,64,256> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 64, 256, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 64, 256, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 64, 256, 8, 48, 128, 8, 8, 128, 4);
// cta<1,96,256> warp<8,72,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 96, 256, 8, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 96, 256, 8, 72, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 96, 256, 8, 72, 128, 8, 8, 128, 4);
// cta<1,128,256> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 128, 256, 8, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 128, 256, 8, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 128, 256, 8, 96, 128, 8, 8, 128, 4);
// cta<4,32,256> warp<32,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 256, 32, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 256, 32, 24, 128, 8, 8, 128, 3);
// cta<4,64,256> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 64, 256, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 64, 256, 32, 48, 128, 8, 8, 128, 3);
// cta<4,96,256> warp<32,72,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 96, 256, 32, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 96, 256, 32, 72, 128, 8, 8, 128, 3);
// cta<4,128,256> warp<32,96,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 128, 256, 32, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 128, 256, 32, 96, 128, 8, 8, 128, 3);
// cta<8,32,256> warp<64,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 256, 64, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 256, 64, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 256, 64, 24, 128, 8, 8, 128, 4);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 256, 64, 24, 128, 8, 8, 128, 5);
// cta<8,64,256> warp<64,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 256, 64, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 256, 64, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 256, 64, 48, 128, 8, 8, 128, 4);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 256, 64, 48, 128, 8, 8, 128, 5);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 256, 64, 48, 128, 8, 8, 128, 6);
// cta<8,96,256> warp<64,72,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 96, 256, 64, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 96, 256, 64, 72, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 96, 256, 64, 72, 128, 8, 8, 128, 4);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 96, 256, 64, 72, 128, 8, 8, 128, 5);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 96, 256, 64, 72, 128, 8, 8, 128, 6);
// cta<8,128,256> warp<64,96,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 128, 256, 64, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 128, 256, 64, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 128, 256, 64, 96, 128, 8, 8, 128, 4);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 128, 256, 64, 96, 128, 8, 8, 128, 5);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 128, 256, 64, 96, 128, 8, 8, 128, 6);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 128, 256, 64, 96, 128, 8, 8, 128, 7);
// cta<4,32,256> warp<16,96,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 256, 16, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 256, 16, 96, 128, 8, 8, 128, 3);
// cta<8,32,256> warp<32,96,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 256, 32, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 256, 32, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 256, 32, 96, 128, 8, 8, 128, 4);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 256, 32, 96, 128, 8, 8, 128, 5);
// cta<4,32,256> warp<16,48,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 256, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 256, 16, 48, 128, 8, 8, 128, 3);
// cta<4,48,256> warp<16,72,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 48, 256, 16, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 48, 256, 16, 72, 128, 8, 8, 128, 3);
// cta<4,64,256> warp<16,96,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 64, 256, 16, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 64, 256, 16, 96, 128, 8, 8, 128, 3);
// cta<4,80,256> warp<16,120,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 80, 256, 16, 120, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 80, 256, 16, 120, 128, 8, 8, 128, 3);
// cta<8,32,256> warp<32,48,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 256, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 256, 32, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 256, 32, 48, 128, 8, 8, 128, 4);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 256, 32, 48, 128, 8, 8, 128, 5);
// cta<8,48,256> warp<32,72,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 48, 256, 32, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 48, 256, 32, 72, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 48, 256, 32, 72, 128, 8, 8, 128, 4);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 48, 256, 32, 72, 128, 8, 8, 128, 5);
// cta<8,64,256> warp<32,96,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 256, 32, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 256, 32, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 256, 32, 96, 128, 8, 8, 128, 4);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 256, 32, 96, 128, 8, 8, 128, 5);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 256, 32, 96, 128, 8, 8, 128, 6);
// cta<8,80,256> warp<32,120,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 80, 256, 32, 120, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 80, 256, 32, 120, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 80, 256, 32, 120, 128, 8, 8, 128, 4);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 80, 256, 32, 120, 128, 8, 8, 128, 5);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 80, 256, 32, 120, 128, 8, 8, 128, 6);
// cta<4,32,256> warp<8,96,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 256, 8, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 256, 8, 96, 128, 8, 8, 128, 3);
// cta<8,32,256> warp<16,96,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 256, 16, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 256, 16, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 256, 16, 96, 128, 8, 8, 128, 4);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 256, 16, 96, 128, 8, 8, 128, 5);
// cta<1,32,384> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 32, 384, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 32, 384, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 32, 384, 8, 48, 128, 8, 8, 128, 4);
// cta<1,48,384> warp<8,72,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 48, 384, 8, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 48, 384, 8, 72, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 48, 384, 8, 72, 128, 8, 8, 128, 4);
// cta<1,64,384> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 64, 384, 8, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 64, 384, 8, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 64, 384, 8, 96, 128, 8, 8, 128, 4);
// cta<1,80,384> warp<8,120,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 80, 384, 8, 120, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 80, 384, 8, 120, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 80, 384, 8, 120, 128, 8, 8, 128, 4);
// cta<4,32,384> warp<32,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 384, 32, 48, 128, 8, 8, 128, 2);
// cta<4,48,384> warp<32,72,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 48, 384, 32, 72, 128, 8, 8, 128, 2);
// cta<4,64,384> warp<32,96,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 64, 384, 32, 96, 128, 8, 8, 128, 2);
// cta<4,80,384> warp<32,120,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 80, 384, 32, 120, 128, 8, 8, 128, 2);
// cta<8,32,384> warp<64,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 384, 64, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 384, 64, 48, 128, 8, 8, 128, 3);
// cta<8,48,384> warp<64,72,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 48, 384, 64, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 48, 384, 64, 72, 128, 8, 8, 128, 3);
// cta<8,64,384> warp<64,96,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 384, 64, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 384, 64, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 384, 64, 96, 128, 8, 8, 128, 4);
// cta<1,32,384> warp<8,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 32, 384, 8, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 32, 384, 8, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 32, 384, 8, 24, 128, 8, 8, 128, 4);
// cta<1,64,384> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 64, 384, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 64, 384, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 64, 384, 8, 48, 128, 8, 8, 128, 4);
// cta<1,96,384> warp<8,72,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 96, 384, 8, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 96, 384, 8, 72, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 96, 384, 8, 72, 128, 8, 8, 128, 4);
// cta<1,128,384> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 128, 384, 8, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 128, 384, 8, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 128, 384, 8, 96, 128, 8, 8, 128, 4);
// cta<4,32,384> warp<32,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 384, 32, 24, 128, 8, 8, 128, 2);
// cta<4,64,384> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 64, 384, 32, 48, 128, 8, 8, 128, 2);
// cta<4,96,384> warp<32,72,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 96, 384, 32, 72, 128, 8, 8, 128, 2);
// cta<4,128,384> warp<32,96,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 128, 384, 32, 96, 128, 8, 8, 128, 2);
// cta<8,32,384> warp<64,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 384, 64, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 384, 64, 24, 128, 8, 8, 128, 3);
// cta<8,64,384> warp<64,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 384, 64, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 384, 64, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 384, 64, 48, 128, 8, 8, 128, 4);
// cta<8,96,384> warp<64,72,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 96, 384, 64, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 96, 384, 64, 72, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 96, 384, 64, 72, 128, 8, 8, 128, 4);
// cta<8,128,384> warp<64,96,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 128, 384, 64, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 128, 384, 64, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 128, 384, 64, 96, 128, 8, 8, 128, 4);
// cta<4,32,384> warp<16,96,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 384, 16, 96, 128, 8, 8, 128, 2);
// cta<8,32,384> warp<32,96,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 384, 32, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 384, 32, 96, 128, 8, 8, 128, 3);
// cta<4,32,384> warp<16,48,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 384, 16, 48, 128, 8, 8, 128, 2);
// cta<4,48,384> warp<16,72,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 48, 384, 16, 72, 128, 8, 8, 128, 2);
// cta<4,64,384> warp<16,96,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 64, 384, 16, 96, 128, 8, 8, 128, 2);
// cta<4,80,384> warp<16,120,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 80, 384, 16, 120, 128, 8, 8, 128, 2);
// cta<8,32,384> warp<32,48,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 384, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 384, 32, 48, 128, 8, 8, 128, 3);
// cta<8,48,384> warp<32,72,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 48, 384, 32, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 48, 384, 32, 72, 128, 8, 8, 128, 3);
// cta<8,64,384> warp<32,96,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 384, 32, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 384, 32, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 384, 32, 96, 128, 8, 8, 128, 4);
// cta<8,80,384> warp<32,120,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 80, 384, 32, 120, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 80, 384, 32, 120, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 80, 384, 32, 120, 128, 8, 8, 128, 4);
// cta<4,32,384> warp<8,96,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 384, 8, 96, 128, 8, 8, 128, 2);
// cta<8,32,384> warp<16,96,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 384, 16, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 384, 16, 96, 128, 8, 8, 128, 3);
// cta<1,32,512> warp<8,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 32, 512, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 32, 512, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 32, 512, 8, 48, 128, 8, 8, 128, 4);
// cta<1,48,512> warp<8,72,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 48, 512, 8, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 48, 512, 8, 72, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 48, 512, 8, 72, 128, 8, 8, 128, 4);
// cta<1,64,512> warp<8,96,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 64, 512, 8, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 64, 512, 8, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 64, 512, 8, 96, 128, 8, 8, 128, 4);
// cta<1,80,512> warp<8,120,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 80, 512, 8, 120, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 80, 512, 8, 120, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 80, 512, 8, 120, 128, 8, 8, 128, 4);
// cta<4,32,512> warp<32,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 512, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 512, 32, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 512, 32, 48, 128, 8, 8, 128, 4);
// cta<4,48,512> warp<32,72,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 48, 512, 32, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 48, 512, 32, 72, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 48, 512, 32, 72, 128, 8, 8, 128, 4);
// cta<4,64,512> warp<32,96,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 64, 512, 32, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 64, 512, 32, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 64, 512, 32, 96, 128, 8, 8, 128, 4);
// cta<4,80,512> warp<32,120,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 80, 512, 32, 120, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 80, 512, 32, 120, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 80, 512, 32, 120, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<64,48,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 512, 64, 48, 128, 8, 8, 128, 2);
// cta<8,48,512> warp<64,72,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 48, 512, 64, 72, 128, 8, 8, 128, 2);
// cta<8,64,512> warp<64,96,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 512, 64, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 512, 64, 96, 128, 8, 8, 128, 3);
// cta<1,32,512> warp<8,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 32, 512, 8, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 32, 512, 8, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 32, 512, 8, 24, 128, 8, 8, 128, 4);
// cta<1,64,512> warp<8,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 64, 512, 8, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 64, 512, 8, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 64, 512, 8, 48, 128, 8, 8, 128, 4);
// cta<1,96,512> warp<8,72,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 96, 512, 8, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 96, 512, 8, 72, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 96, 512, 8, 72, 128, 8, 8, 128, 4);
// cta<1,128,512> warp<8,96,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 128, 512, 8, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 128, 512, 8, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 1, 128, 512, 8, 96, 128, 8, 8, 128, 4);
// cta<4,32,512> warp<32,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 512, 32, 24, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 512, 32, 24, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 512, 32, 24, 128, 8, 8, 128, 4);
// cta<4,64,512> warp<32,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 64, 512, 32, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 64, 512, 32, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 64, 512, 32, 48, 128, 8, 8, 128, 4);
// cta<4,96,512> warp<32,72,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 96, 512, 32, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 96, 512, 32, 72, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 96, 512, 32, 72, 128, 8, 8, 128, 4);
// cta<4,128,512> warp<32,96,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 128, 512, 32, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 128, 512, 32, 96, 128, 8, 8, 128, 3);
// cta<8,32,512> warp<64,24,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 512, 64, 24, 128, 8, 8, 128, 2);
// cta<8,64,512> warp<64,48,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 512, 64, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 512, 64, 48, 128, 8, 8, 128, 3);
// cta<8,96,512> warp<64,72,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 96, 512, 64, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 96, 512, 64, 72, 128, 8, 8, 128, 3);
// cta<8,128,512> warp<64,96,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 128, 512, 64, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 128, 512, 64, 96, 128, 8, 8, 128, 3);
// cta<4,32,512> warp<16,96,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 512, 16, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 512, 16, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 512, 16, 96, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<32,96,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 512, 32, 96, 128, 8, 8, 128, 2);
// cta<4,32,512> warp<16,48,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 512, 16, 48, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 512, 16, 48, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 512, 16, 48, 128, 8, 8, 128, 4);
// cta<4,48,512> warp<16,72,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 48, 512, 16, 72, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 48, 512, 16, 72, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 48, 512, 16, 72, 128, 8, 8, 128, 4);
// cta<4,64,512> warp<16,96,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 64, 512, 16, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 64, 512, 16, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 64, 512, 16, 96, 128, 8, 8, 128, 4);
// cta<4,80,512> warp<16,120,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 80, 512, 16, 120, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 80, 512, 16, 120, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 80, 512, 16, 120, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<32,48,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 512, 32, 48, 128, 8, 8, 128, 2);
// cta<8,48,512> warp<32,72,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 48, 512, 32, 72, 128, 8, 8, 128, 2);
// cta<8,64,512> warp<32,96,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 512, 32, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 64, 512, 32, 96, 128, 8, 8, 128, 3);
// cta<8,80,512> warp<32,120,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 80, 512, 32, 120, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 80, 512, 32, 120, 128, 8, 8, 128, 3);
// cta<4,32,512> warp<8,96,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 512, 8, 96, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 512, 8, 96, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBMMA, 8, 3, true, 4, 32, 512, 8, 96, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<16,96,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBMMA, 8, 3, true, 8, 32, 512, 16, 96, 128, 8, 8, 128, 2);
#endif