// Copyright (C) ABQ.2024 (xieyusheng.12@bytedance.com)
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

#ifdef W7A7
////// W7A7 int
// cta<8,8,128> warp<56,56,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 8, 128, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 8, 128, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 8, 128, 56, 56, 128, 8, 8, 128, 4);
// cta<8,16,128> warp<56,56,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 16, 128, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 16, 128, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 16, 128, 56, 56, 128, 8, 8, 128, 4);
// cta<8,32,128> warp<56,56,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 32, 128, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 32, 128, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 32, 128, 56, 56, 128, 8, 8, 128, 4);
// cta<16,8,128> warp<56,56,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 8, 128, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 8, 128, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 8, 128, 56, 56, 128, 8, 8, 128, 4);
// cta<16,16,128> warp<56,56,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 16, 128, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 16, 128, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 16, 128, 56, 56, 128, 8, 8, 128, 4);
// cta<32,8,128> warp<56,56,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 32, 8, 128, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 32, 8, 128, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 32, 8, 128, 56, 56, 128, 8, 8, 128, 4);
// cta<8,8,256> warp<56,56,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 8, 256, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 8, 256, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 8, 256, 56, 56, 128, 8, 8, 128, 4);
// cta<8,16,256> warp<56,56,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 16, 256, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 16, 256, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 16, 256, 56, 56, 128, 8, 8, 128, 4);
// cta<8,32,256> warp<56,56,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 32, 256, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 32, 256, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 32, 256, 56, 56, 128, 8, 8, 128, 4);
// cta<16,8,256> warp<56,56,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 8, 256, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 8, 256, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 8, 256, 56, 56, 128, 8, 8, 128, 4);
// cta<16,16,256> warp<56,56,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 16, 256, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 16, 256, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 16, 256, 56, 56, 128, 8, 8, 128, 4);
// cta<32,8,256> warp<56,56,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 32, 8, 256, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 32, 8, 256, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 32, 8, 256, 56, 56, 128, 8, 8, 128, 4);
// cta<8,8,384> warp<56,56,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 8, 384, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 8, 384, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 8, 384, 56, 56, 128, 8, 8, 128, 4);
// cta<8,16,384> warp<56,56,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 16, 384, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 16, 384, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 16, 384, 56, 56, 128, 8, 8, 128, 4);
// cta<8,32,384> warp<56,56,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 32, 384, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 32, 384, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 32, 384, 56, 56, 128, 8, 8, 128, 4);
// cta<16,8,384> warp<56,56,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 8, 384, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 8, 384, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 8, 384, 56, 56, 128, 8, 8, 128, 4);
// cta<16,16,384> warp<56,56,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 16, 384, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 16, 384, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 16, 384, 56, 56, 128, 8, 8, 128, 4);
// cta<32,8,384> warp<56,56,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 32, 8, 384, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 32, 8, 384, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 32, 8, 384, 56, 56, 128, 8, 8, 128, 4);
// cta<8,8,512> warp<56,56,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 8, 512, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 8, 512, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 8, 512, 56, 56, 128, 8, 8, 128, 4);
// cta<8,16,512> warp<56,56,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 16, 512, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 16, 512, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 16, 512, 56, 56, 128, 8, 8, 128, 4);
// cta<8,32,512> warp<56,56,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 32, 512, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 32, 512, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 32, 512, 56, 56, 128, 8, 8, 128, 4);
// cta<16,8,512> warp<56,56,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 8, 512, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 8, 512, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 8, 512, 56, 56, 128, 8, 8, 128, 4);
// cta<16,16,512> warp<56,56,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 16, 512, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 16, 512, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 16, 512, 56, 56, 128, 8, 8, 128, 4);
// cta<32,8,512> warp<56,56,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 32, 8, 512, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 32, 8, 512, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 32, 8, 512, 56, 56, 128, 8, 8, 128, 4);
// cta<8,8,640> warp<56,56,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 8, 640, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 8, 640, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 8, 640, 56, 56, 128, 8, 8, 128, 4);
// cta<8,16,640> warp<56,56,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 16, 640, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 16, 640, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 16, 640, 56, 56, 128, 8, 8, 128, 4);
// cta<8,32,640> warp<56,56,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 32, 640, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 32, 640, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 32, 640, 56, 56, 128, 8, 8, 128, 4);
// cta<16,8,640> warp<56,56,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 8, 640, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 8, 640, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 8, 640, 56, 56, 128, 8, 8, 128, 4);
// cta<16,16,640> warp<56,56,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 16, 640, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 16, 640, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 16, 640, 56, 56, 128, 8, 8, 128, 4);
// cta<32,8,640> warp<56,56,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 32, 8, 640, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 32, 8, 640, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 32, 8, 640, 56, 56, 128, 8, 8, 128, 4);
// cta<8,8,768> warp<56,56,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 8, 768, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 8, 768, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 8, 768, 56, 56, 128, 8, 8, 128, 4);
// cta<8,16,768> warp<56,56,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 16, 768, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 16, 768, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 16, 768, 56, 56, 128, 8, 8, 128, 4);
// cta<8,32,768> warp<56,56,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 32, 768, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 32, 768, 56, 56, 128, 8, 8, 128, 3);
// cta<16,8,768> warp<56,56,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 8, 768, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 8, 768, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 8, 768, 56, 56, 128, 8, 8, 128, 4);
// cta<16,16,768> warp<56,56,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 16, 768, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 16, 768, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 16, 768, 56, 56, 128, 8, 8, 128, 4);
// cta<32,8,768> warp<56,56,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 32, 8, 768, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 32, 8, 768, 56, 56, 128, 8, 8, 128, 3);
// cta<8,8,896> warp<56,56,128> mma<8,8,128>   WARPS[1x1]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 8, 896, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 8, 896, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 8, 896, 56, 56, 128, 8, 8, 128, 4);
// cta<8,16,896> warp<56,56,128> mma<8,8,128>   WARPS[1x2]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 16, 896, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 16, 896, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 16, 896, 56, 56, 128, 8, 8, 128, 4);
// cta<8,32,896> warp<56,56,128> mma<8,8,128>   WARPS[1x4]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 32, 896, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 8, 32, 896, 56, 56, 128, 8, 8, 128, 3);
// cta<16,8,896> warp<56,56,128> mma<8,8,128>   WARPS[2x1]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 8, 896, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 8, 896, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 8, 896, 56, 56, 128, 8, 8, 128, 4);
// cta<16,16,896> warp<56,56,128> mma<8,8,128>   WARPS[2x2]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 16, 896, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 16, 896, 56, 56, 128, 8, 8, 128, 3);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 16, 16, 896, 56, 56, 128, 8, 8, 128, 4);
// cta<32,8,896> warp<56,56,128> mma<8,8,128>   WARPS[4x1]
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 32, 8, 896, 56, 56, 128, 8, 8, 128, 2);
AQ_DECL_FUN(AqBWMMA, 7, 7, true, 32, 8, 896, 56, 56, 128, 8, 8, 128, 3);
#endif