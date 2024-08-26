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

// cta<1,48,512> warp<8,24,128> mma<8,8,128>   WARPS[1x4]
AQ_INSTANTIATE_FUN(AqBMMA, 8, 2, true, 1, 48, 512, 8, 24, 128, 8, 8, 128, 4);
AQ_INSTANTIATE_FUN(AqBMMA, 8, 2, false, 1, 48, 512, 8, 24, 128, 8, 8, 128, 4);
// cta<8,48,256> warp<32,48,128> mma<8,8,128>   WARPS[2x2]
AQ_INSTANTIATE_FUN(AqBMMA, 8, 2, true, 8, 48, 256, 32, 48, 128, 8, 8, 128, 5);
AQ_INSTANTIATE_FUN(AqBMMA, 8, 2, false, 8, 48, 256, 32, 48, 128, 8, 8, 128, 5);