// Copyright (C) ABQ.2024 (liusongwei.zju@bytedance.com)
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

// pack 4 bit
cudaError_t launch_pack4(const int *in_data, int *pack_data, int m, int k, int BIT,
                         cudaStream_t stream = 0);

cudaError_t launch_pack(const int *in_data, int *pack_data, int m, int k, int BIT,
                        cudaStream_t stream = 0);