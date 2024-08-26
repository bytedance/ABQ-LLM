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
#include <cuda_runtime.h>

struct CudaTimer {
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    cudaStream_t stream;
    CudaTimer(cudaStream_t stream = 0)
    {
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
        this->stream = stream;
    }

    ~CudaTimer()
    {
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }

    void start()
    {
        cudaEventRecord(startEvent, this->stream);
    }

    void stop()
    {
        cudaEventRecord(stopEvent, this->stream);
        cudaEventSynchronize(stopEvent);
    }

    float elapsed_msecs()
    {
        float elapsed;
        cudaEventElapsedTime(&elapsed, startEvent, stopEvent);
        return elapsed;
    }
};
