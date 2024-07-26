
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
