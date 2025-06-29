#include "solve.h"
#include <cuda_runtime.h>

__global__ void sum_kernel(const float* y, float* totalSum, int n) {
    extern __shared__ float sharedData[];
    int tIdx = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + tIdx;
    int stride = gridDim.x * blockDim.x;
    float threadSum = 0.0f;

    for (int idx = globalIdx; idx < n; idx += stride) threadSum += y[idx];

    sharedData[tIdx] = threadSum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tIdx >= s) continue;
        sharedData[tIdx] += sharedData[tIdx + s];
        __syncthreads();
    }

    if (tIdx == 0) atomicAdd(totalSum, sharedData[0]);
}

__global__ void res_kernel(
    float* res, 
    const float* totalSum, 
    const float interval, 
    int nSamples
) {
    float avg = *totalSum / nSamples;
    *res = interval * avg;
}

// y_samples, result are device pointers
void solve(const float* y_samples, float* result, float a, float b, int n_samples) {
    const float interval = b - a;
    float* dSum;

    cudaMalloc(&dSum, sizeof(float));
    cudaMemset(dSum, 0, sizeof(float));

    const int blockSize = 256;
    int gridSize = (n_samples + blockSize - 1) / blockSize;

    // Limit the number of blocks
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (gridSize > prop.maxGridSize[0]) gridSize = prop.maxGridSize[0];

    size_t sharedMemSize = blockSize * sizeof(float);
    
    sum_kernel<<<gridSize, blockSize, sharedMemSize>>>(y_samples, dSum, n_samples);
    cudaDeviceSynchronize();

    res_kernel<<<1, 1>>>(result, dSum, interval, n_samples);
    cudaDeviceSynchronize();

    cudaFree(dSum);
}
