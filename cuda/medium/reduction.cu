#include "solve.h"
#include <cuda_runtime.h>

__global__ void reduction_kernel(const float* input, float* output, int N) {
    extern __shared__ float sharedData[];

    int tIdx = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x * 2 + tIdx;

    sharedData[tIdx] = (globalIdx < N) ? input[globalIdx] : 0;

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tIdx < s) sharedData[tIdx] += sharedData[tIdx + s];
    }

    if (tIdx == 0) atomicAdd(output, sharedData[0]);
}

// input, output are device pointers
void solve(const float* input, float* output, int N) {  
    int threads = 1 << 10;
    int blocks = (N + 2 * threads - 1) / (2 * threads);
    reduction_kernel<<<blocks, threads, threads * sizeof(float)>>>(input, output, N);
    cudaDeviceSynchronize();
}
