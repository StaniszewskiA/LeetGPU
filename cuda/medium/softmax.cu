#include "solve.h"
#include <cuda_runtime.h>

__global__ void softmax_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float maxVal = input[0];
    for (int i = 1; i < N; ++i)
        if (input[i] > maxVal) maxVal = input[i];

    float expVal = expf(input[idx] - maxVal);
    output[idx] = expVal;

    __threadfence();

    float sum = 0.0f;
    for (int j = 0; j < N; ++j) sum += output[j];

    output[idx] = expVal / sum;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}