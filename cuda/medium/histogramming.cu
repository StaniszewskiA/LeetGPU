#include "solve.h"
#include <cuda_runtime.h>

__global__ void histogram_kernel(const int* input, int* histogram, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int val = input[idx];
    atomicAdd(&histogram[val], 1);
}

// input, histogram are device pointers
void solve(const int* input, int* histogram, int N, int num_bins) {
    cudaMemset(histogram, 0, num_bins * sizeof(int));

    const int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    histogram_kernel<<<numBlocks, blockSize>>>(input, histogram, n);

    cudaDeviceSynchronize();
}
