#include "solve.h"
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

__global__ void bitonic_sort_step(float* data, int N, int j, int k) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int l = i ^ j;

    if (l > i && i < N && l < N) {
        if ((i & k) == 0) {
            if (data[i] > data[l]) {
                float tmp = data[i];
                data[i] = data[l];
                data[l] = tmp;
            }
        } else {
            if (data[i] < data[l]) {
                float tmp = data[i];
                data[i] = data[l];
                data[l] = tmp;
            }
        }
    }
}

__device__ __host__ int next_power_of_two(int n) {
    int power = 1;
    while (power < n) power <<= 1;
    return power;
}

// data is device pointer
void solve(float* data, int N) {
    if (N <= 1) return;

    int paddedN = next_power_of_two(N);
    float* sortData = NULL;

    if (paddedN > N) {
        cudaMalloc(&sortData, paddedN * sizeof(float));
        cudaMemcpy(sortData, data, N * sizeof(float), cudaMemcpyDeviceToDevice);

        float* hostPadding = (float*)malloc((paddedN - N) * sizeof(float));
        for (int i = 0; i < paddedN - N; i++) hostPadding[i] = FLT_MAX;

        cudaMemcpy(&sortData[N], hostPadding, (paddedN - N) * sizeof(float), cudaMemcpyHostToDevice);
        free(hostPadding);

    } else sortData = data;

    const int blockSize = 256;
    int gridSize = (paddedN + blockSize - 1) / blockSize;

    for (int k = 2; k <= paddedN; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_sort_step<<<gridSize, blockSize>>>(sortData, paddedN, j, k);
            cudaDeviceSynchronize();
        }
    }

    if (paddedN > N) {
        cudaMemcpy(data, sortData, N * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaFree(sortData);
    }
}
