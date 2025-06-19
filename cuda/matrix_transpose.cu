#include "solve.h"
#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        output[x * rows + y] = input[y * cols + x];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* input, float* output, int rows, int cols) {
    float *dInput, *dOutput;

    // Allocate
    cudaMalloc(&dInput, rows * cols * sizeof(float));
    cudaMalloc(&dOutput, rows * cols * sizeof(float));

    // Copy mem to device
    cudaMemcpy(dInput, input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(dInput, dOutput, rows, cols);
    cudaDeviceSynchronize();

    // Copy res
    cudaMemcpy(output, dOutput, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dInput);
    cudaFree(dOutput);
}