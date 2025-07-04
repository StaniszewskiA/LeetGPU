#include "solve.h"
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(
    const float* input, 
    const float* kernel, 
    float* output,
    int input_size, 
    int kernel_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = input_size - kernel_size + 1;

    if (idx >= output_size) return;

    float res = 0.0f;

    for (int j = 0; j < kernel_size; j++) {
        res += input[idx + j] * kernel[j];
    }

    output[idx] = res;      
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
void solve(
    const float* input, 
    const float* kernel, 
    float* output, 
    int input_size, 
    int kernel_size
) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}