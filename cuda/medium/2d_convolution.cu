#include "solve.h"
#include <cuda_runtime.h>

__global__ void convolution_kernel_2d(
    const float* input,
    const float* kernel,
    float* output,
    int inputRows,
    int inputCols,
    int kernelRows,
    int kernelCols,
    int outputRows,
    int outputCols
) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= outputRows || j >= outputCols) return;

    float sum = 0.0f;

    for (int m = 0; m < kernelRows; ++m) {
        for (int n = 0; n < kernelCols; ++n) {
            int inputRow = i + m;
            int inputCol = j + n;
            sum += input[inputRow * inputCols + inputCol] * kernel[m * kernelCols + n];
        }
    }

    output[i * outputCols + j] = sum;
}

// input, kernel, output are device pointers
void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    int outputRows = input_rows - kernel_rows + 1;
    int outputCols = input_cols - kernel_cols + 1;

    const int blockSize = 16;
    dim3 block(blockSize, blockSize);
    dim3 grid((outputCols + block.x - 1) / block.x, (outputRows + block.y - 1) / block.y);

    convolution_kernel_2d<<<grid, block>>>(input, kernel, output, input_rows, input_cols,
                                            kernel_rows, kernel_cols, outputRows, outputCols);
}
