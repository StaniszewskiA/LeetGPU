#include "solve.h"
#include <cuda_runtime.h>

__global__ void qkt_kernel(
    const float* Q,
    const float* K,
    float* attnScores,
    int M,
    int N,
    int d
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= M || j >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < d; ++k) sum += Q[i * d + k] * K[j * d + k];
    sum /= sqrtf((float)d);
    attnScores[i * N + j] = sum;
}

__global__ void softmax_kernel(
    const float* attnScores,
    float* attnWeight,
    int M,
    int N
) {
    int row = blockIdx.x;
    if (row >= M) return;

    int tIdx = threadIdx.x;
    int numThreads = blockDim.x;

    float localMax = -INFINITY;
    for (int j = tIdx; j < N; j += numThreads) 
        localMax = fmaxf(localMax, attnScores[row * N + j]);

    __shared__ float sharedMax[256];
    sharedMax[tIdx] = localMax;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tIdx < s) 
            sharedMax[tIdx] = fmaxf(sharedMax[tIdx], sharedMax[tIdx + s]);
        __syncthreads();
    }

    float rowMax = sharedMax[0];

    float localSum = 0.0f;
    for (int j = tIdx; j < N; j += numThreads) {
        float expVal = expf(attnScores[row * N + j] - rowMax);
        attnWeight[row * N + j] = expVal;
        localSum += expVal;
    }

    __shared__ float sharedSum[256];
    sharedSum[tIdx] = localSum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tIdx < s)
            sharedSum[tIdx] += sharedSum[tIdx + s];
        __syncthreads();
    }

    float rowSum = sharedSum[0];

    for (int j = tIdx; j < N; j += numThreads)
        attnWeight[row * N + j] /= rowSum;
}

__global__ void sum_kernel(
    const float* attnWeights,
    const float* V,
    float* output,
    int M,
    int N,
    int d
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= M || j >= d) return;

    float sum = 0.0f;
    for (int k = 0; k < N; ++k) sum += attnWeights[i * N + k] * V[k * d + j];
    output[i * d + j] = sum;
}

// Q, K, V, output are device pointers
void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    float* attnScores;
    float* attnWeights;

    cudaMalloc(&attnScores, M * N * sizeof(float));
    cudaMalloc(&attnWeights, M * N * sizeof(float));

    const int blockSize = 16;

    // Q * K^T
    dim3 qktBlock(blockSize, blockSize);
    dim3 qktGrid((M + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);
    qkt_kernel<<<qktGrid, qktBlock>>>(Q, K, attnScores, M, N, d);
    cudaDeviceSynchronize();

    // softmax
    const int softmaxThreads = 256;
    softmax_kernel<<<M, softmaxThreads>>>(attnScores, attnWeights, M, N);
    cudaDeviceSynchronize();

    // attnWeights * V
    dim3 sumBlock(blockSize, blockSize);
    dim3 sumGrid((M + 15) / 16, (d + 15) / 16);
    sum_kernel<<<sumGrid, sumBlock>>>(attnWeights, V, output, M, N, d);
    cudaDeviceSynchronize();

    cudaFree(attnScores);
    cudaFree(attnWeights);
}
