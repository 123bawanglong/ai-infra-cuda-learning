#include <cuda_runtime.h>

#include <iostream>

constexpr int BLOCK_SIZE = 256;

__global__ void reduceSequential(const float* input, float* output, int n) {
    __shared__ float sharedData[BLOCK_SIZE];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;

    sharedData[tid] = index < n ? input[index] : 0.0f;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride >= 1; stride /= 2) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

int main() {
    int n;
    std::cin >> n;

    if (n <= 0 || n > BLOCK_SIZE * BLOCK_SIZE) {
        std::cerr << "N must be in [1, 65536]" << std::endl;
        return 1;
    }

    float* hostInput = new float[n];
    for (int i = 0; i < n; ++i) {
        std::cin >> hostInput[i];
    }

    int blockCount = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float* deviceInput = nullptr;
    float* devicePartialSums = nullptr;
    float* deviceResult = nullptr;
    float hostResult = 0.0f;

    cudaMalloc(reinterpret_cast<void**>(&deviceInput), sizeof(float) * n);
    cudaMalloc(reinterpret_cast<void**>(&devicePartialSums),
               sizeof(float) * blockCount);
    cudaMalloc(reinterpret_cast<void**>(&deviceResult), sizeof(float));

    cudaMemcpy(deviceInput, hostInput, sizeof(float) * n,
               cudaMemcpyHostToDevice);

    cudaEvent_t begin;
    cudaEvent_t end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    cudaEventRecord(begin);

    reduceSequential<<<blockCount, BLOCK_SIZE>>>(
        deviceInput, devicePartialSums, n);
    reduceSequential<<<1, BLOCK_SIZE>>>(
        devicePartialSums, deviceResult, blockCount);

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, begin, end);
    cudaMemcpy(&hostResult, deviceResult, sizeof(float),
               cudaMemcpyDeviceToHost);

    std::cout << milliseconds << std::endl;
    std::cout << hostResult << std::endl;

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaFree(deviceInput);
    cudaFree(devicePartialSums);
    cudaFree(deviceResult);
    delete[] hostInput;

    return 0;
}
