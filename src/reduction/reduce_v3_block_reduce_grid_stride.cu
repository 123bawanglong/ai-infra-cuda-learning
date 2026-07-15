#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
constexpr unsigned int FULL_MASK = 0xffffffffu;

__device__ __forceinline__ float warpReduceSum(float value) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(FULL_MASK, value, offset);
    }
    return value;
}

__device__ __forceinline__ float blockReduceSum(float value) {
    __shared__ float warpSums[WARPS_PER_BLOCK];

    int tid = threadIdx.x;
    int laneId = tid % WARP_SIZE;
    int warpId = tid / WARP_SIZE;

    value = warpReduceSum(value);

    if (laneId == 0) {
        warpSums[warpId] = value;
    }
    __syncthreads();

    if (warpId == 0) {
        value = laneId < WARPS_PER_BLOCK ? warpSums[laneId] : 0.0f;
        value = warpReduceSum(value);
    }

    return value;
}

__global__ void reduceBlockGridStride(const float* input, float* output,
                                      int n) {
    float localSum = 0.0f;

    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = gridDim.x * blockDim.x;

    for (int index = globalThreadId; index < n; index += gridStride) {
        localSum += input[index];
    }

    float blockSum = blockReduceSum(localSum);

    if (threadIdx.x == 0) {
        output[blockIdx.x] = blockSum;
    }
}

int main() {
    int n;
    std::cin >> n;

    if (n <= 0) {
        std::cerr << "N must be positive" << std::endl;
        return 1;
    }

    float* hostInput = new float[n];
    for (int i = 0; i < n; ++i) {
        std::cin >> hostInput[i];
    }

    int requiredBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blockCount = std::min(requiredBlocks, BLOCK_SIZE);

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

    reduceBlockGridStride<<<blockCount, BLOCK_SIZE>>>(
        deviceInput, devicePartialSums, n);
    reduceBlockGridStride<<<1, BLOCK_SIZE>>>(
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
