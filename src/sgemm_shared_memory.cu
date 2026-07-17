#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        const cudaError_t error = (call);                                    \
        if (error != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(error)        \
                      << " (" << __FILE__ << ':' << __LINE__ << ")\n";      \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (false)

template <int TILE_SIZE>
__global__ void sgemm_shared_memory(int M, int N, int K, float alpha,
                                    const float* A, const float* B, float beta,
                                    float* C) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + tx;
    const int row = blockIdx.y * TILE_SIZE + ty;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE_SIZE) {
        As[ty][tx] =
            (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.0f;
        Bs[ty][tx] =
            (col < N && k0 + ty < K) ? B[(k0 + ty) * N + col] : 0.0f;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        const int index = row * N + col;
        C[index] = alpha * sum + beta * C[index];
    }
}

int main() {
    constexpr int kTileSize = 16;

    int M = 0;
    int N = 0;
    int K = 0;
    std::cin >> M >> N >> K;
    if (!std::cin || M <= 0 || N <= 0 || K <= 0) {
        std::cerr << "M, N, and K must be positive integers.\n";
        return EXIT_FAILURE;
    }

    std::vector<float> h_A(static_cast<size_t>(M) * K);
    std::vector<float> h_B(static_cast<size_t>(K) * N);
    std::vector<float> h_C(static_cast<size_t>(M) * N);

    for (float& value : h_A) {
        std::cin >> value;
    }
    for (float& value : h_B) {
        std::cin >> value;
    }
    if (!std::cin) {
        std::cerr << "Insufficient matrix input data.\n";
        return EXIT_FAILURE;
    }

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    const size_t bytes_A = h_A.size() * sizeof(float);
    const size_t bytes_B = h_B.size() * sizeof(float);
    const size_t bytes_C = h_C.size() * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, bytes_C));

    const dim3 threads(kTileSize, kTileSize);
    const dim3 blocks((N + kTileSize - 1) / kTileSize,
                      (M + kTileSize - 1) / kTileSize);

    sgemm_shared_memory<kTileSize><<<blocks, threads>>>(
        M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(
        cudaMemcpy(h_C.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));

    for (float value : h_C) {
        std::cout << value << '\n';
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return EXIT_SUCCESS;
}
