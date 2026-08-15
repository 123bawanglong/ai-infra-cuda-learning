#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA error: "                           \
                  << cudaGetErrorString(err)                   \
                  << std::endl;                               \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while (0)

constexpr int HIST_SIZE = 256;

__global__ void histogram_kernel(
    const int *data,
    int *hist,
    int N)
{
    __shared__ int private_hist[HIST_SIZE];

    for (int i = threadIdx.x;
         i < HIST_SIZE;
         i += blockDim.x)
    {
        private_hist[i] = 0;
    }

    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tid;
         i < N;
         i += gridDim.x * blockDim.x)
    {
        int value = data[i];

        if (value >= 0 && value < HIST_SIZE)
        {
            atomicAdd(&private_hist[value], 1);
        }
    }

    __syncthreads();

    for (int i = threadIdx.x;
         i < HIST_SIZE;
         i += blockDim.x)
    {
        atomicAdd(&hist[i], private_hist[i]);
    }
}

int main()
{
    int N;
    std::cin >> N;

    if (N <= 0)
    {
        return EXIT_FAILURE;
    }

    int *h_data = new int[N];
    int h_hist[HIST_SIZE] = {0};

    for (int i = 0; i < N; ++i)
    {
        std::cin >> h_data[i];
    }

    int *d_data = nullptr;
    int *d_hist = nullptr;

    CUDA_CHECK(cudaMalloc(&d_data, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_hist, sizeof(int) * HIST_SIZE));

    CUDA_CHECK(
        cudaMemset(
            d_hist,
            0,
            sizeof(int) * HIST_SIZE
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_data,
            h_data,
            sizeof(int) * N,
            cudaMemcpyHostToDevice
        )
    );

    constexpr int BLOCK_SIZE = 256;

    int grid_size =
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    if (grid_size > 1024)
    {
        grid_size = 1024;
    }

    histogram_kernel<<<grid_size, BLOCK_SIZE>>>(
        d_data,
        d_hist,
        N
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMemcpy(
            h_hist,
            d_hist,
            sizeof(int) * HIST_SIZE,
            cudaMemcpyDeviceToHost
        )
    );

    for (int i = 0; i < HIST_SIZE; ++i)
    {
        std::cout
            << i
            << " : "
            << h_hist[i]
            << std::endl;
    }

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_hist));

    delete[] h_data;

    return 0;
}