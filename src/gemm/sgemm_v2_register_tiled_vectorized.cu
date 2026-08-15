#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstdint>

#define CUDA_CHECK(call)                                                   \
do {                                                                       \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)             \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)

#define LOAD_FLOAT4(ptr) \
    (*reinterpret_cast<const float4 *>(ptr))

#define STORE_FLOAT4(ptr, value) \
    (*reinterpret_cast<float4 *>(ptr) = (value))

template<int BM, int BN, int BK, int TM, int TN>
__global__ void mysgemm_v4(
    int M,
    int N,
    int K,
    float alpha,
    const float *A,
    const float *B,
    float beta,
    float *C)
{
    static_assert(BM % TM == 0);
    static_assert(BN % TN == 0);
    static_assert(BK % 4 == 0);
    static_assert(BN % 4 == 0);
    static_assert(TM % 4 == 0);
    static_assert(TN % 4 == 0);

    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.x;
    const int thread_num = blockDim.x;

    const int thread_row = tid / (BN / TN);
    const int thread_col = tid % (BN / TN);

    const int ty = thread_row * TM;
    const int tx = thread_col * TN;

    float sum[TM][TN] = {0.0f};

    for (int k0 = 0; k0 < K; k0 += BK) {

        for (int index = tid * 4;
             index < BM * BK;
             index += thread_num * 4) {

            const int a_row = index / BK;
            const int a_col = index % BK;

            const int global_row =
                blockIdx.y * BM + a_row;

            const int global_col =
                k0 + a_col;

            float4 value =
                make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            if (global_row < M) {

                const int base =
                    global_row * K + global_col;

                if (global_col + 3 < K &&
                    (base & 3) == 0) {

                    value =
                        LOAD_FLOAT4(A + base);

                } else {

                    if (global_col < K)
                        value.x =
                            A[global_row * K +
                              global_col];

                    if (global_col + 1 < K)
                        value.y =
                            A[global_row * K +
                              global_col + 1];

                    if (global_col + 2 < K)
                        value.z =
                            A[global_row * K +
                              global_col + 2];

                    if (global_col + 3 < K)
                        value.w =
                            A[global_row * K +
                              global_col + 3];
                }
            }

            As[a_col][a_row] =
                value.x;

            As[a_col + 1][a_row] =
                value.y;

            As[a_col + 2][a_row] =
                value.z;

            As[a_col + 3][a_row] =
                value.w;
        }

        for (int index = tid * 4;
             index < BK * BN;
             index += thread_num * 4) {

            const int b_row =
                index / BN;

            const int b_col =
                index % BN;

            const int global_row =
                k0 + b_row;

            const int global_col =
                blockIdx.x * BN + b_col;

            float4 value =
                make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            if (global_row < K) {

                const int base =
                    global_row * N +
                    global_col;

                if (global_col + 3 < N &&
                    (base & 3) == 0) {

                    value =
                        LOAD_FLOAT4(B + base);

                } else {

                    if (global_col < N)
                        value.x =
                            B[global_row * N +
                              global_col];

                    if (global_col + 1 < N)
                        value.y =
                            B[global_row * N +
                              global_col + 1];

                    if (global_col + 2 < N)
                        value.z =
                            B[global_row * N +
                              global_col + 2];

                    if (global_col + 3 < N)
                        value.w =
                            B[global_row * N +
                              global_col + 3];
                }
            }

            STORE_FLOAT4(
                &Bs[b_row][b_col],
                value);
        }

        __syncthreads();

        for (int ks = 0;
             ks < BK;
             ++ks) {

            float a_frag[TM];
            float b_frag[TN];

            for (int m = 0;
                 m < TM;
                 m += 4) {

                float4 v =
                    LOAD_FLOAT4(
                        &As[ks][ty + m]);

                a_frag[m] =
                    v.x;

                a_frag[m + 1] =
                    v.y;

                a_frag[m + 2] =
                    v.z;

                a_frag[m + 3] =
                    v.w;
            }

            for (int n = 0;
                 n < TN;
                 n += 4) {

                float4 v =
                    LOAD_FLOAT4(
                        &Bs[ks][tx + n]);

                b_frag[n] =
                    v.x;

                b_frag[n + 1] =
                    v.y;

                b_frag[n + 2] =
                    v.z;

                b_frag[n + 3] =
                    v.w;
            }

            #pragma unroll
            for (int m = 0;
                 m < TM;
                 ++m) {

                #pragma unroll
                for (int n = 0;
                     n < TN;
                     ++n) {

                    sum[m][n] +=
                        a_frag[m] *
                        b_frag[n];
                }
            }
        }

        __syncthreads();
    }

    for (int m = 0;
         m < TM;
         ++m) {

        const int row =
            blockIdx.y * BM +
            ty + m;

        if (row >= M)
            continue;

        for (int n = 0;
             n < TN;
             n += 4) {

            const int col =
                blockIdx.x * BN +
                tx + n;

            const int base =
                row * N + col;

            if (col + 3 < N &&
                (base & 3) == 0) {

                float4 old_value =
                    make_float4(
                        0.0f,
                        0.0f,
                        0.0f,
                        0.0f);

                if (beta != 0.0f)
                    old_value =
                        LOAD_FLOAT4(C + base);

                float4 out;

                out.x =
                    alpha * sum[m][n]
                    + beta * old_value.x;

                out.y =
                    alpha * sum[m][n + 1]
                    + beta * old_value.y;

                out.z =
                    alpha * sum[m][n + 2]
                    + beta * old_value.z;

                out.w =
                    alpha * sum[m][n + 3]
                    + beta * old_value.w;

                STORE_FLOAT4(
                    C + base,
                    out);

            } else {

                for (int t = 0;
                     t < 4;
                     ++t) {

                    if (col + t < N) {

                        const int idx =
                            row * N +
                            col + t;

                        const float old_value =
                            beta == 0.0f
                            ? 0.0f
                            : C[idx];

                        C[idx] =
                            alpha *
                            sum[m][n + t]
                            +
                            beta *
                            old_value;
                    }
                }
            }
        }
    }
}

int main()
{
    int M;
    int N;
    int K;

    std::cin >> M >> N >> K;

    float *A =
        new float[M * K];

    float *B =
        new float[K * N];

    float *C =
        new float[M * N];

    for (int i = 0;
         i < M * K;
         ++i)
        std::cin >> A[i];

    for (int i = 0;
         i < K * N;
         ++i)
        std::cin >> B[i];

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_A,
            sizeof(float) * M * K));

    CUDA_CHECK(
        cudaMalloc(
            &d_B,
            sizeof(float) * K * N));

    CUDA_CHECK(
        cudaMalloc(
            &d_C,
            sizeof(float) * M * N));

    CUDA_CHECK(
        cudaMemcpy(
            d_A,
            A,
            sizeof(float) * M * K,
            cudaMemcpyHostToDevice));

    CUDA_CHECK(
        cudaMemcpy(
            d_B,
            B,
            sizeof(float) * K * N,
            cudaMemcpyHostToDevice));

    CUDA_CHECK(
        cudaMemset(
            d_C,
            0,
            sizeof(float) * M * N));

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    constexpr int THREADS =
        (BM / TM) *
        (BN / TN);

    dim3 block(
        THREADS);

    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM);

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(
        cudaEventCreate(&start));

    CUDA_CHECK(
        cudaEventCreate(&stop));

    CUDA_CHECK(
        cudaEventRecord(start));

    mysgemm_v4<
        BM,
        BN,
        BK,
        TM,
        TN
    ><<<grid, block>>>(
        M,
        N,
        K,
        1.0f,
        d_A,
        d_B,
        0.0f,
        d_C);

    CUDA_CHECK(
        cudaGetLastError());

    CUDA_CHECK(
        cudaEventRecord(stop));

    CUDA_CHECK(
        cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &elapsed_ms,
            start,
            stop));

    CUDA_CHECK(
        cudaMemcpy(
            C,
            d_C,
            sizeof(float) * M * N,
            cudaMemcpyDeviceToHost));

    std::cout
        << elapsed_ms
        << '\n';

    for (int i = 0;
         i < M;
         ++i) {

        for (int j = 0;
             j < N;
             ++j) {

            std::cout
                << C[i * N + j];

            if (j + 1 < N)
                std::cout << ' ';
        }

        std::cout << '\n';
    }

    CUDA_CHECK(
        cudaEventDestroy(start));

    CUDA_CHECK(
        cudaEventDestroy(stop));

    CUDA_CHECK(
        cudaFree(d_A));

    CUDA_CHECK(
        cudaFree(d_B));

    CUDA_CHECK(
        cudaFree(d_C));

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}