#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << cudaGetErrorString(err) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

template<
    int BM,int BN,int BK,
    int TM,int TN,
    int WM,int WN,
    int WNITER
>
__global__ void mysgemm_v6(
    int M,int N,int K,
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C
){
    constexpr int WARP_SIZE = 32;

    constexpr int THREAD_NUM =
        (BM / WM)
        *
        (BN / WN)
        *
        WARP_SIZE;

    constexpr int WMITER =
        WM * WN /
        (
            WARP_SIZE
            *
            TM
            *
            TN
            *
            WNITER
        );

    constexpr int WSUBM =
        WM / WMITER;

    constexpr int WSUBN =
        WN / WNITER;

    __shared__ float As[2][BK][BM];
    __shared__ float Bs[2][BK][BN];

    int tid = threadIdx.x;

    int warp_idx =
        tid / WARP_SIZE;

    int warp_col =
        warp_idx % (BN / WN);

    int warp_row =
        warp_idx / (BN / WN);

    int lane =
        tid % WARP_SIZE;

    int thread_col =
        lane % (WSUBN / TN);

    int thread_row =
        lane / (WSUBN / TN);

    float sum[
        WMITER * TM
    ][
        WNITER * TN
    ] = {0.0f};

    for(int idx = tid * 4;
        idx < BM * BK;
        idx += THREAD_NUM * 4){

        int row = idx / BK;
        int col = idx % BK;

        float4 v =
            *reinterpret_cast<const float4*>(
                &A[
                    (blockIdx.y * BM + row) * K
                    + col
                ]
            );

        As[0][col][row] = v.x;
        As[0][col + 1][row] = v.y;
        As[0][col + 2][row] = v.z;
        As[0][col + 3][row] = v.w;
    }

    for(int idx = tid * 4;
        idx < BK * BN;
        idx += THREAD_NUM * 4){

        int row = idx / BN;
        int col = idx % BN;

        float4 v =
            *reinterpret_cast<const float4*>(
                &B[
                    row * N
                    + blockIdx.x * BN
                    + col
                ]
            );

        *reinterpret_cast<float4*>(
            &Bs[0][row][col]
        ) = v;
    }

    __syncthreads();

    int load_index = 0;
    int write_index = 1;

    for(int block_k=0;
        block_k<K;
        block_k+=BK){

        constexpr int A_REG_NUM =
            (BM * BK) / THREAD_NUM;

        constexpr int B_REG_NUM =
            (BK * BN) / THREAD_NUM;

        float ldg_a_reg[A_REG_NUM];
        float ldg_b_reg[B_REG_NUM];

        bool has_next =
            block_k + BK < K;

        if(has_next){

            int reg_index = 0;

            for(int idx = tid * 4;
                idx < BM * BK;
                idx += THREAD_NUM * 4){

                int row = idx / BK;
                int col = idx % BK;

                float4 v =
                    *reinterpret_cast<const float4*>(
                        &A[
                            (blockIdx.y * BM + row) * K
                            + block_k
                            + BK
                            + col
                        ]
                    );

                ldg_a_reg[reg_index] = v.x;
                ldg_a_reg[reg_index + 1] = v.y;
                ldg_a_reg[reg_index + 2] = v.z;
                ldg_a_reg[reg_index + 3] = v.w;

                reg_index += 4;
            }

            reg_index = 0;

            for(int idx = tid * 4;
                idx < BK * BN;
                idx += THREAD_NUM * 4){

                int row = idx / BN;
                int col = idx % BN;

                float4 v =
                    *reinterpret_cast<const float4*>(
                        &B[
                            (block_k + BK + row) * N
                            + blockIdx.x * BN
                            + col
                        ]
                    );

                ldg_b_reg[reg_index] = v.x;
                ldg_b_reg[reg_index + 1] = v.y;
                ldg_b_reg[reg_index + 2] = v.z;
                ldg_b_reg[reg_index + 3] = v.w;

                reg_index += 4;
            }
        }

        float a_frag[
            2
        ][
            WMITER * TM
        ];

        float b_frag[
            2
        ][
            WNITER * TN
        ];

        for(int wr=0;
            wr<WMITER;
            wr++){

            for(int m=0;
                m<TM;
                m+=4){

                int row =
                    warp_row * WM
                    + wr * WSUBM
                    + thread_row * TM
                    + m;

                float4 v =
                    *reinterpret_cast<float4*>(
                        &As[
                            load_index
                        ][0][row]
                    );

                a_frag[0][wr*TM+m] =
                    v.x;

                a_frag[0][wr*TM+m+1] =
                    v.y;

                a_frag[0][wr*TM+m+2] =
                    v.z;

                a_frag[0][wr*TM+m+3] =
                    v.w;
            }
        }

        for(int wc=0;
            wc<WNITER;
            wc++){

            for(int n=0;
                n<TN;
                n+=4){

                int col =
                    warp_col * WN
                    + wc * WSUBN
                    + thread_col * TN
                    + n;

                float4 v =
                    *reinterpret_cast<float4*>(
                        &Bs[
                            load_index
                        ][0][col]
                    );

                b_frag[0][wc*TN+n] =
                    v.x;

                b_frag[0][wc*TN+n+1] =
                    v.y;

                b_frag[0][wc*TN+n+2] =
                    v.z;

                b_frag[0][wc*TN+n+3] =
                    v.w;
            }
        }

        for(int ks=0;
            ks<BK-1;
            ks++){

            for(int wr=0;
                wr<WMITER;
                wr++){

                for(int m=0;
                    m<TM;
                    m+=4){

                    int row =
                        warp_row * WM
                        + wr * WSUBM
                        + thread_row * TM
                        + m;

                    float4 v =
                        *reinterpret_cast<float4*>(
                            &As[
                                load_index
                            ][ks+1][row]
                        );

                    a_frag[(ks+1)&1][wr*TM+m] =
                        v.x;

                    a_frag[(ks+1)&1][wr*TM+m+1] =
                        v.y;

                    a_frag[(ks+1)&1][wr*TM+m+2] =
                        v.z;

                    a_frag[(ks+1)&1][wr*TM+m+3] =
                        v.w;
                }
            }

            for(int wc=0;
                wc<WNITER;
                wc++){

                for(int n=0;
                    n<TN;
                    n+=4){

                    int col =
                        warp_col * WN
                        + wc * WSUBN
                        + thread_col * TN
                        + n;

                    float4 v =
                        *reinterpret_cast<float4*>(
                            &Bs[
                                load_index
                            ][ks+1][col]
                        );

                    b_frag[(ks+1)&1][wc*TN+n] =
                        v.x;

                    b_frag[(ks+1)&1][wc*TN+n+1] =
                        v.y;

                    b_frag[(ks+1)&1][wc*TN+n+2] =
                        v.z;

                    b_frag[(ks+1)&1][wc*TN+n+3] =
                        v.w;
                }
            }

            for(int wr=0;
                wr<WMITER;
                wr++){

                for(int wc=0;
                    wc<WNITER;
                    wc++){

                    for(int m=0;
                        m<TM;
                        m++){

                        for(int n=0;
                            n<TN;
                            n++){

                            sum[
                                wr*TM+m
                            ][
                                wc*TN+n
                            ]
                            +=
                            a_frag[
                                ks&1
                            ][
                                wr*TM+m
                            ]
                            *
                            b_frag[
                                ks&1
                            ][
                                wc*TN+n
                            ];
                        }
                    }
                }
            }
        }

        for(int wr=0;
            wr<WMITER;
            wr++){

            for(int wc=0;
                wc<WNITER;
                wc++){

                for(int m=0;
                    m<TM;
                    m++){

                    for(int n=0;
                        n<TN;
                        n++){

                        sum[
                            wr*TM+m
                        ][
                            wc*TN+n
                        ]
                        +=
                        a_frag[
                            (BK-1)&1
                        ][
                            wr*TM+m
                        ]
                        *
                        b_frag[
                            (BK-1)&1
                        ][
                            wc*TN+n
                        ];
                    }
                }
            }
        }

        if(has_next){

            int reg_index = 0;

            for(int idx = tid * 4;
                idx < BM * BK;
                idx += THREAD_NUM * 4){

                int row = idx / BK;
                int col = idx % BK;

                As[write_index][col][row] =
                    ldg_a_reg[reg_index];

                As[write_index][col+1][row] =
                    ldg_a_reg[reg_index+1];

                As[write_index][col+2][row] =
                    ldg_a_reg[reg_index+2];

                As[write_index][col+3][row] =
                    ldg_a_reg[reg_index+3];

                reg_index += 4;
            }

            reg_index = 0;

            for(int idx = tid * 4;
                idx < BK * BN;
                idx += THREAD_NUM * 4){

                int row = idx / BN;
                int col = idx % BN;

                float4 v;

                v.x =
                    ldg_b_reg[reg_index];

                v.y =
                    ldg_b_reg[reg_index+1];

                v.z =
                    ldg_b_reg[reg_index+2];

                v.w =
                    ldg_b_reg[reg_index+3];

                *reinterpret_cast<float4*>(
                    &Bs[
                        write_index
                    ][row][col]
                ) = v;

                reg_index += 4;
            }

            __syncthreads();

            load_index ^= 1;
            write_index ^= 1;
        }
    }

    for(int wr=0;
        wr<WMITER;
        wr++){

        for(int wc=0;
            wc<WNITER;
            wc++){

            for(int m=0;
                m<TM;
                m++){

                for(int n=0;
                    n<TN;
                    n+=4){

                    int row =
                        blockIdx.y * BM
                        + warp_row * WM
                        + wr * WSUBM
                        + thread_row * TM
                        + m;

                    int col =
                        blockIdx.x * BN
                        + warp_col * WN
                        + wc * WSUBN
                        + thread_col * TN
                        + n;

                    int sr =
                        wr * TM + m;

                    int sc =
                        wc * TN + n;

                    float4 old =
                        *reinterpret_cast<float4*>(
                            &C[row*N+col]
                        );

                    float4 out;

                    out.x =
                        alpha * sum[sr][sc]
                        + beta * old.x;

                    out.y =
                        alpha * sum[sr][sc+1]
                        + beta * old.y;

                    out.z =
                        alpha * sum[sr][sc+2]
                        + beta * old.z;

                    out.w =
                        alpha * sum[sr][sc+3]
                        + beta * old.w;

                    *reinterpret_cast<float4*>(
                        &C[row*N+col]
                    ) = out;
                }
            }
        }
    }
}

int main(){

    int M,N,K;

    std::cin >> M >> N >> K;

    if(M % 128 != 0 ||
       N % 128 != 0 ||
       K % 8 != 0){
        std::cerr << "M,N must be multiples of 128 and K multiple of 8\n";
        return 1;
    }

    float* A =
        new float[M*K];

    float* B =
        new float[K*N];

    float* C =
        new float[M*N];

    for(int i=0;i<M*K;i++)
        std::cin >> A[i];

    for(int i=0;i<K*N;i++)
        std::cin >> B[i];

    float *d_A,*d_B,*d_C;

    CUDA_CHECK(
        cudaMalloc(
            &d_A,
            sizeof(float)*M*K
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_B,
            sizeof(float)*K*N
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_C,
            sizeof(float)*M*N
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_A,A,
            sizeof(float)*M*K,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_B,B,
            sizeof(float)*K*N,
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemset(
            d_C,0,
            sizeof(float)*M*N
        )
    );

    constexpr int BM=128;
    constexpr int BN=128;
    constexpr int BK=8;

    constexpr int TM=8;
    constexpr int TN=8;

    constexpr int WM=64;
    constexpr int WN=64;

    constexpr int WNITER=1;

    constexpr int WARP_SIZE=32;

    dim3 block(
        (BM/WM)
        *
        (BN/WN)
        *
        WARP_SIZE
    );

    dim3 grid(
        N/BN,
        M/BM
    );

    mysgemm_v6<
        BM,BN,BK,
        TM,TN,
        WM,WN,
        WNITER
    ><<<grid,block>>>(
        M,N,K,
        1.0f,
        d_A,d_B,
        0.0f,
        d_C
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start,end;

    CUDA_CHECK(
        cudaEventCreate(&start)
    );

    CUDA_CHECK(
        cudaEventCreate(&end)
    );

    CUDA_CHECK(
        cudaEventRecord(start)
    );

    mysgemm_v6<
        BM,BN,BK,
        TM,TN,
        WM,WN,
        WNITER
    ><<<grid,block>>>(
        M,N,K,
        1.0f,
        d_A,d_B,
        0.0f,
        d_C
    );

    CUDA_CHECK(
        cudaEventRecord(end)
    );

    CUDA_CHECK(
        cudaEventSynchronize(end)
    );

    float ms;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &ms,
            start,
            end
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            C,d_C,
            sizeof(float)*M*N,
            cudaMemcpyDeviceToHost
        )
    );

    std::cout << ms << '\n';

    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++)
            std::cout
                << C[i*N+j]
                << ' ';

        std::cout << '\n';
    }

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}