#pragma once
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <iostream>
#include <iomanip>
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
template<int BM,int BN,int BK,int TM,int TN,int WM,int WN>
__global__ void mysgemm_v7(int M,int N,int K,float alpha,float *A,float beta,float *B,float *C){
    constexpr int THREADNUM=(BM/WM)*(BN/WN)*32;
    __shared__ __align__(16) float As[2][BM*BK];
    __shared__ __align__(16) float Bs[2][BK*BN];
    int a_row=threadIdx.x/(BK/4);
    int a_col=threadIdx.x%(BK/4)*4;
    int b_row=threadIdx.x/(BN/4);
    int b_col=threadIdx.x%(BN/4)*4;
    constexpr int a_stride=THREADNUM/(BK/4);
    constexpr int b_stride=THREADNUM/(BN/4);
    int warp_id=threadIdx.x/32;
    int lane_id=threadIdx.x%32;
    int warp_row=warp_id/(BN/WN);
    int warp_col=warp_id%(BN/WN);
    int thread_row_in_warp=lane_id/(WN/TN);
    int thread_col_in_warp=lane_id%(WN/TN);
    int thread_row=warp_row*(WM/TM)+thread_row_in_warp;
    int thread_col=warp_col*(WN/TN)+thread_col_in_warp;
    __align__(16) float sum[TM][TN]={0.f};
    __align__(16) float a_frag[2][TM];
    __align__(16) float b_frag[2][TN];
    int load_index=0;
        for(int i=0;a_row+i<BM;i+=a_stride){
            int global_row=blockIdx.y*BM+a_row+i;
            int global_col=a_col;
            size_t index=static_cast<size_t>(global_row)*K+global_col;
            if(global_row<M&&global_col+3<K&&
               (reinterpret_cast<uintptr_t>(A+index)&15)==0){
                __pipeline_memcpy_async(
                    &As[load_index][OFFSET(a_row+i,a_col,BK)],
                    &A[index],
                    16
                );
            }else{
                #pragma unroll
                for(int j=0;j<4;j++){
                    As[load_index][OFFSET(a_row+i,a_col+j,BK)]=(global_row<M&&global_col+j<K)?
                        A[static_cast<size_t>(global_row)*K+global_col+j]:0.0f;
                }
            }
        }
        for(int i=0;b_row+i<BK;i+=b_stride){
            int global_row=b_row+i;
            int global_col=blockIdx.x*BN+b_col;
            size_t index=static_cast<size_t>(global_row)*N+global_col;
            if(global_row<K&&global_col+3<N&&
               (reinterpret_cast<uintptr_t>(B+index)&15)==0){
                __pipeline_memcpy_async(
                    &Bs[load_index][OFFSET(b_row+i,b_col,BN)],
                    &B[index],
                    16
                );
            }else{
                #pragma unroll
                for(int j=0;j<4;j++){
                    Bs[load_index][OFFSET(b_row+i,b_col+j,BN)]=(global_row<K&&global_col+j<N)?
                        B[static_cast<size_t>(global_row)*N+global_col+j]:0.0f;
                }
            }
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
    for(int k=0;k<K;k+=BK){
        int write_index=load_index^1;
        if(k+BK<K){
            for(int i=0;a_row+i<BM;i+=a_stride){
                int global_row=blockIdx.y*BM+a_row+i;
                int global_col=k+BK+a_col;
                size_t index=static_cast<size_t>(global_row)*K+global_col;
                if(global_row<M&&global_col+3<K&&
                   (reinterpret_cast<uintptr_t>(A+index)&15)==0){
                    __pipeline_memcpy_async(
                        &As[write_index][OFFSET(a_row+i,a_col,BK)],
                        &A[index],
                        16
                    );
                }else{
                    #pragma unroll
                    for(int j=0;j<4;j++){
                        As[write_index][OFFSET(a_row+i,a_col+j,BK)]=(global_row<M&&global_col+j<K)?
                            A[static_cast<size_t>(global_row)*K+global_col+j]:0.0f;
                    }
                }
            }
            for(int i=0;b_row+i<BK;i+=b_stride){
                int global_row=k+BK+b_row+i;
                int global_col=blockIdx.x*BN+b_col;
                size_t index=static_cast<size_t>(global_row)*N+global_col;
                if(global_row<K&&global_col+3<N&&
                   (reinterpret_cast<uintptr_t>(B+index)&15)==0){
                    __pipeline_memcpy_async(
                        &Bs[write_index][OFFSET(b_row+i,b_col,BN)],
                        &B[index],
                        16
                    );
                }else{
                    #pragma unroll
                    for(int j=0;j<4;j++){
                        Bs[write_index][OFFSET(b_row+i,b_col+j,BN)]=(global_row<K&&global_col+j<N)?
                            B[static_cast<size_t>(global_row)*N+global_col+j]:0.0f;
                    }
                }
            }
            __pipeline_commit();
        }
        #pragma unroll
        for(int m=0;m<TM;m++){
            a_frag[0][m]=As[load_index][OFFSET(
                thread_row*TM+m,
                0,
                BK
            )];
        }
        #pragma unroll
        for(int n=0;n<TN;n+=4){
            FETCH_FLOAT4(b_frag[0][n])=
            FETCH_FLOAT4(Bs[load_index][OFFSET(
                0,
                thread_col*TN+n,
                BN
            )]);
        }
        #pragma unroll
        for(int bk=0;bk<BK;bk++){
            if(bk+1<BK){
                #pragma unroll
                for(int m=0;m<TM;m++){
                    a_frag[(bk+1)%2][m]=As[load_index][OFFSET(
                        thread_row*TM+m,
                        bk+1,
                        BK
                    )];
                }
                #pragma unroll
                for(int n=0;n<TN;n+=4){
                    FETCH_FLOAT4(b_frag[(bk+1)%2][n])=
                    FETCH_FLOAT4(Bs[load_index][OFFSET(
                        bk+1,
                        thread_col*TN+n,
                        BN
                    )]);
                }
            }
            #pragma unroll
            for(int l=0;l<TM;l++){
                #pragma unroll
                for(int j=0;j<TN;j++){
                    sum[l][j]+=a_frag[bk%2][l]*b_frag[bk%2][j];
                }
            }
        }
        if(k+BK<K){
            __pipeline_wait_prior(0);
        }
        __syncthreads();
        load_index=write_index;
    }
    #pragma unroll
    for(int l=0;l<TM;l++){
        int row=blockIdx.y*BM+thread_row*TM+l;
        #pragma unroll
        for(int j=0;j<TN;j+=4){
            int col=blockIdx.x*BN+thread_col*TN+j;
            size_t index=static_cast<size_t>(row)*N+col;
            if(row<M&&col+3<N&&
               (reinterpret_cast<uintptr_t>(C+index)&15)==0){
                float4 mid;
                if(beta==0.0f){
                    mid.x=alpha*sum[l][j];
                    mid.y=alpha*sum[l][j+1];
                    mid.z=alpha*sum[l][j+2];
                    mid.w=alpha*sum[l][j+3];
                }else{
                    mid=FETCH_FLOAT4(C[index]);
                    mid.x=alpha*sum[l][j]+beta*mid.x;
                    mid.y=alpha*sum[l][j+1]+beta*mid.y;
                    mid.z=alpha*sum[l][j+2]+beta*mid.z;
                    mid.w=alpha*sum[l][j+3]+beta*mid.w;
                }
                FETCH_FLOAT4(C[index])=mid;
            }else{
                #pragma unroll
                for(int n=0;n<4;n++){
                    if(row<M&&col+n<N){
                        size_t scalar_index=static_cast<size_t>(row)*N+col+n;
                        C[scalar_index]=(beta==0.0f)?alpha*sum[l][j+n]:
                            alpha*sum[l][j+n]+beta*C[scalar_index];
                    }
                }
            }
        }
    }
}
int main()
{
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    constexpr int BM=32;
    constexpr int BN=128;
    constexpr int BK=32;
    constexpr int TM=4;
    constexpr int TN=4;
    constexpr int WM=16;
    constexpr int WN=32;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int warmup = 10;
    const int repeat = 100;
    size_t bytesA = static_cast<size_t>(M) * K * sizeof(float);
    size_t bytesB = static_cast<size_t>(K) * N * sizeof(float);
    size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);
    std::vector<float> hA(static_cast<size_t>(M) * K, 1.0f);
    std::vector<float> hB(static_cast<size_t>(K) * N, 1.0f);
    float *A = nullptr;
    float *B = nullptr;
    float *C = nullptr;
    cudaMalloc(&A, bytesA);
    cudaMalloc(&B, bytesB);
    cudaMalloc(&C, bytesC);
    cudaMemcpy(A, hA.data(), bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(B, hB.data(), bytesB, cudaMemcpyHostToDevice);
    dim3 block((BM/WM)*(BN/WN)*32);
    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );
    for (int i = 0; i < warmup; ++i) {
        mysgemm_v7<BM,BN,BK,TM,TN,WM,WN><<<grid, block>>>(
            M, N, K,
            alpha, A,
            beta, B, C
        );
    }
    cudaDeviceSynchronize();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; ++i) {
        mysgemm_v7<BM,BN,BK,TM,TN,WM,WN><<<grid, block>>>(
            M, N, K,
            alpha, A,
            beta, B, C
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / repeat;
    double flops =
        2.0 * static_cast<double>(M) *
        static_cast<double>(N) *
        static_cast<double>(K);
    double tflops =
        flops / (avg_ms * 1e9);
    float result = 0.0f;
    cudaMemcpy(
        &result,
        C,
        sizeof(float),
        cudaMemcpyDeviceToHost
    );
    std::cout << "M=N=K=" << M << '\n';
    std::cout << "block=(" << block.x << "," << block.y << ")\n";
    std::cout << "grid=(" << grid.x << "," << grid.y << ")\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "average kernel time = " << avg_ms << " ms\n";
    std::cout << "performance = " << tflops << " TFLOP/s\n";
    std::cout << std::setprecision(1);
    std::cout << "C[0] = " << result << ", expected = " << static_cast<float>(K) << '\n';
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}




