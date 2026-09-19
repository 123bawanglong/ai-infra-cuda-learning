#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <iostream>
#include <iomanip>
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
template<int BM,int BN,int BK,int TM,int TN>
__global__ void mysgemm_v3(int M,int N,int K,float alpha,float *A,float beta,float *B,float *C){
    __shared__ float As[BM*BK];
    __shared__ float Bs[BK*BN];
    constexpr int THREAD_X=BN/TN;
    constexpr int THREAD_Y=BM/TM;
    constexpr int THREADNUM=THREAD_X*THREAD_Y;
    int thread_row=threadIdx.y;
    int thread_col=threadIdx.x;
    int tid=thread_row*THREAD_X+thread_col;
    float sum[TM][TN]={0.f};
    float a_frag[TM];
    float b_frag[TN];
    for(int bk=0;bk<K;bk+=BK){
        for(int i=tid;i<BM*BK;i+=THREADNUM){
            int a_row=i/BK;
            int a_col=i%BK;
            int global_row=blockIdx.y*BM+a_row;
            int global_col=bk+a_col;
            if(global_row<M&&global_col<K){
                As[OFFSET(a_row,a_col,BK)]=A[OFFSET(global_row,global_col,K)];
            }else{
                As[OFFSET(a_row,a_col,BK)]=0.0f;
            }
        }
        for(int i=tid;i<BK*BN;i+=THREADNUM){
            int b_row=i/BN;
            int b_col=i%BN;
            int global_row=bk+b_row;
            int global_col=blockIdx.x*BN+b_col;
            if(global_row<K&&global_col<N){
                Bs[OFFSET(b_row,b_col,BN)]=B[OFFSET(global_row,global_col,N)];
            }else{
                Bs[OFFSET(b_row,b_col,BN)]=0.0f;
            }
        }
        __syncthreads();
        #pragma unroll
        for(int k=0;k<BK;k++){
            #pragma unroll
            for(int l=0;l<TM;l++){
                a_frag[l]=As[OFFSET(
                    thread_row+l*THREAD_Y,
                    k,
                    BK
                )];
            }
            #pragma unroll
            for(int j=0;j<TN;j++){
                b_frag[j]=Bs[OFFSET(
                    k,
                    thread_col+j*THREAD_X,
                    BN
                )];
            }
            #pragma unroll
            for(int l=0;l<TM;l++){
                #pragma unroll
                for(int j=0;j<TN;j++){
                    sum[l][j]+=a_frag[l]*b_frag[j];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int l=0;l<TM;l++){
        int row=blockIdx.y*BM+thread_row+l*THREAD_Y;
        #pragma unroll
        for(int j=0;j<TN;j++){
            int col=blockIdx.x*BN+thread_col+j*THREAD_X;
            if(row<M&&col<N){
                int index=OFFSET(row,col,N);
                C[index]=(beta==0.0f)?alpha*sum[l][j]:alpha*sum[l][j]+beta*C[index];
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
    dim3 block(BN/TN, BM/TM);
    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );
    for (int i = 0; i < warmup; ++i) {
        mysgemm_v3<BM,BN,BK,TM,TN><<<grid, block>>>(
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
        mysgemm_v3<BM,BN,BK,TM,TN><<<grid, block>>>(
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
