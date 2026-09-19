#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <iostream>
#include <iomanip>
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
template<int BM,int BN,int BK,int TM,int TN>
__global__ void mysgemm_v4(int M,int N,int K,float alpha,float *A,float beta,float *B,float *C){
    constexpr int THREADNUM=(BM/TM)*(BN/TN);
    __shared__ __align__(16) float As[BK*BM];
    __shared__ __align__(16) float Bs[BK*BN];
    int a_row=threadIdx.x/(BK/4);
    int a_col=threadIdx.x%(BK/4)*4;
    int b_row=threadIdx.x/(BN/4);
    int b_col=threadIdx.x%(BN/4)*4;
    constexpr int a_stride=THREADNUM/(BK/4);
    constexpr int b_stride=THREADNUM/(BN/4);
    int thread_row=threadIdx.x/(BN/TN);
    int thread_col=threadIdx.x%(BN/TN);
    __align__(16) float sum[TM][TN]={0.f};
    __align__(16) float a_frag[TM];
    __align__(16) float b_frag[TN];
    __align__(16) float ldg_a_reg[4];
    for(int k=0;k<K;k+=BK){
        for(int i=0;a_row+i<BM;i+=a_stride){
            int global_row=blockIdx.y*BM+a_row+i;
            int global_col=k+a_col;
            size_t index=static_cast<size_t>(global_row)*K+global_col;
            if(global_row<M&&global_col+3<K&&
               (reinterpret_cast<uintptr_t>(A+index)&15)==0){
                FETCH_FLOAT4(ldg_a_reg[0])=FETCH_FLOAT4(A[index]);
            }else{
                #pragma unroll
                for(int j=0;j<4;j++){
                    ldg_a_reg[j]=(global_row<M&&global_col+j<K)?
                        A[static_cast<size_t>(global_row)*K+global_col+j]:0.0f;
                }
            }
            As[OFFSET(a_col,a_row+i,BM)]=ldg_a_reg[0];
            As[OFFSET(a_col+1,a_row+i,BM)]=ldg_a_reg[1];
            As[OFFSET(a_col+2,a_row+i,BM)]=ldg_a_reg[2];
            As[OFFSET(a_col+3,a_row+i,BM)]=ldg_a_reg[3];
        }
        for(int i=0;b_row+i<BK;i+=b_stride){
            int global_row=k+b_row+i;
            int global_col=blockIdx.x*BN+b_col;
            size_t index=static_cast<size_t>(global_row)*N+global_col;
            if(global_row<K&&global_col+3<N&&
               (reinterpret_cast<uintptr_t>(B+index)&15)==0){
                FETCH_FLOAT4(Bs[OFFSET(b_row+i,b_col,BN)])=
                FETCH_FLOAT4(B[index]);
            }else{
                #pragma unroll
                for(int j=0;j<4;j++){
                    Bs[OFFSET(b_row+i,b_col+j,BN)]=(global_row<K&&global_col+j<N)?
                        B[static_cast<size_t>(global_row)*N+global_col+j]:0.0f;
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for(int bk=0;bk<BK;bk++){
            #pragma unroll
            for(int m=0;m<TM;m+=4){
                FETCH_FLOAT4(a_frag[m])=
                FETCH_FLOAT4(As[OFFSET(
                    bk,
                    thread_row*TM+m,
                    BM
                )]);
            }
            #pragma unroll
            for(int n=0;n<TN;n+=4){
                FETCH_FLOAT4(b_frag[n])=
                FETCH_FLOAT4(Bs[OFFSET(
                    bk,
                    thread_col*TN+n,
                    BN
                )]);
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
    dim3 block((BM/TM)*(BN/TN));
    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );
    for (int i = 0; i < warmup; ++i) {
        mysgemm_v4<BM,BN,BK,TM,TN><<<grid, block>>>(
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
        mysgemm_v4<BM,BN,BK,TM,TN><<<grid, block>>>(
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



