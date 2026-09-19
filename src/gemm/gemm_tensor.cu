#pragma once
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <mma.h>
using namespace nvcuda;
#include <cstdio>
#include <cstdint>
#include <vector>
#include <iostream>
#include <iomanip>
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
template<int BM,int BN,int BK>
__global__ void mysgemm_v6_tensor(int M,int N,int K,float alpha,float *A,float beta,float *B,float *C){
    constexpr int THREADNUM=(BM/16)*(BN/16)*32;
    static_assert(BM%16==0&&BN%16==0&&BK%8==0,"WMMA tile sizes");
    static_assert(THREADNUM<=1024&&THREADNUM%(BK/4)==0&&THREADNUM%(BN/4)==0,"copy thread mapping");
    __shared__ __align__(32) float As[2][BM*BK];
    __shared__ __align__(32) float Bs[2][BK*BN];
    int a_row=threadIdx.x/(BK/4);
    int a_col=threadIdx.x%(BK/4)*4;
    int b_row=threadIdx.x/(BN/4);
    int b_col=threadIdx.x%(BN/4)*4;
    constexpr int a_stride=THREADNUM/(BK/4);
    constexpr int b_stride=THREADNUM/(BN/4);
    __shared__ __align__(32) float Cs[BM*BN];
    int warp_id=threadIdx.x/32;
    int warp_row=warp_id/(BN/16);
    int warp_col=warp_id%(BN/16);
    wmma::fragment<wmma::matrix_a,16,16,8,wmma::precision::tf32,wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b,16,16,8,wmma::precision::tf32,wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator,16,16,8,float> sum;
    wmma::fill_fragment(sum,0.0f);
    int load_index=0;
        for(int i=0;a_row+i<BM;i+=a_stride){
            int global_row=blockIdx.y*BM+a_row+i;
            int global_col=a_col;
            size_t index=static_cast<size_t>(global_row)*K+global_col;
            if(global_row<M&&global_col+3<K&&
               (index&3)==0){
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
               (index&3)==0){
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
                   (index&3)==0){
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
                   (index&3)==0){
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
        wmma::load_matrix_sync(a_frag[0],&As[load_index][OFFSET(warp_row*16,0,BK)],BK);
        wmma::load_matrix_sync(b_frag[0],&Bs[load_index][OFFSET(0,warp_col*16,BN)],BN);
        #pragma unroll
        for(int i=0;i<a_frag[0].num_elements;i++){
            a_frag[0].x[i]=wmma::__float_to_tf32(a_frag[0].x[i]);
        }
        #pragma unroll
        for(int i=0;i<b_frag[0].num_elements;i++){
            b_frag[0].x[i]=wmma::__float_to_tf32(b_frag[0].x[i]);
        }
        #pragma unroll
        for(int bk=0;bk<BK;bk+=8){
            int frag_index=(bk/8)%2;
            int next_index=frag_index^1;
            if(bk+8<BK){
                wmma::load_matrix_sync(a_frag[next_index],
                    &As[load_index][OFFSET(warp_row*16,bk+8,BK)],BK);
                wmma::load_matrix_sync(b_frag[next_index],
                    &Bs[load_index][OFFSET(bk+8,warp_col*16,BN)],BN);
                #pragma unroll
                for(int i=0;i<a_frag[next_index].num_elements;i++){
                    a_frag[next_index].x[i]=wmma::__float_to_tf32(a_frag[next_index].x[i]);
                }
                #pragma unroll
                for(int i=0;i<b_frag[next_index].num_elements;i++){
                    b_frag[next_index].x[i]=wmma::__float_to_tf32(b_frag[next_index].x[i]);
                }
            }
            wmma::mma_sync(sum,a_frag[frag_index],b_frag[frag_index],sum);
        }
        if(k+BK<K){
            __pipeline_wait_prior(0);
        }
        __syncthreads();
        load_index=write_index;
    }
    wmma::store_matrix_sync(&Cs[OFFSET(warp_row*16,warp_col*16,BN)],
                            sum,BN,wmma::mem_row_major);
    __syncthreads();
    for(int i=threadIdx.x*4;i<BM*BN;i+=THREADNUM*4){
        int row=blockIdx.y*BM+i/BN;
        int col=blockIdx.x*BN+i%BN;
        size_t index=static_cast<size_t>(row)*N+col;
        if(row<M&&col+3<N&&(index&3)==0){
            float4 value=FETCH_FLOAT4(Cs[i]);
            float4 old={0.f,0.f,0.f,0.f};
            if(beta!=0.0f)old=FETCH_FLOAT4(C[index]);
            value.x=alpha*value.x+beta*old.x;
            value.y=alpha*value.y+beta*old.y;
            value.z=alpha*value.z+beta*old.z;
            value.w=alpha*value.w+beta*old.w;
            FETCH_FLOAT4(C[index])=value;
        }else{
            #pragma unroll
            for(int j=0;j<4;j++){
                if(row<M&&col+j<N){
                    C[index+j]=(beta==0.0f)?alpha*Cs[i+j]:
                        alpha*Cs[i+j]+beta*C[index+j];
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
    constexpr int BN=64;
    constexpr int BK=32;
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
    dim3 block((BM/16)*(BN/16)*32);
    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );
    for (int i = 0; i < warmup; ++i) {
        mysgemm_v6_tensor<BM,BN,BK><<<grid, block>>>(
            M, N, K,
            alpha, A,
            beta, B, C
        );
    }
    cudaError_t status=cudaDeviceSynchronize();
    if(status!=cudaSuccess){std::cerr<<cudaGetErrorString(status)<<"\n";return 1;}
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; ++i) {
        mysgemm_v6_tensor<BM,BN,BK><<<grid, block>>>(
            M, N, K,
            alpha, A,
            beta, B, C
        );
    }
    cudaEventRecord(stop);
    status=cudaEventSynchronize(stop);
    if(status!=cudaSuccess){std::cerr<<cudaGetErrorString(status)<<"\n";return 1;}
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
    std::cout << "TF32 Tensor Core multiplication, FP32 accumulation\n";
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





