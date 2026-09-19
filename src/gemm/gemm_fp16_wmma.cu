#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cuda_fp16.h>
#include <mma.h>
namespace wmma=nvcuda::wmma;
#include <cstdio>
#include <vector>
#include <iostream>
#include <iomanip>
template<int BM,int BN,int BK,int WM,int WN>
__global__ void __launch_bounds__((BM/WM)*(BN/WN)*32)
mysgemm_wmma(int M,int N,int K,float alpha,const half *A,float beta,const half *B,float *C){
    constexpr int THREADNUM=(BM/WM)*(BN/WN)*32;
    constexpr int MITER=WM/16;
    constexpr int NITER=WN/16;
    constexpr int ACPR=BK/8;
    constexpr int BCPR=BN/8;
    __shared__ __align__(32) half As[2][BM*BK];
    __shared__ __align__(32) half Bs[2][BK*BN];
    int tid=threadIdx.x;
    int lane=tid%32;
    int warp_id=tid/32;
    int warp_row=warp_id/(BN/WN);
    int warp_col=warp_id%(BN/WN);
    wmma::fragment<wmma::accumulator,16,16,16,float> sum[MITER][NITER];
    #pragma unroll
    for(int m=0;m<MITER;m++){
        #pragma unroll
        for(int n=0;n<NITER;n++)wmma::fill_fragment(sum[m][n],0.f);
    }
    int load_index=0;
    for(int i=tid;i<BM*BK/8;i+=THREADNUM){
        int row=i/ACPR;
        int chunk=i%ACPR;
        int dst=row*BK+chunk*8;
        int global_row=blockIdx.y*BM+row;
        int global_col=0+chunk*8;
        size_t index=static_cast<size_t>(global_row)*K+global_col;
        if(global_row<M&&global_col+7<K&&(index&7)==0){
            __pipeline_memcpy_async(&As[load_index][dst],&A[index],16);
        }else{
            #pragma unroll
            for(int j=0;j<8;j++)As[load_index][dst+j]=(global_row<M&&global_col+j<K)?A[index+j]:__float2half(0.f);
        }
    }
    for(int i=tid;i<BK*BN/8;i+=THREADNUM){
        int row=i/BCPR;
        int chunk=i%BCPR;
        int dst=row*BN+chunk*8;
        int global_row=0+row;
        int global_col=blockIdx.x*BN+chunk*8;
        size_t index=static_cast<size_t>(global_row)*N+global_col;
        if(global_row<K&&global_col+7<N&&(index&7)==0){
            __pipeline_memcpy_async(&Bs[load_index][dst],&B[index],16);
        }else{
            #pragma unroll
            for(int j=0;j<8;j++)Bs[load_index][dst+j]=(global_row<K&&global_col+j<N)?B[index+j]:__float2half(0.f);
        }
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    for(int k=0;k<K;k+=BK){
        int write_index=load_index^1;
        if(k+BK<K){
            for(int i=tid;i<BM*BK/8;i+=THREADNUM){
                int row=i/ACPR;
                int chunk=i%ACPR;
                int dst=row*BK+chunk*8;
                int global_row=blockIdx.y*BM+row;
                int global_col=k+BK+chunk*8;
                size_t index=static_cast<size_t>(global_row)*K+global_col;
                if(global_row<M&&global_col+7<K&&(index&7)==0){
                    __pipeline_memcpy_async(&As[write_index][dst],&A[index],16);
                }else{
                    #pragma unroll
                    for(int j=0;j<8;j++)As[write_index][dst+j]=(global_row<M&&global_col+j<K)?A[index+j]:__float2half(0.f);
                }
            }
            for(int i=tid;i<BK*BN/8;i+=THREADNUM){
                int row=i/BCPR;
                int chunk=i%BCPR;
                int dst=row*BN+chunk*8;
                int global_row=k+BK+row;
                int global_col=blockIdx.x*BN+chunk*8;
                size_t index=static_cast<size_t>(global_row)*N+global_col;
                if(global_row<K&&global_col+7<N&&(index&7)==0){
                    __pipeline_memcpy_async(&Bs[write_index][dst],&B[index],16);
                }else{
                    #pragma unroll
                    for(int j=0;j<8;j++)Bs[write_index][dst+j]=(global_row<K&&global_col+j<N)?B[index+j]:__float2half(0.f);
                }
            }
            __pipeline_commit();
        }
        #pragma unroll
        for(int bk=0;bk<BK;bk+=16){
            wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a_frag[MITER];
            wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b_frag[NITER];
            #pragma unroll
            for(int m=0;m<MITER;m++){
                int row=warp_row*WM+m*16;
                wmma::load_matrix_sync(a_frag[m],&As[load_index][row*BK+bk],BK);
            }
            #pragma unroll
            for(int n=0;n<NITER;n++){
                int col=warp_col*WN+n*16;
                wmma::load_matrix_sync(b_frag[n],&Bs[load_index][bk*BN+col],BN);
            }
            #pragma unroll
            for(int m=0;m<MITER;m++){
                #pragma unroll
                for(int n=0;n<NITER;n++){
                    wmma::mma_sync(sum[m][n],a_frag[m],b_frag[n],sum[m][n]);
                }
            }
        }
        if(k+BK<K)__pipeline_wait_prior(0);
        __syncthreads();
        load_index=write_index;
    }
    float *tile=reinterpret_cast<float*>(&As[0][0])+warp_id*256;
    #pragma unroll
    for(int m=0;m<MITER;m++){
        #pragma unroll
        for(int n=0;n<NITER;n++){
            wmma::store_matrix_sync(tile,sum[m][n],16,wmma::mem_row_major);
            __syncwarp();
            #pragma unroll
            for(int i=lane;i<128;i+=32){
                int row=blockIdx.y*BM+warp_row*WM+m*16+i/8;
                int col=blockIdx.x*BN+warp_col*WN+n*16+(i%8)*2;
                size_t index=static_cast<size_t>(row)*N+col;
                if(row<M&&col+1<N&&(index&1)==0){
                    float2 value=reinterpret_cast<float2*>(tile)[i];
                    float2 old={0.f,0.f};
                    if(beta!=0.f)old=reinterpret_cast<float2*>(&C[index])[0];
                    value.x=alpha*value.x+beta*old.x;
                    value.y=alpha*value.y+beta*old.y;
                    reinterpret_cast<float2*>(&C[index])[0]=value;
                }else{
                    #pragma unroll
                    for(int j=0;j<2;j++){
                        if(row<M&&col+j<N)C[index+j]=(beta==0.f)?alpha*tile[i*2+j]:alpha*tile[i*2+j]+beta*C[index+j];
                    }
                }
            }
            __syncwarp();
        }
    }
}
int main()
{
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    constexpr int BM=128;
    constexpr int BN=128;
    constexpr int BK=32;
    constexpr int WM=64;
    constexpr int WN=32;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int warmup = 10;
    const int repeat = 100;
    size_t bytesA = static_cast<size_t>(M) * K * sizeof(half);
    size_t bytesB = static_cast<size_t>(K) * N * sizeof(half);
    size_t bytesC = static_cast<size_t>(M) * N * sizeof(float);
    std::vector<half> hA(static_cast<size_t>(M) * K, __float2half(1.0f));
    std::vector<half> hB(static_cast<size_t>(K) * N, __float2half(1.0f));
    half *A = nullptr;
    half *B = nullptr;
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
        mysgemm_wmma<BM,BN,BK,WM,WN><<<grid, block>>>(
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
        mysgemm_wmma<BM,BN,BK,WM,WN><<<grid, block>>>(
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
    std::cout << "FP16 Tensor Core multiplication, FP32 accumulation\n";
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