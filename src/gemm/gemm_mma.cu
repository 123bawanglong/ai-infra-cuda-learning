


#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <iostream>
#include <iomanip>
template<int BM,int BN,int BK,int WM,int WN>
__global__ void __launch_bounds__((BM/WM)*(BN/WN)*32)
mysgemm_mma(int M,int N,int K,float alpha,const half *A,float beta,const half *B,float *C){
    constexpr int THREADNUM=(BM/WM)*(BN/WN)*32;
    constexpr int MITER=WM/16;
    constexpr int NITER=WN/8;
    constexpr int ACPR=BK/8;
    constexpr int BCPR=BN/8;
    __shared__ __align__(16) half As[2][BM*BK];
    __shared__ __align__(16) half Bs[2][BK*BN];
    int tid=threadIdx.x;
    int lane=tid%32;
    int warp_id=tid/32;
    int warp_row=warp_id/(BN/WN);
    int warp_col=warp_id%(BN/WN);
    int load_row=(lane&7)+((lane>>3)&1)*8;
    int load_col=(lane>>4)*8;
    float sum[MITER][NITER][4]={0.f};
    int load_index=0;
    for(int i=tid;i<BM*BK/8;i+=THREADNUM){
        int row=i/ACPR;
        int chunk=i%ACPR;
        int mask=(ACPR==4)?((row>>1)&3):(row&7);
        int dst=row*BK+(chunk^mask)*8;
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
        int dst=row*BN+(chunk^(row&7))*8;
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
                int mask=(ACPR==4)?((row>>1)&3):(row&7);
                int dst=row*BK+(chunk^mask)*8;
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
                int dst=row*BN+(chunk^(row&7))*8;
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
            unsigned a_frag[MITER][4];
            unsigned b_frag[NITER][2];
            #pragma unroll
            for(int m=0;m<MITER;m++){
                int row=warp_row*WM+m*16+load_row;
                int col=bk+load_col;
                int mask=(ACPR==4)?((row>>1)&3):(row&7);
                int index=row*BK+((col/8)^mask)*8+col%8;
                unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(&As[load_index][index]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                    :"=r"(a_frag[m][0]),"=r"(a_frag[m][1]),"=r"(a_frag[m][2]),"=r"(a_frag[m][3]):"r"(addr));
            }
            #pragma unroll
            for(int n=0;n<NITER;n+=2){
                int row=bk+load_row;
                int col=warp_col*WN+n*8+load_col;
                int index=row*BN+((col/8)^(row&7))*8+col%8;
                unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(&Bs[load_index][index]));
                unsigned reg[4];
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                    :"=r"(reg[0]),"=r"(reg[1]),"=r"(reg[2]),"=r"(reg[3]):"r"(addr));
                b_frag[n][0]=reg[0];b_frag[n][1]=reg[1];
                b_frag[n+1][0]=reg[2];b_frag[n+1][1]=reg[3];
            }
            #pragma unroll
            for(int m=0;m<MITER;m++){
                #pragma unroll
                for(int n=0;n<NITER;n++){
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        :"+f"(sum[m][n][0]),"+f"(sum[m][n][1]),"+f"(sum[m][n][2]),"+f"(sum[m][n][3])
                        :"r"(a_frag[m][0]),"r"(a_frag[m][1]),"r"(a_frag[m][2]),"r"(a_frag[m][3]),"r"(b_frag[n][0]),"r"(b_frag[n][1]));
                }
            }
        }
        if(k+BK<K)__pipeline_wait_prior(0);
        __syncthreads();
        load_index=write_index;
    }
    #pragma unroll
    for(int m=0;m<MITER;m++){
        #pragma unroll
        for(int n=0;n<NITER;n++){
            int row=blockIdx.y*BM+warp_row*WM+m*16+lane/4;
            int col=blockIdx.x*BN+warp_col*WN+n*8+2*(lane%4);
            #pragma unroll
            for(int r=0;r<2;r++){
                int global_row=row+r*8;
                size_t index=static_cast<size_t>(global_row)*N+col;
                if(global_row<M&&col+1<N&&(index&1)==0){
                    float2 value;
                    float2 old={0.f,0.f};
                    if(beta!=0.f)old=reinterpret_cast<float2*>(&C[index])[0];
                    value.x=alpha*sum[m][n][r*2]+beta*old.x;
                    value.y=alpha*sum[m][n][r*2+1]+beta*old.y;
                    reinterpret_cast<float2*>(&C[index])[0]=value;
                }else{
                    #pragma unroll
                    for(int j=0;j<2;j++){
                        if(global_row<M&&col+j<N)C[index+j]=(beta==0.f)?alpha*sum[m][n][r*2+j]:alpha*sum[m][n][r*2+j]+beta*C[index+j];
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
        mysgemm_fp16_mma<BM,BN,BK,WM,WN><<<grid, block>>>(
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
        mysgemm_fp16_mma<BM,BN,BK,WM,WN><<<grid, block>>>(
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





