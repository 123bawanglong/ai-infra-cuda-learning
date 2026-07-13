#include <iostream>
#include <cuda_runtime.h>
#define BLOCK_SIZE 256
__device__ float warpReduceSum(float value){
    value+=__shfl_down_sync(0xffffffff,value,16);
    value+=__shfl_down_sync(0xffffffff,value,8);
    value+=__shfl_down_sync(0xffffffff,value,4);
    value+=__shfl_down_sync(0xffffffff,value,2);
    value+=__shfl_down_sync(0xffffffff,value,1);
    return value;
}
__global__ void reduce(const float *g_idata,float *g_odata,int N){
__shared__ float sidata[BLOCK_SIZE];
int tid=threadIdx.x;
int j=blockDim.x*blockIdx.x+threadIdx.x;
if(j<N){
sidata[tid]=g_idata[j];}
else{
sidata[tid]=0.f;
}
__syncthreads();
for(int i=BLOCK_SIZE/2;i>32;i/=2){
    if(tid<i){
        sidata[tid]+=sidata[tid+i];
    }
    __syncthreads();
}
if(tid<32){
float value=sidata[tid]+sidata[tid+32];
value=warpReduceSum(value);
if(tid==0){
    g_odata[blockIdx.x]=value;}
}}
int main(){
    int N;
    std::cin>>N;
    if(N<=0||N>BLOCK_SIZE*BLOCK_SIZE){
        std::cerr<<"666演都不演了"<<std::endl;
        return 1;
    }
    float *h_idata=new float[N];
    for(int i=0;i<N;i++){
        std::cin>>h_idata[i];
    }
    float *d_idata=nullptr;
    float *d_odata=nullptr;
    int blockNum=(N+BLOCK_SIZE-1)/BLOCK_SIZE;
    float* d_out=nullptr;
    float h_out=0.f;
    cudaMalloc((void**)&d_idata,sizeof(float)*N);
    cudaMalloc((void**)&d_odata,sizeof(float)*blockNum);
    cudaMalloc((void**)&d_out,sizeof(float));
    cudaMemcpy(d_idata,h_idata,sizeof(float)*N,cudaMemcpyHostToDevice);
    cudaEvent_t begin,end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    cudaEventRecord(begin);
    reduce<<<blockNum,BLOCK_SIZE>>>(d_idata,d_odata,N);
    reduce<<<1,BLOCK_SIZE>>>(d_odata,d_out,blockNum);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaMemcpy(&h_out,d_out,sizeof(float),cudaMemcpyDeviceToHost);
    cudaError_t err=cudaGetLastError();
    if(err!=cudaSuccess){
        std::cerr<<cudaGetErrorString(err)<<std::endl;
    }
    float time=0.f;
    cudaEventElapsedTime(&time,begin,end);
    std::cout<<time<<std::endl;
    std::cout<<h_out<<std::endl;
    cudaFree(d_idata);
    cudaFree(d_odata);
    cudaFree(d_out);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    delete[] h_idata;
    return 0;
}
