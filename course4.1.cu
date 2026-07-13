#include <iostream>
#include <cuda_runtime.h>
#define BLOCK_SIZE 256
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
for(int i=BLOCK_SIZE/2;i>=1;i/=2){
    if(tid<i){
        sidata[tid]+=sidata[tid+i];
    }
    __syncthreads();
}
if(tid==0){
    g_odata[blockIdx.x]=sidata[0];
}
}
int main(){
    int N;
    std::cin>>N;
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
    delete[] h_idata;
    return 0;
}
