#include <iostream>
#include <cuda_runtime.h>
__global__ void reduce(float *g_idata,float *g_odata){
    extern __shared__ float sdata[];
    int tid=threadIdx.x;
    sdata[tid]=g_idata[tid];
    __syncthreads();
    for(int s=1;s<blockDim.x;s*=2){
        if(tid%(s*2)==0){
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }
    if(tid==0){
    g_odata[blockIdx.x]=sdata[0];
    }
}
int main(){
    int N;
    std::cin>>N;
    float *h_idata=new float[N];
    float h_odata=0.f;
    for(int i=0;i<N;i++){
        std::cin>>h_idata[i];
    }
    float *d_idata=nullptr;
    float *d_odata=nullptr;
    cudaMalloc((void**)&d_idata,sizeof(float)*N);
    cudaMalloc((void**)&d_odata,sizeof(float));
    cudaMemcpy(d_idata,h_idata,sizeof(float)*N,cudaMemcpyHostToDevice);
    cudaEvent_t begin,end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    cudaEventRecord(begin);
    reduce<<<1,N,sizeof(float)*N>>>(d_idata,d_odata);
    cudaError_t err=cudaGetLastError();
    if(err!=cudaSuccess){
        std::cerr<<cudaGetErrorString(err)<<std::endl;
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time=0.f;
    cudaEventElapsedTime(&time,begin,end);
    std::cout<<time<<std::endl;
    cudaMemcpy(&h_odata,d_odata,sizeof(float),cudaMemcpyDeviceToHost);
    std::cout<<h_odata<<std::endl;
    cudaFree(d_idata);
    cudaFree(d_odata);
    delete[] h_idata;
    return 0;
}
