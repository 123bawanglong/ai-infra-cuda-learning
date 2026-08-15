#include <iostream>
#include <math.h>
#include <cuda_runtime.h>
__global__ void softmax(float *out,const float *inp,int N,int C){
extern __shared__ float shared[]; 
int row=blockIdx.x;
int tid=threadIdx.x;
int block_size=blockDim.x;
if(row>=N)return;
    const float *inp_row=inp+row*C;
    float *out_row=out+row*C;
     float maxval=-INFINITY;
     for(int j=tid;j<C;j+=block_size){
        if(inp_row[j]>maxval){
            maxval=inp_row[j];
        }
     }
     shared[tid]=maxval;
     __syncthreads();
     for(int stride=block_size/2;stride>=1;stride/=2){
        if(tid<stride){
        shared[tid]=fmaxf(shared[tid],shared[tid+stride]);}
        __syncthreads();
     }
     float row_max=shared[0];
     float sumval=0.f;
     for(int j=tid;j<C;j+=block_size){
        out_row[j]=expf(inp_row[j]-row_max);
        sumval+=out_row[j];
     }
     shared[tid]=sumval;
     __syncthreads();
     for(int stride=block_size/2;stride>=1;stride/=2){
        if(tid<stride){
            shared[tid]+=shared[tid+stride];
        }
        __syncthreads();
     }
     float row_sum=shared[0];
     for(int j=tid;j<C;j+=block_size){
        out_row[j]/=row_sum;
     }
}
int main(){
int N=0,C=0;
std::cin>>N>>C;
int blockSize=256;
int numblocks=N;
size_t sharedMemSize=blockSize*sizeof(float);
float *H_inp=new float[N*C];
float *H_out=new float[N*C];
float *D_inp=nullptr;
float *D_out=nullptr;
for(int i=0;i<N*C;i++){
    std::cin>>H_inp[i];
}
cudaMalloc((void**)&D_inp,sizeof(float)*N*C);
cudaMalloc((void**)&D_out,sizeof(float)*N*C);
cudaMemcpy(D_inp,H_inp,sizeof(float)*N*C,cudaMemcpyHostToDevice);
cudaEvent_t begay,belala;
cudaEventCreate(&begay);
cudaEventCreate(&belala);
cudaEventRecord(begay);
softmax<<<numblocks,blockSize,sharedMemSize>>>(D_out,D_inp,N,C);
cudaError_t err=cudaGetLastError();
if(err!=cudaSuccess){
    std::cerr<<cudaGetErrorString(err)<<std::endl;
}
cudaEventRecord(belala);
cudaEventSynchronize(belala);
float timi=0.f;
cudaEventElapsedTime(&timi,begay,belala);
std::cout<<timi<<std::endl;
cudaMemcpy(H_out,D_out,sizeof(float)*N*C,cudaMemcpyDeviceToHost);
for(int i=0;i<N*C;i++){
    std::cout<<H_out[i]<<std::endl;
}
cudaEventDestroy(begay);
cudaEventDestroy(belala);
delete[] H_inp;
delete[] H_out;
cudaFree(D_inp);
cudaFree(D_out);
return 0;
}