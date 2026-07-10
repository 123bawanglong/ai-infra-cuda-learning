#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
__device__ float warpReduceMax(float val){
    for(int offset=16;offset>=1;offset/=2){
    val=fmaxf(val,__shfl_down_sync(0xffffffff,val,offset));
 }
 return val;
}
__device__ float warpReduceSum(float val){
    for(int offset=16;offset>=1;offset/=2){
        val+=__shfl_down_sync(0xffffffff,val,offset);
    }
    return val;
}
__global__ void reduce(const float *d_idata,float *d_odata,int N,int C){
    int row=blockIdx.x;
    int col=threadIdx.x;
    int block_size=blockDim.x;
    const float *idata=d_idata+row*C;
    float *odata=d_odata+row*C;
if(row>=N)
return;
   float maxval=-INFINITY;
   for(int stride=col;stride<C;stride+=block_size){
             if(idata[stride]>maxval){
                maxval=idata[stride];
             }
   }
maxval=warpReduceMax(maxval);
float maxnum=__shfl_sync(0xffffffff,maxval,0,32);
float sum=0;
for(int stride=col;stride<C;stride+=block_size){
    odata[stride]=expf(idata[stride]-maxnum);
    sum+=odata[stride];
}
sum=warpReduceSum(sum);
float maxsum=__shfl_sync(0xffffffff,sum,0);
for(int stride=col;stride<C;stride+=block_size){
    odata[stride]/=maxsum;
}
}
int main(){
int N,C;
std::cin>>N>>C;
float *h_idata=new float[N*C];
float *h_odata=new float[N*C];
float *d_idata=nullptr;
float *d_odata=nullptr;
for(int i=0;i<N*C;i++){
    std::cin>>h_idata[i];
}
int blocknum=N;
int blocksize=32;
cudaMalloc((void**)&d_idata,sizeof(float)*N*C);
cudaMalloc((void**)&d_odata,sizeof(float)*N*C);
cudaMemcpy(d_idata,h_idata,sizeof(float)*N*C,cudaMemcpyHostToDevice);
cudaEvent_t begin,end;
cudaEventCreate(&begin);
cudaEventCreate(&end);
cudaEventRecord(begin);
reduce<<<blocknum,blocksize>>>(d_idata,d_odata,N,C);
cudaEventRecord(end);
cudaEventSynchronize(end);
float time=0.f;
cudaEventElapsedTime(&time,begin,end);
std::cout<<time<<std::endl;
cudaError_t err=cudaGetLastError();
if(err!=cudaSuccess){
    std::cerr<<cudaGetErrorString(err)<<std::endl;
}
cudaMemcpy(h_odata, d_odata, sizeof(float) * N * C, cudaMemcpyDeviceToHost);
for (int i = 0; i < N * C; i++) {
    std::cout << h_odata[i] << std::endl;
}
cudaFree(d_idata);
cudaFree(d_odata);
delete[] h_idata;
delete[] h_odata;
cudaEventDestroy(begin);
cudaEventDestroy(end);
return 0;
}