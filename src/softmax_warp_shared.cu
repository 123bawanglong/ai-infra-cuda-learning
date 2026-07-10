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
    extern __shared__ float shared[];
    int row=blockIdx.x;
    int col=threadIdx.x;
    int block_size=blockDim.x;
    int warpId=col/32;
    int laneId=col%32;
    int warpsPerBlock=block_size/32;
    float *maxvals=shared;
    float *sumvals=shared+warpsPerBlock;
if(row>=N)
return;
    const float *idata=d_idata+row*C;
    float *odata=d_odata+row*C;
   float maxval=-INFINITY;
   for(int stride=col;stride<C;stride+=block_size){
             if(idata[stride]>maxval){
                maxval=idata[stride];
             }
   }
maxval=warpReduceMax(maxval);
if(laneId==0){
    maxvals[warpId]=maxval;
}
__syncthreads();
if(col==0){
    float val=maxvals[0];
    for(int i=1;i<warpsPerBlock;i++){
        val=fmaxf(val,maxvals[i]);
    }
    maxvals[0]=val;
}
__syncthreads();
float maxnum=maxvals[0];
float sum=0;
for(int stride=col;stride<C;stride+=block_size){
    odata[stride]=expf(idata[stride]-maxnum);
    sum+=odata[stride];
}
sum=warpReduceSum(sum);
if(laneId==0){
    sumvals[warpId]=sum;
}
__syncthreads();
if(col==0){
    float val=sumvals[0];
    for(int i=1;i<warpsPerBlock;i++){
val+=sumvals[i];
    }
    sumvals[0]=val;
}
__syncthreads();
float maxsum=sumvals[0];
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
int blocksize=128;
int warpsPerBlock=blocksize/32;
size_t sharedMemsize=warpsPerBlock*2*sizeof(float);
cudaMalloc((void**)&d_idata,sizeof(float)*N*C);
cudaMalloc((void**)&d_odata,sizeof(float)*N*C);
cudaMemcpy(d_idata,h_idata,sizeof(float)*N*C,cudaMemcpyHostToDevice);
cudaEvent_t begin,end;
cudaEventCreate(&begin);
cudaEventCreate(&end);
cudaEventRecord(begin);
reduce<<<blocknum,blocksize,sharedMemsize>>>(d_idata,d_odata,N,C);
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