#include <cuda_runtime.h>
#include <iostream>
__global__ void v1(float* out,const float* in,int nx,int ny){
    int ix=blockIdx.x*blockDim.x+threadIdx.x;
    int iy=blockIdx.y*blockDim.y+threadIdx.y;
    if(ix<nx&&iy<ny){
        unsigned int ti=iy*nx+ix;
        unsigned int to=ix*ny+iy;
        out[to]=in[ti];
    }
}
int main(){
    int nx,ny;
    std::cout<<"input nx ny: ";
    std::cin>>nx>>ny;
    if(nx<=0||ny<=0){
        std::cerr<<"nx and ny must be positive\n";
        return 1;
    }
    size_t num_elements=static_cast<size_t>(nx)*static_cast<size_t>(ny);
    size_t bytes=num_elements*sizeof(float);
    float* h_in=new float[num_elements];
    float* h_out=new float[num_elements];
    for(size_t i=0;i<num_elements;i++) h_in[i]=static_cast<float>(i);
    float* d_in=nullptr;
    float* d_out=nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_in),bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_out),bytes);
    cudaMemcpy(d_in,h_in,bytes,cudaMemcpyHostToDevice);
    dim3 block(32,8);
    dim3 grid((nx+block.x-1)/block.x,(ny+block.y-1)/block.y);
    v1<<<grid,block>>>(d_out,d_in,nx,ny);
    cudaGetLastError();
    cudaDeviceSynchronize();
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    v1<<<grid,block>>>(d_out,d_in,nx,ny);
    cudaGetLastError();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms=0.0f;
    cudaEventElapsedTime(&ms,start,stop);
    cudaMemcpy(h_out,d_out,bytes,cudaMemcpyDeviceToHost);
    std::cout<<"kernel time: "<<ms<<" ms\n";
    if(nx<=16&&ny<=16){
        std::cout<<"\ninput:\n";
        for(int i=0;i<ny;i++){
            for(int j=0;j<nx;j++) std::cout<<h_in[i*nx+j]<<' ';
            std::cout<<'\n';
        }
        std::cout<<"\ntranspose:\n";
        for(int i=0;i<nx;i++){
            for(int j=0;j<ny;j++) std::cout<<h_out[i*ny+j]<<' ';
            std::cout<<'\n';
        }
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;
    return 0;
}
