#include <cuda_runtime.h>
#include <iostream>
#define BDIMX 32
#define BDIMY 8
__global__ void v4(float* out,const float* in,int nx,int ny){
    __shared__ float tile[BDIMY][2*BDIMX+1];
    int ix=2*blockDim.x*blockIdx.x+threadIdx.x;
    int iy=blockDim.y*blockIdx.y+threadIdx.y;
    if(ix<nx&&iy<ny){
        unsigned int ti=iy*nx+ix;
        tile[threadIdx.y][threadIdx.x]=in[ti];
    }else{
        tile[threadIdx.y][threadIdx.x]=0.0f;
    }
    if(ix+BDIMX<nx&&iy<ny){
        unsigned int ti2=iy*nx+ix+BDIMX;
        tile[threadIdx.y][threadIdx.x+BDIMX]=in[ti2];
    }else{
        tile[threadIdx.y][threadIdx.x+BDIMX]=0.0f;
    }
    __syncthreads();
    unsigned int bidx=threadIdx.y*blockDim.x+threadIdx.x;
    int row=bidx/blockDim.y;
    int col=bidx%blockDim.y;
    int out_x=blockIdx.y*blockDim.y+col;
    int out_y=2*blockIdx.x*blockDim.x+row;
    unsigned int to=out_y*ny+out_x;
    if(out_x<ny&&out_y<nx) out[to]=tile[col][row];
    if(out_x<ny&&out_y+BDIMX<nx) out[to+ny*BDIMX]=tile[col][row+BDIMX];
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
    cudaMemset(d_out,0,bytes);
    dim3 block(BDIMX,BDIMY);
    dim3 grid((nx+2*BDIMX-1)/(2*BDIMX),(ny+BDIMY-1)/BDIMY);
    v4<<<grid,block>>>(d_out,d_in,nx,ny);
    cudaGetLastError();
    cudaDeviceSynchronize();
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    v4<<<grid,block>>>(d_out,d_in,nx,ny);
    cudaGetLastError();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms=0.0f;
    cudaEventElapsedTime(&ms,start,stop);
    cudaMemcpy(h_out,d_out,bytes,cudaMemcpyDeviceToHost);
    std::cout<<"kernel time: "<<ms<<" ms\n";
    if(nx<=16&&ny<=16){
        std::cout<<"\ninput:\n";
        for(int y=0;y<ny;y++){
            for(int x=0;x<nx;x++) std::cout<<h_in[y*nx+x]<<' ';
            std::cout<<'\n';
        }
        std::cout<<"\ntranspose:\n";
        for(int y=0;y<nx;y++){
            for(int x=0;x<ny;x++) std::cout<<h_out[y*ny+x]<<' ';
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
