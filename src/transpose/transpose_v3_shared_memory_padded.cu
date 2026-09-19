#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#define BDIMX 32
#define BDIMY 8
__global__ void v3(float* out,const float* in,int nx,int ny){
    __shared__ float tile[BDIMY][BDIMX+1];
    int ix=blockIdx.x*blockDim.x+threadIdx.x;
    int iy=blockIdx.y*blockDim.y+threadIdx.y;
    if(ix<nx&&iy<ny){
        unsigned int ti=iy*nx+ix;
        tile[threadIdx.y][threadIdx.x]=in[ti];
    }else{
        tile[threadIdx.y][threadIdx.x]=0.0f;
    }
    __syncthreads();
    unsigned int bidx=threadIdx.y*blockDim.x+threadIdx.x;
    int row=bidx/blockDim.y;
    int col=bidx%blockDim.y;
    int out_x=blockIdx.y*blockDim.y+col;
    int out_y=blockIdx.x*blockDim.x+row;
    if(out_x<ny&&out_y<nx){
        unsigned int to=out_y*ny+out_x;
        out[to]=tile[col][row];
    }
}
void cpu_transpose(const float* in,float* out,int nx,int ny){
    for(int y=0;y<ny;y++){
        for(int x=0;x<nx;x++){
            int ti=y*nx+x;
            int to=x*ny+y;
            out[to]=in[ti];
        }
    }
}
bool check_result(const float* gpu,const float* cpu,size_t size){
    for(size_t i=0;i<size;i++){
        if(std::fabs(gpu[i]-cpu[i])>1e-5f){
            std::cerr<<"Mismatch at index "<<i<<": GPU = "<<gpu[i]<<", CPU = "<<cpu[i]<<'\n';
            return false;
        }
    }
    return true;
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
    float* h_ref=new float[num_elements];
    for(size_t i=0;i<num_elements;i++) h_in[i]=static_cast<float>(i);
    float* d_in=nullptr;
    float* d_out=nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_in),bytes);
    cudaMalloc(reinterpret_cast<void**>(&d_out),bytes);
    cudaMemcpy(d_in,h_in,bytes,cudaMemcpyHostToDevice);
    cudaMemset(d_out,0,bytes);
    dim3 block(BDIMX,BDIMY);
    dim3 grid((nx+BDIMX-1)/BDIMX,(ny+BDIMY-1)/BDIMY);
    v3<<<grid,block>>>(d_out,d_in,nx,ny);
    cudaGetLastError();
    cudaDeviceSynchronize();
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    v3<<<grid,block>>>(d_out,d_in,nx,ny);
    cudaGetLastError();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms=0.0f;
    cudaEventElapsedTime(&ms,start,stop);
    cudaMemcpy(h_out,d_out,bytes,cudaMemcpyDeviceToHost);
    cpu_transpose(h_in,h_ref,nx,ny);
    bool correct=check_result(h_out,h_ref,num_elements);
    std::cout<<"kernel time: "<<ms<<" ms\n";
    std::cout<<"result: "<<(correct?"PASS":"FAIL")<<'\n';
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
    delete[] h_ref;
    return 0;
}
