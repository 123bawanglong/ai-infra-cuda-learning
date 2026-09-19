#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <vector>
__device__ __forceinline__ float warp_reduce_sum(float value){
    #pragma unroll
    for(int offset=16;offset>0;offset/=2){
        value+=__shfl_down_sync(0xffffffffu,value,offset);
    }
    return value;
}
template<int BlockSize>
__global__ void rms_norm_vectorized(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    float epsilon,
    int batch_size,
    int hidden_size
){
    static_assert(BlockSize%32==0,"BlockSize must be a multiple of 32");
    constexpr int kWarpCount=BlockSize/32;
    __shared__ float warp_sums[kWarpCount];
    int row=blockIdx.x;
    if(row>=batch_size){
        return;
    }
    int tid=threadIdx.x;
    int lane=tid&31;
    int warp=tid>>5;
    const float* row_input=input+static_cast<size_t>(row)*hidden_size;
    float* row_output=output+static_cast<size_t>(row)*hidden_size;
    bool can_vectorize=(hidden_size&3)==0;
    float sum_of_squares=0.0f;
    if(can_vectorize){
        int vector_count=hidden_size/4;
        const float4* vector_input=reinterpret_cast<const float4*>(row_input);
        for(int index=tid;index<vector_count;index+=BlockSize){
            float4 value=vector_input[index];
            sum_of_squares=fmaf(value.x,value.x,sum_of_squares);
            sum_of_squares=fmaf(value.y,value.y,sum_of_squares);
            sum_of_squares=fmaf(value.z,value.z,sum_of_squares);
            sum_of_squares=fmaf(value.w,value.w,sum_of_squares);
        }
    }else{
        for(int index=tid;index<hidden_size;index+=BlockSize){
            float value=row_input[index];
            sum_of_squares=fmaf(value,value,sum_of_squares);
        }
    }
    sum_of_squares=warp_reduce_sum(sum_of_squares);
    if(lane==0){
        warp_sums[warp]=sum_of_squares;
    }
    __syncthreads();
    if(warp==0){
        float block_sum=lane<kWarpCount?warp_sums[lane]:0.0f;
        block_sum=warp_reduce_sum(block_sum);
        if(lane==0){
            warp_sums[0]=block_sum;
        }
    }
    __syncthreads();
    float scale=rsqrtf(
        warp_sums[0]/static_cast<float>(hidden_size)+epsilon
    );
    if(can_vectorize){
        int vector_count=hidden_size/4;
        const float4* vector_input=reinterpret_cast<const float4*>(row_input);
        const float4* vector_weight=reinterpret_cast<const float4*>(weight);
        float4* vector_output=reinterpret_cast<float4*>(row_output);
        for(int index=tid;index<vector_count;index+=BlockSize){
            float4 value=vector_input[index];
            float4 gamma=vector_weight[index];
            vector_output[index]=make_float4(
                value.x*gamma.x*scale,
                value.y*gamma.y*scale,
                value.z*gamma.z*scale,
                value.w*gamma.w*scale
            );
        }
    }else{
        for(int index=tid;index<hidden_size;index+=BlockSize){
            row_output[index]=row_input[index]*weight[index]*scale;
        }
    }
}
int main(){
    int batch_size=0;
    int hidden_size=0;
    float epsilon=0.0f;
    std::cin>>batch_size>>hidden_size>>epsilon;
    if(batch_size<=0||hidden_size<=0||epsilon<0.0f){
        std::cerr<<"invalid dimensions or epsilon\n";
        return EXIT_FAILURE;
    }
    size_t count=static_cast<size_t>(batch_size)*hidden_size;
    size_t data_bytes=count*sizeof(float);
    size_t weight_bytes=static_cast<size_t>(hidden_size)*sizeof(float);
    std::vector<float> host_input(count);
    std::vector<float> host_weight(hidden_size);
    std::vector<float> host_output(count);
    for(float& value:host_input) std::cin>>value;
    for(float& value:host_weight) std::cin>>value;
    float* device_input=nullptr;
    float* device_output=nullptr;
    float* device_weight=nullptr;
    cudaMalloc(reinterpret_cast<void**>(&device_input),data_bytes);
    cudaMalloc(reinterpret_cast<void**>(&device_output),data_bytes);
    cudaMalloc(reinterpret_cast<void**>(&device_weight),weight_bytes);
    cudaMemcpy(
        device_input,
        host_input.data(),
        data_bytes,
        cudaMemcpyHostToDevice
    );
    cudaMemcpy(
        device_weight,
        host_weight.data(),
        weight_bytes,
        cudaMemcpyHostToDevice
    );
    constexpr int kBlockSize=256;
    rms_norm_vectorized<kBlockSize><<<batch_size,kBlockSize>>>(
        device_input,
        device_output,
        device_weight,
        epsilon,
        batch_size,
        hidden_size
    );
    cudaMemcpy(
        host_output.data(),
        device_output,
        data_bytes,
        cudaMemcpyDeviceToHost
    );
    for(float value:host_output){
        std::cout<<value<<'\n';
    }
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_weight);
    return EXIT_SUCCESS;
}
