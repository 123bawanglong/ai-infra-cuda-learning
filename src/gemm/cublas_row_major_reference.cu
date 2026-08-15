#include <cublas_v2.h>
#include <cuda_runtime.h>

int main(){
    constexpr int A_N=3;
    constexpr int A_C=2;
    constexpr int B_N=2;
    constexpr int B_C=3;
    static_assert(A_C == B_N);
    float h_A[A_N*A_C]={
        1.0f,2.0f,
        3.0f,4.0f,
        5.0f,6.0f,
    };
    float h_B[B_N*B_C]{
        8.0f,9.0f,10.0f,
        11.0f,12.0f,13.0f,
    };
    float *h_C=new float[A_N*B_C]{};
    float *d_A=nullptr;
    float *d_B=nullptr;
    float *d_C=nullptr;
cudaMalloc(&d_A,sizeof(float)*A_N*A_C);
cudaMalloc(&d_B,sizeof(float)*B_N*B_C);
cudaMalloc(&d_C,sizeof(float)*B_C*A_N);
cudaMemcpy(d_A,h_A,sizeof(float)*A_N*A_C,cudaMemcpyHostToDevice);
cudaMemcpy(d_B,h_B,sizeof(float)*B_N*B_C,cudaMemcpyHostToDevice);
cublasHandle_t handle=nullptr;
cublasCreate(&handle);
const float alpha=1.0f;
const float beta=0.0f;
cublasSgemm(
    handle,
    CUBLAS_OP_N,
    CUBLAS_OP_N,
    B_C,A_N,A_C,
    &alpha,
    d_B,
    B_C,
    d_A,
    A_C,
    &beta,
    d_C,
    B_C
);
cudaMemcpy(h_C,d_C,sizeof(float)*A_N*B_C,cudaMemcpyDeviceToHost);
cublasDestroy(handle);
cudaFree(d_A);
cudaFree(d_B);
cudaFree(d_C);
delete[] h_C;
return 0;
}
