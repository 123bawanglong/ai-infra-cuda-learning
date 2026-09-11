#include <cuda_runtime.h>
#include <math_constants.h>

__device__ __forceinline__ float warp_reduce_max(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

template <int D, int Br, int Bc>
__global__ void flash_attention_warp_online_softmax(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    float* __restrict__ output,
    int sequence_length
) {
    static_assert(Bc == 32, "One warp maps to one 32-key tile, so Bc must equal 32");
    static_assert(D > 0 && Br > 0, "Template sizes must be positive");
    if (blockDim.x < 32 || (blockDim.x & 31) != 0) {
        return;
    }

    __shared__ float q_tile[Br][D];
    __shared__ float k_tile[Bc][D];
    __shared__ float v_tile[Bc][D];
    __shared__ float o_tile[Br][D];
    __shared__ float row_sum[Br];
    __shared__ float row_max[Br];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int warp_count = blockDim.x >> 5;
    const float softmax_scale = rsqrtf(static_cast<float>(D));

    for (int q_local = warp; q_local < Br; q_local += warp_count) {
        const int q_row = blockIdx.x * Br + q_local;
        const bool q_valid = q_row < sequence_length;
        for (int column = lane; column < D; column += 32) {
            q_tile[q_local][column] =
                q_valid ? query[static_cast<size_t>(q_row) * D + column] : 0.0f;
            o_tile[q_local][column] = 0.0f;
        }
        if (lane == 0) {
            row_sum[q_local] = 0.0f;
            row_max[q_local] = -CUDART_INF_F;
        }
    }
    __syncthreads();

    for (int k_start = 0; k_start < sequence_length; k_start += Bc) {
        for (int index = tid; index < Bc * D; index += blockDim.x) {
            const int k_local = index / D;
            const int column = index % D;
            const int k_row = k_start + k_local;
            const bool k_valid = k_row < sequence_length;
            k_tile[k_local][column] =
                k_valid ? key[static_cast<size_t>(k_row) * D + column] : 0.0f;
            v_tile[k_local][column] =
                k_valid ? value[static_cast<size_t>(k_row) * D + column] : 0.0f;
        }
        __syncthreads();

        for (int q_local = warp; q_local < Br; q_local += warp_count) {
            const int q_row = blockIdx.x * Br + q_local;
            const bool q_valid = q_row < sequence_length;
            const int k_row = k_start + lane;
            const bool k_valid = k_row < sequence_length;

            float score = -CUDART_INF_F;
            if (q_valid && k_valid) {
                score = 0.0f;
                #pragma unroll
                for (int column = 0; column < D; ++column) {
                    score = fmaf(q_tile[q_local][column], k_tile[lane][column], score);
                }
                score *= softmax_scale;
            }

            const float tile_max = warp_reduce_max(score);
            const float new_max = fmaxf(row_max[q_local], tile_max);
            const float old_scale = expf(row_max[q_local] - new_max);
            const float probability =
                q_valid && k_valid ? expf(score - new_max) : 0.0f;
            const float tile_sum = warp_reduce_sum(probability);
            const float new_sum = row_sum[q_local] * old_scale + tile_sum;

            for (int column_base = 0; column_base < D; column_base += 32) {
                const int column = column_base + lane;
                float weighted_value = 0.0f;
                #pragma unroll
                for (int k_local = 0; k_local < Bc; ++k_local) {
                    const float key_probability =
                        __shfl_sync(0xffffffffu, probability, k_local);
                    if (column < D) {
                        weighted_value = fmaf(
                            key_probability, v_tile[k_local][column], weighted_value
                        );
                    }
                }
                if (column < D) {
                    o_tile[q_local][column] =
                        o_tile[q_local][column] * old_scale + weighted_value;
                }
            }
            if (lane == 0) {
                row_max[q_local] = new_max;
                row_sum[q_local] = new_sum;
            }
        }
        __syncthreads();
    }

    for (int q_local = warp; q_local < Br; q_local += warp_count) {
        const int q_row = blockIdx.x * Br + q_local;
        if (q_row < sequence_length) {
            const float inverse_sum = 1.0f / row_sum[q_local];
            for (int column = lane; column < D; column += 32) {
                output[static_cast<size_t>(q_row) * D + column] =
                    o_tile[q_local][column] * inverse_sum;
            }
        }
    }
}

// Launch with a one-dimensional block whose size is a positive multiple of 32,
// for example 128 threads, and grid.x = (sequence_length + Br - 1) / Br.
