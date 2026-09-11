#include <cuda_runtime.h>
#include <math_constants.h>

template <int Br, int Bc, int D>
__global__ void flash_attention_tiled_online_softmax(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    float* __restrict__ output,
    int sequence_length
) {
    static_assert(Br > 0 && Bc > 0 && D > 0, "Template sizes must be positive");

    __shared__ float q_tile[Br][D];
    __shared__ float k_tile[Bc][D];
    __shared__ float v_tile[Bc][D];
    __shared__ float o_tile[Br][D];
    __shared__ float scores[Br][Bc];
    __shared__ float probabilities[Br][Bc];
    __shared__ float row_max[Br];
    __shared__ float row_sum[Br];
    __shared__ float old_scale[Br];

    const int k_local = threadIdx.x;
    const int q_local = threadIdx.y;
    const int q_row = blockIdx.x * Br + q_local;
    const bool q_valid = q_row < sequence_length;
    const float softmax_scale = rsqrtf(static_cast<float>(D));

    for (int column = k_local; column < D; column += blockDim.x) {
        q_tile[q_local][column] =
            q_valid ? query[static_cast<size_t>(q_row) * D + column] : 0.0f;
        o_tile[q_local][column] = 0.0f;
    }
    if (k_local == 0) {
        row_max[q_local] = -CUDART_INF_F;
        row_sum[q_local] = 0.0f;
    }
    __syncthreads();

    for (int k_start = 0; k_start < sequence_length; k_start += Bc) {
        const int k_row = k_start + k_local;
        const bool k_valid = k_row < sequence_length;
        for (int column = q_local; column < D; column += blockDim.y) {
            k_tile[k_local][column] =
                k_valid ? key[static_cast<size_t>(k_row) * D + column] : 0.0f;
            v_tile[k_local][column] =
                k_valid ? value[static_cast<size_t>(k_row) * D + column] : 0.0f;
        }
        __syncthreads();

        float score = -CUDART_INF_F;
        if (q_valid && k_valid) {
            score = 0.0f;
            #pragma unroll
            for (int column = 0; column < D; ++column) {
                score = fmaf(q_tile[q_local][column], k_tile[k_local][column], score);
            }
            score *= softmax_scale;
        }
        scores[q_local][k_local] = score;
        __syncthreads();

        if (k_local == 0) {
            float tile_max = -CUDART_INF_F;
            #pragma unroll
            for (int index = 0; index < Bc; ++index) {
                tile_max = fmaxf(tile_max, scores[q_local][index]);
            }
            const float new_max = fmaxf(row_max[q_local], tile_max);
            old_scale[q_local] = expf(row_max[q_local] - new_max);
            row_max[q_local] = new_max;
        }
        __syncthreads();

        probabilities[q_local][k_local] =
            q_valid && k_valid ? expf(score - row_max[q_local]) : 0.0f;
        __syncthreads();

        if (k_local == 0) {
            float tile_sum = 0.0f;
            #pragma unroll
            for (int index = 0; index < Bc; ++index) {
                tile_sum += probabilities[q_local][index];
            }
            row_sum[q_local] = row_sum[q_local] * old_scale[q_local] + tile_sum;
        }
        __syncthreads();

        for (int column = k_local; column < D; column += blockDim.x) {
            float weighted_value = 0.0f;
            #pragma unroll
            for (int index = 0; index < Bc; ++index) {
                weighted_value = fmaf(
                    probabilities[q_local][index],
                    v_tile[index][column],
                    weighted_value
                );
            }
            o_tile[q_local][column] =
                o_tile[q_local][column] * old_scale[q_local] + weighted_value;
        }
        __syncthreads();
    }

    if (q_valid) {
        const float inverse_sum = 1.0f / row_sum[q_local];
        for (int column = k_local; column < D; column += blockDim.x) {
            output[static_cast<size_t>(q_row) * D + column] =
                o_tile[q_local][column] * inverse_sum;
        }
    }
}

// Launch configuration:
// dim3 block(Bc, Br);
// dim3 grid((sequence_length + Br - 1) / Br);
