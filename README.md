# AI Infra CUDA Learning

CUDA kernel implementations and optimization notes for AI infrastructure and
LLM inference workloads. The repository follows an optimization-first path:
start with a correct baseline, identify bottlenecks, and introduce CUDA
optimizations step by step.

## Highlights

- Implements SGEMM, Reduction, Softmax, RMSNorm, Transpose, Histogram, and FlashAttention.
- Keeps numbered versions so every optimization step can be reviewed independently.
- Includes CPU or cuBLAS references and correctness checks in standalone examples.
- Covers boundary-safe access, stable softmax, warp reductions, and tiled memory reuse.
- Provides build, test, and Nsight Compute profiling commands.

## Optimization Map

| Operator | Optimization path |
|---|---|
| SGEMM | Shared-memory tiling → register tiling and `float4` access → double buffering → warp tiling |
| Reduction | Shared-memory reduction → warp shuffle → grid-stride block reduction |
| Softmax | Shared-memory reduction → one warp per row → multiple warps per row |
| Transpose | Naive transpose → shared-memory coalescing → bank-conflict padding → two elements per thread |
| RMSNorm | Vectorized I/O → warp/block reduction → alignment-safe scalar fallback |
| FlashAttention | Tiled Q/K/V reuse → online softmax → warp-level score reduction |
| Histogram | Block-local shared-memory atomics → global merge |

## Repository Layout

```text
.
├── src/
│   ├── attention/    # FlashAttention kernels
│   ├── gemm/         # cuBLAS reference and SGEMM stages
│   ├── histogram/    # shared-memory histogram
│   ├── reduction/    # block and warp reduction stages
│   ├── rmsnorm/      # vectorized RMSNorm
│   ├── softmax/      # row-wise softmax stages
│   └── transpose/    # matrix-transpose stages
├── docs/
│   ├── benchmarking.md
│   └── softmax/profiling.md
├── scripts/softmax/
└── PUSH_WORKFLOW.md
```

## Kernel Guide

### SGEMM

| Stage | File | Main idea |
|---|---|---|
| Reference | [`cublas_row_major_reference.cu`](src/gemm/cublas_row_major_reference.cu) | Row-major cuBLAS comparison |
| V1 | [`sgemm_v1_shared_memory.cu`](src/gemm/sgemm_v1_shared_memory.cu) | Shared-memory block tiling |
| V2 | [`sgemm_v2_register_tiled_vectorized.cu`](src/gemm/sgemm_v2_register_tiled_vectorized.cu) | Register tiles and `float4` access |
| V3 | [`sgemm_v3_double_buffered.cu`](src/gemm/sgemm_v3_double_buffered.cu) | Double buffering and register prefetch |
| V4 | [`sgemm_v4_warp_tiled_double_buffered.cu`](src/gemm/sgemm_v4_warp_tiled_double_buffered.cu) | Warp tiling and double buffering |

V3 and V4 require `M` and `N` to be multiples of 128 and `K` to be a multiple of 8.

### Reduction and Softmax

| Operator | Stage | File | Main idea |
|---|---|---|---|
| Reduction | V1 | [`reduce_v1_sequential.cu`](src/reduction/reduce_v1_sequential.cu) | Continuous active-thread reduction |
| Reduction | V2 | [`reduce_v2_last_warp_shuffle.cu`](src/reduction/reduce_v2_last_warp_shuffle.cu) | Shared memory followed by warp shuffle |
| Reduction | V3 | [`reduce_v3_block_reduce_grid_stride.cu`](src/reduction/reduce_v3_block_reduce_grid_stride.cu) | Grid-stride block reduction |
| Softmax | V1 | [`softmax_v1_shared_memory.cu`](src/softmax/softmax_v1_shared_memory.cu) | Shared-memory max/sum reduction |
| Softmax | V2 | [`softmax_v2_warp_shuffle.cu`](src/softmax/softmax_v2_warp_shuffle.cu) | One warp per row |
| Softmax | V3 | [`softmax_v3_multi_warp_shared.cu`](src/softmax/softmax_v3_multi_warp_shared.cu) | Multiple warps per row |

### Transpose and AI Inference Operators

| Operator | Stage | File | Main idea |
|---|---|---|---|
| Transpose | V1 | [`transpose_v1_naive.cu`](src/transpose/transpose_v1_naive.cu) | Coalesced reads, strided writes |
| Transpose | V2 | [`transpose_v2_shared_memory.cu`](src/transpose/transpose_v2_shared_memory.cu) | Shared-memory coalescing |
| Transpose | V3 | [`transpose_v3_shared_memory_padded.cu`](src/transpose/transpose_v3_shared_memory_padded.cu) | Padding against bank conflicts |
| Transpose | V4 | [`transpose_v4_shared_memory_padded_two_elements.cu`](src/transpose/transpose_v4_shared_memory_padded_two_elements.cu) | Two elements per thread |
| RMSNorm | Optimized | [`rmsnorm_vectorized_warp_reduce.cu`](src/rmsnorm/rmsnorm_vectorized_warp_reduce.cu) | Vectorized I/O and two-level reduction |
| FlashAttention | V1 | [`flash_attention_v1_tiled_online_softmax.cu`](src/attention/flash_attention_v1_tiled_online_softmax.cu) | Tiled reuse and online softmax |
| FlashAttention | V2 | [`flash_attention_v2_warp_online_softmax.cu`](src/attention/flash_attention_v2_warp_online_softmax.cu) | Warp-level max/sum reduction |
| Histogram | Optimized | [`histogram_shared_memory_atomics.cu`](src/histogram/histogram_shared_memory_atomics.cu) | Shared histogram and global merge |

FlashAttention files are reusable kernels rather than executables. Required
launch configurations are documented at the bottom of each source file.

## Requirements

- NVIDIA GPU and CUDA Toolkit with `nvcc`
- Nsight Compute (`ncu`) for profiling
- Python 3 for input generation

## Build

```bash
mkdir -p build

nvcc -O3 -lineinfo src/gemm/sgemm_v1_shared_memory.cu -o build/sgemm_v1
nvcc -O3 -lineinfo src/gemm/sgemm_v4_warp_tiled_double_buffered.cu -o build/sgemm_v4
nvcc -O3 -lineinfo src/gemm/cublas_row_major_reference.cu -lcublas -o build/cublas_gemm
nvcc -O3 -lineinfo src/softmax/softmax_v3_multi_warp_shared.cu -o build/softmax_v3
nvcc -O3 -lineinfo src/transpose/transpose_v4_shared_memory_padded_two_elements.cu -o build/transpose_v4
nvcc -O3 -lineinfo src/rmsnorm/rmsnorm_vectorized_warp_reduce.cu -o build/rmsnorm
nvcc -O3 -lineinfo src/histogram/histogram_shared_memory_atomics.cu -o build/histogram
```

Compile the reusable FlashAttention kernels as object files:

```bash
nvcc -O3 -lineinfo -c src/attention/flash_attention_v1_tiled_online_softmax.cu -o build/flash_attention_v1.o
nvcc -O3 -lineinfo -c src/attention/flash_attention_v2_warp_online_softmax.cu -o build/flash_attention_v2.o
```

Add the architecture flag for your GPU, for example `-arch=sm_89`.

## Quick Correctness Checks

```bash
# SGEMM V1
printf "2 3 4\n1 2 3 4 5 6 7 8\n1 0 0 0 1 0 0 0 1 1 1 1\n" | build/sgemm_v1

# Softmax V3
printf "2 3\n1 2 3\n4 5 6\n" | build/softmax_v3

# Transpose V4 (prints result: PASS)
printf "4 3\n" | build/transpose_v4

# RMSNorm
printf "1 4 0.00001\n1 2 3 4\n1 1 1 1\n" | build/rmsnorm
```

## Profiling

```bash
python3 scripts/softmax/gen_input.py --rows 320 --cols 4096 > input_320x4096.txt
mkdir -p reports
ncu --set full --force-overwrite \
  -o reports/softmax_v3_320x4096 \
  build/softmax_v3 < input_320x4096.txt
```

Compare duration, memory bandwidth, global-memory efficiency, shared-memory
bank conflicts, occupancy, register use, and warp efficiency. Record results
with [`docs/benchmarking.md`](docs/benchmarking.md); Softmax-specific guidance is in
[`docs/softmax/profiling.md`](docs/softmax/profiling.md).

## Current Scope

- Kernels use FP32 to focus on CUDA optimization fundamentals.
- FlashAttention models one head without causal masking, batching, or mixed precision.
- Performance numbers are not published until measured reproducibly on a specified GPU.
- This educational repository is not a replacement for production libraries.

## Resume Summary

> Implemented and optimized CUDA kernels for SGEMM, Reduction, Softmax,
> RMSNorm, Transpose, Histogram, and FlashAttention. Applied shared-memory
> tiling, vectorized access, register blocking, warp shuffle, double buffering,
> and online softmax, with correctness checks and Nsight Compute workflows.

## Roadmap

- Add automated correctness tests for every operator.
- Benchmark all optimization stages on the same NVIDIA GPU.
- Add FP16/BF16 and Tensor Core implementations.
- Add batched, multi-head, and causal FlashAttention variants.
- Compare custom kernels against cuBLAS and framework baselines.
