# AI Infra CUDA Learning

CUDA kernel implementations and optimization notes for AI infrastructure and
LLM inference workloads. The repository follows an optimization-first path:
start with a correct baseline, identify bottlenecks, and introduce CUDA
optimizations step by step.

## Highlights

- Implements SGEMM, Reduction, Softmax, RMSNorm, Transpose, Histogram, and FlashAttention.
- Keeps numbered versions so every optimization step can be reviewed independently.
- Includes CPU references and correctness checks in selected standalone examples.
- Covers boundary-safe access, stable softmax, warp reductions, and tiled memory reuse.
- Provides build, test, and Nsight Compute profiling commands.

## Optimization Map

| Operator | Optimization path |
|---|---|
| GEMM | Naive FP32 → shared-memory tiling → register tiling → vectorized access → async copy and double buffering → register prefetch → warp tiling; TF32 Tensor Core variant |
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
│   ├── gemm/         # GEMM V1–V7 and TF32 Tensor Core example
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

### GEMM

| Stage | File | Main idea |
|---|---|---|
| V1 | [`gemm_v1.cu`](src/gemm/gemm_v1.cu) | Naive FP32, one output element per thread |
| V2 | [`gemm_v2.cu`](src/gemm/gemm_v2.cu) | Shared-memory block tiling |
| V3 | [`gemm_v3.cu`](src/gemm/gemm_v3.cu) | Per-thread register tiles |
| V4 | [`gemm_v4.cu`](src/gemm/gemm_v4.cu) | `float4` access and transposed A shared-memory layout |
| V5 | [`gemm_v5.cu`](src/gemm/gemm_v5.cu) | Asynchronous global-to-shared copies and shared-memory double buffering |
| V6 | [`gemm_v6.cu`](src/gemm/gemm_v6.cu) | Double-buffered register fragments |
| V7 | [`gemm_v7.cu`](src/gemm/gemm_v7.cu) | Explicit warp tiling |
| Tensor Core | [`gemm_tensor.cu`](src/gemm/gemm_tensor.cu) | WMMA TF32 multiplication with FP32 accumulation |

All eight files are standalone programs with `M=N=K=4096`, all-one inputs,
10 warm-up iterations, and 100 timed iterations. They take no stdin input and
print average kernel time, throughput, and `C[0]` (expected: 4096).
This single-element check is a smoke test, not a full correctness test.
The Tensor Core variant explicitly rounds inputs to TF32 before multiplication;
it does not have the same multiplication precision as the FP32 versions.

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

# Set this to your GPU architecture (sm_120 for RTX 5080).
CUDA_ARCH=sm_120
for version in v1 v2 v3 v4 v5 v6 v7 tensor; do
  nvcc -O3 -lineinfo -arch=$CUDA_ARCH src/gemm/gemm_${version}.cu -o build/gemm_${version}
done
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
# GEMM smoke tests (fixed 4096 x 4096 inputs, expected C[0] = 4096)
build/gemm_v1
build/gemm_tensor

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

- Kernels use FP32, except GEMM Tensor Core multiplication uses TF32 with FP32 accumulation.
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
- Add FP16/BF16 Tensor Core implementations.
- Add batched, multi-head, and causal FlashAttention variants.
- Compare custom kernels against cuBLAS and framework baselines.
