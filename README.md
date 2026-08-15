# AI Infra CUDA Learning

CUDA learning examples for AI infrastructure and LLM inference optimization. The repository groups kernels by operator and keeps numbered implementations in optimization order.

## Repository Layout

```text
.
├── src/
│   ├── gemm/          # cuBLAS reference and SGEMM optimization stages
│   ├── softmax/       # row-wise softmax optimization stages
│   ├── reduction/     # block-reduction learning stages
│   └── histogram/     # shared-memory histogram
├── docs/
│   └── softmax/       # profiling workflow and results template
├── scripts/
│   └── softmax/       # softmax input generator
└── PUSH_WORKFLOW.md   # repository contribution checklist
```

## Kernels

### Matrix Multiplication

| File | Stage | Main idea |
|---|---|---|
| `src/gemm/cublas_row_major_reference.cu` | Reference | Row-major GEMM implemented with cuBLAS for comparison. |
| `src/gemm/sgemm_v1_shared_memory.cu` | V1 | Shared-memory block tiling; one thread computes one output element. |
| `src/gemm/sgemm_v2_register_tiled_vectorized.cu` | V2 | Per-thread register tiles and `float4` vectorized memory access, with boundary fallback paths. |
| `src/gemm/sgemm_v3_double_buffered.cu` | V3 | Double-buffered shared memory and register prefetching overlap tile loading with computation. |
| `src/gemm/sgemm_v4_warp_tiled_double_buffered.cu` | V4 | Adds warp-level output tiling to the double-buffered pipeline. |

V3 and V4 currently require `M` and `N` to be multiples of 128 and `K` to be a multiple of 8.

### Softmax

| File | Stage | Main idea |
|---|---|---|
| `src/softmax/softmax_v1_shared_memory.cu` | V1 | One block per row with shared-memory max and sum reductions. |
| `src/softmax/softmax_v2_warp_shuffle.cu` | V2 | One warp per row; shuffle instructions replace intra-warp shared-memory reductions. |
| `src/softmax/softmax_v3_multi_warp_shared.cu` | V3 | Multiple warps per row; shuffle reduces within warps and shared memory combines warp results. |

### Reduction

| File | Stage | Main idea |
|---|---|---|
| `src/reduction/reduce_v1_sequential.cu` | V1 | Continuous active threads reduce through shared memory. |
| `src/reduction/reduce_v2_last_warp_shuffle.cu` | V2 | Shared memory reduces to 64 values, then warp 0 finishes with `__shfl_down_sync`. |
| `src/reduction/reduce_v3_block_reduce_grid_stride.cu` | V3 | Grid-stride accumulation, independent warp reductions, and a final warp-level merge. |

### Histogram

| File | Main idea |
|---|---|
| `src/histogram/histogram_shared_memory_atomics.cu` | Each block accumulates a private shared-memory histogram before atomically merging it into the global result. |

## Build

Create a local output directory:

```bash
mkdir -p build
```

Compile an individual kernel with `nvcc`:

```bash
nvcc -O3 -lineinfo src/gemm/sgemm_v1_shared_memory.cu -o build/sgemm_v1
nvcc -O3 -lineinfo src/gemm/sgemm_v2_register_tiled_vectorized.cu -o build/sgemm_v2
nvcc -O3 -lineinfo src/gemm/sgemm_v3_double_buffered.cu -o build/sgemm_v3
nvcc -O3 -lineinfo src/gemm/sgemm_v4_warp_tiled_double_buffered.cu -o build/sgemm_v4

nvcc -O3 -lineinfo src/softmax/softmax_v1_shared_memory.cu -o build/softmax_v1
nvcc -O3 -lineinfo src/softmax/softmax_v2_warp_shuffle.cu -o build/softmax_v2
nvcc -O3 -lineinfo src/softmax/softmax_v3_multi_warp_shared.cu -o build/softmax_v3

nvcc -O3 -lineinfo src/histogram/histogram_shared_memory_atomics.cu -o build/histogram
```

The cuBLAS reference also needs the cuBLAS library:

```bash
nvcc -O3 -lineinfo src/gemm/cublas_row_major_reference.cu -lcublas -o build/cublas_gemm
```

## Small Correctness Tests

Boundary-safe SGEMM V1 test:

```bash
printf "2 3 4\n1 2 3 4 5 6 7 8\n1 0 0 0 1 0 0 0 1 1 1 1\n" | build/sgemm_v1
```

Expected matrix values after the timing line:

```text
5 6 7
13 14 15
```

Softmax test:

```bash
printf "2 3\n1 2 3\n4 5 6\n" | build/softmax_v3
```

Expected values after the timing line:

```text
0.0900306
0.244728
0.665241
0.0900306
0.244728
0.665241
```

Histogram test:

```bash
printf "8\n0 1 1 2 2 2 255 300\n" | build/histogram
```

The relevant bins should be `0 : 1`, `1 : 2`, `2 : 3`, and `255 : 1`; the out-of-range value is ignored.

## Softmax Profiling

Generate a larger input:

```bash
python3 scripts/softmax/gen_input.py --rows 320 --cols 4096 > input_320x4096.txt
```

Profile all three stages with Nsight Compute:

```bash
ncu --set basic --force-overwrite -o reports/softmax_v1_320x4096 build/softmax_v1 < input_320x4096.txt
ncu --set basic --force-overwrite -o reports/softmax_v2_320x4096 build/softmax_v2 < input_320x4096.txt
ncu --set basic --force-overwrite -o reports/softmax_v3_320x4096 build/softmax_v3 < input_320x4096.txt
```

See `docs/softmax/profiling.md` for the analysis checklist and results table.

## Learning Focus

- CUDA grid, block, thread, warp, and lane organization
- global memory, shared memory, and register reuse
- tiled matrix multiplication and vectorized access
- double buffering and register prefetching
- warp-level communication with shuffle instructions
- reduction patterns for max, sum, and histograms
- numerical stability in softmax
- Nsight Compute profiling and bottleneck analysis

## Contribution Workflow

Portfolio updates follow `PUSH_WORKFLOW.md`.
