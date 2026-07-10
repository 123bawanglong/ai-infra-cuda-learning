# AI Infra CUDA Learning

This repository records my CUDA learning path for AI infrastructure and LLM inference optimization.

The current focus is implementing and optimizing a row-wise softmax kernel with CUDA. The project compares different reduction strategies, including shared memory, warp shuffle, and multi-warp cooperation.

## Kernels

| File | Version | Main Idea |
|---|---|---|
| `src/softmax_shared.cu` | Shared-memory reduction | One block handles one row. Threads compute local max/sum, then reduce through shared memory. |
| `src/softmax_warp.cu` | Warp shuffle reduction | One warp handles one row. Warp-level shuffle replaces shared-memory reduction inside the warp. |
| `src/softmax_warp_shared.cu` | Multi-warp shuffle + shared memory | Multiple warps handle one row. Shuffle reduces inside each warp, shared memory combines warp-level results. |

The current best learning target is `src/softmax_warp_shared.cu`, which uses:

- one CUDA block per row
- multiple warps per block
- warp-level reduction with `__shfl_down_sync`
- shared memory for inter-warp reduction
- numerically stable softmax using `exp(x - row_max)`

## Why This Matters

Softmax is a common operation in deep learning workloads, especially in attention mechanisms. Optimizing softmax helps build intuition for GPU execution, memory hierarchy, synchronization, and profiling, all of which are important for AI infrastructure and LLM inference systems.

## Build

```bash
nvcc -lineinfo src/softmax_shared.cu -o softmax_shared
nvcc -lineinfo src/softmax_warp.cu -o softmax_warp
nvcc -lineinfo src/softmax_warp_shared.cu -o softmax_warp_shared
```

## Run A Small Test

```bash
printf "2 3\n1 2 3\n4 5 6\n" | ./softmax_warp_shared
```

Expected softmax values after the timing line:

```text
0.0900306
0.244728
0.665241
0.0900306
0.244728
0.665241
```

## Profiling

Generate a larger input:

```bash
python3 scripts/gen_input.py --rows 320 --cols 4096 > input_320x4096.txt
```

Profile with Nsight Compute:

```bash
ncu --set basic --force-overwrite -o reports/softmax_shared_320x4096 ./softmax_shared < input_320x4096.txt
ncu --set basic --force-overwrite -o reports/softmax_warp_320x4096 ./softmax_warp < input_320x4096.txt
ncu --set basic --force-overwrite -o reports/softmax_warp_shared_320x4096 ./softmax_warp_shared < input_320x4096.txt
```

See `docs/profiling.md` for the analysis template.

## Optimization Path

| Step | Kernel | What Improved | Main Limitation |
|---|---|---|---|
| 1 | `softmax_shared.cu` | Uses many threads in a block and shared memory reduction. | Shared-memory reduction needs repeated block-level synchronization. |
| 2 | `softmax_warp.cu` | Uses warp shuffle for lower-latency warp-level communication. | `blockDim.x = 32`, so row-level parallelism is limited for large `C`. |
| 3 | `softmax_warp_shared.cu` | Uses multiple warps per row. Shuffle handles intra-warp reduction; shared memory handles inter-warp reduction. | Final inter-warp reduction is still simple and can be further optimized. |

## Learning Notes

Key concepts practiced in this project:

- CUDA grid, block, thread, warp, and lane
- global memory vs shared memory vs registers
- warp-level communication with shuffle instructions
- block-level synchronization with `__syncthreads`
- reduction patterns for max and sum
- softmax numerical stability
- basic Nsight Compute profiling workflow

## Contribution Workflow

Portfolio updates follow `PUSH_WORKFLOW.md`.

## Next Steps

- Add profiling screenshots and performance comparison tables
- Refactor repeated CUDA error checking
- Compare different block sizes such as 128, 256, 512, and 1024
