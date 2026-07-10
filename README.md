# AI Infra CUDA Learning

This repository records my CUDA learning path for AI infrastructure and LLM inference optimization.

The current focus is implementing and optimizing a row-wise softmax kernel with CUDA. The project compares different reduction strategies, including shared memory, warp shuffle, and multi-warp cooperation.

## Current Kernel

`src/softmax_warp_shared.cu` implements a softmax kernel using:

- one CUDA block per row
- multiple warps per block
- warp-level reduction with `__shfl_down_sync`
- shared memory for inter-warp reduction
- numerically stable softmax using `exp(x - row_max)`

## Why This Matters

Softmax is a common operation in deep learning workloads, especially in attention mechanisms. Optimizing softmax helps build intuition for GPU execution, memory hierarchy, synchronization, and profiling, all of which are important for AI infrastructure and LLM inference systems.

## Build

```bash
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

The kernel can be profiled with Nsight Compute:

```bash
ncu --set basic --force-overwrite -o ncu/course3/softmax_warp_shared_320x4096 ./softmax_warp_shared < ncu/course3/input_320x4096.txt
```

## Learning Notes

Key concepts practiced in this project:

- CUDA grid, block, thread, warp, and lane
- global memory vs shared memory vs registers
- warp-level communication with shuffle instructions
- block-level synchronization with `__syncthreads`
- reduction patterns for max and sum
- softmax numerical stability
- basic Nsight Compute profiling workflow

## Next Steps

- Add earlier versions: naive, shared-memory reduction, and warp-only reduction
- Add profiling screenshots and performance comparison tables
- Refactor repeated CUDA error checking
- Compare different block sizes such as 128, 256, 512, and 1024
