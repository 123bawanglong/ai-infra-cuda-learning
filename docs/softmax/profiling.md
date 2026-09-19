# Profiling Notes

This document records how I analyze the CUDA softmax kernels with Nsight Compute.

## Test Shape

Default profiling shape:

```text
N = 320
C = 4096
```

This shape is large enough to make row-wise parallelism visible while keeping the experiment easy to reproduce on a local GPU.

## Commands

Prepare `input_320x4096.txt` with `320 4096` on the first line, followed by
320 × 4096 floating-point values in row-major order, separated by whitespace.

```bash
mkdir -p build reports
nvcc -O3 -lineinfo src/softmax/softmax_v1_shared_memory.cu -o build/softmax_v1
nvcc -O3 -lineinfo src/softmax/softmax_v2_warp_shuffle.cu -o build/softmax_v2
nvcc -O3 -lineinfo src/softmax/softmax_v3_multi_warp_shared.cu -o build/softmax_v3

ncu --set basic --force-overwrite -o reports/softmax_v1_320x4096 build/softmax_v1 < input_320x4096.txt
ncu --set basic --force-overwrite -o reports/softmax_v2_320x4096 build/softmax_v2 < input_320x4096.txt
ncu --set basic --force-overwrite -o reports/softmax_v3_320x4096 build/softmax_v3 < input_320x4096.txt
```

## Analysis Checklist

- Check kernel duration and compare versions under the same input shape.
- Check block size and grid size.
- Check memory throughput and compute throughput.
- Check active warps per scheduler.
- Check stall reasons, especially long scoreboard stalls and synchronization stalls.
- Connect the stall reason back to source code using `-lineinfo`.

## Expected Learning Conclusion

The warp-only version reduces shared-memory synchronization overhead, but it also restricts one block to one warp. For large `C`, this can reduce row-level parallelism too much. The multi-warp version restores more parallelism while still using shuffle instructions inside each warp.

## Results

Fill this table after collecting Nsight Compute results on the target machine:

| Kernel | Block Size | Duration | Compute Throughput | Memory Throughput | Notes |
|---|---:|---:|---:|---:|---|
| `softmax_v1` | 256 | TBD | TBD | TBD | Shared-memory block reduction |
| `softmax_v2` | 32 | TBD | TBD | TBD | Warp-only shuffle reduction |
| `softmax_v3` | 128 | TBD | TBD | TBD | Multi-warp shuffle + shared memory |
