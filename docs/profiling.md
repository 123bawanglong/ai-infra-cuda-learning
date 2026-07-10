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

```bash
python3 scripts/gen_input.py --rows 320 --cols 4096 > input_320x4096.txt

nvcc -lineinfo src/softmax_shared.cu -o softmax_shared
nvcc -lineinfo src/softmax_warp.cu -o softmax_warp
nvcc -lineinfo src/softmax_warp_shared.cu -o softmax_warp_shared

ncu --set basic --force-overwrite -o reports/softmax_shared_320x4096 ./softmax_shared < input_320x4096.txt
ncu --set basic --force-overwrite -o reports/softmax_warp_320x4096 ./softmax_warp < input_320x4096.txt
ncu --set basic --force-overwrite -o reports/softmax_warp_shared_320x4096 ./softmax_warp_shared < input_320x4096.txt
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
| `softmax_shared` | 256 | TBD | TBD | TBD | Shared-memory block reduction |
| `softmax_warp` | 32 | TBD | TBD | TBD | Warp-only shuffle reduction |
| `softmax_warp_shared` | 128 | TBD | TBD | TBD | Multi-warp shuffle + shared memory |
