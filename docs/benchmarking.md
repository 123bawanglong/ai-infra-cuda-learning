# Benchmarking Template

Use this template for every performance claim so results remain reproducible.

## Environment

| Item | Value |
|---|---|
| GPU | |
| GPU memory | |
| Driver version | |
| CUDA Toolkit | |
| Compiler flags | `-O3 -lineinfo -arch=sm_XX` |
| Power mode / clock settings | |
| Operating system | |

## Method

1. Run at least 10 warm-up iterations.
2. Measure at least 100 timed iterations with CUDA events.
3. Report the median and a tail statistic such as P95.
4. Validate each version against the same CPU, cuBLAS, or framework reference.
5. Keep shapes, data type, launch configuration, and flags identical across versions.

## Results

| Operator | Shape | Version | Median (ms) | P95 (ms) | Bandwidth / TFLOP/s | Correct |
|---|---|---|---:|---:|---:|---|
| | | | | | | |

## Nsight Compute Notes

| Metric | Baseline | Optimized | Interpretation |
|---|---:|---:|---|
| Duration | | | |
| DRAM throughput | | | |
| L1/TEX hit rate | | | |
| Shared-memory bank conflicts | | | |
| Achieved occupancy | | | |
| Registers per thread | | | |
| Warp execution efficiency | | | |

## Reproduction Command

```bash
ncu --set full --force-overwrite -o reports/<report-name> <executable> < <input-file>
```

Record the exact Git commit used for the measurement:

```text
Commit:
```
