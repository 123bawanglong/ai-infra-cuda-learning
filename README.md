# AI Infra CUDA Learning

CUDA kernel implementations and optimization notes for AI infrastructure and
LLM inference workloads. The repository follows an optimization-first path:
start with a correct baseline, identify bottlenecks, and introduce CUDA
optimizations step by step.

## GEMM 优化实验：从 Naive 到 Warp Tiling

在 `M=N=K=4096` 的实验记录中，调参后的 V7 相对 V1 达到约 **8.45×** 加速。

| 版本 | 耗时 | 计算吞吐量 |
|---|---:|---:|
| Naive（V1） | 44.26 ms | 3.11 TFLOP/s |
| V7（调参后） | 5.24 ms | 26.23 TFLOP/s |

下面按版本保留实验分析和 Nsight Compute 截图，点击展开。
<details>
<summary><strong>V1 实验分析与 Profiling 截图</strong></summary>

### roofline

![image-20260918000003623](images/image-20260918000003623.png)

![image-20260918000011345](images/image-20260918000011345.png)

V1 的 FP32 吞吐只有理论峰值约 5.83%，L2 和 DRAM 都远未达到带宽 Roof，L1/TEX 相对更接近 Roof，说明 Naive GEMM 存在较重的近端 Load 流量和较低的数据复用，但又没有真正打满 L1 带宽。因此目前只能判断整体执行效率较低

### speed of light

`L1/TEX ≈ 97.09%`，而 `L2 ≈ 31.43%`、`DRAM ≈ 18.49%`，说明主要压力集中在靠近 SM 的 Load/L1/TEX 路径，`Compute(SM) ≈ 93.29%` 表示 SM 很忙，但是 Roofline 显示 FP32 只有约 5.83% 峰值，所以说明 GPU 很忙，但是计算都没花在 FMA 上，而可能是花在取数、Load/Store 相关执行和地址计算上。

![image-20260917235940083](images/image-20260917235940083.png)

### Compute Workload Analysis

![image-20260917235920664](images/image-20260917235920664.png)

LSU=93.29%，说明主要是 Load/Store 相关执行流水线很忙，而不是 FP32 FMA 很忙。

### Memory Workload Analysis

![image-20260917235857282](images/image-20260917235857282.png)

ncu警告：

L1TEX Global Load Access Pattern
Estimated Speedup: 16.33%

也就是：

thread0 → A[0]
thread1 → A[64]
thread2 → A[128]
thread3 → A[192]
...

等于约16.33%带宽/管线工作是在搬没用到的数据。

这还会使得内存访问量增加，计算强度变小

### Scheduler Statistics

![image-20260917235816516](images/image-20260917235816516.png)

Active Warps很多，但是Eligible warp少，说明可以发射的很少，还有就是No Eligible 很高 约 73.39% 的周期，Scheduler 手里一个可以发射的 Warp 都没有。

### Warp State Statistics

![image-20260917235751235](images/image-20260917235751235.png)

LG Throttle 占比大，说明 Global/Local Memory 请求过多。

### SASS sourse

![image-20260917235707414](images/image-20260917235707414.png)

Attributed Stalls（归因停顿）里，`LDG` = Load Global（从全局内存加载）占比很高，IMAD.WIDE（整数计算指令）占比却很低，FFMA（浮点乘加）也很低

### 诊断

LSU很高，说明主要compute在load/store上，FMA低，Warp State Statistics 中 LG Throttle 是最主要的 Stall；SASS 中热点又集中在大量 `LDG.E` 全局加载指令，考虑引入shared。

</details>

<details>
<summary><strong>V2 实验分析与 Profiling 截图</strong></summary>

### roofline

![image-20260918001701144](images/image-20260918001701144.png)

l1的计算强度向右移，说明数据传输量较大幅度减少，引入shared起了作用。

### speed of light

情况和v1差不多 l1依旧很高，compute也很高。

### Compute Workload Analysis

![image-20260918003624439](images/image-20260918003624439.png)

LSU任然很忙，但是构成不一样了：

| 动态 warp 指令数 | V1       | V2       |
| ---------------- | -------- | -------- |
| Global load      | 42.95 亿 | 3.36 亿  |
| Shared load      | 0        | 26.84 亿 |
| Shared store     | 0        | 3.36 亿  |

### Memory Workload Analysis

![image-20260918004924511](images/image-20260918004924511.png)

global load确实少了很多

### Scheduler Statistics

![image-20260918005004170](images/image-20260918005004170.png)

no eligible得到了改善，有更多的周期能发射的warp变多了

### Warp State Statistics

![image-20260918005534024](images/image-20260918005534024.png)

LG几乎直接消失了 现在mio成为最大等待，结合之前 No Eligible 仍有约 53%，也就是说现在重点处理shared的load store，bank conflict。

### SASS sourse

![image-20260918161610664](images/image-20260918161610664.png)

![image-20260918161710391](images/image-20260918161710391.png)

| 类别        | 常见指令        | 含义                                              |
| ----------- | --------------- | ------------------------------------------------- |
| Global 访存 | `LDG` / `LDG.E` | 从 global memory 加载到寄存器                     |
| Global 访存 | `STG` / `STG.E` | 将寄存器数据写入 global memory                    |
| Shared 访存 | `LDS`           | 从 shared memory 加载到寄存器                     |
| Shared 访存 | `STS`           | 将寄存器数据写入 shared memory                    |
| Local 访存  | `LDL` / `STL`   | 访问线程私有的 local memory，可能与寄存器溢出有关 |
| 浮点计算    | `FFMA`          | FP32 融合乘加：`a × b + c`                        |
| 浮点计算    | `FADD` / `FMUL` | FP32 加法 / 乘法                                  |
| 整数计算    | `IADD3`         | 整数加法，常用于索引和地址计算                    |
| 整数计算    | `IMAD`          | 整数乘加                                          |
| 整数计算    | `IMAD.WIDE`     | 生成较宽的整数结果，常用于构造 64 位地址          |
| 数据移动    | `MOV`           | 将数据或立即数移动到寄存器                        |
| 索引获取    | `S2R`           | 读取线程编号、block 编号等特殊寄存器              |
| 条件判断    | `ISETP`         | 整数比较，并设置谓词                              |
| 控制流      | `BRA` / `EXIT`  | 跳转 / 线程退出                                   |
| 同步        | `BAR.SYNC`      | block 内同步，常对应 `__syncthreads()`            |

可以看出lds 比较多，而且检查的`LDS` 有 12,531 个未发射采样，其中 12,446 个是 MIO Throttle，也就是说mio队列太满导致无法发射，是v2受阻的重要原因。

### 诊断

也就可以考虑在完成相同 FMA 数量的前提下，减少所需 shared 加载，能否降低 MIO 等待并缩短耗时。

所以考虑加入thread tile。

</details>

<details>
<summary><strong>V3 实验分析与 Profiling 截图</strong></summary>

### roofline

![image-20260918170416153](images/image-20260918170416153.png)

吞吐有明显提高 性能也有比较显著的提升

### speed of light

![image-20260918171927907](images/image-20260918171927907.png)

l1 cache还是比较高

### Compute Workload Analysis

![image-20260918172817660](images/image-20260918172817660.png)

![image-20260918172723195](images/image-20260918172723195.png)

FMA更活跃，LG比较高，显示LDS比较多

### Memory Workload Analysis

![image-20260918173347689](images/image-20260918173347689.png)

![image-20260918173859040](images/image-20260918173859040.png)

shared load大幅度减少 FFMA次数不变

### Scheduler Statistics

![image-20260918174341670](images/image-20260918174341670.png)

V3 用更少的驻留 warp，获得了更好的指令发射情况

### Warp State Statistics

![image-20260918174527890](images/image-20260918174527890.png)

MIO有比较大幅度优化，但还是主要的压力来源

### SASS sourse

![image-20260918174850359](images/image-20260918174850359.png)

![image-20260918181726812](images/image-20260918181726812.png)

LDS还是占主要部分。L1 Conflicts Shared N-Way` = `1说明bank conflict也少

### 诊断

LSU很高 SASS里load/store多 考虑向量化访存

</details>

<details>
<summary><strong>V4 实验分析与 Profiling 截图</strong></summary>

### roofline

![image-20260918191031591](images/image-20260918191031591.png)

吞吐有明显提高

### speed of light

![image-20260918191259375](images/image-20260918191259375.png)

Memory 汇总指标更接近其峰值，需要优先查看访存相关的细分指标。

### Compute Workload Analysis

![image-20260918191149861](images/image-20260918191149861.png)

有比较明显的优化

### Memory Workload Analysis

![image-20260918192057951](images/image-20260918192057951.png)

提示有 bank conflict

### Scheduler Statistics

![image-20260918192234950](images/image-20260918192234950.png)



### Warp State Statistics

![image-20260918192531269](images/image-20260918192531269.png)

mio明显下降 stall  Long Scoreboard 上升：等待访存结果的依赖更突出 Short Scoreboard 上升：短延迟操作的结果依赖更突出  Barrier 上升：warp 等待同一个 block 的其他 warp 到达同步点更多

### SASS sourse

![image-20260918193905634](images/image-20260918193905634.png)

### 诊断

A转置写入出现bank conflict，需要解决，float4 一定程度上导致了long scoreboard 考虑引入双缓冲和异步

</details>

<details>
<summary><strong>V5 实验分析与 Profiling 截图</strong></summary>

### roofline

![image-20260918200748407](images/image-20260918200748407.png)

### speed of light

![image-20260918201036083](images/image-20260918201036083.png)

DRAM有提高

### Compute Workload Analysis

![image-20260918201158928](images/image-20260918201158928.png)

### Memory Workload Analysis

![image-20260918202512359](images/image-20260918202512359.png)

bank conflict没了

### Warp State Statistics

![image-20260918202545056](images/image-20260918202545056.png)

### Warp State Statistics

![image-20260918202554444](images/image-20260918202554444.png)

long scoreboard大幅下降 异步起到了效果 Short Scoreboard 仍是较突出的等待项

### SASS sourse

![image-20260918203400554](images/image-20260918203400554.png)

sum[l][j]+=a_frag[l]*b_frag[j]，有 2,074 个 Short Scoreboard 未发射样本，这里的乘加在等待前面 shared load 的结果，数据尚未就绪时无法发射。

</details>

<details>
<summary><strong>V6 实验分析与 Profiling 截图</strong></summary>

在v5的异步拷贝和shared双缓冲基础上，给 `a_frag`、`b_frag` 加入寄存器双缓冲，提前加载下一轮要用的数据，再计算当前轮。M=N=K=4096，BM=32、BN=128、BK=32、TM=TN=4，与v5相同。

### roofline

![image-20260918221201859](images/image-20260918221201859.png)

### speed of light

![image-20260918221126973](images/image-20260918221126973.png)

### Compute Workload Analysis

![image-20260918221115905](images/image-20260918221115905.png)

### Memory Workload Analysis

![image-20260918221058411](images/image-20260918221058411.png)

### Scheduler Statistics

![image-20260918221030474](images/image-20260918221030474.png)

### Warp State Statistics

![image-20260918220947388](images/image-20260918220947388.png)

Short Scoreboard实现了有所下降。

### SASS sourse

![image-20260918220909523](images/image-20260918220909523.png)

可以看到 `LDS.128 R28, [R42+0x1a00]` 与使用R28的 `FFMA R25, R5, R28, R50` 之间隔着一组计算，加载和使用之间留出了时间。

但这条FFMA仍有1,999个Short Scoreboard未发射样本，说明等待还没有完全隐藏。不同版本的单条指令采样数不能直接当成加速比，主要结合前面Warp State的整体指标判断。

### 诊断

这次采样中，Short Scoreboard和No Eligible有小幅下降，shared加载总数不变，整体FP32利用率变化不大。寄存器预取有缓解等待的迹象，但还不能据此认定性能有明显提升。

</details>

<details>
<summary><strong>V7 实验分析与 Profiling 截图</strong></summary>

### roofline

![image-20260918223930255](images/image-20260918223930255.png)

### speed of light

![image-20260918224040916](images/image-20260918224040916.png)

### Compute Workload Analysis

![image-20260918224017199](images/image-20260918224017199.png)

### Memory Workload Analysis

![image-20260918224058811](images/image-20260918224058811.png)

### Scheduler Statistics

![image-20260918224109814](images/image-20260918224109814.png)

### Warp State Statistics

![image-20260918224125829](images/image-20260918224125829.png)

最终在M=N=K=4096的场景下达到cublas的84.1%，调参后的V7 约为 naive（V1）的 8.45 倍性能。

| 版本        | 耗时     | 计算吞吐量    |
| ----------- | -------- | ------------- |
| naive（V1） | 44.26 ms | 3.11 TFLOP/s  |
| V7调参后    | 5.24 ms  | 26.23 TFLOP/s |

- **加速比**：`44.26 ÷ 5.24 ≈ 8.45×`
- **耗时降低**：约 **88.16%**

</details>

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
├── images/          # GEMM profiling screenshots
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

Prepare `input_320x4096.txt` with `320 4096` on the first line, followed by
320 × 4096 floating-point values in row-major order, separated by whitespace.

```bash
mkdir -p reports
ncu --set full --force-overwrite \
  -o reports/softmax_v3_320x4096 \
  build/softmax_v3 < input_320x4096.txt
```

Compare duration, memory bandwidth, global-memory efficiency, shared-memory
bank conflicts, occupancy, register use, and warp efficiency.

## Current Scope

- Kernels use FP32, except GEMM Tensor Core multiplication uses TF32 with FP32 accumulation.
- FlashAttention models one head without causal masking, batching, or mixed precision.
- The GEMM experiment section preserves recorded results; a reproducible hardware, compiler, and code-version mapping remains to be documented.
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
