# GPU Computing Fundamentals

Before writing a single line of CUDA, HIP, or Metal, it is worth understanding what a GPU actually is and why it behaves the way it does. The programming model makes more sense once you know what the hardware is doing, and most of the performance mistakes engineers make trace back to a mismatch between their mental model of the GPU and reality.

## The Wrong Mental Model

The common misconception is that a GPU is a fast CPU with many cores. It is not. A GPU is an architecture optimized for a very different workload profile, and the differences are more fundamental than clock speed or core count.

A modern high-end CPU — say, an AMD EPYC Genoa — might have 96 cores running at 3–4 GHz, each with large private caches, out-of-order execution engines, branch predictors, and all the machinery that makes single-threaded latency low. That CPU is designed to make any given thread of execution run as fast as possible, even if it means burning transistors on speculation, prefetching, and reordering instructions.

A GPU — say, an NVIDIA H100 SXM — has 16,896 CUDA cores organized into 132 Streaming Multiprocessors. Each individual core is simple. It does not speculate. It does not reorder instructions. It executes one instruction at a time in lockstep with 31 other cores in its warp. What the GPU does instead of latency reduction is latency hiding: it maintains thousands of in-flight warps simultaneously, and when one warp stalls waiting for memory, it switches instantly to another. Zero context-switch overhead. The latency is still there; the GPU just ignores it by having enough other work to do.

This is the fundamental tradeoff: **throughput vs. latency**.

## GPU Architecture Components

### Streaming Multiprocessors (SMs) / Compute Units (CUs) / GPU Cores

The top-level compute unit in NVIDIA GPUs is the Streaming Multiprocessor (SM). In AMD terminology it is a Compute Unit (CU). In Apple Silicon it is a GPU Core. The names differ; the concept is similar.

Each SM on an H100 contains:
- 128 CUDA cores (FP32 ALUs)
- 64 FP64 units (double-precision)
- 4 Tensor Cores (for matrix operations)
- 256 KB of register file
- 228 KB of shared memory / L1 cache
- Load/store units and special function units (SFUs)

The SM is the scheduling unit. It manages warps. It is the level at which occupancy — the fraction of maximum possible warps that are resident at once — is calculated.

### Warps and SIMT

NVIDIA executes threads in groups of 32 called **warps**. AMD's equivalent is a **wavefront** of 64 threads (though RDNA3 and later support both 32 and 64). This is the Single Instruction, Multiple Thread (SIMT) execution model.

Within a warp, all 32 threads execute the same instruction at the same time. When threads diverge — taking different branches based on per-thread data — the GPU executes both branches and masks off the inactive threads in each path. This is called **warp divergence**, and it is the most common source of GPU performance loss that programmers have direct control over.

If you have an `if/else` where half the threads take each branch, you have effectively halved your throughput for that section of code. If you can reorganize your data so that threads in the same warp make the same branch decision, you can recover that performance. This is one of the reasons GPU kernels often have awkward-looking data layouts.

### Memory Hierarchy

The memory hierarchy is where most GPU performance is won or lost.

**Global memory** (HBM on datacenter GPUs): This is your main GPU memory. H100 SXM5 has 80 GB of HBM3 with ~3.35 TB/s bandwidth. Fast by any CPU memory standard. Slow from the perspective of a compute unit sitting next to it wanting 100+ TB/s of throughput. Every kernel that runs on the GPU is, at some level, fighting to keep the math units fed faster than global memory can supply them.

**L2 cache**: Shared across all SMs. On H100, 50 MB. This is the critical staging area for data that multiple SMs access. If your working set fits in L2, your effective bandwidth approaches what you'd get on-chip.

**Shared memory / L1 cache**: This is per-SM, explicitly managed (for the shared memory portion), and fast — on the order of 19 TB/s for the H100. When you see `__shared__` in CUDA code, this is where data lives. The programmer controls what goes in shared memory and when, which is both the power and the burden of the model.

**Registers**: The fastest storage. Per-thread. On H100, each SM has 256 KB of register file shared across all resident warps. Register pressure — using too many registers per thread — reduces occupancy because fewer warps can be resident simultaneously.

```
Registers    ~19 TB/s (effectively infinite bandwidth, per-thread)
Shared Mem   ~19 TB/s (per-SM, programmer-managed)
L1 Cache     Merged with shared memory in recent NVIDIA designs
L2 Cache     ~12 TB/s (across all SMs)
HBM (VRAM)   ~3.35 TB/s (H100 SXM5)
PCIe/NVLink  ~900 GB/s (NVLink 4.0, bidirectional)
CPU DRAM     ~460 GB/s (DDR5-4800, 8-channel)
```

Latencies follow the inverse pattern. Registers are essentially free. Global memory accesses that miss all caches take 400–800 cycles. This is why the GPU needs thousands of in-flight warps just to keep its arithmetic units busy.

### Memory Coalescing

When threads in a warp access global memory, the hardware combines individual thread accesses into as few memory transactions as possible. If 32 threads each access a contiguous 4-byte element at addresses that span a 128-byte aligned region, that is one memory transaction. If they access random addresses scattered across memory, that is 32 separate transactions — 32x the bandwidth consumption, 32x the latency exposure.

**Coalesced access** is not optional. It is one of the first things to check when a kernel underperforms.

```c
// Coalesced: thread i accesses element i
float val = data[threadIdx.x + blockIdx.x * blockDim.x];

// Not coalesced: thread i accesses element i * stride (with large stride)
float val = data[(threadIdx.x + blockIdx.x * blockDim.x) * stride];
```

The second pattern can easily run at 1/32 of the bandwidth of the first.

## The Compute Throughput Problem

Modern GPUs are computationally powerful in a way that creates a new problem: it has become very easy to be memory-bound.

The H100 delivers ~67 TFLOPS of FP64 and ~989 TFLOPS of FP16. The memory bandwidth is ~3.35 TB/s. For an operation that reads A bytes and performs N floating-point operations, the arithmetic intensity is N/A FLOP/byte. Below the hardware's roofline — the ratio of peak compute to peak bandwidth — you are memory-bound. Above it, compute-bound.

For FP64: 67e12 / 3.35e12 ≈ 20 FLOP/byte is the crossover. A dense matrix-matrix multiply (GEMM) at large sizes achieves much higher arithmetic intensity (O(N) flops per O(N²) memory reads as tile sizes grow), which is why GEMM is one of the few operations that can actually saturate a GPU's compute units. A simple element-wise operation on a large array (read, compute, write) has an arithmetic intensity of roughly 1–2 FLOP/byte — deep in memory-bound territory regardless of how fast the ALUs are.

This is not unique to GPUs; the same analysis applies to CPUs. It is just more dramatic on GPUs where the compute-to-bandwidth ratio is higher.

## The CPU-GPU Interface

Every GPU kernel invocation involves the CPU. The CPU launches the kernel, manages memory transfers, and handles synchronization. The latency of this interface matters.

**PCIe bandwidth** between CPU and GPU is the bottleneck for workloads that move data frequently across the interface. PCIe 5.0 x16 provides ~64 GB/s bidirectional, compared to 3350 GB/s of HBM bandwidth on-device. The rule is simple: data that moves across PCIe frequently should not be on the GPU.

**NVLink** (NVIDIA) and **XGMI** (AMD) provide high-bandwidth GPU-to-GPU interconnects — up to 900 GB/s for NVLink 4.0. These enable multi-GPU workloads to share memory across GPUs with substantially lower bandwidth penalty than going through host memory. On a DGX H100, 8 GPUs are connected in an all-reduce-friendly topology via NVLink, enabling collective operations (AllReduce, AllGather) that are fast enough to make distributed training practical at scale.

**CUDA Unified Memory** (and its equivalents in ROCm and Metal) provides a programming abstraction that hides explicit data transfers. The hardware migrates pages between CPU and GPU on demand. This is convenient for getting code working. It is not an optimization strategy; it is a tool for avoiding having to think about data movement, which carries the usual costs of not thinking about data movement.

## Occupancy and Its Limits

Occupancy is the fraction of maximum warps that are simultaneously resident on an SM. High occupancy enables better latency hiding. But high occupancy is not the same thing as high performance.

The limits on occupancy are:

1. **Register pressure**: More registers per thread → fewer threads per SM → lower occupancy
2. **Shared memory usage**: More shared memory per block → fewer blocks per SM → lower occupancy
3. **Block size**: Blocks must be sized such that they can fill an SM without remainder

Tools like `nvcc --ptxas-options=-v` (CUDA) and `rocprof` (ROCm) report register usage. The CUDA occupancy calculator (also available as a spreadsheet from NVIDIA) will tell you the theoretical maximum occupancy given your kernel's resource usage.

The caveat: a kernel with 100% theoretical occupancy and poor memory access patterns will underperform a kernel with 50% occupancy and well-optimized memory access. Occupancy is a prerequisite for latency hiding, not a substitute for correct behavior.

## A Framework for Thinking About GPU Performance

When a GPU kernel is slow, the cause is almost always one of the following:

1. **Memory bandwidth saturation** — the kernel is reading/writing too much data relative to computation (low arithmetic intensity). Profile with `ncu` or `rocprof` and look at memory throughput vs. peak.

2. **Warp divergence** — threads in a warp are taking different code paths. Profile for "active warps" vs "eligible warps."

3. **Uncoalesced memory access** — threads in a warp are not accessing contiguous memory. L2 cache hit rate will be low; memory transactions per warp will be high.

4. **Occupancy too low** — not enough warps to hide latency. Register or shared memory usage is too high. Reduce resource usage per thread or accept that you need a different algorithmic approach.

5. **Kernel launch overhead** — too many small kernel launches. Launch overhead is on the order of microseconds; for work measured in tens of microseconds, this matters. Fuse kernels or use CUDA graphs.

6. **PCIe transfer overhead** — moving data between CPU and GPU too frequently. Restructure to do more work on-device between transfers.

These categories apply equally to CUDA, ROCm, and Metal — the profiling tools differ but the underlying physics does not.

## The Next Chapters

The following chapters go deep on each of the three major GPU programming ecosystems: CUDA (Chapter 3), ROCm/HIP (Chapter 4), and Metal (Chapter 5). Each covers the programming model, memory management, kernel writing, and performance analysis tools specific to that platform.

If you already understand one platform well, you can read the others with the mental model from this chapter as context and note what is the same and what differs. The underlying physics is the same. The tradeoffs are the same. The syntax and toolchain are where they diverge.
