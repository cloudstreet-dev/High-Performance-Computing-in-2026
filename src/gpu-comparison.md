# GPU Framework Shootout

The three previous chapters made the case for each platform on its own terms. This chapter makes them compete. The goal is a practical decision framework — not a winner-declaration, because the winner depends on your constraints — but a structured analysis of what you gain and what you give up with each choice.

## The Decision Matrix

Before benchmarks, acknowledge the meta-question: are you choosing a programming model or a hardware platform? They are coupled but not identical.

- **CUDA** = NVIDIA hardware + CUDA programming model + full ecosystem
- **HIP/ROCm** = NVIDIA or AMD hardware + HIP programming model + ROCm ecosystem (AMD) or CUDA ecosystem (NVIDIA)
- **Metal** = Apple Silicon hardware + Metal/MSL programming model + Apple ecosystem

You can write HIP code that runs on NVIDIA hardware via the CUDA backend. You cannot run Metal on non-Apple hardware. You can write CUDA code that runs on AMD via porting tools, but it is a one-time port, not ongoing portability.

## Side-by-Side: Language and Syntax

A simple reduction kernel in all three languages reveals where the models align and where they diverge.

### CUDA

```c
#include <cub/cub.cuh>

// Using CUB for a proper reduction
__global__ void reduce_sum(const float* input, float* output, int n) {
    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage temp;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? input[idx] : 0.0f;

    float block_sum = BlockReduce(temp).Sum(val);

    if (threadIdx.x == 0)
        atomicAdd(output, block_sum);
}

// Launch:
int threads = 256;
int blocks = (n + threads - 1) / threads;
float* d_output;
cudaMalloc(&d_output, sizeof(float));
cudaMemset(d_output, 0, sizeof(float));
reduce_sum<<<blocks, threads>>>(d_input, d_output, n);
cudaDeviceSynchronize();
```

### HIP (ROCm)

```cpp
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>

__global__ void reduce_sum(const float* input, float* output, int n) {
    using BlockReduce = hipcub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage temp;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? input[idx] : 0.0f;

    float block_sum = BlockReduce(temp).Sum(val);

    if (threadIdx.x == 0)
        atomicAdd(output, block_sum);
}

// Launch — identical to CUDA
int threads = 256;
int blocks = (n + threads - 1) / threads;
float* d_output;
hipMalloc(&d_output, sizeof(float));
hipMemset(d_output, 0, sizeof(float));
hipLaunchKernelGGL(reduce_sum, dim3(blocks), dim3(threads), 0, 0,
                    d_input, d_output, n);
hipDeviceSynchronize();
```

The mechanical similarity is intentional. `hipcub` is a port of CUB to HIP; the interface is the same. This is the best-case scenario for HIP portability: the code is nearly identical.

### Metal (MSL)

```cpp
// MSL kernel — reduction using threadgroup memory
#include <metal_stdlib>
using namespace metal;

kernel void reduce_sum(
    device const float* input  [[buffer(0)]],
    device atomic_float* output [[buffer(1)]],
    constant uint& n            [[buffer(2)]],
    threadgroup float* shared   [[threadgroup(0)]],
    uint local_idx              [[thread_position_in_threadgroup]],
    uint global_idx             [[thread_position_in_grid]],
    uint group_size             [[threads_per_threadgroup]])
{
    shared[local_idx] = (global_idx < n) ? input[global_idx] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (local_idx < stride)
            shared[local_idx] += shared[local_idx + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (local_idx == 0)
        atomic_fetch_add_explicit(output, shared[0], memory_order_relaxed);
}
```

The Metal version is more verbose at the kernel signature level (every parameter requires attribute decoration) and uses different synchronization primitives. The reduction algorithm is the same; the idioms differ.

## Performance Comparison

Benchmarking GPU kernels is an exercise in controlled disappointment: results vary significantly by hardware generation, problem size, precision, memory access pattern, and which version of which library you compiled against. The numbers below are indicative, not contractual.

### GEMM (Matrix Multiplication)

Dense GEMM at FP32, 4096×4096 matrices, measured in TFLOPS effective throughput:

| Platform | Hardware | Library | Effective TFLOPS |
|----------|----------|---------|-----------------|
| CUDA | H100 SXM5 | cuBLAS 12 | ~60 TFLOPS |
| HIP | MI300X | rocBLAS 6.x | ~55 TFLOPS |
| HIP | H100 (CUDA backend) | cuBLAS | ~59 TFLOPS |
| Metal | M2 Ultra | MPS | ~12 TFLOPS |
| Metal | M3 Max | MPS | ~8 TFLOPS |

The H100/MI300X gap in GEMM has narrowed considerably in ROCm 6.x. MI300X's memory bandwidth advantage does not help large GEMMs much (they are compute-bound at this size), but its 192 GB capacity allows working at much larger problem sizes without spilling to host memory.

### Memory Bandwidth (Stream Benchmark)

Sustained memory bandwidth, measured in GB/s:

| Platform | Hardware | Achieved BW | % of Peak |
|----------|----------|-------------|-----------|
| CUDA | H100 SXM5 | ~3.1 TB/s | ~92% |
| HIP | MI300X | ~4.9 TB/s | ~92% |
| Metal | M2 Ultra | ~720 GB/s | ~90% |
| Metal | M3 Max | ~360 GB/s | ~90% |

MI300X's bandwidth advantage is real and consistent. For memory-bound workloads (element-wise ops, reductions, gather/scatter), MI300X outperforms H100 by roughly 45%.

### Inference Throughput (LLM, Large Batch)

For transformer inference at FP16, a representative large language model (70B parameters):

| Setup | Tokens/sec (batch=32) |
|-------|----------------------|
| 1× H100 SXM5 (requires quantization or offload) | ~3,200 |
| 1× MI300X (full FP16, fits in 192 GB) | ~4,100 |
| 2× H100 NVLink (full FP16) | ~5,800 |
| M2 Ultra (full FP16, fits in 192 GB) | ~320 |

The MI300X's single-card advantage for large model inference is significant. Not needing to split across GPUs eliminates the NVLink/XGMI communication overhead and simplifies deployment. The M2 Ultra — despite also fitting the model in unified memory — is outpaced on raw throughput; it is relevant for inference where latency and power matter more than throughput.

## Ecosystem Comparison

| Capability | CUDA | ROCm/HIP | Metal |
|-----------|------|----------|-------|
| PyTorch | First-class | Supported, slight lag | Via MPS backend (limited) |
| JAX | First-class | Supported | Experimental |
| TensorFlow | First-class | Supported | Via Metal plugin |
| Triton kernels | First-class | ROCm backend (growing) | No |
| BLAS library | cuBLAS | rocBLAS | MPS |
| Profiler quality | Excellent (ncu, nsys) | Good (Omniperf) | Excellent (Instruments) |
| Community knowledge | Vast | Growing | Limited (HPC) |
| Debugger | cuda-gdb, compute-sanitizer | rocgdb | Xcode Metal Debugger |
| Docker support | `--gpus all` | `--device /dev/kfd /dev/dri` | macOS only |
| CI/CD integration | Mature | Workable | macOS runners only |

## The Portability Spectrum

GPU portability is a spectrum, not a binary:

**Write-once, run-anywhere**: Does not exist at the kernel level in 2026. WebGPU is the closest thing (Metal/Vulkan/D3D12 backends, runs in browsers), and it has restrictions that make it unsuitable for serious HPC.

**Portable high-level code**: Possible with frameworks that abstract hardware (PyTorch, JAX, MLX). You write in Python/Python-adjacent, the framework dispatches to the right backend. This works well for ML workloads and reasonably well for array-style numerical computing. It does not work if you need to write custom kernels.

**Portable custom kernels via HIP**: HIP code compiles for both NVIDIA and AMD. This is the practical portability option for kernel authors. It requires maintaining one codebase but accepting that architecture-specific optimizations (wavefront-64 vs warp-32) may need conditional compilation.

**Platform-specific kernels with shared logic**: The most common real-world pattern for performance-critical code. Write CUDA and HIP separately, share the algorithmic logic in header files or via an abstraction layer. More code, better per-platform performance.

## Decision Guide

**Choose CUDA if:**
- You need maximum ecosystem compatibility (third-party libraries, tutorials, hiring)
- You are running on NVIDIA hardware and do not need AMD compatibility
- You are working on ML training at scale (NCCL, cuDNN, cuBLAS are unmatched)
- Your team already knows CUDA and the portability cost is not justified

**Choose HIP/ROCm if:**
- You need code that runs on both NVIDIA and AMD hardware
- Your deployment hardware is or will be AMD (MI300X, future Instinct parts)
- The MI300X memory capacity or bandwidth is the decisive factor for your workload
- You have a principled preference for open-source GPU software stacks
- You are working with cloud providers who have made AMD competitive on price

**Choose Metal if:**
- Your deployment target is Apple Silicon (macOS application, Apple-specific service)
- You are developing and debugging on a MacBook and want GPU acceleration there
- The power-efficiency of Apple Silicon matters for your use case
- You are working with MLX and need to write custom kernels

**None of the above if:**
- You can express your workload in terms of existing library operations (cuBLAS, MPS, rocBLAS). The best GPU kernel is often the one you did not write.

## On the Fragmentation Tax

Every organization that uses more than one GPU platform pays a fragmentation tax. It manifests as:
- Duplicate kernel implementations that must be maintained in sync
- CI pipelines that test on multiple hardware targets
- Engineers who know platform A but not platform B, creating knowledge silos
- Library version skew where feature parity is not guaranteed across platforms

This tax is real, measurable in engineering time, and tends to grow as the codebase grows. Frameworks like PyTorch partially absorb the tax by presenting a unified API over multiple backends. But the moment you drop below the framework layer — which HPC work regularly requires — the tax reappears.

The honest answer to "which GPU platform should we use?" is often: pick one and commit, unless you have a specific, quantified reason to support multiple. The grass on the other side of the PCIe bus is sometimes genuinely greener, but moving there still costs moving expenses.
