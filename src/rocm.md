# ROCm: AMD's Open Platform

ROCm (Radeon Open Compute) has had a difficult adolescence. For most of its life — releases 1.x through 4.x — it was technically promising and practically painful. The installation process was fragile, hardware support was narrow, the ecosystem was sparse, and asking anyone to deploy it in production required either genuine idealism about the open-source GPU future or a procurement decision that precluded NVIDIA.

ROCm 5.x (2022–2023) was where things changed. ROCm 6.x continued the maturation. By 2025, it reached a state where the question shifted from "can we make this work?" to "is this the right tool for this job?" — which is the only question worth asking about any mature tool. This chapter answers that question with specifics.

## HIP: The Portability Layer

Before the ROCm platform itself, it is useful to understand HIP, because HIP is how most CUDA code migrates to AMD hardware.

**HIP (Heterogeneous-compute Interface for Portability)** is a C++ runtime API and kernel language that is syntactically near-identical to CUDA. A valid HIP program can compile and run on NVIDIA GPUs (via the CUDA backend) or AMD GPUs (via the ROCm backend) with minimal or no source changes. Most of the CUDA concepts map directly:

| CUDA | HIP |
|------|-----|
| `cudaMalloc` | `hipMalloc` |
| `cudaMemcpy` | `hipMemcpy` |
| `__global__` | `__global__` |
| `threadIdx.x` | `threadIdx.x` |
| `cudaDeviceSynchronize` | `hipDeviceSynchronize` |
| `nvcc` | `hipcc` |

The mechanical translation from CUDA to HIP can be done with the `hipify-perl` or `hipify-clang` tools:

```bash
hipify-clang my_cuda_kernel.cu --cuda-gpu-arch=sm_80 -- -I/usr/local/cuda/include
```

This produces HIP source that compiles with `hipcc`. The tool handles the syntactic translation correctly in most cases. What it does not handle: CUDA-specific library calls (cuBLAS → rocBLAS, cuDNN → MIOpen), NVIDIA-specific intrinsics with no AMD equivalent, and architecture-specific tuning that worked well on NVIDIA and may not on AMD.

### HIP Kernel Example

```cpp
#include <hip/hip_runtime.h>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int n = 1 << 20;
    const size_t size = n * sizeof(float);

    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, size);
    hipMalloc(&d_b, size);
    hipMalloc(&d_c, size);

    // ... host allocation, initialization, memcpy ...

    dim3 block(256);
    dim3 grid((n + 255) / 256);
    hipLaunchKernelGGL(vector_add, grid, block, 0, 0, d_a, d_b, d_c, n);

    hipDeviceSynchronize();

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
}
```

`hipLaunchKernelGGL` is the HIP kernel launch syntax. The `<<<...>>>` syntax also works with `hipcc` but the explicit form is preferred for portability with strict C++ compilers.

## AMD GPU Architecture

Understanding the differences between AMD and NVIDIA microarchitecture helps explain where ROCm code needs tuning beyond a mechanical CUDA port.

### Compute Units and Wavefronts

AMD Compute Units (CUs) are conceptually similar to NVIDIA SMs but differ in important details. On RDNA3 (RX 7000 series) and CDNA3 (Instinct MI300 series), the wavefront size is 64 threads — double NVIDIA's warp size of 32. RDNA3 also supports wave32 mode.

The larger wavefront size means:
- More register file per wavefront (64 threads worth vs 32)
- Branch divergence costs more when it occurs (64 masked threads vs 32)
- Warp-shuffle-equivalent operations (`__shfl_sync` equivalent in HIP) operate on 64 lanes

Code tuned for 32-wide warps may not be optimal on 64-wide wavefronts without adjustment to block sizes and reduction patterns.

### MI300X: The Datacenter Part

The AMD Instinct MI300X is the primary AMD competitor to the H100 in 2025–2026. Key specs:

- 192 GB of HBM3 unified memory (vs. H100's 80 GB)
- ~5.3 TB/s memory bandwidth (vs. H100's 3.35 TB/s)
- 304 compute units, ~164 TFLOPS FP64
- 5632 AI accelerators (matrix cores), up to ~1.3 PFLOPS FP8

The MI300X's headline advantage is memory capacity. For inference workloads where model weight size is the binding constraint — large language models, large multimodal models — fitting more of the model on a single device with no PCIe data movement is genuinely valuable. A single MI300X can hold a 70B parameter model in FP16 with room for KV cache, where an H100 requires multiple devices or quantization.

The memory bandwidth advantage (5.3 TB/s vs 3.35 TB/s) is also relevant for memory-bound operations. Inference is typically memory-bound for batch sizes where the arithmetic intensity is low; higher bandwidth directly translates to higher token throughput.

## The ROCm Stack

ROCm is not just HIP. It is a full software stack:

```
Your Code
  ↓
HIP / OpenCL / OpenMP offload
  ↓
ROCr (ROCm Runtime)
  ↓
HSA (Heterogeneous System Architecture)
  ↓
AMDKFD (kernel driver) / AMDGPU (DRM driver)
  ↓
AMD GPU Hardware
```

The key components:

**rocBLAS**: AMD's BLAS implementation. Generally competitive with cuBLAS for GEMM on MI-series hardware. Some operations have parity, some do not; benchmark for your specific operation.

**MIOpen**: AMD's equivalent to cuDNN. Covers convolutions, attention, pooling, batch normalization. Performance varies by operation and hardware generation. The MI300X saw significant MIOpen improvements in ROCm 6.x.

**rocFFT**: FFT operations. Comparable to cuFFT in most scenarios.

**RCCL**: ROCm Collective Communications Library — the AMD equivalent of NCCL. Implements AllReduce, Broadcast, etc. over AMD's XGMI interconnect. On systems with multiple MI300X connected via Infinity Fabric, this enables multi-GPU training.

**rocSPARSE**: Sparse linear algebra.

**Composable Kernel (CK)**: AMD's equivalent to CUTLASS — template-based high-performance kernels for GEMM and related operations, designed to express operations that exploit AMD-specific hardware (matrix cores, wavefront operations).

## Installation and Environment

ROCm's installation story has improved substantially from its early years but remains more involved than CUDA.

```bash
# Ubuntu 22.04 / 24.04
sudo apt-get update
sudo apt-get install -y rocm

# Verify installation
rocminfo
rocm-smi

# Check GPU visibility
/opt/rocm/bin/rocminfo | grep -A5 "Agent"
```

Environment variables that matter:

```bash
export ROCM_PATH=/opt/rocm
export HIP_PLATFORM=amd          # or nvidia when testing HIP on CUDA
export GPU_TARGETS=gfx942        # MI300X target architecture
export HIP_VISIBLE_DEVICES=0,1   # GPU selection (analogous to CUDA_VISIBLE_DEVICES)
```

Docker images are the practical approach for deployments:

```bash
docker pull rocm/rocm-terminal:6.3-ubuntu22.04
docker run -it --device /dev/kfd --device /dev/dri \
    --group-add video --group-add render \
    rocm/rocm-terminal:6.3-ubuntu22.04
```

The `--device /dev/kfd` and `--device /dev/dri` flags are AMD-specific — different from NVIDIA's `--gpus all`. This trips up people porting Docker-based workflows from CUDA environments.

## Profiling with ROCm

### rocprof

`rocprof` is the primary profiling tool. It collects hardware performance counters and generates trace files.

```bash
# Basic counter collection
rocprof --stats ./my_program

# Specific counters
rocprof --pmc SQ_INSTS_VALU SQ_INSTS_VMEM TCC_HIT TCC_MISS ./my_program

# Output CSV with per-dispatch data
rocprof -o results.csv ./my_program
```

The counter names on AMD hardware are different from NVIDIA's. The concepts map but the naming does not. `SQ_INSTS_VALU` counts vector ALU instructions; `SQ_INSTS_VMEM` counts vector memory instructions. The ratio between them is the hardware-level arithmetic intensity indicator.

### Omniperf

**Omniperf** is AMD's newer profiling framework, designed to provide the kind of detailed per-kernel analysis that Nsight Compute provides on NVIDIA. It runs `rocprof` under the hood but adds normalization, roofline analysis, and a web-based visualization:

```bash
# Collect profile
omniperf profile -n my_workload -- ./my_program

# Analyze
omniperf analyze -p workloads/my_workload/MI300X_A1/
```

Omniperf's roofline view is particularly useful for the same reason as on NVIDIA: it immediately tells you whether a kernel is compute-bound or memory-bound, and how far from the hardware ceiling it is operating.

## Where ROCm Wins and Where It Doesn't

### ROCm wins on:

**Memory capacity (MI300X)**: No single NVIDIA GPU in the H100/H200 class matches 192 GB of on-package HBM. For inference workloads where model size is the constraint, this is a significant operational advantage.

**Memory bandwidth (MI300X)**: 5.3 TB/s is genuinely faster than 3.35 TB/s. For memory-bound operations, this directly translates to throughput.

**Price-to-performance**: AMD hardware is generally available at better pricing than equivalent NVIDIA hardware, particularly in the cloud market where providers like Google, Microsoft, and AWS have deployed MI300X at scale.

**Openness**: ROCm is Apache-licensed. You can read the implementation, file bugs, contribute patches. This matters for research groups and organizations where the ability to modify the stack is important.

### ROCm lags on:

**Ecosystem depth**: CUDA has 17 years of libraries, tutorials, Stack Overflow answers, and code examples. ROCm has fewer high-quality third-party libraries, less community knowledge, and more "works in theory, needs a patch in practice" scenarios.

**PyTorch/JAX integration**: Both frameworks have first-class CUDA support. ROCm support is second-class — functional, but frequently trailing by a release or two, and occasionally requiring workarounds. The gap is closing in 2025–2026 but has not closed.

**Triton**: OpenAI's Triton (the Python-to-GPU-kernel compiler) added ROCm support in 2023, but kernel generation quality on AMD hardware still lags the NVIDIA backend for many operations.

**Small workloads and tooling polish**: For production inference serving, debugging tools (roc-gdb, memory sanitizers), and integration with observability stacks, NVIDIA's tooling is more mature.

## The Portability Question

The strongest case for using HIP/ROCm is **hardware portability**. If your code runs on both AMD and NVIDIA hardware, you have negotiating leverage with cloud providers, can take advantage of spot pricing across hardware types, and are not locked into a single vendor's GPU supply chain.

For new code, HIP costs roughly the same effort as CUDA — the APIs are nearly identical. For existing CUDA code, the hipify tools handle 80–90% of the mechanical translation; the remaining 10–20% is library porting and architecture-specific tuning.

Whether that portability is worth the ongoing cost of maintaining compatibility with two hardware targets depends on your organization's operational situation and procurement strategy. There is no universal answer. There is only the honest accounting of what portability requires versus what it provides.
