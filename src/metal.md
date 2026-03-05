# Metal: Apple Silicon in the HPC Arena

Apple Silicon was not supposed to matter for HPC. It was a laptop chip. A phone chip. Something you used for consumer workloads — video editing, maybe some light ML inference. The HPC community was politely uninterested.

Then people started running benchmarks.

The M-series chips, particularly the M2 Ultra and M3 Max/Ultra, deliver memory bandwidth that competes with discrete GPUs, FP16 throughput in the same neighborhood as mid-range datacenter parts, and a unified memory architecture that eliminates the PCIe bottleneck that makes CPU-GPU data movement painful. They do this in a laptop, drawing under 100W.

The relevance for HPC is specific. Apple Silicon is not going to replace H100 clusters. It is, however, a credible workstation compute platform for development, for workloads with large working sets that benefit from unified memory, and for inference where the power envelope matters. Understanding Metal — Apple's GPU programming API — is increasingly a useful skill.

## The Unified Memory Architecture

The most distinctive feature of Apple Silicon is unified memory: the CPU cores, GPU cores, and Neural Engine all share the same physical DRAM with direct cache-coherent access. There is no discrete GPU with its own VRAM pool. There is no PCIe bus between CPU and GPU.

This changes the analysis of data movement cost fundamentally.

On a conventional discrete GPU setup:
- CPU allocates data in DRAM
- Transfer to GPU VRAM over PCIe (~64 GB/s peak, often much less)
- GPU processes
- Transfer results back over PCIe
- CPU reads results from DRAM

On Apple Silicon:
- CPU allocates data in shared DRAM
- GPU accesses the same allocation directly
- No transfer cost

For workloads where the data movement is the bottleneck — which is many workloads — this eliminates a fundamental constraint. The cost you pay is bandwidth: 400 GB/s (M3 Max) or 800 GB/s (M2 Ultra) instead of 3+ TB/s on a discrete datacenter GPU. For workloads that are memory-bandwidth-bound, unified memory does not help and may hurt if the bandwidth is the constraint.

The practical implication: workloads with high data-reuse are well-served by Apple Silicon. Workloads that need raw throughput on very large matrices with high arithmetic intensity are not.

## Metal Shading Language

Metal compute shaders are written in Metal Shading Language (MSL), a C++14-derived language with GPU-specific extensions. If you have written HLSL or GLSL compute shaders, MSL will feel familiar. If your background is CUDA, it is syntactically different but conceptually similar.

```cpp
// A simple vector addition kernel in MSL
#include <metal_stdlib>
using namespace metal;

kernel void vector_add(
    device const float* a  [[buffer(0)]],
    device const float* b  [[buffer(1)]],
    device float* c        [[buffer(2)]],
    constant uint& n       [[buffer(3)]],
    uint index             [[thread_position_in_grid]])
{
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}
```

The `[[attribute]]` syntax is how MSL decorates parameters with their binding points and semantics:

- `device`: Main GPU-accessible memory (shared with CPU in unified memory)
- `constant`: Read-only memory, typically small (uniforms, parameters)
- `threadgroup`: Shared memory within a threadgroup (analogous to CUDA shared memory)
- `[[buffer(N)]]`: Binding slot for buffer arguments
- `[[thread_position_in_grid]]`: Built-in — the thread's global index
- `[[threadgroup_position_in_grid]]`: Block index (analogous to `blockIdx`)
- `[[thread_position_in_threadgroup]]`: Thread index within block (analogous to `threadIdx`)

### Threadgroup Memory (Shared Memory)

```cpp
kernel void tiled_matmul(
    device const float* A   [[buffer(0)]],
    device const float* B   [[buffer(1)]],
    device float* C         [[buffer(2)]],
    constant uint3& dims    [[buffer(3)]],
    threadgroup float* As   [[threadgroup(0)]],
    threadgroup float* Bs   [[threadgroup(1)]],
    uint2 thread_pos        [[thread_position_in_grid]],
    uint2 thread_in_group   [[thread_position_in_threadgroup]],
    uint2 group_pos         [[threadgroup_position_in_grid]])
{
    const uint TILE = 16;
    uint M = dims.x, N = dims.y, K = dims.z;

    uint row = thread_pos.y;
    uint col = thread_pos.x;
    float acc = 0.0;

    for (uint t = 0; t < (K + TILE - 1) / TILE; ++t) {
        // Load tile cooperatively
        uint a_col = t * TILE + thread_in_group.x;
        uint b_row = t * TILE + thread_in_group.y;

        As[thread_in_group.y * TILE + thread_in_group.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0;
        Bs[thread_in_group.y * TILE + thread_in_group.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; ++k)
            acc += As[thread_in_group.y * TILE + k] * Bs[k * TILE + thread_in_group.x];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}
```

`threadgroup_barrier(mem_flags::mem_threadgroup)` is the MSL equivalent of `__syncthreads()`. The `mem_flags` argument specifies which memory domains to synchronize.

## The Metal API (Host Side)

The host-side Metal API is Objective-C or Swift. This is where the CUDA parallel breaks down for C/C++ shops. There is no C API for Metal; interfacing from C++ requires either Objective-C++ (`.mm` files) or wrapper libraries.

```objc
// Objective-C++ Metal setup
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

id<MTLDevice> device = MTLCreateSystemDefaultDevice();
id<MTLCommandQueue> queue = [device newCommandQueue];

// Load shader library
NSError* error = nil;
id<MTLLibrary> library = [device newDefaultLibrary];
id<MTLFunction> function = [library newFunctionWithName:@"vector_add"];
id<MTLComputePipelineState> pipeline =
    [device newComputePipelineStateWithFunction:function error:&error];

// Allocate shared buffers (accessible by both CPU and GPU)
id<MTLBuffer> buf_a = [device newBufferWithLength:size
                                          options:MTLResourceStorageModeShared];
id<MTLBuffer> buf_b = [device newBufferWithLength:size
                                          options:MTLResourceStorageModeShared];
id<MTLBuffer> buf_c = [device newBufferWithLength:size
                                          options:MTLResourceStorageModeShared];

// Write data into buf_a and buf_b via their contents pointer
float* data_a = (float*)[buf_a contents];
// ... fill data_a ...

// Encode and submit
id<MTLCommandBuffer> cmd = [queue commandBuffer];
id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];

[encoder setComputePipelineState:pipeline];
[encoder setBuffer:buf_a offset:0 atIndex:0];
[encoder setBuffer:buf_b offset:0 atIndex:1];
[encoder setBuffer:buf_c offset:0 atIndex:2];

uint n_val = n;
[encoder setBytes:&n_val length:sizeof(uint) atIndex:3];

MTLSize threadgroup_size = {256, 1, 1};
MTLSize grid_size = {(n + 255) / 256 * 256, 1, 1};
[encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];

[encoder endEncoding];
[cmd commit];
[cmd waitUntilCompleted];
```

For C++ codebases, the practical options are:

1. **Metal-cpp**: Apple's official C++ wrapper for the Metal API. Header-only, fairly mechanical translation from Objective-C. Use this for new C++ code.
2. **Objective-C++**: Mix `.mm` files into your build. Messy but functional for wrapping Metal in a C++ interface.
3. **MLX**: Apple's ML framework uses Metal internally and exposes a NumPy-like C++/Python API. Good for ML workloads; not a general GPU compute library.

## Storage Modes and Memory Management

Metal's `MTLResourceStorageMode` controls where memory lives and how it is accessed:

| Mode | Description |
|------|-------------|
| `MTLResourceStorageModeShared` | Single allocation, accessible by CPU and GPU |
| `MTLResourceStorageModePrivate` | GPU-only; fastest GPU access, no CPU access |
| `MTLResourceStorageModeManaged` | Explicit synchronization required (macOS only) |
| `MTLResourceStorageModeMemoryless` | On-chip only; exists only during a render/compute pass |

On Apple Silicon (unified memory), `Shared` mode does not involve a copy — the CPU and GPU literally access the same memory. The performance difference between `Shared` and `Private` is typically small for compute workloads; `Private` mode can be faster for GPU-only buffers that the CPU never reads.

On Macs with discrete AMD GPUs (older Intel Macs), `Shared` mode requires managed synchronization, which is why `MTLResourceStorageModeManaged` exists. This mode is effectively obsolete for Apple Silicon development.

## Matrix Multiplication: MPS

**Metal Performance Shaders (MPS)** is Apple's library of optimized GPU compute operations. For matrix multiplication, `MPSMatrixMultiplication` is the relevant class:

```objc
MPSMatrixMultiplication* gemm = [[MPSMatrixMultiplication alloc]
    initWithDevice:device
    resultRows:M
    resultColumns:N
    interiorColumns:K];

// MPSMatrix descriptors
MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M
    columns:K rowBytes:K*sizeof(float) dataType:MPSDataTypeFloat32];
MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K
    columns:N rowBytes:N*sizeof(float) dataType:MPSDataTypeFloat32];
MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M
    columns:N rowBytes:N*sizeof(float) dataType:MPSDataTypeFloat32];

MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:buf_a descriptor:descA];
MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:buf_b descriptor:descB];
MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:buf_c descriptor:descC];

[gemm encodeToCommandBuffer:cmd primaryMatrix:matA
    secondaryMatrix:matB resultMatrix:matC];
```

MPS handles float32, float16, and (on recent hardware) bfloat16. For ML workloads, this is generally the right level of abstraction — hand-written Metal kernels rarely beat Apple's tuned implementations.

## Performance Characteristics

Concrete numbers for Apple M-series GPUs (as of late 2025):

| Chip | GPU Cores | FP32 TFLOPS | FP16 TFLOPS | Memory BW |
|------|-----------|-------------|-------------|-----------|
| M3 | 10 | ~3.6 | ~7.2 | 100 GB/s |
| M3 Max | 40 | ~14.2 | ~28.4 | 400 GB/s |
| M2 Ultra | 76 | ~27.2 | ~54.4 | 800 GB/s |

For comparison: an H100 SXM5 delivers ~67 TFLOPS FP64, ~989 TFLOPS FP16, at ~3.35 TB/s. The H100 wins on raw throughput by roughly 20–30×. The M2 Ultra wins on memory bandwidth per watt by a substantial margin and eliminates PCIe overhead.

**Where Apple Silicon is competitive:**
- Inference with large models that fit in 192 GB of unified memory
- Workloads with irregular memory access patterns that benefit from cache coherence
- Development and prototyping where the developer's laptop is the compute platform
- Power-constrained deployments (edge, on-device)

**Where it is not competitive:**
- Training at scale (no NVLink equivalent, limited tensor core equivalent)
- Large batch dense linear algebra where raw TFLOPS dominate
- Workloads that need multi-node MPI distribution

## The Developer Experience

Metal's development story is tight with Xcode. The Metal Debugger and GPU Frame Capture tools in Xcode are genuinely excellent for inspecting shader execution, visualizing buffer contents, and identifying performance bottlenecks. For GPU debugging, it is arguably the best DX of any platform.

The profiling story is also good: Instruments with the GPU counter template provides timeline views, occupancy data, and per-shader metrics. On Apple Silicon, you can profile compute kernels the same way you profile render work.

The limitation is the Apple-only toolchain. There is no Metal on Linux or Windows. Shaders written in MSL do not run anywhere else. If you need portability across GPU vendors, look at WebGPU (which has a Metal backend and runs in browsers), Vulkan compute (which has broad hardware support but not Apple), or HIP (NVIDIA/AMD only).

## MLX: The Modern Python Interface

For ML-adjacent HPC work on Apple Silicon, **MLX** (Apple's open-source array framework) provides the most convenient interface:

```python
import mlx.core as mx

a = mx.array([1.0, 2.0, 3.0])
b = mx.array([4.0, 5.0, 6.0])
c = a + b  # Runs on GPU via Metal, lazy evaluation
mx.eval(c)  # Triggers execution

# Matrix multiplication
A = mx.random.normal((1024, 512))
B = mx.random.normal((512, 1024))
C = A @ B  # Uses MPS internally
mx.eval(C)
```

MLX uses lazy evaluation: operations build a computation graph, and `mx.eval()` triggers execution on the GPU. It handles memory management, precision, and device scheduling. For numerical computing that happens to need GPU acceleration on a Mac, this is the right starting point — not hand-written Metal shaders.

## Verdict

Metal is worth knowing if you do serious GPU compute development on Apple hardware, ship products that target Apple Silicon, or are evaluating Apple devices for inference workloads. The programming model is coherent, the tools are polished, and the unified memory architecture creates genuine advantages for certain workload shapes.

It is not a path to CUDA-ecosystem compatibility or multi-vendor portability. It is a platform for Apple hardware specifically, and it is a good one for that narrower scope.
