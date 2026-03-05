# The CUDA Programming Model

CUDA is 17 years old, controls roughly 90% of the GPU compute market, and remains the standard against which everything else is measured. Understanding CUDA is not optional for HPC in 2026 — even if you end up using ROCm or Metal, you will be reading CUDA documentation, porting CUDA code, or debugging problems that are explained in CUDA terms.

This chapter covers the CUDA programming model in depth: the execution hierarchy, memory management, synchronization, and the performance tools you need to stop guessing and start measuring.

## The Execution Hierarchy

CUDA organizes computation into a three-level hierarchy: **grids**, **blocks**, and **threads**.

A **kernel** is a function that runs on the GPU. When you launch a kernel, you specify a grid of thread blocks. Each block contains threads. The threads within a block can share memory and synchronize with each other. Threads in different blocks cannot directly communicate during a kernel.

```
Grid (one per kernel launch)
└── Block (0, 0, 0), Block (1, 0, 0), Block (2, 0, 0), ...
    └── Thread (0,0,0), Thread (1,0,0), ..., Thread (blockDim.x-1, 0, 0)
```

The grid and block dimensions can be up to 3D. In practice, most compute kernels use 1D or 2D layouts matching the problem structure (arrays, matrices, images).

```c
// Kernel definition: runs on the GPU
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Kernel launch from the host (CPU)
int block_size = 256;
int grid_size = (n + block_size - 1) / block_size;  // ceiling division
vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
```

The `<<<grid_size, block_size>>>` syntax is the kernel launch configuration. It tells CUDA how many blocks to create and how many threads per block. Block sizes of 128, 256, or 512 are conventional starting points; optimal values depend on register pressure, shared memory usage, and the specific hardware.

### Thread Indexing

`blockIdx` and `threadIdx` are built-in variables that give each thread its position in the grid. A thread's globally unique index in a 1D launch is:

```c
int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
```

For 2D:

```c
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

The bounds check (`if (idx < n)`) matters. Grid sizes are always rounded up to fill blocks, so the last block may contain threads with no valid work. Without the bounds check, those threads will access out-of-bounds memory.

## Memory Management

CUDA operates with a host (CPU) address space and a device (GPU) address space. In the general case, these are separate. Data must be explicitly moved between them.

### Basic Allocation and Transfer

```c
float *h_a = (float*)malloc(n * sizeof(float));     // host allocation
float *d_a;
cudaMalloc(&d_a, n * sizeof(float));                 // device allocation

// Host → Device
cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);

// Device → Host
cudaMemcpy(h_a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost);

// Cleanup
cudaFree(d_a);
free(h_a);
```

`cudaMemcpy` is synchronous by default — it blocks until the transfer completes. For overlapping computation and data transfer, use `cudaMemcpyAsync` with streams (covered below).

### Pinned Memory

Regular host memory allocated with `malloc` is pageable. The CUDA driver must stage transfers through a locked buffer, costing an extra copy. **Pinned memory** (also called page-locked memory) removes this staging copy:

```c
float *h_a;
cudaMallocHost(&h_a, n * sizeof(float));  // pinned allocation
// ... use h_a ...
cudaFreeHost(h_a);
```

Pinned transfers are typically 2× faster than pageable transfers and are a prerequisite for async transfers. The downside: pinned memory is scarce. Over-allocating it degrades overall system performance because the OS can't swap it out.

### Shared Memory

Shared memory is the most important tool for GPU kernel optimization. It is fast (on-chip), explicitly managed by the programmer, and scoped to a thread block.

The canonical use case is **tiled matrix multiplication**: instead of each thread reading from global memory independently, the block cooperatively loads a tile into shared memory, computes from the tile, then loads the next tile.

```c
__global__ void matmul_tiled(const float* A, const float* B, float* C,
                               int M, int N, int K) {
    const int TILE = 16;
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        // Cooperatively load tile into shared memory
        if (row < M && t * TILE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();  // Wait for all threads to finish loading

        for (int k = 0; k < TILE; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();  // Wait before overwriting shared memory
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}
```

`__syncthreads()` is a block-level barrier. Every thread in the block must reach this point before any thread proceeds. Calling it inside a divergent branch (inside an `if` statement where some threads don't execute it) is undefined behavior.

### Unified Memory

`cudaMallocManaged` allocates memory accessible from both CPU and GPU, with the driver handling page migration automatically.

```c
float *data;
cudaMallocManaged(&data, n * sizeof(float));
// data is accessible on CPU and GPU
// driver migrates pages as needed
```

Use this for prototyping and correctness testing. For production code, explicit transfers with careful placement typically outperform the driver's heuristic page migration — especially for workloads with predictable access patterns.

## Streams and Concurrency

A **CUDA stream** is a sequence of operations that execute in order on the device. Operations in different streams can overlap.

```c
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// These can overlap if the GPU has capacity
cudaMemcpyAsync(d_a, h_a, size, cudaMemcpyHostToDevice, stream1);
cudaMemcpyAsync(d_b, h_b, size, cudaMemcpyHostToDevice, stream2);

kernel_a<<<grid, block, 0, stream1>>>(d_a, d_result_a);
kernel_b<<<grid, block, 0, stream2>>>(d_b, d_result_b);

cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);

cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

The classic use of streams is the **copy-compute overlap** pattern: while the GPU is processing batch N, stream in batch N+1. This requires pinned host memory and enough device memory to hold two batches simultaneously. The speedup is real — typically 20–40% for data-bound workloads — and the code complexity is manageable.

### CUDA Graphs

For workloads consisting of many small kernels launched in a fixed pattern, kernel launch overhead accumulates. **CUDA Graphs** capture a set of kernel launches and their dependencies as a graph, then replay the entire graph with a single API call.

```c
// Capture phase
cudaGraph_t graph;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  kernel_a<<<grid, block, 0, stream>>>(args_a);
  kernel_b<<<grid, block, 0, stream>>>(args_b);
  kernel_c<<<grid, block, 0, stream>>>(args_c);
cudaStreamEndCapture(stream, &graph);

// Instantiate (compile the graph)
cudaGraphExec_t graph_exec;
cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);

// Execute (much lower overhead than launching kernels individually)
cudaGraphLaunch(graph_exec, stream);
cudaStreamSynchronize(stream);
```

CUDA Graphs are essential for inference workloads where the same computation pattern repeats across many inputs. They are less useful when the computation graph changes shape (variable sequence lengths, dynamic shapes).

## Synchronization Primitives

### Device-wide Synchronization

`cudaDeviceSynchronize()` blocks the host until all GPU operations complete. Use it for correctness testing and timing. Do not use it in performance-critical paths.

### Events

**CUDA events** are the right tool for timing GPU operations:

```c
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(args);
cudaEventRecord(stop, stream);

cudaEventSynchronize(stop);  // Wait for stop event to complete

float elapsed_ms;
cudaEventElapsedTime(&elapsed_ms, start, stop);
printf("Kernel time: %.3f ms\n", elapsed_ms);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

Events are placed in the stream's command queue. `cudaEventRecord` is asynchronous from the host's perspective — it does not block. `cudaEventSynchronize` blocks until the GPU has processed that event.

## Warp-Level Primitives

Modern CUDA (compute capability 7.0+, Volta and later) exposes warp-level operations that do not require shared memory for intra-warp communication.

### Warp Shuffle

Threads in a warp can directly exchange registers without going through shared memory:

```c
// Warp reduction using shuffle
__device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;  // Lane 0 holds the sum of all 32 values
}
```

`__shfl_down_sync` exchanges values within a warp. The first argument is the mask of participating threads (0xffffffff = all 32). The second is the value being shared. The third is the lane offset. This is faster than a shared-memory reduction and simpler to write.

### Cooperative Groups

The Cooperative Groups API provides a structured way to work with groups of threads at any granularity:

```c
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void reduce(float* data, float* result, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? data[idx] : 0.0f;

    // Warp-level reduce
    val = cg::reduce(warp, val, cg::plus<float>());

    // One thread per warp writes to shared memory
    __shared__ float warp_sums[32];
    if (warp.thread_rank() == 0)
        warp_sums[warp.meta_group_rank()] = val;

    block.sync();

    // Final reduction across warp sums (in first warp)
    if (warp.meta_group_rank() == 0) {
        val = (warp.thread_rank() < block.num_threads() / 32)
                  ? warp_sums[warp.thread_rank()]
                  : 0.0f;
        val = cg::reduce(warp, val, cg::plus<float>());
        if (warp.thread_rank() == 0)
            atomicAdd(result, val);
    }
}
```

Cooperative Groups is the modern way to write reductions, scans, and other patterns that require coordination across subsets of threads. It is more composable than the older `__syncthreads` + shared memory approach and easier to reason about correctness.

## Tensor Cores

Volta (2017) introduced Tensor Cores: specialized matrix multiplication units that operate on 4×4 matrix tiles in mixed precision. On H100, Tensor Cores deliver up to 3958 TFLOPS of FP8 throughput — roughly 50× the FP64 throughput on the same chip.

Tensor Cores are exposed through:
- **WMMA API**: Warp-level matrix operations, explicit control
- **cuBLAS**: High-level GEMM, handles Tensor Core usage automatically
- **CUTLASS**: Template library for efficient GEMM and related operations

For most users, cuBLAS is the right answer. CUTLASS when you need custom epilogue fusions. WMMA when you are writing a research implementation and need explicit control. Hand-written Tensor Core code that beats cuBLAS is rare and difficult.

```c
// Let cuBLAS handle Tensor Cores
cublasHandle_t handle;
cublasCreate(&handle);
cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);  // Enable Tensor Cores

float alpha = 1.0f, beta = 0.0f;
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            d_B, N,
            d_A, K,
            &beta,
            d_C, N);
```

## Error Handling

CUDA functions return `cudaError_t`. Ignoring return values is how bugs become hour-long debugging sessions.

```c
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = (call);                                 \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Usage
CUDA_CHECK(cudaMalloc(&d_data, size));
CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
```

After kernel launches, check for errors with:

```c
kernel<<<grid, block>>>(args);
CUDA_CHECK(cudaGetLastError());      // Check for launch configuration errors
CUDA_CHECK(cudaDeviceSynchronize()); // Check for runtime errors during execution
```

`cudaGetLastError` catches errors set during kernel launch (bad grid/block dimensions, etc.). `cudaDeviceSynchronize` is required to surface errors that occur during kernel execution, since kernel launches are asynchronous.

## Profiling with Nsight Compute

`ncu` (Nsight Compute) is the authoritative profiling tool for CUDA kernels. It collects hardware performance counters per kernel.

```bash
# Profile all kernels in a binary
ncu --set full ./my_program

# Profile a specific kernel
ncu --kernel-name my_kernel --set full ./my_program

# Key metrics to examine
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
              l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
              sm__warps_active.avg.pct_of_peak_sustained_active \
./my_program
```

The three questions `ncu` answers:

1. **Is the kernel compute-bound or memory-bound?** Look at `sm__throughput` vs `l1tex__t_bytes`.
2. **Is the memory pattern efficient?** Look at L1/L2 hit rates and coalescing metrics.
3. **Is occupancy limiting latency hiding?** Look at active warps vs theoretical maximum.

Nsight Systems (`nsys`) is the higher-level profiler for understanding the interaction between CPU and GPU, stream behavior, and data transfer overlap:

```bash
nsys profile --trace=cuda,nvtx ./my_program
nsys-ui report.nsys-rep  # Open GUI
```

Both tools are free and ship with the CUDA toolkit.

## The CUDA Ecosystem

CUDA's real moat is not the programming model — it is the ecosystem built on top of it.

- **cuBLAS**: Dense linear algebra. The gold standard for GEMM.
- **cuSPARSE**: Sparse matrix operations.
- **cuFFT**: Fast Fourier transforms.
- **cuDNN**: Deep learning primitives (convolutions, attention, normalization).
- **NCCL**: GPU-to-GPU collective communications (AllReduce, Broadcast).
- **Thrust**: STL-like algorithms for GPUs.
- **CUB**: Block- and warp-level primitives for high-performance kernel building.

This ecosystem is the primary reason teams stay on CUDA even when they have access to AMD or other hardware. Porting a workflow is not just a matter of porting CUDA kernels to HIP; it is also porting or replacing all the library dependencies that sit between your code and the hardware.

## What CUDA Does Not Do Well

No honest treatment of CUDA omits the frustrations.

**The toolchain is NVIDIA-specific.** `nvcc` compiles CUDA C++. It is a compiler extension to standard C++, and the interaction between CUDA extensions and modern C++ features is occasionally surprising. Clang has CUDA support that is generally good but not perfect.

**Debugging is painful.** `cuda-gdb` exists. It works. It is not pleasant. Memory errors produce `CUDA error: an illegal memory access was encountered` with no indication of which access or why. Tools like `compute-sanitizer` (`--tool memcheck`) help, but GPU debugging in 2026 is still more painful than CPU debugging.

**The API surface is enormous.** CUDA 12 has hundreds of API functions. Knowing which abstraction to use when is not obvious from documentation. The answer is usually: use a library (cuBLAS, cuDNN, NCCL) before writing your own kernel; write CUB primitives before writing raw CUDA; write raw CUDA only when the library doesn't expose what you need.

**Vendor lock-in is real.** CUDA code runs on NVIDIA hardware only. The portability problem is solved by HIP (Chapter 4), which provides a CUDA-compatible API that compiles for both NVIDIA and AMD, at the cost of some NVIDIA-specific features.

Despite these rough edges, CUDA remains the most complete, best-documented, and most-supported GPU computing platform in existence. The frustrations are well-understood and generally manageable. If you are starting a new GPU compute project in 2026 and do not have specific reasons to choose otherwise, CUDA is still the reasonable default.
