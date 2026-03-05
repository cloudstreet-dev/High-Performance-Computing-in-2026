# OpenMP: Shared Memory Without the Ceremony

OpenMP is how most HPC code adds parallelism to a loop without rewriting it. A pragma, a compile flag, and suddenly code that ran on one core runs on all of them. This is either elegant engineering or dangerous magic, depending on whether you understood what the loop was doing before you added the pragma.

OpenMP dates from 1997 and has accreted features steadily since. The core — parallel loops and task parallelism — is simple and useful. The periphery — SIMD directives, GPU offloading (target), memory model clauses — is increasingly complex and occasionally surprising. This chapter covers the core and the most useful features without pretending the complexity does not exist.

## The Execution Model

OpenMP uses a fork-join model. The program begins as a single **master thread**. When it encounters a parallel region, it forks a **team** of threads. At the end of the parallel region, threads join back to one. Multiple parallel regions can be nested; each creates its own team.

```
Single thread → PARALLEL REGION → n threads → join → single thread → ...
```

The number of threads is controlled by:
1. The `OMP_NUM_THREADS` environment variable
2. The `num_threads(n)` clause on a parallel directive
3. `omp_set_num_threads(n)` called before the parallel region

By default, OpenMP uses all available logical cores. On a 16-core/32-thread system, `OMP_NUM_THREADS` defaults to 32. Whether 32 threads is sensible depends on whether your workload benefits from hyperthreading — it often does not for floating-point-heavy HPC code.

## The Parallel Loop

The single most common OpenMP pattern:

```c
#include <omp.h>

// Without OpenMP: serial
for (int i = 0; i < n; i++) {
    result[i] = a[i] * b[i] + c[i];
}

// With OpenMP: parallel across all threads
#pragma omp parallel for
for (int i = 0; i < n; i++) {
    result[i] = a[i] * b[i] + c[i];
}
```

`#pragma omp parallel for` combines `parallel` (fork a team) and `for` (distribute loop iterations). The loop iterations are divided among threads; each thread executes its share.

**The implicit assumption**: iterations are independent. If iteration `i` depends on the result of iteration `i-1`, you cannot parallelize it this way. The compiler will not always catch this.

### Scheduling

The `schedule` clause controls how iterations are assigned to threads:

```c
// Static: divide iterations into equal chunks, assigned at compile time
#pragma omp parallel for schedule(static)
for (int i = 0; i < n; i++) { /* ... */ }

// Dynamic: assign chunks of 16 iterations to threads as they finish
// Good for irregular workloads where some iterations take longer
#pragma omp parallel for schedule(dynamic, 16)
for (int i = 0; i < n; i++) { /* ... */ }

// Guided: decreasing chunk sizes; good for loops with decreasing work
#pragma omp parallel for schedule(guided, 8)
for (int i = 0; i < n; i++) { /* ... */ }
```

`static` scheduling has zero synchronization overhead and is cache-friendly (threads access contiguous memory). Use it for uniform workloads. `dynamic` adds synchronization overhead but balances load for non-uniform work; useful for tree traversal, sparse operations, and anything where iteration time varies.

## Data Sharing

In a parallel region, variables are either **shared** (all threads see the same object) or **private** (each thread has its own copy).

The defaults:
- Variables declared outside the parallel region: **shared**
- Variables declared inside the parallel region: **private**
- Loop iteration variable: **private** (automatically)

```c
int n = 1000;
double sum = 0.0;  // shared by default — DANGER if written concurrently

#pragma omp parallel for
for (int i = 0; i < n; i++) {
    sum += compute(i);  // DATA RACE: multiple threads writing sum
}
```

The reduction clause fixes this:

```c
double sum = 0.0;

#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < n; i++) {
    sum += compute(i);  // Each thread has a private copy; combined at end
}
// sum now contains the total
```

`reduction(op:var)` creates a private copy of `var` for each thread (initialized to the identity element for `op`), lets each thread accumulate independently, and combines at the end. Operations: `+`, `*`, `-`, `min`, `max`, `&&`, `||`, bitwise `&`, `|`, `^`.

### The `private`, `firstprivate`, and `lastprivate` Clauses

```c
double x = 10.0;  // initialized before the parallel region

#pragma omp parallel private(x)
{
    // x is private but UNINITIALIZED — x is not copied in
    x = omp_get_thread_num() * 2.0;  // fine, each thread sets its own x
}

#pragma omp parallel firstprivate(x)
{
    // x is private AND initialized to the value from before the region (10.0)
    x += omp_get_thread_num();  // starts from 10.0 + thread_id
}

double last;
#pragma omp parallel for lastprivate(last)
for (int i = 0; i < n; i++) {
    last = i;  // The value from the last iteration (i = n-1) is copied out
}
```

## Synchronization

### Barrier

An implicit barrier exists at the end of every `parallel for` construct — threads wait until all have finished before the master continues. Explicit barriers:

```c
#pragma omp parallel
{
    do_first_phase();
    #pragma omp barrier  // All threads reach here before any proceed
    do_second_phase();
}
```

### Critical Section

Only one thread executes at a time:

```c
#pragma omp parallel
{
    double local_result = compute();
    #pragma omp critical
    {
        global_sum += local_result;  // Serial; only one thread at a time
    }
}
```

Critical sections serialize execution — use them sparingly. For the common case of accumulation, prefer `reduction`.

### Atomic Operations

For simple updates to a shared variable, `atomic` is faster than `critical`:

```c
#pragma omp atomic
count++;  // Atomic increment; no full lock needed

#pragma omp atomic update
histogram[bin]++;  // Also atomic update
```

`atomic` works for simple read-modify-write operations on scalar variables. The hardware provides atomic instructions for these; it is substantially cheaper than acquiring a lock.

## Tasks

OpenMP tasks (introduced in 3.0) enable parallelism that does not map neatly to loop iterations — recursive algorithms, irregular graphs, producer-consumer patterns.

```c
#pragma omp parallel
{
    #pragma omp single  // Only one thread creates tasks
    {
        for (int i = 0; i < n; i++) {
            #pragma omp task firstprivate(i)
            {
                process_item(i);  // Executes in parallel as threads are available
            }
        }
    }
    // Implicit taskwait at end of single region
}
```

A classic recursive use: parallel merge sort.

```c
void merge_sort(int* data, int n) {
    if (n < THRESHOLD) {
        insertion_sort(data, n);
        return;
    }

    int mid = n / 2;

    #pragma omp task shared(data)
    merge_sort(data, mid);

    #pragma omp task shared(data)
    merge_sort(data + mid, n - mid);

    #pragma omp taskwait  // Wait for both halves before merging

    merge(data, mid, n - mid);
}

// Call site:
#pragma omp parallel
#pragma omp single
merge_sort(data, n);
```

`#pragma omp taskwait` blocks the current task until all child tasks complete. `#pragma omp taskgroup` provides finer control when you want to wait for a subset.

## Thread Affinity and NUMA

On NUMA systems (multiple sockets, each with local memory), thread placement affects memory access latency significantly. A thread on socket 0 accessing memory allocated by a thread on socket 1 pays a 2–3× latency penalty.

```bash
# Control affinity with environment variables (OpenMP 4.0+)
export OMP_PROC_BIND=close    # Bind threads close together (same socket)
export OMP_PROC_BIND=spread   # Spread threads across sockets
export OMP_PLACES=cores       # Bind to core granularity
export OMP_PLACES=sockets     # Bind to socket granularity

# Or with numactl for finer control
numactl --cpunodebind=0 --membind=0 ./my_program
```

First-touch policy: on Linux, pages are allocated in the memory bank of the thread that first accesses them. In NUMA-aware code, initialize arrays in parallel so that each thread touches its own portion of the data, placing that portion in local memory:

```c
// NUMA-aware initialization: each thread initializes its own data
#pragma omp parallel for schedule(static)
for (int i = 0; i < n; i++) {
    data[i] = 0.0;  // Thread i/chunk touches this page → allocated locally
}

// Now the parallel computation has good locality
#pragma omp parallel for schedule(static)
for (int i = 0; i < n; i++) {
    data[i] = compute(i);  // Same thread as init → local memory access
}
```

The schedule must match between initialization and computation. If static scheduling assigns different iterations to different threads in the two loops, you get remote memory access.

## SIMD Directives

OpenMP 4.0+ includes SIMD directives for vectorizing loops at the instruction level:

```c
#pragma omp simd
for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
}

// Combined parallel+simd: threads each execute a SIMD loop
#pragma omp parallel for simd
for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
}
```

In practice, modern compilers auto-vectorize simple loops without the `simd` directive. The directive is useful for loops the compiler would not vectorize by default — typically because it cannot prove the absence of aliasing.

```c
void add(float* restrict a, float* restrict b, float* restrict c, int n) {
    // 'restrict' tells the compiler pointers don't alias — enables auto-vectorization
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
```

The `restrict` keyword (or `-fno-strict-aliasing` on the compiler) and careful loop structure often get you further than OpenMP SIMD directives. See the Vectorization chapter for more depth on this.

## OpenMP Target Offload (GPU)

OpenMP 4.5+ introduced `target` directives for GPU offloading. The intent is to express GPU parallelism using the same pragma model as CPU parallelism.

```c
#pragma omp target map(to:a[0:n], b[0:n]) map(from:c[0:n])
#pragma omp teams distribute parallel for simd
for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
}
```

The `target` pragma executes the following block on the GPU. `map(to:...)` copies data to the device before execution; `map(from:...)` copies results back after. `teams distribute` creates the grid of thread blocks; `parallel for simd` handles the parallel loop and vectorization within a block.

The honest assessment: OpenMP target offload exists, works, and is supported by GCC (via OpenACC backend), Clang, and LLVM. Performance on complex kernels is typically 20–50% behind hand-tuned CUDA. The value proposition is code portability — the same code targets CPU and GPU — at the cost of performance headroom.

For GPU-specific kernels where performance is critical, write CUDA or HIP. For codes where you want GPU acceleration on some platforms and CPU execution on others, and the performance gap is acceptable, OpenMP target is a reasonable choice.

## Profiling OpenMP Code

Before optimizing, measure. Tools for OpenMP profiling:

```bash
# Intel VTune (Linux, free download)
vtune -collect threading ./my_program
vtune -report summary

# LLVM XRay
clang -fxray-instrument -fxray-instruction-threshold=1 ./my_program.c -o my_program
XRAY_OPTIONS="patch_premain=true" ./my_program
llvm-xray account xray-log.* --sort=sum --top=10

# Score-P with Scalasca (cluster profiling)
scalasca -analyze mpirun -n 8 scorep-instrumented-binary

# Simple: time with OMP_NUM_THREADS variation
for t in 1 2 4 8 16; do
    OMP_NUM_THREADS=$t time ./my_program
done
```

**Amdahl's Law** is your constant companion when reading these profiles. If the parallel section takes 90% of serial runtime, the theoretical maximum speedup with infinite threads is 10×. With 16 threads, you might get 7–8×. If you are seeing 3×, the issue is not thread count — it is synchronization, load imbalance, or memory bandwidth saturation.

## A Note on Thread Safety

Adding `#pragma omp parallel for` to a loop that calls external functions is dangerous if those functions are not thread-safe. Standard library functions that use global state (`strtok`, `rand`, some `stdio` functions) will produce incorrect results or crashes. Thread-safe alternatives exist (`strtok_r`, `rand_r`), but the burden is on the programmer to know which functions are safe.

Third-party libraries may or may not be thread-safe. BLAS implementations (OpenBLAS, MKL) are thread-safe when called from multiple threads but may internally use their own thread pools — nested parallelism. When BLAS is called from inside an OpenMP region, you often want to disable BLAS's own threading:

```bash
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=1  # Disable OpenBLAS threading
export MKL_NUM_THREADS=1       # Disable MKL threading
```

Otherwise you get thread oversubscription: 16 OpenMP threads each spawning 16 BLAS threads = 256 threads on 16 cores.

## Summary

OpenMP's strength is low-friction parallelism for existing code. The annotation model means you can add parallelism incrementally, verify correctness at each step, and fall back to serial execution by disabling the flags.

Its limitation is the shared-memory constraint: OpenMP works within a single node. For multi-node parallelism, combine with MPI (as covered in the previous chapter). For GPU parallelism with better performance guarantees, use CUDA or HIP directly and treat OpenMP as the CPU-side complement.

The programmer's responsibility is to understand the code before adding pragmas. OpenMP does not protect you from data races — it just makes them faster.
