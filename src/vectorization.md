# Vectorization and SIMD

SIMD (Single Instruction, Multiple Data) is the most consistently underexploited source of performance in production HPC code. Unlike GPU parallelism — which requires rearchitecting algorithms — or OpenMP — which requires restructuring loops — vectorization often requires very little: write the code correctly, and the compiler does the work.

When it works, vectorization delivers 4–16× speedup on floating-point loops at zero programmer effort. When it does not work, engineers spend days hand-writing intrinsics and end up with code that is brittle, architecture-specific, and barely faster than the scalar version they were complaining about.

This chapter covers both paths.

## What SIMD Is

Modern CPUs contain SIMD units — wide arithmetic units that operate on multiple data elements simultaneously. Instead of adding two floats, they add eight. Instead of multiplying one double-precision pair, they multiply four.

The relevant ISA extensions by platform:

| Extension | Width | Elements (float) | Platform |
|-----------|-------|-------------------|----------|
| SSE2 | 128-bit | 4 × float32 | x86 (2000+) |
| AVX | 256-bit | 8 × float32 | x86 (2011+) |
| AVX2 | 256-bit | 8 × float32, integer | x86 (2013+) |
| AVX-512 | 512-bit | 16 × float32 | x86 (2017+, servers) |
| NEON | 128-bit | 4 × float32 | ARM (v7+) |
| SVE | Scalable | 4–64 × float32 | ARM (v8.2+) |
| VSX | 128-bit | 4 × float32 | POWER |

A Skylake-X Xeon supporting AVX-512 with two FMA units can execute 32 float32 FMA operations per cycle. At 3.5 GHz, that is 112 GFLOPS per core — before considering multiple cores. Leaving AVX-512 unused means leaving most of that peak performance on the table.

## Auto-Vectorization

The compiler is the first line of vectorization. GCC, Clang, and MSVC all auto-vectorize loops that meet certain criteria. The criteria are simple to state and surprisingly easy to violate accidentally:

1. Loop count must be known or estimable at compile time (or at loop entry for runtime vectorization)
2. No loop-carried dependencies (iteration N depends on result of iteration N-1)
3. No aliasing between input and output buffers
4. Memory access must be contiguous (stride-1)

```c
// Auto-vectorizes cleanly
void add(float* c, const float* a, const float* b, int n) {
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}
```

Check whether the compiler vectorized with:

```bash
# GCC
gcc -O3 -march=native -fopt-info-vec-optimized add.c

# Clang
clang -O3 -march=native -Rpass=loop-vectorize add.c

# Check the assembly
objdump -d add.o | grep -E "vmovaps|vaddps|vmulps|vfmadd"
```

SIMD instructions in the output (ymm/zmm registers, v-prefixed instructions) confirm vectorization.

### Why Auto-Vectorization Fails

**Aliasing**: The compiler cannot prove that `a`, `b`, and `c` do not overlap. If they overlap, vectorizing would be incorrect. The `restrict` keyword is the fix:

```c
// Without restrict: compiler may not vectorize (assumes aliasing possible)
void add(float* c, const float* a, const float* b, int n) { ... }

// With restrict: compiler knows pointers don't alias, can vectorize
void add(float* __restrict__ c,
         const float* __restrict__ a,
         const float* __restrict__ b, int n) { ... }
```

**Loop-carried dependency**: A recurrence where each iteration depends on the previous:

```c
// Cannot vectorize: c[i] depends on c[i-1]
for (int i = 1; i < n; i++)
    c[i] = c[i-1] * alpha + b[i];
```

This is a genuine algorithmic constraint; you cannot vectorize this with standard SIMD without restructuring the algorithm (some recurrences admit vectorized reformulations, but not trivially).

**Non-contiguous access**: Stride-2 or gather access prevents the compiler from generating efficient vector loads:

```c
// Stride-2 access: difficult to auto-vectorize efficiently
for (int i = 0; i < n; i += 2)
    result += a[i] * b[i];
```

Restructure data to be contiguous where possible. Separate arrays (structure of arrays) over interleaved layouts (array of structures) is the standard recommendation for SIMD-friendly data.

**Function calls**: Most function calls prevent vectorization because the compiler cannot inspect their internals. Mark small functions `inline` or `__attribute__((always_inline))`.

## Enabling Vectorization

### Compiler Flags

```bash
# Enable AVX-512 on a server with Skylake/Cascade Lake Xeon
gcc -O3 -march=native -funroll-loops -ftree-vectorize

# Target a specific architecture
gcc -O3 -march=skylake-avx512

# Report missed vectorizations
gcc -O3 -march=native -fopt-info-vec-missed
```

`-march=native` enables all instruction set extensions of the current CPU. Do not use it for binaries deployed to heterogeneous clusters — use the oldest common denominator (`-march=znver3` for AMD Zen 3, `-march=icelake-server` for Intel Icelake).

### Compiler Hints

```c
// Hint that the pointer is aligned (enables aligned loads/stores)
float* a = (float*)aligned_alloc(64, n * sizeof(float));
__builtin_assume_aligned(a, 64);

// Loop vectorization hints (GCC)
#pragma GCC ivdep           // Ignore vector dependencies (use when safe)
#pragma GCC unroll 4        // Unroll 4 times before vectorizing

// Clang
#pragma clang loop vectorize(enable)
#pragma clang loop interleave_count(4)
```

## Intel Intrinsics: When You Need Control

When the compiler fails to vectorize efficiently, or when you need to guarantee specific instruction sequences, intrinsics provide direct access to SIMD instructions.

AVX-512 example: computing the sum of squares of a float array.

```c
#include <immintrin.h>  // AVX-512 intrinsics

float sum_of_squares_avx512(const float* data, int n) {
    __m512 acc = _mm512_setzero_ps();  // 16 × float32, initialized to 0

    int i = 0;
    // Main loop: 16 elements per iteration
    for (; i <= n - 16; i += 16) {
        __m512 v = _mm512_loadu_ps(&data[i]);     // Load 16 floats
        acc = _mm512_fmadd_ps(v, v, acc);          // acc += v * v (FMA)
    }

    // Horizontal sum of the 16 accumulators
    float result = _mm512_reduce_add_ps(acc);

    // Scalar tail for remaining elements
    for (; i < n; i++)
        result += data[i] * data[i];

    return result;
}
```

Key intrinsic naming conventions (Intel):
- `_mm_` prefix: 128-bit (SSE)
- `_mm256_` prefix: 256-bit (AVX/AVX2)
- `_mm512_` prefix: 512-bit (AVX-512)
- `_ps` suffix: packed single (float32)
- `_pd` suffix: packed double (float64)
- `_epi32` suffix: packed 32-bit integers

### ARM NEON Intrinsics

```c
#include <arm_neon.h>  // ARM NEON intrinsics

float dot_product_neon(const float* a, const float* b, int n) {
    float32x4_t acc = vdupq_n_f32(0.0f);  // 4 × float32, all zeros

    int i = 0;
    for (; i <= n - 4; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);  // Load 4 floats
        float32x4_t vb = vld1q_f32(&b[i]);  // Load 4 floats
        acc = vfmaq_f32(acc, va, vb);        // acc += va * vb (FMA)
    }

    // Horizontal sum
    float result = vaddvq_f32(acc);

    // Scalar tail
    for (; i < n; i++)
        result += a[i] * b[i];

    return result;
}
```

NEON is available on all AArch64 processors: Apple Silicon, AWS Graviton, Ampere Altra. It delivers 4 × float32 per cycle with FMA, substantially less than AVX-512 but with lower power consumption and better portability in the ARM world.

### ARM SVE: Scalable Vectors

SVE (Scalable Vector Extension) is ARM's answer to AVX-512, with a twist: the vector width is implementation-defined and discoverable at runtime. Code written for SVE runs on 128-bit implementations (4 × float32) and 512-bit implementations (16 × float32) without recompilation.

```c
#include <arm_sve.h>

float sum_sve(const float* data, int n) {
    svfloat32_t acc = svdup_f32(0.0f);
    svbool_t pg;  // Predicate (mask)

    for (int i = 0; i < n; i += svcntw()) {  // svcntw() = vector length in uint32s
        pg = svwhilelt_b32(i, n);             // Predicate: active lanes where i+lane < n
        svfloat32_t v = svld1_f32(pg, &data[i]);  // Masked load
        acc = svmla_f32_m(pg, acc, v, v);    // acc += v * v (masked FMA)
    }

    return svaddv_f32(svptrue_b32(), acc);    // Horizontal sum
}
```

SVE's predicated execution handles loop tails naturally — no separate scalar tail needed. This makes the code cleaner at the cost of a more complex programming model.

## The Roofline Model for CPUs

The same arithmetic intensity analysis from the GPU fundamentals chapter applies to CPUs.

For a Cascade Lake Xeon at 3.5 GHz with AVX-512 and two FMA units:
- Peak compute: 3.5 GHz × 16 (AVX-512 width) × 2 (FMA) × 2 (FMAs/cycle) = 224 GFLOPS/core
- Peak bandwidth (DDR4-2933, 6-channel): ~140 GB/s per socket

Crossover (roofline knee): 224 / 140 ≈ 1.6 FLOP/byte

Operations with arithmetic intensity above 1.6 are compute-bound and benefit from SIMD. Operations below 1.6 are memory-bandwidth-bound, and SIMD helps only insofar as it reduces the number of load/store instructions that touch the bus.

This is why streaming operations (memcpy, simple element-wise ops) do not see large speedups from SIMD on large arrays: they are already memory-bound, and wider SIMD just saturates the memory bus faster without changing the ceiling.

## Practical Vectorization Workflow

1. **Measure first.** Profile to confirm the loop is actually hot. Vectorizing a function that takes 0.1% of runtime provides 0.1% improvement in the best case.

2. **Check compiler output.** Use `-fopt-info-vec-optimized` and `-fopt-info-vec-missed`. Read the disassembly. Confirm you are seeing ymm/zmm registers in hot loops.

3. **Fix obvious blockers.** Add `restrict`, restructure data layouts, eliminate function calls in loops.

4. **Enable the right flags.** `-O3 -march=native` on development hardware; target a specific `-march` for deployment.

5. **Use intrinsics only when necessary.** If the compiler generates correct SIMD with decent performance, stop. Intrinsics code is harder to maintain and architecture-specific.

6. **Test on target hardware.** Performance of SIMD code varies significantly between CPU generations. AVX-512 downclocks on some Intel parts (frequency throttling when AVX-512 is active). Measure on the hardware that matters.

## AVX-512 and the Frequency Throttling Problem

A notable gotcha on Intel Skylake and Cascade Lake Xeons: executing AVX-512 instructions causes the CPU to reduce its clock frequency by 100–300 MHz. This "license 2" frequency state is designed to manage power consumption during wide SIMD execution.

The implications:
- A loop that uses AVX-512 may run slower than AVX2 if the frequency reduction offsets the wider SIMD width
- The throttle applies to the entire core for a settling period (~1 ms) after AVX-512 execution — affecting non-SIMD code that runs afterward
- This is specific to older Intel microarchitectures; Ice Lake and later have significantly reduced throttling; AMD does not throttle on AVX-512

Measurement is mandatory. Do not assume AVX-512 is always faster than AVX2 on Intel hardware without benchmarking on the target CPU.

## Integration with HPC Libraries

Many HPC libraries handle SIMD internally and expose architecture-agnostic interfaces:

- **FFTW**: Detects and uses SSE2/AVX/AVX-512/NEON at runtime based on the CPU
- **OpenBLAS/MKL**: Architecture-optimized BLAS with runtime dispatch
- **Eigen**: C++ template library that generates SIMD code; use `-DEIGEN_VECTORIZE_AVX512` for AVX-512
- **Highway**: Google's portable SIMD library; same code, multiple targets

Highway is worth highlighting for cross-architecture portability:

```cpp
#include "highway/highway.h"
namespace hn = hwy::HWY_NAMESPACE;

// This code compiles for SSE2, AVX2, AVX-512, NEON, SVE, WASM SIMD
// The right version is selected at runtime
HWY_ATTR void add(float* HWY_RESTRICT c,
                   const float* HWY_RESTRICT a,
                   const float* HWY_RESTRICT b, int n) {
    const hn::ScalableTag<float> d;
    const int lanes = hn::Lanes(d);

    int i = 0;
    for (; i + lanes <= n; i += lanes) {
        auto va = hn::Load(d, a + i);
        auto vb = hn::Load(d, b + i);
        hn::Store(hn::Add(va, vb), d, c + i);
    }
    // Scalar tail
    for (; i < n; i++) c[i] = a[i] + b[i];
}
```

Highway's abstraction works well. The generated code quality is competitive with hand-written intrinsics in most cases. If you are writing a library that needs to run efficiently across x86, ARM, and RISC-V, Highway is the right tool.

## Summary

Vectorization is free performance if you write the right code and compile with the right flags. The steps in order:

1. Structure data for contiguous access (SoA over AoS)
2. Annotate away aliasing with `restrict`
3. Compile with `-O3 -march=native`
4. Verify with disassembly or compiler reports
5. Reach for intrinsics only when the compiler falls short and the profiler shows it matters

The gap between scalar and fully vectorized code can be 4–16×. The engineer who ignores vectorization and instead buys a bigger GPU is wasting money that the compiler would have given them for free.
