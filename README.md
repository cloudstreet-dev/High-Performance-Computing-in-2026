# High Performance Computing in 2026

A practical guide to GPU computing, parallel programming, and the intersection of HPC with modern ML workloads — written for engineers who want to understand what's actually happening on the silicon, not just which API call to reach for.

## What This Book Covers

The HPC landscape in 2026 is messier and more interesting than it's been in decades. NVIDIA still holds the CUDA moat, but AMD's ROCm has become genuinely usable, Apple Silicon has crashed the party with unified memory architectures that challenge conventional wisdom, and the boundary between "HPC cluster" and "ML training farm" has effectively dissolved.

This book covers:

- **GPU architecture and programming models** — CUDA, ROCm, and Metal, with honest assessments of where each shines and where each will make you regret your life choices
- **Classical HPC primitives** — MPI for distributed memory, OpenMP for shared memory, and vectorization/SIMD for wringing the last few percent out of a core
- **The ML/HPC convergence** — what it means for workload design when your "scientific computing cluster" and your "training cluster" are the same hardware

## Who This Is For

Engineers. People who write code that runs on real hardware, who care about memory bandwidth and latency, and who have been burned at least once by assuming the compiler would figure it out.

This is not an introductory programming book. It assumes comfort with C/C++ and at least passing familiarity with Linux. It does not assume prior HPC experience.

## Read Online

The book is published at: **https://cloudstreet-dev.github.io/High-Performance-Computing-in-2026/**

## Building Locally

```bash
cargo install mdbook
mdbook serve
```

The book will be available at `http://localhost:3000`.

## License

[CC0 1.0 Universal](LICENSE) — public domain.
