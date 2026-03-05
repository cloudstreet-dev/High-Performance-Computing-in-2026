# Introduction

There is a particular kind of frustration reserved for the engineer who opens a job posting for "HPC" and discovers it is asking for PyTorch experience. There is an equal and opposite frustration reserved for the ML engineer who is handed a cluster and told to "just use MPI" without further guidance.

Both of these engineers are now, for better or worse, the same person.

This book exists because the HPC and ML communities spent the better part of a decade insisting they had nothing to do with each other, and then spent 2022–2025 building the same data center, buying the same accelerators, and filing the same support tickets. The reconciliation is still ongoing. This guide attempts to give working engineers the vocabulary and the mental models to operate in both worlds — and to stop being confused by people who act like they are different worlds.

## The State of HPC in 2026

The hardware landscape has consolidated in some ways and fragmented in others.

**GPU dominance is complete** — at least for workloads that can be parallelized. The era of debating whether GPUs were suitable for "real" HPC is over. H100s and their successors run everything from molecular dynamics to transformer training. The question is no longer whether to use a GPU but which GPU, which memory tier, and which programming model.

**The GPU monoculture has cracked.** For most of the 2010s, "GPU computing" meant CUDA, and CUDA meant NVIDIA. That is no longer a safe assumption. AMD's ROCm ecosystem reached a level of practical maturity around 2024–2025 that made it genuinely deployable — not a heroic research project, but something you could put into production without a dedicated champion willing to suffer for the cause. And Apple Silicon — with its unified memory architecture and Metal compute shaders — introduced a third credible path that matters most at the workstation level but has begun influencing how people think about memory bandwidth more broadly.

**Classical HPC primitives are not going away.** MPI turns 30 in 2024 and shows no signs of retirement. OpenMP is still how you thread a loop when you do not want to write a GPU kernel. Vectorization remains one of the most reliable sources of free performance that engineers consistently leave on the table. The ML community's habit of treating the CPU as a staging area for GPU work ignores a lot of performance sitting in plain sight.

**The infrastructure has normalized.** Kubernetes-based HPC, once a punchline, is now common. Job schedulers still dominate the traditional HPC center, but the boundaries are porous. Engineers increasingly need to understand both paradigms.

## What This Book Is and Is Not

This is a practical technical guide. It contains real code. It makes real recommendations. It will sometimes tell you that a tool or platform is worse than the marketing suggests, because that is useful information.

It is not a comprehensive API reference. The official documentation exists and you should use it. It is not an academic survey with 200 citations per chapter. It is also not a beginner's guide to programming; it assumes you write code for a living and have a passing familiarity with systems programming concepts.

The target reader is an engineer who:
- Has shipped software that runs on real hardware under real constraints
- Understands memory allocation, cache locality, and why pointer aliasing matters
- Wants to understand GPU computing at the model level, not just the API level
- Is comfortable reading C, C++, or Python and can tolerate the others

## A Note on the Platform Landscape

Throughout this book, code examples will be provided in CUDA, HIP (ROCm's CUDA-compatible API), Metal Shading Language, C with OpenMP, and C with MPI. Where frameworks diverge, we'll say so explicitly rather than pretending a single abstraction covers everything.

Benchmarks are hard to make meaningful and easy to make misleading. Where we quote performance numbers, we quote them with hardware context and an acknowledgment that your results will differ. Take any benchmark in any HPC book — including this one — as a directional signal, not a contract.

## How to Read This Book

The chapters are roughly ordered from hardware-adjacent to workload-level. The GPU chapters (Foundations) can be read in any order if you already have context on one or two of the platforms. The parallel programming models section (MPI, OpenMP, vectorization) is self-contained. The final section on ML/HPC convergence draws on everything that came before.

If you are new to GPU computing entirely, read the chapters in order.

If you have CUDA experience and want to understand ROCm or Metal, skip to the relevant chapter and refer back as needed.

If you are a traditional HPC engineer trying to understand why everyone wants to put a transformer on your cluster, start with the final two chapters and work backward.

Let us get into it.
