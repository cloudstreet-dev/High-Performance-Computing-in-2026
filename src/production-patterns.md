# Production Patterns and Workload Design

Everything in the preceding chapters was about individual tools: CUDA, ROCm, Metal, MPI, OpenMP, vectorization. This chapter is about how those tools combine into systems that run reliably at scale — the engineering decisions that separate code that works in a demo from code that runs for months in production without someone babysitting it.

The patterns here are not theoretical. They are the kind of things that get written up after incidents.

## Know Your Workload's Critical Path

Before designing a parallel system, identify the critical path: the sequence of operations that determines end-to-end latency. Operations not on the critical path can be optimized indefinitely without improving overall performance.

A common mistake: optimizing GPU kernel throughput in a pipeline where the critical path is IO. A common parallel mistake: optimizing IO in a pipeline where the critical path is an AllReduce that is network-bound.

The tool for finding the critical path is a distributed trace, not a local profiler. Nsight Systems, Perfetto, or a custom timeline built on NVTX markers can show the timeline of events across all GPUs and the CPU simultaneously. A timeline view answers the question "what is the GPU waiting for and how long?" more directly than any per-kernel metric.

```python
# NVTX markers: annotate code regions that appear in Nsight timeline
import torch.cuda.nvtx as nvtx

nvtx.range_push("data_loading")
batch = next(data_iter)
nvtx.range_pop()

nvtx.range_push("forward_pass")
with torch.cuda.amp.autocast():
    output = model(batch)
nvtx.range_pop()

nvtx.range_push("backward_pass")
scaler.scale(output).backward()
nvtx.range_pop()
```

When you open this trace in Nsight Systems, the colored ranges appear on the timeline alongside CUDA kernel launches, memory transfers, and NCCL collectives. The gaps between GPU activity are immediately visible.

## Roofline-Driven Optimization

The roofline model (introduced in the GPU Fundamentals chapter) is a practical tool for directing optimization effort. For each kernel:

1. Measure actual arithmetic intensity (FLOPs / bytes transferred)
2. Look up the hardware roofline (peak FLOPs / peak bandwidth)
3. If below the roofline: memory-bound → optimize memory access patterns, reduce data movement, increase reuse
4. If at or near the roofline: compute-bound → reduce arithmetic operations, use lower precision, or accept that you are near optimal

```python
# Measure FLOPs and bandwidth for a PyTorch operation
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA],
             with_flops=True,
             profile_memory=True) as prof:
    with record_function("matmul"):
        result = torch.mm(A, B)

print(prof.key_averages().table(
    sort_by="cuda_time_total", row_limit=10))
```

The key metric from this output is `Self CUDA time` vs `FLOPs` — compute the ratio, compare to the hardware crossover.

## Kernel Fusion

Every kernel launch has overhead (~3–10 μs on modern GPUs). For workloads composed of many small operations, this overhead accumulates. More importantly, separate kernels for adjacent operations read and write global memory between them — bandwidth that could be eliminated by fusing.

**Manual fusion**: Combine operations into a single kernel that reads inputs once, computes multiple operations, and writes outputs once:

```cuda
// Instead of: norm_kernel → activation_kernel → scale_kernel
// Write one fused kernel:
__global__ void layer_norm_gelu_scale(
    const float* input, const float* weight, const float* bias,
    float* output, float scale, int n, int d)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // 1. Compute mean and variance (using shared memory reduction)
    __shared__ float mean_var[2];
    // ... (reduction code) ...

    // 2. Normalize
    float val = (input[row * d + tid] - mean_var[0]) /
                sqrtf(mean_var[1] + 1e-5f);

    // 3. Apply learned parameters
    val = val * weight[tid] + bias[tid];

    // 4. GELU activation
    val = 0.5f * val * (1.0f + tanhf(0.7978845608f * (val + 0.044715f * val * val * val)));

    // 5. Scale
    output[row * d + tid] = val * scale;
}
```

**Automatic fusion via compilers**:

```python
# PyTorch 2.0+: torch.compile fuses ops automatically
import torch

@torch.compile
def fused_forward(x, weight, bias):
    return torch.nn.functional.gelu(torch.nn.functional.layer_norm(x, [x.size(-1)], weight, bias))
```

`torch.compile` with TorchInductor generates Triton kernels that fuse operations across layer boundaries, achieving results close to hand-written fused kernels for common patterns. It is the right starting point before writing custom CUDA.

**Triton for custom kernels**:

```python
import triton
import triton.language as tl

@triton.jit
def fused_add_relu_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = tl.maximum(x + y, 0.0)  # Fused add+ReLU
    tl.store(output_ptr + offsets, output, mask=mask)
```

Triton compiles to PTX (NVIDIA) or AMDGPU ISA (ROCm), handles shared memory management automatically, and generates code that competes with hand-written CUDA for many memory-bound patterns.

## Batching and Throughput Optimization

**Static vs. dynamic batching**: Fixed batch sizes allow the compiler and runtime to optimize for known shapes. Variable batch sizes (common in inference serving) require padding or bucketing.

Padding to a fixed size wastes compute on padding tokens. Bucketing — grouping sequences of similar length into the same batch — reduces waste. `torch.nn.utils.rnn.pad_sequence` and custom dataset samplers implement this:

```python
class BucketSampler(torch.utils.data.Sampler):
    """Groups samples by length to minimize padding waste."""
    def __init__(self, lengths, batch_size, shuffle=True):
        self.indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        batches = [self.indices[i:i+self.batch_size]
                   for i in range(0, len(self.indices), self.batch_size)]
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch
```

**Continuous batching (inference)**: For LLM serving, requests arrive dynamically and have variable completion lengths. Naively waiting until a full batch is assembled increases first-token latency. Continuous batching (PagedAttention, vLLM) interleaves requests at the token level — a GPU iteration processes tokens from multiple partially-complete requests simultaneously. This increases GPU utilization at the cost of implementation complexity.

## Memory Management in Production

### Avoiding Memory Fragmentation

GPU memory allocators are less sophisticated than CPU allocators. Fragmentation — where total free memory exceeds what any single allocation can use — is a real failure mode.

```python
# PyTorch memory management
torch.cuda.empty_cache()  # Return cached memory to the allocator (not OS)

# Check fragmentation
print(torch.cuda.memory_stats())
# Look at 'active_bytes.all.current' vs 'reserved_bytes.all.current'

# Set memory fraction to avoid OOM on shared GPUs
torch.cuda.set_per_process_memory_fraction(0.9)
```

For CUDA C++ applications, using a pool allocator (rmm, CUB DeviceAllocator) with a pre-allocated pool avoids the overhead and fragmentation of repeated `cudaMalloc`/`cudaFree` calls.

### Gradient Checkpointing

Trading compute for memory: rather than storing all activations during the forward pass, recompute them during the backward pass.

```python
# PyTorch gradient checkpointing
from torch.utils.checkpoint import checkpoint

class TransformerLayer(torch.nn.Module):
    def forward(self, x):
        # Checkpointed: activations not stored, recomputed in backward
        return checkpoint(self._forward_impl, x, use_reentrant=False)

    def _forward_impl(self, x):
        # ... expensive forward computation ...
        return x
```

Gradient checkpointing typically increases compute by 33% (one extra forward pass per layer) while reducing activation memory by 60–70%. For memory-constrained training, this is a good trade.

### ZeRO: Distributed Optimizer State

For very large models where optimizer state (Adam: 2× model parameters in FP32) does not fit on one GPU, ZeRO shards it across GPUs:

- **ZeRO-1**: Shard optimizer states
- **ZeRO-2**: Shard optimizer states + gradients
- **ZeRO-3**: Shard optimizer states + gradients + parameters

```python
# DeepSpeed ZeRO-3
ds_config = {
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e8,
        "prefetch_bucket_size": 5e7,
    }
}

model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config,
    model_parameters=model.parameters()
)
```

ZeRO-3 enables training models that are 8–16× larger than a single GPU can hold, at the cost of additional AllGather/ReduceScatter communication overhead. The communication overhead is partially hidden by pipelining parameter fetches with computation.

## Monitoring and Observability

Production HPC systems need observability: what are the GPUs doing, when are they idle, why did this job crash?

### GPU Utilization Monitoring

```bash
# Basic: nvidia-smi in watch mode
watch -n1 nvidia-smi

# Better: dcgm-exporter for Prometheus metrics
docker run -d --gpus all \
  -p 9400:9400 \
  nvcr.io/nvidia/k8s/dcgm-exporter:latest

# AMD equivalent: amdsmi
amd-smi monitor --watch 1000 --watch_time 0
```

Key metrics to monitor:
- `SM Activity` (GPU utilization): Percentage of SMs active. Below 80% suggests underutilization.
- `Memory Bandwidth Utilization`: Approaching 100% means memory-bound.
- `NVLink Bandwidth`: Active NVLink traffic indicates inter-GPU communication.
- `GPU Memory Used`: Approaching 100% risks OOM.
- `GPU Temperature` and `Power Draw`: For thermal and power budget awareness.

### Distributed Tracing

For multi-node jobs, distributed tracing connects events across processes:

```python
# PyTorch profiler with distributed tracing
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler('./log/profiler'),
    with_stack=True,
) as prof:
    for step, batch in enumerate(train_loader):
        train_step(batch)
        prof.step()
```

The resulting trace in TensorBoard shows CPU and GPU timelines, memory allocation history, and kernel execution details.

## The Hardware-Software Co-Design Mindset

At scale, the boundary between software optimization and hardware selection blurs. Algorithm changes that reduce communication volume matter more than network upgrades at some problem sizes. Precision reductions (FP16 → FP8) that double throughput matter more than kernel micro-optimizations.

The questions worth asking when performance is insufficient:

**Is the arithmetic intensity high enough to justify the hardware?** A workload with 2 FLOP/byte arithmetic intensity on an H100 (roofline knee at ~20 FLOP/byte) is GPU-memory-bound. More GPUs will not help proportionally; the bottleneck is bandwidth, not compute.

**Is the communication to compute ratio acceptable?** For data parallelism, the ratio is (gradient size) / (compute time). Larger batches, more computation per parameter update, and lower precision all improve this ratio.

**Is the IO system the bottleneck?** If data loading saturates before compute, buying faster GPUs helps nothing. If checkpointing creates IO spikes that stall training, the fix is async checkpointing and staggering, not GPU upgrades.

**Is the job scheduler part of the problem?** Slurm's default fairshare policy can fragment large-job allocations across multiple scheduler cycles. Understanding your scheduler's topology-aware placement policies and requesting them explicitly (`--constraint=gpu_h100`, `--switches=1@10:00:00` in Slurm) avoids jobs with bad network topology.

## Putting It Together: A Production Checklist

When deploying a new HPC or ML workload to production:

**Performance baseline**:
- [ ] Profile on a single GPU/node to establish baseline efficiency
- [ ] Verify the bottleneck (compute-bound vs memory-bound vs IO-bound)
- [ ] Check vectorization (CPU-bound paths)
- [ ] Confirm memory access patterns (coalesced access, shared memory usage)

**Scaling**:
- [ ] Measure scaling efficiency at 2, 4, 8, 16, 32 GPUs
- [ ] Identify the communication overhead fraction at each scale point
- [ ] Enable GPU-aware MPI / NCCL for GPU-GPU communication
- [ ] Verify process/thread affinity and NUMA placement

**Reliability**:
- [ ] Implement checkpoint-restart with automatic resume
- [ ] Set checkpoint frequency based on MTBF of cluster hardware
- [ ] Test recovery from checkpoint on a subset of training

**IO**:
- [ ] Profile data loading throughput; confirm GPU never starves
- [ ] Async checkpoint writes; stagger across parallel jobs
- [ ] Set storage quotas and checkpoint retention policies

**Monitoring**:
- [ ] GPU utilization, bandwidth utilization, temperature, memory via dcgm-exporter or equivalent
- [ ] Distributed traces for multi-node jobs
- [ ] Alerting on OOM, NaN loss, gradient explosion, hardware errors (ECC)

**Reproducibility**:
- [ ] Fix random seeds for deterministic training runs
- [ ] Log software versions (framework, CUDA/ROCm, MPI, libraries)
- [ ] Log hardware configuration (GPU type, count, interconnect)
- [ ] Archive checkpoints at major milestones

Production HPC is not a solved problem. Hardware fails. Libraries have bugs. Communication patterns change when you scale from 16 to 128 GPUs. The checklist does not prevent all problems; it prevents the ones that have already burned someone else.

The literature that survives long enough to become conventional wisdom — checkpoint early, profile before optimizing, co-locate communicating processes, measure your arithmetic intensity — exists because someone did not do those things and paid for it. The good news is that the community's collective understanding of these failure modes continues to improve, and the tooling for catching them earlier continues to get better.

The bad news is that the hardware keeps getting faster, which keeps raising the ceiling on how wrong you can be before it becomes a problem.
