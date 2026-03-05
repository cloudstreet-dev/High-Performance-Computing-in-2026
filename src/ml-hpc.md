# When Training Clusters Became Research Clusters

There is a moment, roughly 2022–2023, when HPC centers across the world looked at their queue systems and noticed something unusual: the jobs were getting longer, the memory requests were getting larger, and an increasing fraction of them wanted GPUs. Not just GPUs as accelerators — GPUs exclusively. The CPU allocation was one core. For the scheduler.

These were ML training jobs.

The HPC community's first instinct was often to treat this as an infestation. ML jobs do not use MPI properly. They do not use the filesystem correctly (too many small files, or one enormous checkpoint). They run for days in an undivided block rather than checkpointing frequently. They consume GPU memory in ways that prevent packing multiple jobs on one node. They are, from the perspective of a traditional HPC operator, rude.

The ML community's first instinct was to build their own infrastructure. Why deal with Slurm when you can use Kubernetes? Why request GPUs through a scheduler when you can reserve a dedicated machine?

Both instincts were wrong, or at least insufficient. By 2025, HPC operators had adapted their workflows to accommodate ML training, and ML practitioners had learned — sometimes painfully — that the HPC community's hard-won knowledge about distributed computing, fault tolerance, and IO performance was not actually useless. The reconciliation was messy and ongoing. This chapter is about the intersection that emerged.

## What Makes ML Training Different from Classical HPC

Classical HPC workloads — computational fluid dynamics, molecular dynamics, climate modeling — share certain characteristics:

- **Deterministic communication patterns**: MPI collectives at known points in the algorithm
- **Floating-point precision matters**: FP64 is standard; results must be bit-reproducible
- **Job duration is predictable**: "This will run for approximately N hours"
- **Output is data**: Files written to a shared filesystem

ML training differs on all of these:

- **Non-deterministic communication patterns**: The gradient aggregation pattern is regular, but dynamic batching, variable sequence lengths, and data-parallel scheduling create variability
- **Lower precision is often better**: FP16 and BF16 converge faster for many models; FP8 is in use for inference and increasingly for training
- **Job duration is unpredictable**: Training continues until loss converges or the researcher gets impatient, whichever comes first
- **Intermediate output is enormous**: Model checkpoints for a 70B parameter model in BF16 are ~140 GB; checkpointing every hour × weeks of training × multiple experiments fills filesystems

The implications for cluster operators:
- GPU allocation policies need to favor long-running exclusive reservations, not the small-job fairness typically optimized by HPC schedulers
- The IO system needs to handle massive sequential writes (checkpointing) and massive sequential reads (data loading) simultaneously
- Network topology matters differently: ML training uses AllReduce almost exclusively, which benefits from fat-tree or dragonfly topologies; HPC uses a broader variety of collectives

## Data Parallelism: The Dominant Pattern

The dominant ML training pattern is **data parallelism**: each GPU holds a complete copy of the model; each GPU processes a different batch of data; gradients are aggregated (averaged) across GPUs after each step.

```python
# PyTorch DDP: the standard data parallelism API
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def train(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    model = MyModel().cuda(rank)
    model = DDP(model, device_ids=[rank])  # Wraps model with AllReduce

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()  # DDP AllReduces gradients across GPUs here
        optimizer.step()

    dist.destroy_process_group()
```

DDP's gradient synchronization uses NCCL's AllReduce under the hood. NCCL is highly optimized for the NVLink topology of DGX nodes; on InfiniBand clusters, it uses RDMA to avoid host memory involvement. The communication overlaps with the backward pass where possible (buckets of gradients are reduced as they become ready, not all at once at the end).

The scaling efficiency of pure data parallelism is good but not perfect. At large batch sizes, the model may not train well (generalization degrades). Communication overhead grows as a fraction of compute time. In practice, linear scaling efficiency of 85–95% across 8 GPUs drops to 70–80% at 64 GPUs and 50–70% at 512+ GPUs, depending on model size and interconnect.

## Tensor Parallelism: Splitting the Model

When a model does not fit in a single GPU's memory, data parallelism is insufficient. **Tensor parallelism** (also called model parallelism) splits the model itself across GPUs.

The prototypical form: split a large matrix multiplication across GPUs. For a weight matrix W with shape [4096, 4096], split it column-wise across 4 GPUs, each holding [4096, 1024]. Each GPU computes its portion of the output; an AllGather reconstructs the full result.

```
GPU 0: W[:,  0:1024]   →   partial output [B, 1024]
GPU 1: W[:,1024:2048]  →   partial output [B, 1024]
GPU 2: W[:,2048:3072]  →   partial output [B, 1024]
GPU 3: W[:,3072:4096]  →   partial output [B, 1024]
                             AllGather → full output [B, 4096]
```

Megatron-LM (NVIDIA's large model training library) popularized this approach for transformer layers. The attention and feed-forward layers in each transformer block are split across tensor-parallel GPUs; an AllReduce (or AllGather + ReduceScatter) synchronizes after each layer.

The cost: tensor parallelism requires extremely low-latency, high-bandwidth communication. It works well within an NVLink-connected DGX node (all 8 GPUs, 900 GB/s bidirectional). It does not scale across nodes well — the per-layer communication latency becomes the bottleneck.

## Pipeline Parallelism: Splitting by Layer

**Pipeline parallelism** assigns different layers of the model to different GPUs (or groups of GPUs). GPU 0 computes layers 1–4, passes the activations to GPU 1 which computes layers 5–8, and so on.

```
GPU 0: layers 1-4   → activations → GPU 1: layers 5-8   → ...
```

Naively, this leaves GPUs idle while waiting for upstream stages to finish the forward pass. **Micro-batching** (GPipe, PipeDream) addresses this by splitting the batch into micro-batches that flow through the pipeline simultaneously:

```
Time:  1  2  3  4  5  6  7  8
GPU 0: F1 F2 F3 F4 B4 B3 B2 B1
GPU 1:    F1 F2 F3 F4 B4 B3 B2
GPU 2:       F1 F2 F3 F4 B4 B3
GPU 3:          F1 F2 F3 F4 B4
(F=forward, B=backward, numbers=micro-batch)
```

Pipeline parallelism communicates activations (not gradients) between pipeline stages — the data size is typically much smaller than gradients, enabling effective use of inter-node InfiniBand.

## 3D Parallelism

Large model training at scale (GPT-4 scale, Llama-3 scale) uses all three forms of parallelism simultaneously:

- **Data parallelism**: across replica groups (each replica is a full copy of the model)
- **Tensor parallelism**: within a node (NVLink provides the bandwidth)
- **Pipeline parallelism**: across nodes (InfiniBand handles inter-node activation transfer)

The interplay between these dimensions is complex and the subject of ongoing research in systems papers. The key insight: match the communication pattern to the available interconnect:

- NVLink (~900 GB/s): tensor parallelism (high-bandwidth, low-latency collectives)
- InfiniBand (~400 Gb/s, ~50 GB/s): pipeline stages (moderate data, point-to-point)
- Ethernet: data parallel replicas (large messages, tolerance for some latency)

## Mixed Precision Training

FP32 training is largely obsolete. The standard since 2018 is **mixed precision**: forward and backward passes in FP16 or BF16; master weights and optimizer state in FP32.

```python
# PyTorch AMP (Automatic Mixed Precision)
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()  # Handles loss scaling for FP16

for batch in dataloader:
    optimizer.zero_grad()

    with autocast():  # Forward pass in FP16
        output = model(batch)
        loss = criterion(output, targets)

    scaler.scale(loss).backward()   # Scale loss to prevent FP16 underflow
    scaler.step(optimizer)          # Unscales gradients before optimizer step
    scaler.update()                 # Adjusts scale factor
```

**BF16 vs FP16**: BF16 (bfloat16, Brain Float) has the same exponent range as FP32 but lower mantissa precision. It is less susceptible to overflow/underflow than FP16, which requires gradient scaling (the `GradScaler` above). On hardware that supports BF16 natively (A100, H100, MI300X, Apple Neural Engine), BF16 is generally preferred for training — simpler code, equivalent convergence, no scaling overhead.

**FP8**: Transformer Engine (NVIDIA) and equivalent frameworks enable FP8 training with dynamic scaling per tensor. Effective throughput on H100 with FP8 approaches 4× the FP16 throughput via Tensor Cores. The stability of FP8 training is an active research area; it works well for transformers with careful scaling; not universally applicable.

## The IO Bottleneck

ML training at scale reveals IO bottlenecks that classical HPC jobs — which often have well-characterized sequential access patterns — manage to avoid.

**Data loading**: Training on large datasets (Common Crawl, RedPajama, LAION) requires reading terabytes of data with minimal stalls. The pattern is sequential within a shard, random across shards. Distributed filesystems (Lustre, GPFS) handle sequential reads well; object storage (S3, GCS) handles the random-shard pattern better with parallel prefetch.

Tools like **WebDataset** (PyTorch) and **tf.data** (TensorFlow) provide async prefetching pipelines that overlap data loading with compute. On a properly configured training pipeline, the GPU never waits for data. Improperly configured, data loading is the bottleneck — a $50,000 H100 idle while waiting for slow IO.

```python
# WebDataset: streaming from object storage or filesystem
import webdataset as wds

dataset = (
    wds.WebDataset("data/train-{000000..001023}.tar")
    .shuffle(10000)
    .decode("pil")
    .to_tuple("jpg", "json")
    .map(preprocess)
    .batched(64)
)

loader = wds.WebLoader(dataset, num_workers=8, prefetch_factor=4)
```

**Checkpointing**: A 70B parameter model in BF16 = ~140 GB. Checkpointing every N steps writes 140 GB to shared storage. With 64 GPUs running simultaneously and checkpointing every 1000 steps, checkpoint IO can saturate shared filesystems in ways that degrade performance for all jobs on the cluster.

Asynchronous checkpointing (writing in the background while training continues) and distributed checkpointing (each GPU writes its shard, reassembled on load) are the standard mitigations.

## Fault Tolerance at Scale

A 1024-GPU training run that experiences a GPU failure every 24 hours (conservative for old hardware) will see a failure every 1.4 hours on average. Without checkpointing and restart automation, the training job dies.

**Elastic training**: PyTorch's `torchrun` (successor to `torch.distributed.launch`) supports membership changes — adding or removing nodes from a running training job without restart. Implemented via `TORCHELASTIC_AGENT_SIDE_MONITOR` and `torch.distributed.elastic`.

```bash
# Launch with torchrun: auto-restarts on failure, supports elasticity
torchrun \
  --nnodes=8:16 \        # Min 8, max 16 nodes (elastic)
  --nproc_per_node=8 \
  --max_restarts=3 \
  train.py
```

**Checkpoint frequency and recovery time**: With 140 GB checkpoints, recovery time from checkpoint includes the IO time to read the checkpoint. Fast checkpointing (NVMe local storage, then async copy to shared FS) versus slow (write directly to Lustre) can mean the difference between 5-minute recovery and 45-minute recovery.

## HPC Patterns That ML Engineers Should Know

The HPC community's experience with large-scale distributed computing contains hard-won knowledge that has been independently rediscovered (sometimes at great cost) by ML practitioners.

**Checkpoint-restart is not optional.** HPC jobs have always checkpointed regularly; ML training is learning this lesson through hardware failures at scale.

**IO is a shared resource.** Writing checkpoints from 1024 GPUs simultaneously destroys performance for everyone on the cluster. Job scheduling systems have bandwidth reservation; use it. Stagger checkpoint writes if the system does not.

**Profiling distributed programs is different.** A 5% compute imbalance across 512 ranks that would be insignificant in isolation causes the entire job to slow by 5% at every synchronization point. Trace-based profilers (Nsight Systems, Perfetto) that show all ranks simultaneously are essential for distributed performance debugging.

**Network topology is not flat.** InfiniBand fat-tree topologies have more bandwidth within a spine switch than across spine switches. Co-locating communicating processes on the same rack is not always possible, but it is the right model to have in your head when looking at why AllReduce performance varies.

## ML Patterns That HPC Engineers Should Know

**Automatic differentiation is real.** ML frameworks compute gradients automatically via the computational graph. Understanding backpropagation at the graph level explains why the backward pass takes 2× the memory of the forward pass (activations must be retained) and why gradient checkpointing (recomputing activations during the backward pass instead of storing them) is a memory/compute tradeoff.

**The optimizer state is large.** Adam maintains two moment estimates per parameter: m (first moment) and v (second moment). For a 70B parameter model in FP32, the optimizer state is 2 × 70B × 4 bytes = 560 GB. This is why ZeRO (Zero Redundancy Optimizer) — which shards optimizer state across GPUs — is essential for large model training.

**Quantization is a legitimate tool, not a hack.** INT8 inference of a FP16 model can recover 1.9× throughput with minimal accuracy loss on many tasks. FP8 training is in production. This is different from low-precision arithmetic in scientific computing, where precision is non-negotiable.

## The Unified Cluster

By 2026, the practical reality for most organizations running significant compute is a single GPU cluster that serves both ML training and scientific computing, managed through a unified scheduler (Slurm with GPU extensions, or Kubernetes with GPU operators).

The policies that make this work:
- Priority queues that differentiate elastic ML jobs from rigid HPC jobs
- GPU time limits that force checkpointing on long-running jobs
- Storage quotas that prevent checkpoint accumulation from consuming all available space
- Network partitioning that keeps bulk data transfer from saturating the HPC interconnect
- Fair-share scheduling that prevents one team's training run from monopolizing resources indefinitely

The tools that make this work:
- **Slurm** with `gres` (generic resource) scheduling for GPUs, TRES fairshare for accounting
- **Kubernetes** with NVIDIA GPU Operator or AMD ROCm Device Plugin
- **MLflow** or **Weights & Biases** for experiment tracking (the HPC equivalent of job accounting)
- **Unified checkpoint storage** (parallel filesystem + object storage tier) with automated retention policies

The integration is imperfect and organization-specific. There is no universal solution. There is only the recognition that the workloads converged before the infrastructure did, and the infrastructure is catching up.
