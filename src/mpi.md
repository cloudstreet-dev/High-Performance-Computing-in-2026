# MPI: Distributed Memory Parallelism

MPI (Message Passing Interface) is the load-bearing wall of the HPC world. It is how you make a single computation span multiple nodes in a cluster. It does this through a simple, brutal model: processes do not share memory; they communicate explicitly by sending and receiving messages.

This explicitness is the source of both MPI's power and its reputation for verbosity. When a program is slow or wrong, the communication pattern is visible in the code. There is no hidden coordination happening behind an abstraction. You can look at a section of MPI code and reason about exactly which processes are talking to which other processes and what data is moving. This property is extremely useful when you are debugging a hang on 2,048 nodes at 3 AM.

MPI is also 30 years old and designed by committee. These facts are not unrelated.

## The Programming Model

MPI programs use the Single Program, Multiple Data (SPMD) model: every process runs the same program, but each process has a unique **rank** within a **communicator**. The default communicator `MPI_COMM_WORLD` contains all processes launched by the job scheduler. The rank identifies each process uniquely within that communicator.

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // This process's rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Total number of processes

    printf("Hello from rank %d of %d\n", rank, size);

    MPI_Finalize();
    return 0;
}
```

Compile and run:

```bash
mpicc -o hello hello.c
mpirun -n 4 ./hello
```

Output (order not guaranteed):
```
Hello from rank 0 of 4
Hello from rank 2 of 4
Hello from rank 1 of 4
Hello from rank 3 of 4
```

The critical insight: every process executes `MPI_Comm_rank` independently and gets a different answer. This is how SPMD programs differentiate behavior per process — conditionals on `rank`.

## Point-to-Point Communication

The two fundamental operations are `MPI_Send` and `MPI_Recv`.

```c
if (rank == 0) {
    // Process 0 sends data to process 1
    int data = 42;
    MPI_Send(&data,           // buffer
             1,               // count
             MPI_INT,         // datatype
             1,               // destination rank
             0,               // tag (user-defined message label)
             MPI_COMM_WORLD); // communicator
} else if (rank == 1) {
    // Process 1 receives from process 0
    int data;
    MPI_Status status;
    MPI_Recv(&data,           // buffer
             1,               // count
             MPI_INT,         // datatype
             0,               // source rank (MPI_ANY_SOURCE to accept from any)
             0,               // tag (MPI_ANY_TAG to accept any tag)
             MPI_COMM_WORLD,
             &status);
    printf("Received: %d\n", data);
}
```

### Send Modes and Deadlocks

`MPI_Send` is a blocking send that may or may not buffer the data, depending on the message size and MPI implementation. This creates a classic deadlock scenario:

```c
// DEADLOCK: both processes try to send before receiving
MPI_Send(&data, n, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD);
MPI_Recv(&recv, n, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD, &status);
```

When `n` is large enough that `MPI_Send` cannot buffer the data, it blocks waiting for the receiver to post a receive. But the receiver is also blocked in `MPI_Send`. Deadlock.

The standard fix: use `MPI_Sendrecv` for simultaneous send and receive, or use non-blocking operations.

```c
// MPI_Sendrecv: atomic send+receive, no deadlock
MPI_Sendrecv(
    &send_data, n, MPI_DOUBLE, right_neighbor, 0,   // send
    &recv_data, n, MPI_DOUBLE, left_neighbor, 0,    // receive
    MPI_COMM_WORLD, &status);
```

### Non-Blocking Communication

Non-blocking operations return immediately and allow overlapping communication with computation:

```c
MPI_Request send_req, recv_req;

// Post non-blocking receive first (good practice)
MPI_Irecv(&recv_buf, n, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &recv_req);

// Post non-blocking send
MPI_Isend(&send_buf, n, MPI_DOUBLE, dst, 0, MPI_COMM_WORLD, &send_req);

// Do compute work while communication is in flight
do_local_computation();

// Wait for both to complete
MPI_Wait(&send_req, MPI_STATUS_IGNORE);
MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
```

`MPI_Waitall` blocks until all requests in an array complete:

```c
MPI_Request requests[2] = {send_req, recv_req};
MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
```

The key rule: **never read or write the buffer of a non-blocking operation until it has completed**. The buffer is "owned" by MPI during the operation.

## Collective Operations

Collectives involve all processes in a communicator simultaneously. They are the expressive, efficient way to do common communication patterns.

### MPI_Bcast: One-to-All

```c
int data;
if (rank == 0) data = 99;  // Only rank 0 initializes

MPI_Bcast(&data,          // buffer (source on root, destination on others)
           1,              // count
           MPI_INT,        // datatype
           0,              // root rank
           MPI_COMM_WORLD);

// After Bcast, all ranks have data == 99
```

### MPI_Reduce: All-to-One

```c
double local_sum = compute_local_sum();
double global_sum;

MPI_Reduce(&local_sum,    // send buffer
           &global_sum,   // receive buffer (meaningful only on root)
           1,             // count
           MPI_DOUBLE,
           MPI_SUM,       // operation
           0,             // root rank
           MPI_COMM_WORLD);

if (rank == 0)
    printf("Global sum: %f\n", global_sum);
```

Operations include `MPI_SUM`, `MPI_MAX`, `MPI_MIN`, `MPI_PROD`, `MPI_LAND` (logical and), and user-defined operations via `MPI_Op_create`.

### MPI_Allreduce: All-to-All Reduce

Like `MPI_Reduce` but every process gets the result. This is the operation underlying distributed gradient aggregation in ML training:

```c
double local_grad = compute_local_gradient();
double global_grad;

MPI_Allreduce(&local_grad, &global_grad, 1, MPI_DOUBLE,
              MPI_SUM, MPI_COMM_WORLD);

// All ranks now have the sum of gradients, can update weights
update_weights(global_grad / size);
```

### MPI_Scatter and MPI_Gather

Scatter distributes data from a root to all processes; Gather collects from all processes to a root:

```c
// Root has data[size * chunk_size], others receive chunk_size elements
double *all_data = NULL;
if (rank == 0) all_data = malloc(size * chunk_size * sizeof(double));
double *local_data = malloc(chunk_size * sizeof(double));

MPI_Scatter(all_data, chunk_size, MPI_DOUBLE,   // send
            local_data, chunk_size, MPI_DOUBLE,  // receive
            0, MPI_COMM_WORLD);

// Each process works on local_data
process(local_data, chunk_size);

// Collect results back
MPI_Gather(local_data, chunk_size, MPI_DOUBLE,   // send
           all_data, chunk_size, MPI_DOUBLE,      // receive
           0, MPI_COMM_WORLD);
```

`MPI_Allgather` and `MPI_Alltoall` are the variants where every process receives the collected data.

## Derived Datatypes

Sending non-contiguous data without copying to a temporary buffer uses MPI derived datatypes.

```c
// Send every other element of an array (stride 2)
MPI_Datatype strided_type;
MPI_Type_vector(
    count,            // number of blocks
    1,                // elements per block
    2,                // stride (in elements)
    MPI_DOUBLE,
    &strided_type);
MPI_Type_commit(&strided_type);

MPI_Send(data, 1, strided_type, dst, 0, MPI_COMM_WORLD);

MPI_Type_free(&strided_type);
```

For structured data (C structs), `MPI_Type_create_struct` creates a datatype matching an arbitrary struct layout. This is often more trouble than packing fields manually; in practice, many codebases avoid derived datatypes in favor of explicit packing routines, trading MPI complexity for code they can test and debug independently.

## Communicators and Topologies

`MPI_COMM_WORLD` covers all processes, but real applications often need subsets.

```c
// Split processes into groups based on rank parity
int color = rank % 2;  // 0 for even, 1 for odd
MPI_Comm subcomm;
MPI_Comm_split(MPI_COMM_WORLD, color, rank, &subcomm);

// Each process is now in a 2-process or larger sub-communicator
int sub_rank, sub_size;
MPI_Comm_rank(subcomm, &sub_rank);
MPI_Comm_size(subcomm, &sub_size);

// Collective on the sub-communicator
MPI_Barrier(subcomm);

MPI_Comm_free(&subcomm);
```

**Cartesian topologies** are useful for structured grid computations where each process has neighbors in 2D or 3D space:

```c
// Create a 2D Cartesian grid of processes
int dims[2] = {0, 0};  // Let MPI choose dimensions
MPI_Dims_create(size, 2, dims);

int periods[2] = {1, 1};  // Periodic boundaries (torus)
MPI_Comm cart_comm;
MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

// Find neighbors
int north, south, east, west;
MPI_Cart_shift(cart_comm, 0, 1, &north, &south);  // dimension 0
MPI_Cart_shift(cart_comm, 1, 1, &west, &east);    // dimension 1

// Halo exchange with neighbors
MPI_Sendrecv(south_edge, n, MPI_DOUBLE, south, 0,
             north_halo, n, MPI_DOUBLE, north, 0,
             cart_comm, MPI_STATUS_IGNORE);
```

## GPU-Aware MPI

Modern MPI implementations (OpenMPI 4+, MVAPICH2-GDR, Cray MPICH) support GPU-aware communication: passing device pointers directly to MPI routines without staging through host memory.

```c
// GPU-aware MPI: d_send_buf is a device (GPU) pointer
cudaMalloc(&d_send_buf, n * sizeof(float));
cudaMalloc(&d_recv_buf, n * sizeof(float));

// MPI sees the device pointer and uses GPUDirect RDMA
// No cudaMemcpy to host required
MPI_Sendrecv(d_send_buf, n, MPI_FLOAT, dst, 0,
             d_recv_buf, n, MPI_FLOAT, src, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
```

This requires:
- NVIDIA GPUDirect RDMA (for NVIDIA hardware)
- An MPI implementation compiled with GPU support
- InfiniBand or other RDMA-capable network
- `OMPI_MCA_osc=rdma` or equivalent configuration

When it works, GPU-aware MPI eliminates the device→host→network→host→device copy chain. The bandwidth and latency improvement is significant — often 3–5× on large messages.

When it does not work (driver version mismatch, unsupported hardware, misconfiguration), you get silent corruption or crashes. The debugging experience for GPU-aware MPI failures is not pleasant.

```bash
# Verify GPU-aware MPI is available
ompi_info --parseable | grep -i "MPI extensions"
# Or check for CUDA support
ompi_info | grep -i "cuda"
```

## MPI + OpenMP Hybrid Programming

Single-node parallelism with OpenMP combined with multi-node MPI is the standard hybrid model for CPU-heavy HPC codes.

```c
// Launch with: mpirun -n 8 --bind-to socket ./hybrid
// Each MPI process uses all cores on its socket via OpenMP

#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv) {
    // MPI_THREAD_FUNNELED: only master thread calls MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double local_result = 0.0;

    #pragma omp parallel reduction(+:local_result)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        // Each thread handles a slice of local work
        local_result += compute_slice(rank, size, tid, nthreads);
    }

    double global_result;
    MPI_Reduce(&local_result, &global_result, 1, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Finalize();
}
```

Thread safety levels in MPI (`MPI_THREAD_SINGLE`, `MPI_THREAD_FUNNELED`, `MPI_THREAD_SERIALIZED`, `MPI_THREAD_MULTIPLE`) must match how your code uses MPI from threads. `MPI_THREAD_FUNNELED` is the safest hybrid approach: only the main thread calls MPI functions; OpenMP threads handle computation. `MPI_THREAD_MULTIPLE` (any thread can call MPI) has overhead and is rarely necessary.

## Performance Considerations

**Latency vs. bandwidth**: Small messages are latency-dominated; large messages are bandwidth-dominated. On InfiniBand HDR (200 Gb/s), latency is ~1–2 μs for small messages; bandwidth peaks around 20 GB/s. Structure your communication to minimize the number of small messages, even if it means aggregating messages manually.

**Collective algorithm selection**: MPI implementations select different algorithms for collectives based on message size and process count. An AllReduce of 4 bytes on 1024 processes uses a different algorithm than AllReduce of 1 GB. Understanding which algorithm is selected (usually via `OMPI_MCA_coll_base_verbose=10` or equivalent) matters when tuning.

**Non-blocking collectives (MPI-3)**: `MPI_Iallreduce`, `MPI_Ibcast`, etc. enable overlapping computation and collective communication, similar to non-blocking point-to-point:

```c
MPI_Request req;
MPI_Iallreduce(&local_grad, &global_grad, n, MPI_DOUBLE,
               MPI_SUM, MPI_COMM_WORLD, &req);

// Overlap with other computation
next_batch_forward_pass();

MPI_Wait(&req, MPI_STATUS_IGNORE);
// Now global_grad is ready
```

**Process placement**: Binding MPI processes to specific CPUs (`--bind-to core`, `--bind-to socket`) prevents process migration and ensures NUMA-local memory access. On NUMA systems, a process that migrates across sockets pays a 2–3× memory latency penalty. Always specify process binding in production MPI jobs.

## The Modern Reality

MPI is not glamorous. The syntax is verbose, the error messages are cryptic, and debugging a deadlock on 512 nodes is an experience that makes you reconsider your career choices. It is also the only mature, widely-deployed solution for distributed-memory parallelism at scale.

The ML community has largely reinvented MPI's AllReduce under names like "distributed data parallel" and "gradient synchronization," with libraries like NCCL and RCCL doing the actual communication. NCCL is excellent for GPU collectives on NVIDIA hardware. It is not a replacement for MPI when you need general-purpose distributed-memory programming, process management across heterogeneous nodes, or communication on CPU buffers.

For new code that needs multi-node GPU distribution and fits into the ML training pattern (data parallel, gradient averaging), NCCL + PyTorch DDP or JAX pmap is likely the right abstraction. For anything that does not fit that pattern — general scientific computing, irregular communication, CPU-GPU mixed pipelines — MPI remains the right tool.
