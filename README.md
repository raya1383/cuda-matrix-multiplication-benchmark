# Matrix Multiplication on CPU and GPU Using CUDA

This project implements and benchmarks matrix multiplication on both CPU and NVIDIA GPU using several different approaches.

The goal is to compare simple implementations with optimized CUDA kernels and NVIDIA's native linear algebra libraries while considering not only computation time, but also memory-transfer and kernel-launch overheads.

## Implemented Methods

The following implementations are included:

- Naive CPU matrix multiplication
- Blocked and multithreaded CPU implementation
- Naive CUDA kernel
- Optimized CUDA kernel using shared memory and tiling
- cuBLAS matrix multiplication
- cuSPARSE matrix multiplication

The implementations are evaluated for three data types:

- `int`
- `float`
- `double`

## CUDA Optimization

The optimized CUDA implementation uses a tiled matrix multiplication algorithm.

Each CUDA block processes a tile of the output matrix. Tiles from matrices A and B are first loaded from global memory into shared memory.

```cpp
__shared__ T tile_a[TILE][TILE];
__shared__ T tile_b[TILE][TILE];
```

Threads inside the block then reuse these values during computation.

The implementation uses:

- Shared memory
- 2D CUDA blocks
- Matrix tiling
- Coalesced global-memory access
- Thread synchronization using `__syncthreads()`

The tile size used in the implementation is:

```cpp
TILE = 16
```

This reduces repeated global-memory accesses compared with the naive CUDA implementation.

## Timing Measurements

CUDA streams and events are used to measure the main GPU execution components.

The following metrics are recorded:

| Metric | Description |
|---|---|
| `h2d_ms` | Host-to-device memory transfer |
| `compute_ms` | Kernel or library execution time |
| `d2h_ms` | Device-to-host memory transfer |
| `total_gpu_ms` | Complete GPU execution time |
| `total_wall_ms` | End-to-end wall-clock time |
| `launch_overhead_us` | Kernel/library launch overhead |
| `setup_ms` | Allocation and library setup time |

For CPU/GPU crossover analysis, the complete GPU execution time is considered rather than kernel time alone.

## Performance Analysis

The benchmark scripts generate performance plots for all three data types.

The analysis includes:

- Total execution time
- CUDA timing breakdown
- CPU/GPU crossover point
- Kernel compute throughput
- Comparison of naive CUDA, tiled CUDA, cuBLAS, and cuSPARSE
- Impact of data type
- Effect of matrix sparsity on cuSPARSE

Example result:

![Total execution time](report/total_execution_time_float.png)

## CPU vs GPU Crossover

For small matrices, CPU execution can be faster because GPU execution includes fixed costs such as:

- memory allocation
- host-to-device transfer
- kernel launch
- device-to-host transfer

As the matrix size increases, the computational cost grows approximately as:

```text
O(N^3)
```

while the amount of transferred matrix data grows approximately as:

```text
O(N^2)
```

For sufficiently large matrices, GPU parallelism compensates for these fixed overheads.

The crossover point is calculated separately for `int`, `float`, and `double`.

Example:

![CPU GPU Crossover](report/gpu_vs_cpu_cross_over_point_total_time_basis_float.png)

## Compute Throughput

Kernel throughput is estimated using approximately:

```text
2 * N^3
```

floating-point or integer operations for an `N x N` matrix multiplication.

Throughput is reported in GOPS using kernel execution time only.

Example:

![Compute Throughput](report/compute_throughput_kernel_time_only_tesla_t4_float.png)

## cuBLAS

cuBLAS provides NVIDIA's optimized dense linear-algebra implementation.

Its matrix multiplication routines generally outperform manually written kernels for sufficiently large matrices because they use highly optimized strategies for:

- memory access
- register usage
- shared memory
- instruction scheduling
- architecture-specific kernels

## cuSPARSE

cuSPARSE is designed for sparse matrices.

The project converts the matrix into CSR format before performing sparse matrix multiplication.

The CSR representation stores:

- non-zero values
- column indices
- row offsets

For dense matrices, cuSPARSE may be slower than cuBLAS because of CSR conversion and metadata overhead.

Its benefit becomes more visible as matrix sparsity increases.

## Multi-GPU Extension

Two strategies for extending the implementation to two GPUs are discussed in the project report.

### Row-wise decomposition

The rows of matrix A and matrix C are divided between two GPUs.

```text
GPU 0: C0 = A0 * B
GPU 1: C1 = A1 * B
```

This approach does not require a reduction step.

### Inner-dimension decomposition

The shared matrix dimension is divided between two GPUs.

```text
GPU 0: C0 = A0 * B0
GPU 1: C1 = A1 * B1

C = C0 + C1
```

This approach requires an additional reduction step.

## Project Structure

```text
.
├── src/
│   └── main.cu
├── scripts/
│   ├── analyze_results.py
│   ├── run_benchmarks.sh
│   └── run_sparse_sweep.sh
├── notebooks/
│   └── matrix_cuda_colab_clean.ipynb
├── results/
├── report/
├── Makefile
├── CMakeLists.txt
└── requirements.txt
```

## Build

A CUDA-capable NVIDIA GPU and CUDA Toolkit are required.

Compile using:

```bash
make
```

Check available CUDA devices:

```bash
./matrix_bench --list-devices
```

Example execution:

```bash
./matrix_bench 512 float \
    --iters 7 \
    --warmup 2
```

## Running the Benchmarks

The benchmark script can be executed using:

```bash
BINARY=./matrix_bench \
SIZES="64 128 256 512 768 1024 1536 2048" \
ITERS=7 \
WARMUP=2 \
CPU_THREADS=2 \
bash scripts/run_benchmarks.sh
```

## Generating Plots

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python scripts/analyze_results.py \
    --input results/results.csv \
    --outdir report \
    --gpu-name "Tesla T4"
```

This generates the CSV summaries and performance plots used in the report.

## Google Colab

A Colab notebook is provided in:

```text
notebooks/matrix_cuda_colab_clean.ipynb
```

The notebook can be used to compile and run the CUDA implementation on a GPU-enabled Google Colab runtime.

## Report

The complete project report is available in:

```text
report/report.pdf
```

The corresponding LaTeX source is also included.

## Technologies

- C++
- CUDA
- cuBLAS
- cuSPARSE
- Python
- Matplotlib
- Pandas
- LaTeX
