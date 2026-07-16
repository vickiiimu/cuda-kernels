# Matmul Kernel Progression (CUDA)

Implementing Simon Boehm's CUDA matmul optimization series
(https://siboehm.com/articles/22/CUDA-MMM), benchmarking each
step against cuBLAS — second step in building toward custom
kernels for training nanoGPT from scratch.

## Files
- `common.cuh` — shared CUDA_CHECK macro, CUBLAS_CHECK macro, benchmark harness, verification
- `00_cublas_reference.cu` — cuBLAS baseline, the ground truth to compare against
- `01_naive.cu` — one thread per output element, no optimization
- `02_global_mem_coalescing.cu` — reorder thread indexing for coalesced access
- `03_shared_mem_blocking.cu` — tile into shared memory
- `04_1d_blocktiling.cu` — each thread computes multiple outputs
- `05_2d_blocktiling.cu` — 2D register tiling
- `06_vectorized_mem_access.cu` — float4 loads/stores
- `07_warptiling.cu` — warp-level tiling

## Build
    nvcc -arch=sm_80 -O3 01_naive.cu -o 01_naive -lcublas

## Run
    ./01_naive

(repeat build/run per numbered file, swapping the filename)

## What this does
- `common.cuh` — shared error-checking macro and timing/verification
  helpers reused across every kernel file, so each one only contains
  the kernel-specific logic
- `00_cublas_reference` — calls `cublasSgemm`, establishes the
  performance ceiling every later kernel is compared against
- `01_naive` through `07_warptiling` — each file implements exactly
  one optimization step on top of the previous file's approach,
  benchmarked independently and identically

## Benchmark results
Run on: [GPU name, e.g. "A100, Engaging cluster"]
Matrix size: M = N = K = 4096
Iterations: 20 (+ 1 discarded warm-up run)

    00 cuBLAS reference:        ___ ms   ___ GFLOPS   100% (baseline)
    01 naive:                   ___ ms   ___ GFLOPS   ___% of cuBLAS
    02 global mem coalescing:   ___ ms   ___ GFLOPS   ___% of cuBLAS
    03 shared mem blocking:     ___ ms   ___ GFLOPS   ___% of cuBLAS
    04 1D blocktiling:          ___ ms   ___ GFLOPS   ___% of cuBLAS
    05 2D blocktiling:          ___ ms   ___ GFLOPS   ___% of cuBLAS
    06 vectorized mem access:   ___ ms   ___ GFLOPS   ___% of cuBLAS
    07 warptiling:              ___ ms   ___ GFLOPS   ___% of cuBLAS

Matmul is compute-bound (high arithmetic intensity, grows with matrix
size), so GFLOPS vs. cuBLAS is the meaningful metric here, not bandwidth.

## Peak compute for reference
[GPU name] theoretical peak: ___ TFLOPS ([FP32/TF32/FP16], specify which)
cuBLAS achieved: ___% of theoretical peak