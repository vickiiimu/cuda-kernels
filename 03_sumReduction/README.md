# Sum Reduction Kernel Progression (CUDA)

Implementing the sum reduction optimization progression from Chapter 10
of *Programming Massively Parallel Processors*, benchmarking each step
against the basic reduction kernel — another step toward building custom
kernels for training nanoGPT from scratch.

## Files

* `common.cuh` — shared CUDA_CHECK macro, benchmark harness, and verification
* `01_basic_reduction.cu` — basic parallel sum reduction
* `02_convergent_sum.cu` — reorganize threads to minimize control divergence
* `03_shared_memory_reduction.cu` — perform the reduction using shared memory
* `04_thread_coarsening.cu` — each thread sums multiple input elements

## Build

```
nvcc -arch=sm_89 -O3 01_basic_reduction.cu -o 01_basic_reduction
```

## Run

```
./01_basic_reduction
```

(repeat build/run per numbered file, swapping the filename)

## What this does

* `common.cuh` — shared error-checking, timing, CPU reference sum, and
  verification helpers reused across every kernel file
* `01_basic_reduction` — basic reduction where increasingly fewer
  threads participate during each reduction step
* `02_minimized_divergence` — changes the thread indexing so active
  threads are grouped together, reducing warp divergence
* `03_shared_memory_reduction` — loads input values into shared memory
  and performs the block-level reduction there
* `04_thread_coarsening` — each thread accumulates multiple input
  elements before participating in the shared-memory reduction

Each block produces one partial sum. Additional kernel launches reduce
the partial sums until only one final value remains.

## Benchmark results

Run on: NVIDIA L40S
Input size: N = __________
Threads per block: __________
Iterations: 20 (+ 1 discarded warm-up run)

```
01 basic reduction:          ___ ms   ___ GB/s   1.00× (baseline)
02 minimized divergence:     ___ ms   ___ GB/s   ___× faster
03 shared memory reduction:  ___ ms   ___ GB/s   ___× faster
04 thread coarsening:        ___ ms   ___ GB/s   ___× faster
```

Sum reduction is memory-bound because each input element requires a
global-memory load but only approximately one floating-point addition.

Effective bandwidth is therefore the most meaningful performance metric:

```
Effective bandwidth = N × sizeof(float) / kernel time
```

The final GPU result is verified against a sequential CPU sum. Because
floating-point addition may occur in a different order on the GPU,
verification uses a numerical tolerance rather than exact equality.