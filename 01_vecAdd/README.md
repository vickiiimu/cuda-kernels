# Vector Add (CUDA)

Basic CUDA vector addition kernel — first step in building toward
custom kernels for training nanoGPT from scratch.

## Files
- `vecAdd.cu` — kernel, host wrapper, and benchmark harness

## Build
    nvcc vecAdd.cu -o vecAdd

## Run
    ./vecAdd

## What this does
- `vecAddKernel` — one thread per element, `d_C[i] = d_A[i] + d_B[i]`
- `vecAdd` — single correctness-check run (alloc, copy, launch, copy back, free)
- `vecAddBenchmark` — 20-iteration timed loop (after 1 warm-up run) using
  CUDA events, reports achieved bandwidth (GB/s) and compute throughput (GFLOPS)

## Benchmark results
Run on: [GPU name, e.g. "A100, Engaging cluster"]
n = 1,048,576 elements

    Avg kernel time: ___ ms
    Achieved bandwidth: ___ GB/s
    Achieved compute: ___ GFLOPS
    Arithmetic intensity: 0.083 FLOPs/byte

Vec_add is memory-bound (low arithmetic intensity), so bandwidth vs.
GPU peak is the meaningful metric here, not GFLOPS.

## Peak bandwidth for reference
[GPU name] theoretical peak: ___ GB/s
Achieved: ___% of peak