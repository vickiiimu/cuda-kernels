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
Run on: NVIDIA L40S

n = 1,073,741,824 elements

    Avg kernel time: 18.403294 ms
    Achieved bandwidth: 700.141081 GB/s
    Achieved compute: 58.345090 GFLOPS
    Arithmetic intensity: 0.083333 FLOPs/byte

Vec_add is memory-bound (low arithmetic intensity), so bandwidth vs.
GPU peak is the meaningful metric here, not GFLOPS.

## Peak bandwidth for reference
NVIDIA L40S theoretical peak: 864 GB/s

Achieved: ~81% of peak

Note: A smaller benchmark with n = 1,048,576 reported 2270 GB/s effective
bandwidth, but Nsight Compute showed a 99.9% L2 hit rate, meaning that result
measured warm-cache performance rather than sustained DRAM bandwidth.