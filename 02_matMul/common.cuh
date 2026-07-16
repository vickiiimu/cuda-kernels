#ifndef COMMON_CUH
#define COMMON_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//---------- Error Check ----------
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("Error: %s in %s at line %d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            printf("Error: %d in %s at line %d\n", status, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

//---------- Verification ----------
bool verifyMatMul(const float *C, const float *C_ref, int M, int N, float tol = 1e-4f) {
    for (int i = 0; i < M * N; i++) {
        if (fabs(C[i] - C_ref[i]) > tol) {
            printf("Mismatch at index %d: got %f, expected %f\n", i, C[i], C_ref[i]);
            return false;
        }
    }
    return true;
}

//---------- Benchmark Harness ----------
// Runs `kernelLaunch` (a lambda/function wrapping one kernel launch)
// `iterations` times after 1 warm-up run, returns avg ms.
template <typename KernelLaunchFn>
float benchmarkKernel(KernelLaunchFn kernelLaunch, int iterations = 20) {
    float totalMs = 0;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warm-up launch
    kernelLaunch();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < iterations; i++){
        CUDA_CHECK(cudaEventRecord(start));
        kernelLaunch();
        CUDA_CHECK(cudaEventRecord(stop));

        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        totalMs += ms;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    float avgMs = totalMs / iterations;
    return avgMs;
}

//---------- Metrics ----------
void printMatMulMetrics(int M, int N, int K, float avgMs, int iterations,
                        size_t bytesMoved, const char *label){
    double avgSec = avgMs / 1e3;
    double gbPerSec = (bytesMoved / avgSec) / 1e9;

    double flops = 2.0 * M * N * K;
    double gflops = flops / avgSec / 1e9;
    double arithmeticIntensity = flops / bytesMoved;

    printf("=== %s Benchmark ===\n", label);
    printf("M = %d, N = %d, K = %d\n", M, N, K);
    printf("Iterations: %d\n", iterations);
    printf("Avg kernel time: %f ms\n", avgMs);
    printf("Bytes moved per launch: %zu\n", bytesMoved);
    printf("Achieved bandwidth: %f GB/s\n", gbPerSec);
    printf("Achieved compute: %f GFLOPS\n", gflops);
    printf("Arithmetic intensity: %f FLOPs/byte\n", arithmeticIntensity);
}
#endif