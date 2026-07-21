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

//---------- Verification ----------
template <typename SumReductionLaunchFn>
void runSmallMatMulCheck(SumReductionLaunchFn sumReductionLaunch) {
    // Small input with distinct values so missing, duplicated, or
    // incorrectly indexed elements produce an obviously wrong result.
    int N = 10;
    size_t inputSize = (size_t)N * sizeof(float);

    float h_input[] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
        6.0f, 7.0f, 8.0f, 9.0f, 10.0f
    }; 

    // Hand-computed expected sum:
    // 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 = 55
    float expectedSum = 55.0f;

    float* d_input;
    float* d_output;

    CUDA_CHECK(cudaMalloc((void**)&d_input, inputSize));
    CUDA_CHECK(cudaMalloc((void**)&d_output, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        h_input,
        inputSize,
        cudaMemcpyHostToDevice
    ));

    sumReductionLaunch(d_input, d_output, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float gpuSum;
    CUDA_CHECK(cudaMemcpy(
        &gpuSum,
        d_output,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    float absoluteError = std::fabs(gpuSum - expectedSum);
    bool correct = absoluteError < 1e-5f;

    printf("Expected sum: %.6f\n", expectedSum);
    printf("GPU sum:      %.6f\n", gpuSum);

    printf(correct ? "Small sum reduction check PASSED.\n"
                   : "Small sum reduction check FAILED.\n");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
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