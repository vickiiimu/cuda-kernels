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

template <typename MatMulLaunchFn>
void runSmallMatMulCheck(MatMulLaunchFn matMulLaunch) {
    // Small, non-square, distinct values — chosen specifically so that
    // a transpose or argument-swap bug produces a visibly WRONG result,
    // unlike the all-ones test which can't distinguish correct from wrong.
    int M = 2, K = 3, N = 4;
    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);

    // A (2x3, row-major): [[1, 2, 3], [4, 5, 6]]
    float h_A[] = {1, 2, 3, 4, 5, 6};
    // B (3x4, row-major): [[7, 8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18]]
    float h_B[] = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};

    // Hand-computed expected C = A x B (2x4, row-major):
    // C[0][0] = 1*7 + 2*11 + 3*15 = 74
    // C[0][1] = 1*8 + 2*12 + 3*16 = 80
    // C[0][2] = 1*9 + 2*13 + 3*17 = 86
    // C[0][3] = 1*10 + 2*14 + 3*18 = 92
    // C[1][0] = 4*7 + 5*11 + 6*15 = 173
    // C[1][1] = 4*8 + 5*12 + 6*16 = 188
    // C[1][2] = 4*9 + 5*13 + 6*17 = 203
    // C[1][3] = 4*10 + 5*14 + 6*18 = 218
    float h_C_expected[] = {74, 80, 86, 92, 173, 188, 203, 218};

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&d_B, sizeB));
    CUDA_CHECK(cudaMalloc((void**)&d_C, sizeC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    matMulLaunch( d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_C[8];
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    bool correct = verifyMatMul(h_C, h_C_expected, M, N);
    printf(correct ? "Transpose/argument-order check PASSED.\n"
                   : "Transpose/argument-order check FAILED.\n");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
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