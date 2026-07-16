#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("Error: %s in %s at line %d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void vecAddKernel(const float *d_A, const float *d_B, float *d_C, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n){
        d_C[i] = d_A[i] + d_B[i];
    }
}

void vecAdd(const float *h_A, const float *h_B, float *h_C, int n){
    size_t size = n * sizeof(float);
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void vecAddBenchmark(const float *h_A, const float *h_B, float *h_C, int n){
    size_t size = n * sizeof(float);
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    int iterations = 20;
    float totalMs = 0;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // warm-up launch
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < iterations; i++){
        CUDA_CHECK(cudaEventRecord(start));
        vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
        CUDA_CHECK(cudaEventRecord(stop));

        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        totalMs += ms;
    }

    float avgMs = totalMs / iterations;
    size_t bytesMoved = 3 * (size_t)n * sizeof(float); 
    double avgSec = avgMs / 1e3;
    double gbPerSec = (bytesMoved / avgSec) / 1e9;

    double flops = (double)n;
    double gflops = flops / avgSec / 1e9;
    double arithmeticIntensity = gflops / gbPerSec;

    printf("=== vecAdd Benchmark ===\n");
    printf("n = %d elements (%.2f MB per array)\n", n, (n * sizeof(float)) / 1e6);
    printf("Iterations: %d\n", iterations);
    printf("Avg kernel time: %f ms\n", avgMs);
    printf("Bytes moved per launch: %zu\n", bytesMoved);
    printf("Achieved bandwidth: %f GB/s\n", gbPerSec);
    printf("Achieved compute: %f GFLOPS\n", gflops);
    printf("Arithmetic intensity: %f FLOPs/byte\n", arithmeticIntensity);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int n = 1 << 20;
    size_t size = n * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < n; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    vecAdd(h_A, h_B, h_C, n);

    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (h_C[i] != 3.0f) {
            correct = false;
            printf("Mismatch at index %d: got %f, expected 3.0\n", i, h_C[i]);
            break;
        }
    }
    printf(correct ? "Correctness check passed.\n" : "Correctness check FAILED.\n");

    vecAddBenchmark(h_A, h_B, h_C, n);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}