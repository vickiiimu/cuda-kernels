#include "common.cuh"
#include <cublas_v2.h>

void cublasMatmul(cublasHandle_t handle, const float *d_A, const float *d_B, float *d_C, int M, int N, int K) {
    float alpha = 1.0f;
    float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    N, M, K,
    &alpha,
    d_B, N,
    d_A, K,
    &beta,
    d_C, N));
}

void runTransposeCheck(cublasHandle_t handle) {
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

    cublasMatmul(handle, d_A, d_B, d_C, M, N, K);
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

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    runTransposeCheck(handle);

    int M = 4096, N = 4096, K = 4096;
    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);

    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);

    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&d_B, sizeB));
    CUDA_CHECK(cudaMalloc((void**)&d_C, sizeC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    int iterations = 20;
    float avgMs = benchmarkKernel([&]() {
        cublasMatmul(handle, d_A, d_B, d_C, M, N, K);
    }, iterations);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // theoretical min of bytes moved
    size_t bytesMoved = (size_t)(M * K + K * N + M * N) * sizeof(float);
    printMatMulMetrics(M, N, K, avgMs, iterations, bytesMoved, "00_cublas_reference");

    // Sanity check: with A and B filled entirely with 1.0f, every
    // output element should equal K (sum of K products of 1*1)
    float *h_C_expected = (float*)malloc(sizeC);
    for (int i = 0; i < M * N; i++) h_C_expected[i] = (float)K;

    bool correct = verifyMatMul(h_C, h_C_expected, M, N);
    printf(correct ? "Sanity check passed.\n" : "Sanity check FAILED.\n");

    free(h_C_expected);

    free(h_A);
    free(h_B);
    free(h_C);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    cublasDestroy(handle);
    return 0;
}