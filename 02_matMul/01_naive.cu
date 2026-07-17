#include "common.cuh"

__global__ void naiveMatMulKernel(const float *d_A, const float *d_B, float *d_C, int M, int N, int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < M) && (col < N)){
        float Cvalue = 0;
        for (int k = 0; k < K; ++k){
            Cvalue += d_A[row * K + k] * d_B[k * N + col];
        }
        d_C[row * N + col] = Cvalue;
    }
}

int main(){
    dim3 threadsPerBlock(16, 16);
    
    runSmallMatMulCheck([&](const float *d_A, const float *d_B, float *d_C, int M, int N, int K){
        dim3 smallBlocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
        naiveMatMulKernel<<<smallBlocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    });

    int M = 4096, N = 4096, K = 4096;
    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);

    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);

    for (size_t i = 0; i < (size_t)M * K; ++i) h_A[i] = 1.0f;
    for (size_t i = 0; i < (size_t)K * N; ++i) h_B[i] = 1.0f;

    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&d_B, sizeB));
    CUDA_CHECK(cudaMalloc((void**)&d_C, sizeC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    int iterations = 20;
    float avgMs = benchmarkKernel([&]() {
        naiveMatMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    }, iterations);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Sanity check: with A and B filled entirely with 1.0f, every
    // output element should equal K (sum of K products of 1*1)
    float *h_C_expected = (float*)malloc(sizeC);
    for (size_t i = 0; i < (size_t)M * N; ++i) h_C_expected[i] = (float)K;

    bool correct = verifyMatMul(h_C, h_C_expected, M, N);
    printf(correct ? "Sanity check passed.\n" : "Sanity check FAILED.\n");

    // theoretical min of bytes moved
    size_t bytesMoved = ((size_t)M * K + (size_t)K * N + (size_t)M * N) * sizeof(float);
    printMatMulMetrics(M, N, K, avgMs, iterations, bytesMoved, "01_naive");

    free(h_C_expected);

    free(h_A);
    free(h_B);
    free(h_C);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return 0;
}