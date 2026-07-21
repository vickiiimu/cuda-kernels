#include "common.cuh"

__global__ void CoarsenedSumReductionKernel(float* input, float* output){
    __shared__ float input_s[blockDim.x];
    unsigned int segment = 2 * blockDim.x * blockIdx.x * COARSE_FACTOR;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;

    float sum = input[i];
    for (unsigned int tile = 1; tile < COARSE_FACTOR; ++ tile){
        sum += input[i + tile * blockDim.x];
    }
    input_s[t] = sum;
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2){
        __syncthreads();
        if (threadIdx.x < stride){
            input_s[t] += input_s[t + stride];
        }
    }
    if (t == 0){
        atomicAdd(output, input_s[0]);
    }
}