#include "common.cuh"

__global__ void ConvergentSumReductionKernel(float* input, float* output){
    unsigned int i = threadIdx.x;
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2){
        if (threadIdx.x < stride){
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        *output = input[0];
    }
}