#include "common.cuh"

__global__ void SharedMemoryReductionKernel(float* input, float* output){
    __shared__ float input_s[blockDim.x];
    unsigned int i = threadIdx.x;
    input_s[i] = input[i] + input[i + blockDim.x];
    
    for (unsigned int stride = blockDim.x/2; stride >=1; stride /= 2){
        __syncthreads();
        if (threadIdx.x < stride){
            input[i] += input[i + stride];
        }
    }
    if (threadIdx.x == 0){
        *output = input[0];
    }
}