#include "common.cuh"

__global__ void SegmentedSumReduction(float* input, float* output){
    __shared__ float input_s[blockDim.x];
    unsigned int segment = 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    input_s[t] = input[i] + input[i + blockDim.x];

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