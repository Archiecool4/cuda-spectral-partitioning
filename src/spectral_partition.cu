/**
 * @file spectral_partition.cu
 * @author Archie Shahidullah (archie@caltech.edu)
 * 
 */

#include <cuda_runtime.h>

#include "spectral_partition.cuh"
#include "helper_cuda.h"


int compare_elem(const void *a, const void *b) {
    return (((struct elem *)a)->data) > (((struct elem *)b)->data) ? 1 : -1;
}


__global__
void cudaSubtractKernel(float *A, float *B, float *C, int length) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    while (i < length) {
        C[i] = A[i] - B[i];
        i += blockDim.x * gridDim.x;
    }
}


void cudaCallSubtractKernel(const unsigned int blocks,
                            const unsigned int threadsPerBlock,
                            float *A,
                            float *B, 
                            float *C,
                            const unsigned int length) 
{
        

    cudaSubtractKernel<<<blocks, threadsPerBlock>>>
    (
        A, 
        B, 
        C, 
        length
    );
}


__global__
void cudaDivisionKernel(float *A, float *val, int length) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    while (i < length) {
        A[i] /= *val;
        i += blockDim.x * gridDim.x;
    }
}


void cudaCallDivisionKernel(const unsigned int blocks,
                            const unsigned int threadsPerBlock,
                            float *A,
                            float *val, 
                            const unsigned int length) 
{
    
    cudaDivisionKernel<<<blocks, threadsPerBlock>>>
    (
        A, 
        val,
        length
    );
}


__global__ 
void cudaDiagonalKernel(float *A, float *out, int length) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    while (i < length) {
        out[i+length*i] = A[i];
        i += blockDim.x * gridDim.x;
    }
}


void cudaCallDiagonalKernel(const unsigned int blocks,
                            const unsigned int threadsPerBlock,
                            float *A,
                            float *out,
                            const unsigned int length) 
{
    cudaDiagonalKernel<<<blocks, threadsPerBlock>>>
    (
        A, 
        out, 
        length
    );
}