/**
 * @file spectral_partition.cuh
 * @author Archie Shahidullah (archie@caltech.edu)
 * 
 */

#ifndef __SPECTRAL_PARTITION_CUH__
#define __SPECTRAL_PARTITION_CUH__


struct elem {
    float data;
    int idx;
};


int compare_elem(const void *a, const void *b);


void cudaCallSubtractKernel(const unsigned int blocks,
                            const unsigned int threadsPerBlock,
                            float *A,
                            float *B,
                            float *C,
                            const unsigned int length);


void cudaCallDivisionKernel(const unsigned int blocks,
                            const unsigned int threadsPerBlock,
                            float *A,
                            float *val,
                            const unsigned int length);


void cudaCallDiagonalKernel(const unsigned int blocks,
                            const unsigned int threadsPerBlock,
                            float *A,
                            float *out,
                            const unsigned int length);


#endif // __SPECTRAL_PARTITION_CUH__