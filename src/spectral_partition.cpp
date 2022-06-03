/**
 * @file spectral_partition.cpp
 * @author Archie Shahidullah (archie@caltech.edu)
 * 
 */

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>
#include <cstring>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "spectral_partition.cuh"
#include "helper_cuda.h"
#include "ta_utilities.hpp"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}


cudaEvent_t start;
cudaEvent_t stop;


#define START_TIMER() {                         \
      gpuErrchk(cudaEventCreate(&start));       \
      gpuErrchk(cudaEventCreate(&stop));        \
      gpuErrchk(cudaEventRecord(start));        \
    }


#define STOP_RECORD_TIMER(name) {                           \
      gpuErrchk(cudaEventRecord(stop));                     \
      gpuErrchk(cudaEventSynchronize(stop));                \
      gpuErrchk(cudaEventElapsedTime(&name, start, stop));  \
      gpuErrchk(cudaEventDestroy(start));                   \
      gpuErrchk(cudaEventDestroy(stop));                    \
    }


/**
 * Constructs a graph for spectral partitioning
 * 
 * @param n number of vertices in graph
 * @param x size of desired partition
 * 
 * @return float array of adjacency matrix
 */
float *get_graph(int n, int x) {
    float *A = (float *) malloc(n * n * sizeof(float));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                A[i+n*j] = 0;
            }
            else if ((i < x && j >= x) || (i >= x && j < x)) {
                A[i+n*j] = 0;
            } 
            else {
                A[i+n*j] = 1;
            }
        }
    }
    A[n*x] = 1;
    A[x] = 1;
    A[1+n*(x+1)] = 1;
    A[(x+1)+n] = 1;
    A[2+n*(x+2)] = 1;
    A[(x+2)+n*2] = 1;

    return A;
}


int main(int argc, char *argv[]) {
    std::string usage = " threadsPerBlock numBlocks numVertices partitionSize";
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << usage << std::endl;
        std::exit(-1);
    }

    std::vector<int> args;
    for (int i = 1; i < argc; i++) {
        try {
            args.push_back(std::stoi(argv[i]));
        }
        catch (std::exception &err) {
            std::cerr << "Error: " << err.what() << std::endl;
            std::cerr << argv[0] << usage << std::endl;
            std::exit(-1);
        }
    }

    int threadsPerBlock = args.at(0);
    int numBlocks = args.at(1);
    int numVertices = args.at(2);
    int partitionSize = args.at(3);

    // TA_Utilities::select_coldest_GPU();
    // int max_time_allowed_in_seconds = 90;
    // TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);

    const unsigned int local_size = threadsPerBlock;
    const unsigned int max_blocks = numBlocks;
    const unsigned int blocks = std::min(max_blocks,
                                         (unsigned int)ceil(1e7 / (float)local_size));

    int n = numVertices;
    int cluster = partitionSize;
    float time = -1;

    float *adjacency = get_graph(n, cluster);

    START_TIMER();

    float ones[n];
    for (int i = 0; i < n; i++) {
        ones[i] = 1;
    }


    float *d_adjacency;
    float *d_sum;
    float *d_ones;
    float *d_degree;
    float *d_laplacian;

    // Allocate GPU memory
    CUDA_CALL(cudaMalloc(&d_adjacency, n * n * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_sum, n * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_ones, n * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_degree, n * n * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_laplacian, n * n * sizeof(float)));

    // Copy adjacency matrix
    CUDA_CALL(cudaMemcpy(d_adjacency, adjacency, n * n * sizeof(float), cudaMemcpyHostToDevice));
    // Zero out sum (for counting degrees)
    CUDA_CALL(cudaMemset(d_sum, 0.0, n * sizeof(float)));

    // cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CALL(cublasCreate(&handle));

    // Initialize vector to 1s in order to compute row-sums via matrix-vector multiplication
    CUBLAS_CALL(cublasSetVector(n, sizeof(float), ones, 1, d_ones, 1));
    
    // Compute row-sums
    float one = 1;
    float zero = 0;
    CUBLAS_CALL(cublasSgemv(handle, CUBLAS_OP_N, n, n, &one, d_adjacency, n, d_ones, 1, &zero, d_sum, 1));

    // Compute degree matrix
    cudaCallDiagonalKernel(blocks, local_size, d_sum, d_degree, n);

    // Compute Laplacian matrix
    cudaCallSubtractKernel(blocks, local_size, d_degree, d_adjacency, d_laplacian, n * n);

    // Make a copy of Laplacian since this will store eigenvectors
    float *d_laplacian_copy;
    CUDA_CALL(cudaMalloc(&d_laplacian_copy, n * n * sizeof(float)));
    CUBLAS_CALL(cublasSetMatrix(n, n, sizeof(float), d_laplacian, n, d_laplacian_copy, n));

    // cuSolver handle
    cusolverDnHandle_t solver_handle;

    // Store eigenvalues of Laplacian
    float *d_lambda;
    // Store fiedler vector
    float *d_fiedler;
    // Check solving the eigenvalue problem
    int info;
    int *d_info;
    // Buffer size
    int lwork;
    float *d_work;

    CUDA_CALL(cudaMalloc(&d_lambda, n * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_fiedler, n * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_info, sizeof(int)));

    CUSOLVER_CALL(cusolverDnCreate(&solver_handle));

    // Compute both eigenvalues and eigenvectors
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; 
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    // Compute and allocate buffer
    CUSOLVER_CALL(cusolverDnSsyevd_bufferSize(solver_handle, jobz, uplo, n, d_laplacian_copy, n, d_lambda, &lwork));
    CUDA_CALL(cudaMalloc(&d_work, lwork * sizeof(float)));

    // Compute eigenvalues and eigenvectors
    CUSOLVER_CALL(cusolverDnSsyevd(solver_handle, jobz, uplo, n, d_laplacian_copy, n, d_lambda, d_work, lwork, d_info));

    // Vector to extract Fiedler vector (2nd smallest eigenvalue)
    float extract[n] = {0};
    extract[1] = 1;
    float *d_extract;

    CUDA_CALL(cudaMalloc(&d_extract, n * sizeof(float)));
    CUBLAS_CALL(cublasSetVector(n, sizeof(float), extract, 1, d_extract, 1));
    
    // Get Fiedler vector
    CUBLAS_CALL(cublasSgemv(handle, CUBLAS_OP_N, n, n, &one, d_laplacian_copy, n, d_extract, 1, &zero, d_fiedler, 1));

    // Sort Fiedler vector to get indices
    float fiedler[n];
    CUBLAS_CALL(cublasGetVector(n, sizeof(float), d_fiedler, 1, fiedler, 1));
    struct elem lst[n];
    for (int i = 0; i < n; i++) {
        lst[i] = (struct elem){fiedler[i], i};
    }
    std::qsort(lst, n, sizeof(struct elem), compare_elem);

    // Compute first partition
    float s1[n];
    float *d_s1;

    // -1 if smallest `n - cluster` elements of Fiedler vector
    int k = 0;
    for (auto &e : lst) {
        if (k < n - cluster) {
            s1[e.idx] = -1;
        }
        else {
            s1[e.idx] = 1;
        }
        k++;
    }

    CUDA_CALL(cudaMalloc(&d_s1, n * sizeof(float)));
    CUBLAS_CALL(cublasSetVector(n, sizeof(float), s1, 1, d_s1, 1));

    // Compute second partition
    float s2[n];
    float *d_s2;

    k = 0;
    for (auto &e : lst) {
        if (k < n - cluster) {
            s2[e.idx] = 1;
        }
        else {
            s2[e.idx] = -1;
        }
        k++;
    }

    CUDA_CALL(cudaMalloc(&d_s2, n * sizeof(float)));
    CUBLAS_CALL(cublasSetVector(n, sizeof(float), s2, 1, d_s2, 1));

    // For computing cut size
    float one_fourth = 0.25;

    float *d_r1;
    float r1;

    CUDA_CALL(cudaMalloc(&d_r1, n * sizeof(float)));

    // (1 / 4) .* S1' * L * S1
    CUBLAS_CALL(cublasSgemv(handle, CUBLAS_OP_N, n, n, &one_fourth, d_laplacian, n, d_s1, 1, &zero, d_r1, 1));
    CUBLAS_CALL(cublasSdot(handle, n, d_s1, 1, d_r1, 1, &r1));
    
    float *d_r2;
    float r2;

    CUDA_CALL(cudaMalloc(&d_r2, n * sizeof(float)));

    // (1 / 4) .* S2' * L * S2
    CUBLAS_CALL(cublasSgemv(handle, CUBLAS_OP_N, n, n, &one_fourth, d_laplacian, n, d_s2, 1, &zero, d_r2, 1));
    CUBLAS_CALL(cublasSdot(handle, n, d_s2, 1, d_r2, 1, &r2));

    STOP_RECORD_TIMER(time);

    // Print cut sizes
    std::printf("R1:\t%f\nR2:\t%f\n", r1, r2);
    std::printf("Execution Time: %.3f ms\n", time);

    // Choose partition with the smallest cut size
    float *partition = r1 <= r2 ? s1 : s2;

    // Free resources
    CUDA_CALL(cudaFree(d_adjacency));
    CUDA_CALL(cudaFree(d_sum));
    CUDA_CALL(cudaFree(d_ones));
    CUDA_CALL(cudaFree(d_degree));
    CUDA_CALL(cudaFree(d_laplacian));
    CUDA_CALL(cudaFree(d_laplacian_copy));

    CUDA_CALL(cudaFree(d_lambda));
    CUDA_CALL(cudaFree(d_fiedler));
    CUDA_CALL(cudaFree(d_work));
    CUDA_CALL(cudaFree(d_info));

    CUDA_CALL(cudaFree(d_extract));
    CUDA_CALL(cudaFree(d_s1));
    CUDA_CALL(cudaFree(d_s2));
    CUDA_CALL(cudaFree(d_r1));
    CUDA_CALL(cudaFree(d_r2));

    CUSOLVER_CALL(cusolverDnDestroy(solver_handle));
    CUBLAS_CALL(cublasDestroy(handle));

    free(adjacency);
}