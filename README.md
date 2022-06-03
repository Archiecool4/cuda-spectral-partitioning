# CUDA-Accelerated Spectral Partitioning

Created as the final project for CS 179, the GPU programming course at Caltech

## Introduction

Graph partitioning is the problem of dividing the vertices of a graph into non-overlapping groups such that the number of edges between these groups is minimized. It has been proven the problem is NP-hard, which motivates a search for heuristics to approximate a solution. This project uses CUDA to accelerate [Fiedler's spectral partioning algorithm](https://en.wikipedia.org/wiki/Algebraic_connectivity#Partitioning_a_graph_using_the_Fiedler_vector) for partitioning an undirected, acyclic graph into two connected components.

## Algorithm Overview

- Given: A (adjacency matrix), n1 (partition size), n (number of vertices)
- Objective: Partition graph into groups of size n1 and n - n1
- Compute degree matrix D by calculating the row-sums of A
- Compute Laplacian matrix L = D - A
- Compute the Fiedler vector S, the (normalized) eigenvector corresponding to the second-smallest eigenvalue
- Set partition S1 to 1 for indices that correspond to the largest n1 elements of S and -1 otherwise
- Compute the cut size R1 = (1 / 4) .* S1' * L * S1
- Set partition S2 to 1 for indices that correspond to the smallest n - n1 elements of S and -1 otherwise
- Compute the cut size R2 = (1 / 4) .* S2' * L * S1
- Choose S1 or S2 corresponding to min(R1, R2)

## Dependencies
- CUDA
- cuBLAS
- cuSolver

## Running
In the file `spectral_partition.cpp`, the function `get_graph()` will construct a sample graph for partitioning. If you would like to try a different graph, you can construct your own adjacency matrix in there. The partitions are computed but are not returned as of now.

To compile simply run,

```
make
```

To run the program, use,

```
./spectral_partition threadsPerBlock numBlocks numVertices partitionSize
```

For example,

```
./spectral_partition 128 128 1000 400
```

This should output something along the lines of,

```
R1:     3.000000
R2:     3.000000
Execution Time: 391.230 ms
```

## Performance
A Python script written with NumPy is included to demonstrate a CPU implementation. NumPy was chosen since it uses optimized, pre-compiled C++ modules. This makes it an ideal comparison with this GPU-accelerated implementation. 

To run this Python script, use,

```
python3 spectral_partition.py numVertices partitionSize
```

For example,

```
python3 spectral_partition.py 1000 400
```

This should output something like,

```
R1:     3.0
R2:     3.0
Execution Time: 1.438 s
```

For small graphs, the Python script is much faster due to the overhead in allocating GPU resources. However, once we pass roughly 500 vertices, the GPU implementation becomes increasingly faster. In the example above, the GPU implemention is approximately 4x faster. For 2000 vertices, the GPU implementation is roughly 15x faster. For 3000 vertices, the GPU implementation is roughly 31x faster. In fact, the GPU implementation can compute the algorithm for 10000 vertices faster than the Python script can compute 3000 vertices. This shows the benefit of accelerating large matrix operations on the GPU as the Laplacian matrix (for which we solve the eigenvalue problem) has `numVertices ** 2` elements.