import numpy as np
import time
import sys

def get_graph(n, x):
    A = np.ones((n, n)) - np.eye(n)
    A[:x, x:] = 0
    A[x:, :x] = 0
    A[0, x + 1] = 1
    A[x + 1, 0] = 1
    A[1, x + 2] = 1
    A[x + 2, 1] = 1
    A[2, x + 3] = 1
    A[x + 3, 2] = 1

    return A

if __name__ == '__main__':
    usage = f'Usage: python3 {sys.argv[0]} numVertices partitionSize'

    if len(sys.argv) != 3:
        print(usage)
        sys.exit(-1)

    args = []
    for arg in sys.argv[1:]:
        try:
            args.append(int(arg))
        except Exception as e:
            print(f'Error: {e}')
            print(usage)
            sys.exit(-1)

    n = args[0]
    x = args[1]
    A = get_graph(n, x)

    start = time.time()

    D = np.diag(np.sum(A, axis=1))
    L = D - A
    _, V = np.linalg.eig(L)
    fiedler = V[:, 1]
    idxs = np.argsort(fiedler)
    S1 = np.zeros(n)
    S1[idxs[:x]] = -1
    S1[idxs[x:]] = 1
    R1 = 0.25 * np.dot(S1, L.dot(S1))
    S2 = np.zeros(n)
    S2[idxs[x:]] = -1
    S2[idxs[:x]] = 1
    R2 = 0.25 * np.dot(S2, L.dot(S2))

    end = time.time()

    print(f'R1:\t{R1}\nR2:\t{R2}')
    print(f'Execution Time:\t{end - start:.3f} s')

