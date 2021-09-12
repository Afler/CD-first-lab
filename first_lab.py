import numpy as np


def REF(B):
    A = B.copy()
    m = A.shape[0]  # rows number
    n = A.shape[1]  # columns number
    for j in range(n):  # by columns
        lead = -1
        for i in range(j, m):  # by rows
            if A[i, j] != 0:
                lead = A[i, j]
                if i != j:
                    A[[i, j]] = A[[j, i]]
                A[i] = A[i] / lead % 2
                for k in range(i + 1, m):
                    A[k] = (A[k] - A[j] * A[k, j]) % 2
    return A


if __name__ == '__main__':
    A = np.mat([[0, 0, 0],
                [1, 0, 1],
                [0, 1, 1]], dtype=int)

    A = REF(A)
    print(A)
