import numpy as np


def REF(B):
    A = B.copy()
    m = A.shape[0]  # rows number
    n = A.shape[1]  # columns number
    if m == n == 1:
        return A
    for j in range(n):  # by columns
        lead = -1
        for i in range(j, m):  # by rows
            if A[i, j] != 0:
                lead = A[i, j]
                if i != j:
                    A[[i, 0]] = A[[0, i]]
                A[i] = A[i] / lead % 2
                for k in range(j + 1, m):
                    A[k] -= A[j] * A[k, j] % 2
        # if lead < 0 and not A[:, j].any():
        # A = np.delete(A, j, 1)
        # return REF(A)p
        A[1:n, 1:m] = REF(A[1:n, 1:m])
    return A


if __name__ == '__main__':
    A = np.mat([[0, 1, 0],
                [1, 0, 1],
                [0, 1, 0]], dtype=int)

    A = REF(A)
    print(A)
