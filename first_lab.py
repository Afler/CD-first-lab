import numpy as np


def REF(B):
    A = B.copy()
    m = A.shape[0]  # rows number
    n = A.shape[1]  # columns number
    number_of_leaders = 0
    for j in range(n):  # by columns
        lead = -1
        for i in range(number_of_leaders, m):  # by rows
            if A[i, j] != 0:
                lead = A[i, j]
                A[[i, number_of_leaders]] = A[[number_of_leaders, i]]
                A[i] = A[i] / lead % 2
                for k in range(number_of_leaders + 1, m):
                    A[k] = (A[k] - A[number_of_leaders] * A[k, j]) % 2
                number_of_leaders += 1
    return A

def RREF(A):
    C = A.copy
    m = C.shape[0]
    n = C.shape[1]
    return C


if __name__ == '__main__':
    A = np.mat([[0, 1, 1],
                [0, 0, 1],
                [0, 1, 1]], dtype=int)
    print(REF(A))
