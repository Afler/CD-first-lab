import numpy as np


class LinearCode:
    A = np.mat([[]], dtype=int)
    G = np.mat([[]], dtype=int)
    X = np.mat([[]], dtype=int)
    n = 0
    k = 0
    def __init__(self, A):
        m = A.shape[0]
        n = A.shape[1]
        B = A.copy()
        for i in range(m - 1):
            for j in range(i + 1, m):
                row = (A[i] + A[j]) % 2
                isAppend = True
                for k in range(0, B.shape[0]):
                    if (np.array_equal(B[k], row)):
                        isAppend = False
                        break
                if (isAppend):
                    B = np.vstack([B, row])
        B = np.vstack([B, np.zeros(n, dtype=int)])
        self.A = B
        self.G = RREF(self.A)
        m = self.G.shape[0]
        n = self.G.shape[1]
        for i in range(m):
            if (np.array_equal(np.ravel(self.G[i]), np.zeros(n, dtype=int))):
                self.G = self.G[0: i]
                break
        self.n = self.G.shape[1]
        self.k = self.G.shape[0]
        self.X = RREF(self.G).copy()
        delete_columns =[]
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                if (self.X[i, j] != 0):
                    delete_columns.append(j)
                    break
        print(delete_columns)
        self.X =  np.delete(self.X, delete_columns, 1)

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


def RREF(B):
    C = REF(B).copy()
    m = C.shape[0]
    n = C.shape[1]
    for i in range(m - 1, -1, -1):
        for j in range(n):
            if C[i, j] != 0:
                for k in range(i - 1, -1, -1):
                    C[k, j] = 0

    return C


if __name__ == '__main__':
    A = np.mat([[0, 0, 1, 0, 0],
                [0, 0, 1, 1, 1],
                [1, 1, 1, 0, 1],
                [1, 1, 0, 0, 1]], dtype=int)
    print(REF(A))
    print(RREF(A))
    linearcode = LinearCode(A)
    print(linearcode.A)
    print(linearcode.G)
    print(linearcode.n)
    print(linearcode.k)
    print(linearcode.X)
