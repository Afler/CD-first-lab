import numpy as np


class LinearCode:
    A = np.mat([[]], dtype=int)
    G = np.mat([[]], dtype=int)
    X = np.mat([[]], dtype=int)
    H = np.mat([[]], dtype=int)
    allowed_words = np.mat([[]], dtype=int)
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
                    if np.array_equal(B[k], row):
                        isAppend = False
                        break
                if isAppend:
                    B = np.vstack([B, row])
        B = np.vstack([B, np.zeros(n, dtype=int)])
        self.A = B
        self.G = RREF(self.A)
        m = self.G.shape[0]
        n = self.G.shape[1]
        for i in range(m):
            if np.array_equal(np.ravel(self.G[i]), np.zeros(n, dtype=int)):
                self.G = self.G[0: i]
                break
        print("G =\n", self.G)
        self.n = self.G.shape[1]
        self.k = self.G.shape[0]
        print("n = ", self.n)
        print("k = ", self.k)
        self.X = RREF(self.G)
        delete_columns = []
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                if self.X[i, j] != 0:
                    delete_columns.append(j)
                    break
        print("lead = ", delete_columns)
        self.X = np.delete(self.X, delete_columns, 1)
        I = np.mat(np.eye(self.X.shape[1], dtype=int))
        print("I =\n", I)
        print("X =\n", self.X)
        self.H = np.zeros([self.X.shape[0] + I.shape[0], self.X.shape[1]], dtype=int)
        iter_k = 0
        iter_j = 0
        for i in range(self.X.shape[0] + I.shape[0]):
            if i in delete_columns:
                self.H[[i]] = self.X[iter_j]
                iter_j += 1
            else:
                self.H[[i]] = I[iter_k]
                iter_k += 1
        print("H =\n", self.H)

        # task 1.4.1
        allowed_words = self.G.copy()
        for i in range(self.G.shape[0] - 1):
            for j in range(i + 1, self.G.shape[0]):
                row = (self.G[i] + self.G[j]) % 2
                isAppend = True
                for k in range(0, allowed_words.shape[0]):
                    if np.array_equal(allowed_words[k], row):
                        isAppend = False
                        break
                if isAppend:
                    allowed_words = np.vstack([allowed_words, row])
        print("allowed words =\n", allowed_words)

        # task 1.4.2
        I = np.mat(np.eye(self.k, dtype=int))
        k_length_words = I.copy()
        index_to_add = I.shape[0]
        for i in range(index_to_add):
            for j in range(0, I.shape[0]):
                row = (I[i] + I[j]) % 2
                isNotAppend = True
                for k in range(0, k_length_words.shape[0]):
                    if np.array_equal(k_length_words[k], row):
                        isNotAppend = False
                        break
                if isNotAppend:
                    k_length_words = np.vstack([k_length_words, row])
                    index_to_add += 1

        print("k_length_words =\n", k_length_words)
        print(k_length_words @ self.G)


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
    print("REF(A) =\n", REF(A))
    print("RREF(A) = \n", RREF(A))
    linearcode = LinearCode(A)
