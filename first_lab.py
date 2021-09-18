import numpy as np


def printMatrix(arr, name):
    print(name, " = ")
    print(arr)


class LinearCode:
    G = np.mat([[]], dtype=int)
    n = 0
    k = 0
    X = np.mat([[]], dtype=int)
    H = np.mat([[]], dtype=int)
    allowed_words = np.mat([[]], dtype=int)

    def __init__(self, S):

        # task 1.3.1
        m = S.shape[0]
        for i in range(m - 1):
            for j in range(i + 1, m):
                row = (S[i] + S[j]) % 2
                isAppend = True
                for k in range(0, S.shape[0]):
                    if np.array_equal(S[k], row):
                        isAppend = False
                        break
                if isAppend:
                    S = np.vstack([S, row])
        S = np.vstack([S, np.zeros(S.shape[1], dtype=int)])
        self.G = RREF(S)
        m = self.G.shape[0]
        n = self.G.shape[1]
        for i in range(m):
            if np.array_equal(np.ravel(self.G[i]), np.zeros(n, dtype=int)):
                self.G = self.G[0: i]
                break
        printMatrix(self.G, "G")

        # task 1.3.2
        self.n = self.G.shape[1]
        self.k = self.G.shape[0]
        print("n =", self.n)
        print("k =", self.k)

        # task 1.3.3
        # step 1
        self.X = RREF(self.G)
        # step 2
        delete_columns = []
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                if self.X[i, j] != 0:
                    delete_columns.append(j)
                    break
        print("lead =", delete_columns)
        # step 3
        self.X = np.delete(self.X, delete_columns, 1)
        # step 4
        I = np.mat(np.eye(self.X.shape[1], dtype=int))
        printMatrix(I, "I")
        printMatrix(self.X, "X")
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
        printMatrix(self.H, "H")

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
        printMatrix(allowed_words, "allowed words")

        # task 1.4.2
        I = np.mat(np.eye(self.k, dtype=int))
        k_length_words = I.copy()
        index_to_add = I.shape[0]
        for i in range(index_to_add):
            for j in range(0, I.shape[0]):
                row = (I[i] + I[j]) % 2
                isAppend = True
                for k in range(0, k_length_words.shape[0]):
                    if np.array_equal(k_length_words[k], row):
                        isAppend = False
                        break
                if isAppend:
                    k_length_words = np.vstack([k_length_words, row])
                    index_to_add += 1
        printMatrix(k_length_words, "k_length_words")
        printMatrix(k_length_words @ self.G % 2, "k@G")


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
                break
    return C


if __name__ == '__main__':
    A = np.mat([[0, 0, 1, 0, 0],
                [0, 0, 1, 1, 1],
                [1, 1, 1, 0, 1],
                [1, 1, 0, 0, 1]], dtype=int)
    S = np.mat([[1, 0, 1, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                [1, 0, 1, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 1, 1, 1]])
    linearcode = LinearCode(S)
