import numpy as np


def printMatrix(arr, name):
    print(name, "=")
    print(arr)


class LinearCode:
    G = np.mat([[]], dtype=int)
    Gstar = np.mat([[]], dtype=int)
    n = 0
    k = 0
    X = np.mat([[]], dtype=int)
    H = np.mat([[]], dtype=int)
    allowed_words = np.mat([[]], dtype=int)
    d = -1
    t = -1

    def __init__(self, S):

        # task 1.3.1
        self.G = REF(S)
        printMatrix(self.G, "G")
        self.Gstar = RREF(self.G)
        m = self.Gstar.shape[0]
        n = self.Gstar.shape[1]
        for i in range(m):
            if np.array_equal(np.ravel(self.Gstar[i]), np.zeros(n, dtype=int)):
                self.Gstar = self.Gstar[0: i]
                break
        printMatrix(self.Gstar, "G*")

        # task 1.3.2
        self.n = self.G.shape[1]
        self.k = self.G.shape[0]
        print("n =", self.n)
        print("k =", self.k)

        # task 1.3.3
        # step 1
        self.X = self.Gstar
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
        allowed_words = np.vstack([allowed_words, np.zeros(allowed_words.shape[1], dtype=int)])
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
        printMatrix(k_length_words @ self.G % 2, "k@G")

        # task 1.5
        self.d = self.G.shape[1]
        for i in range(self.G.shape[0]):
            for j in range(i + 1, self.G.shape[0]):
                self.d = min(np.count_nonzero((self.G[j] - self.G[i]) % 2), self.d)
        self.t = self.d - 1
        print("d =", self.d)
        print("t =", self.t)

        # task 1.5.1
        # фиксируем разрешенное кодовое слово
        v = np.mat([[1, 0, 1, 1, 1, 0, 1, 0, 0, 1]])

        # вносим ошибку одинарной кратности
        e1 = np.mat([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
        print("v + e1 =", (v + e1) % 2)
        # она обнаруживается
        print("(v + e1)@H =", ((v + e1) % 2) @ self.H % 2)

        # вносим ошибку двойной кратности
        e2 = np.mat([[0, 0, 0, 1, 0, 1, 0, 0, 0, 0]])
        print("v + e2 =", (v + e2) % 2)
        # она обнаруживается?
        print("(v + e2)@H =", ((v + e2) % 2) @ self.H % 2)


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

    for i in range(m - 1):
        for j in range(n):
            if C[i, j] != 0:
                for k in range(i):
                    C[k] = (C[k] - C[i] * C[k, j]) % 2
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
    # u = [1, 0, 1, 1, 0]
    # v = u @ linearcode.G % 2
    # print("u@G =", v @ linearcode.H % 2)
    # printMatrix(RREF(linearcode.G), "Gstar")
