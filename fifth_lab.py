import numpy as np


def C(n, k):
    if 0 <= k <= n:
        nn = 1
        kk = 1
        for t in range(1, min(k, n - k) + 1):
            nn *= n
            kk *= t
            n -= 1
        return nn // kk
    else:
        return 0


def printMatrix(arr, name):
    print(name, "=")
    print(arr)


def getG(r, m):
    rows_in_block = 0
    I = np.zeros([1, m], dtype=int)
    G = np.mat(np.ones([1, 2 ** m]), dtype=int)
    for i in range(m):
        I[0, i] = i
    for i in range(1, m - 1):
        countInd = I[0, i]
        rows_in_block = C(m, countInd)
        vIndFirst = np.mat(np.zeros([1, countInd]), dtype=int)
        for z in range(rows_in_block):
            a = z
            for j in range(countInd):
                vIndFirst[0, vIndFirst.shape[1] - 1 - j] = I[0, I.shape[1] - 1 - a]
                print(vIndFirst[0, vIndFirst.shape[1] - 1 - j])
                a = a + 1


        # for j in range(vIndFirst.shape[1]):
        #     vIndFirst[0, j] = I[0, I.shape[1] - 1 - j]
        #     print(vIndFirst[0, j])
        #     for y in range(vIndFirst.shape[1]):
        #         G = np.vstack([G, getVValues(vIndFirst, m)])

    return G



def getVValues(I, m):
    v = np.mat(np.zeros([1, 2 ** m]), dtype=int)
    x = fillX(m)
    for i in range(v.shape[1]):
        v[0, i] = f(x[0, i * m:(i + 1) * m], I)
    return v


def fillX(m):
    answer = np.mat([], dtype=int)
    binaryToAdd = np.mat(np.zeros([1, m]), dtype=int)
    for i in range(2 ** m):
        binaryIStr = (bin(i)[2:])[::-1]
        for j in range(len(binaryIStr), m):
            binaryIStr += "0"
        for k in range(len(binaryIStr)):
            binaryToAdd[0, k] = binaryIStr[k]
        answer = np.hstack([answer, binaryToAdd])
    return answer


def f(x, I):
    for i in range(I.shape[1]):
        if x[0, I[0, i]] == 1:
            return int(False)
    return int(True)


def С(n, k):
    if 0 <= k <= n:
        nn = 1
        kk = 1
        for t in range(1, min(k, n - k) + 1):
            nn *= n
            kk *= t
            n -= 1
        return nn // kk
    else:
        return 0


class CanonicalRMCode:
    r = 0
    m = 0
    k = 0
    d = 0
    G = np.mat([[]], dtype=int)

    def __init__(self, r, m):
        self.r = r
        self.m = m
        self.n = 2 ** m
        self.d = 2 ** (self.m - self.r)
        for i in range(r + 1):
            self.k += С(self.n, i)
        self.G = getG(r, m)


if __name__ == '__main__':
    # print(C(4, 3))
    # print(C(4, 2))
    canonicalRM = CanonicalRMCode(3, 4)
    print(canonicalRM.G)
    # print(v)
    print("End")
