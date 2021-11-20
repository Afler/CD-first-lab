import numpy as np


def printMatrix(arr, name):
    print(name, "=")
    print(arr)


def getG(r, m):
    n = 2 ** m
    G = np.mat(np.zeros(m), dtype=int)
    for i in range(n):
        G[[i]] = getVValues(I, m)
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
    canonicalRM = CanonicalRMCode(3, 4)
    v = getVValues(np.mat([2]), 4)
    print(v)
    print("End")
