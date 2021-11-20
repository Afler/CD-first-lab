import numpy as np
import itertools

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

def getIndexMatrix(n, k, I):
    result = np.mat(np.zeros([0, k]), dtype=int)
    tolist = I[0, :].tolist()
    res = itertools.permutations(tolist, k)
    for i in res:
        isCorrectRow = True
        row = i
        for j in range(len(row)):
            for b in range(j + 1, len(row)):
                if row[j] > row[b]:
                    isCorrectRow = False
                    break
            if not isCorrectRow:
                break
        if isCorrectRow:
           result = np.vstack([result, row])
    return result


def getG(r, m):
    rows_in_block = 0
    blockNumber = np.zeros([0, 1], dtype=int)
    I = np.zeros([1, m], dtype=int)
    G = np.mat(np.ones([1, 2 ** m]), dtype=int)
    allIndexesMatrix = np.mat(np.zeros([0, r], dtype=int))
    for i in range(m):
        I[0, i] = i
    for i in range(I.shape[1] - 1, -1, -1):
        v_values = getVValues(np.mat(I[0, i]), m)
        G = np.vstack([G, v_values])
        mat = I[0, i]
        mat = np.hstack([np.zeros([1, r - 1], dtype=int), np.mat(mat, dtype=int)])
        allIndexesMatrix = np.vstack([allIndexesMatrix, mat])
        blockNumber = np.vstack([blockNumber, np.mat([1])])
    for i in range(2, m):
        rows_in_block = C(m, i)
        matrixWithBlockIndex = getIndexMatrix(rows_in_block, i, I)
        matrixWithBlockIndex = np.hstack([np.zeros([matrixWithBlockIndex.shape[0], r - matrixWithBlockIndex.shape[1]], dtype=int), matrixWithBlockIndex])
        allIndexesMatrix = np.vstack([allIndexesMatrix, matrixWithBlockIndex])
        np_mat = np.mat([i])
        np_mat = np.resize(np_mat, (matrixWithBlockIndex.shape[0], 1))
        blockNumber = np.vstack([blockNumber, np_mat])
        for j in range(matrixWithBlockIndex.shape[0]):
            G = np.vstack([G, getVValues(matrixWithBlockIndex[[j]], m)])
            printMatrix(G, "G")
    allIndexesMatrix = np.hstack([blockNumber, allIndexesMatrix])
    c = 0
    c +=1
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
