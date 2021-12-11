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

    blockNumber = np.zeros([0, 1], dtype=int)
    I = np.zeros([1, m], dtype=int)
    allIndexesMatrix = np.mat(np.zeros([0, r], dtype=int))
    for i in range(m):
        I[0, i] = i
    for i in range(I.shape[1] - 1, -1, -1):

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

    allIndexesMatrix = np.hstack([blockNumber, allIndexesMatrix])

    #сортировка матрицы индексов

    column = allIndexesMatrix[:, 0]
    for i in range(1, m):
        for j in range(allIndexesMatrix.shape[0] - 1):  # - 1
            for k in range(j + 1, allIndexesMatrix.shape[0]):
                if i == column[j] and i == column[k]:
                    sumCur = np.sum(allIndexesMatrix[[j]])
                    sumNext = np.sum(allIndexesMatrix[[k]])
                    if (sumNext > sumCur):
                        allIndexesMatrix[[j, k]] = allIndexesMatrix[[k, j]]
                    elif sumNext == sumCur:
                        if allIndexesMatrix[j, allIndexesMatrix.shape[1] - 1] < allIndexesMatrix[k, allIndexesMatrix.shape[1] - 1]:
                            allIndexesMatrix[[j, k]] = allIndexesMatrix[[k, j]]

    # формирование двоичных чисел в обратной записи

    U = np.zeros([0, m], dtype=int)
    binaryToAdd = np.mat(np.zeros([1, m]), dtype=int)
    for i in range(2 ** m):
        binaryIStr = (bin(i)[2:])[::-1]
        for j in range(len(binaryIStr), m):
            binaryIStr += "0"
        for k in range(len(binaryIStr)):
            binaryToAdd[0, k] = binaryIStr[k]
        U = np.vstack([U, binaryToAdd])
    answerG = np.zeros([0, 2 ** m], dtype=int)
    rowG = np.mat(np.zeros([1, 2 ** m]), dtype=int)

    # создание матрицы G

    for i in range(allIndexesMatrix.shape[0]):
        rowIndexx = allIndexesMatrix[i, allIndexesMatrix.shape[1] - allIndexesMatrix[i, 0]: allIndexesMatrix.shape[1]]
        for j in range(0, U.shape[0] - 1):
            #indexesOnes = np.nonzero(U[[j]])
            if np.count_nonzero(U[j, rowIndexx]) > 0:
                rowG[0, j] = 0
            else:
                rowG[0, j] = 1
        answerG = np.vstack([answerG, rowG])
    answerG = np.vstack([np.mat(np.ones([1, 2 ** m]), dtype=int), answerG])
    return answerG


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
    printMatrix(canonicalRM.G, "G:")
    # print(v)
    print("End")
