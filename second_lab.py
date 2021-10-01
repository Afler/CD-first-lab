import numpy as np

from first_lab import LinearCode


def printMatrix(arr, name):
    print(name, "=")
    print(arr)


if __name__ == '__main__':
    # task 2.1
    G = np.mat([[1, 0, 0, 0, 0, 1, 1],
                [0, 1, 0, 0, 1, 1, 0],
                [0, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 1, 1, 1, 1]])
    #    n = G.shape[1]
    #    k = G.shape[0]
    #    d = 3

    # task 2.2
    #    B = np.mat([[0, 1, 1],
    #                [1, 1, 0],
    #                [1, 0, 1],
    #               [1, 1, 1],
    #                [1, 0, 0],
    #                [0, 1, 0],
    #             [0, 0, 1]])
    #  H = np.mat([[]], dtype=int)
    #   y = 0
    #
    #   H = np.zeros((n, n - k), dtype=int)
    #
    #    for i in range(k, n):
    #        H[i][y] = 1
    #        y += 1

    #    for i in range(k):
    #        for j in range(n - k):
    #            H[i, j] = G[i, j + k]

    #    print((G @ H) % 2)
    # linearcode = LinearCode(S)

    linearCode = LinearCode(G)
    # Task 2.3
    I = np.mat(np.eye(linearCode.n, dtype=int))
    printMatrix(I, "I")
    printMatrix(linearCode.H, "H")
    syndromes = (I @ linearCode.H) % 2
    printMatrix(syndromes, "syndromes")
    # Task 2.4
    v1 = np.mat([1, 1, 0, 1])
    v1 = (v1 @ G) % 2
    printMatrix(v1, "исходное v")
    e1 = np.mat([1, 0, 0, 0, 0, 0, 0])
    v1 = (v1 + e1) % 2
    w = np.mat([1, 0, 0, 0, 0, 1, 1])
    printMatrix(v1, "с ошибкой v")
    # вычисляем синдром
    firstSyndrome = (v1 @ linearCode.H) % 2
    printMatrix(linearCode.allowed_words, "allowed_words")
    printMatrix(firstSyndrome, "firstSyndrome")
    # проверка, что кодовое слово умножить на проверочную матрицу даёт нулевой синдром
    # printMatrix((w @ linearCode.H) % 2, "syndrome")
    index = -1
    for i in range(linearCode.H.shape[0]):
        if np.array_equal(linearCode.H[[i]], firstSyndrome):
            index = i
            break
    # индекс синдрома в H
    print(index)
    v1[0, index] = (v1[0, index] ^ 1)
    # исправленное слово
    printMatrix(v1, "Исправленное слово v1")
    printMatrix((v1 @ linearCode.H) % 2, "Check:")
    # Task 2.5
    v2 = np.mat([1, 0, 0, 1])
    v2 = (v2 @ G) % 2
    printMatrix(v2, "исходное v2")
    e2 = np.mat([1, 0, 0, 0, 0, 1, 0])
    v2 = (v2 + e2) % 2
    printMatrix(v2, "с ошибкой v2")
    secondSyndrome = (v2 @ linearCode.H) % 2
    printMatrix(secondSyndrome, "secondSyndrome")
    index = -1
    for i in range(linearCode.H.shape[0]):
        if np.array_equal(linearCode.H[[i]], secondSyndrome):
            index = i
            break
    # индекс синдрома в H
    print(index)
    v2[0, index] = (v2[0, index] ^ 1)
    # исправленное слово
    printMatrix(v2, "Исправленное слово v2")

    # Task 2.6
    n = 9
    d = 5
    k = n - d
    I2 = np.mat(np.eye(k, dtype=int))
    X2 = np.mat([[]])
    X2 = np.ones((k, n - k), dtype=int)
    for i in range(k):
        for j in range(n):
            if i == j:
                X2[i, j] = 0
    printMatrix(X2, "X2")
    G2 = np.hstack([I2, X2])
    printMatrix(G2, "G2")
    linearCode2 = LinearCode(G2)

    # Task 2.3
    ERROR = np.mat(np.eye(linearCode2.n, dtype=int))
    printMatrix(ERROR, "I")
    printMatrix(linearCode2.H, "H")
    syndromes2 = (ERROR @ linearCode2.H) % 2
    printMatrix(syndromes2, "syndromes")
    # Task 2.4

    w1 = np.mat([1, 1, 0, 1])
    w1 = (w1 @ G2) % 2
    printMatrix(w1, "исходное w1")
    error1 = np.mat([1, 0, 0, 0, 0, 0, 0, 0, 0])
    w1 = (w1 + error1) % 2
    printMatrix(w1, "с ошибкой w1")
    # вычисляем синдром
    firstSyndrome = (w1 @ linearCode2.H) % 2
    printMatrix(linearCode2.allowed_words, "allowed_words")
    printMatrix(firstSyndrome, "firstSyndrome")
    index = -1
    for i in range(linearCode2.H.shape[0]):
        if np.array_equal(linearCode2.H[[i]], firstSyndrome):
            index = i
            break
    # индекс синдрома в H
    print(index)
    w1[0, index] = (w1[0, index] ^ 1)
    # исправленное слово
    printMatrix(w1, "Исправленное слово w1")
    printMatrix((w1 @ linearCode2.H) % 2, "Check:")
    # Task 2.5
    # Матрица синдромов для двукртаных
    EYE2 = np.mat(np.eye(linearCode2.n, dtype=int))
    ERROR2 = EYE2.copy()
    for i in range(EYE2.shape[0] - 1):
        for j in range(i + 1, EYE2.shape[0]):
            row = (EYE2[i] + EYE2[j]) % 2
            isAppend = True
            for k in range(0, ERROR2.shape[0]):
                if np.array_equal(ERROR2[k], row):
                    isAppend = False
                    break
            if isAppend:
                ERROR2 = np.vstack([ERROR2, row])
    zeros = np.zeros(n, dtype=int)
    for i in range(zeros.shape[0]):
        zeros[i] = i
    ERROR2 = np.delete(ERROR2, zeros, axis=0)
    printMatrix(ERROR2, "ERROR2")
    # Матрица синдромов для двукратных ошибок
    syndromes2X2 = (ERROR2 @ linearCode2.H) % 2
    printMatrix(syndromes2X2, "syndromes2X2")

    w2 = np.mat([1, 0, 0, 1])
    w2 = (w2 @ G2) % 2
    printMatrix(w2, "исходное w2")
    error2 = np.mat([1, 0, 0, 0, 0, 1, 0, 0, 0])
    w2 = (w2 + error2) % 2
    printMatrix(w2, "с ошибкой w2")
    secondSyndrome = (w2 @ linearCode2.H) % 2
    printMatrix(secondSyndrome, "secondSyndrome")
    index = -1
    for i in range(syndromes2X2.shape[0]):
        if np.array_equal(syndromes2X2[[i]], secondSyndrome):
            index = i
            break
    # индекс синдрома в H
    print(index)
    w2 = (w2 + ERROR2[[index]]) % 2
    # исправленное слово
    printMatrix(w2, "Исправленное слово w2")

    w3 = np.mat([1, 1, 1, 0])
    w3 = (w3 @ G2) % 2
    printMatrix(w3, "исходное w3")
    error3 = np.mat([1, 0, 0, 1, 0, 1, 0, 0, 0])
    w3 = (w3 + error3) % 2
    printMatrix(w3, "с ошибкой w3")
    thirdSyndrome = (w3 @ linearCode2.H) % 2
    printMatrix(thirdSyndrome, "thirdSyndrome")
    index = -1
    for i in range(syndromes2X2.shape[0]):
        if np.array_equal(syndromes2X2[[i]], thirdSyndrome):
            index = i
            break
    print(index)
    w3 = (w3 + ERROR2[[index]]) % 2
    printMatrix(w3, "Исправленное словыо w3")