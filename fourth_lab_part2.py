import numpy as np


def printMatrix(arr, name):
    print(name, "=")
    print(arr)


def KroneckerMultiply(A, B):
    n = A.shape[0] if A.shape[0] >= B.shape[0] else B.shape[0]
    result = np.mat(np.zeros([2 * n, 2 * n]), dtype=int)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            result[i * B.shape[0]:(i + 1) * B.shape[0], j * B.shape[1]:(j + 1) * B.shape[1]] = A[i, j] * B
    return result


def getH(i, m):
    H = np.mat([[1, 1], [1, -1]])
    if i - 1 == 0:
        return KroneckerMultiply(np.eye(2 ** (m - i)), H)
    elif m - i == 0:
        return KroneckerMultiply(H, np.eye(2 ** (i - 1)))
    else:
        return KroneckerMultiply(KroneckerMultiply(np.eye(2 ** (m - i)), H), np.eye(2 ** (i - 1)))


class RidMaller:
    r = 0
    m = 0
    g01 = np.mat([[]], dtype=int)
    g11 = np.mat([[]], dtype=int)

    def __init__(self, r, m):
        self.r = r
        self.m = m
        self.g01 = np.mat([[1, 1]], dtype=int)
        self.g11 = np.mat([[1, 1], [0, 1]], dtype=int)

    # task 4.3
    def G(self, r, m):
        if r == m:
            u = np.mat((self.G(r - 1, m)))
            d = np.mat(np.zeros([1, u.shape[1]]))
            d[0, d.shape[1] - 1] = 1
            return np.mat(np.vstack([u, d]), dtype=int)
        if r == 0 and m == 1:
            return self.g01
        if r == 1 and m == 1:
            return self.g11
        if r == 0:
            return np.mat(np.ones([1, m ** 2]), dtype=int)
        gUp = self.G(r, m - 1)
        up_half = np.mat(np.hstack([gUp, gUp]))
        gDown = self.G(r - 1, m - 1)
        low_half = np.mat(np.hstack([np.zeros([gDown.shape[0], up_half.shape[1] - gDown.shape[1]]), gDown]))
        return np.mat(np.vstack([up_half, low_half]), dtype=int)

    def getG(self):
        return self.G(self.r, self.m)

    def decode(self, w):
        wWave = w.copy()
        for i in range(w.shape[1]):
            if w[0, i] == 0:
                wWave[0, i] = -1
        j, val = self.getIndexWithValue(wWave)
        jBin = (bin(j)[2:])[::-1]
        for i in range(len(jBin), self.m):
            jBin += str(0)
        answer = np.mat(np.zeros([1, len(jBin) + 1]), dtype=int)
        for i in range(1, answer.shape[1]):
            answer[0, i] = int(jBin[i - 1])
        if val > 0:
            answer[0, 0] = 1
            return answer
        else:
            answer[0, 0] = 0
            return answer

    def getIndexWithValue(self, wWave):
        w = np.mat(np.zeros([self.m, wWave.shape[1]]), dtype=int)
        w[[0]] = wWave @ getH(1, self.m)
        #printMatrix(w[[0]], "w" + str(1))
        for i in range(1, self.m):
            w[[i]] = w[[i - 1]] @ getH(i + 1, self.m)
            #printMatrix(w[[i]], "w" + str(i))
        j = np.argmax(np.amax(w, axis=0), axis=1)
        val = np.max(w[:, j])
        return int(j), val


if __name__ == '__main__':
    mallerOneThree = RidMaller(1, 3)
    mallerOneThreeG = mallerOneThree.getG()
    printMatrix(mallerOneThreeG, "MallerGenerateMatrix")
    word = np.mat([1, 0, 1, 0, 1, 0, 1, 1], dtype=int)
    originWord = mallerOneThree.decode(word)
    printMatrix(originWord, "OneErrorMessage")
    codedWord = (originWord @ mallerOneThreeG) % 2
    # В соответствие с полученным видом исходного слова заключем, что в него была внесена однократная ошибка в последнем
    # бите
    # Далее будем увеличивать кратность ошибки
    codedWord[0, 0] = (codedWord[0, 0] + 1) % 2
    codedWord[0, 1] = (codedWord[0, 1] + 1) % 2
    originWord = mallerOneThree.decode(codedWord)
    printMatrix(originWord, "TwoErrorMessage")
    # Ошибки исправляются правильно
    # Теперь проверим для RM(1,4)
    mallerOneFour = RidMaller(1, 4)
    mallerOneFourG = mallerOneFour.getG()

    codeWord = mallerOneFourG[[0]]
    originWord = mallerOneFour.decode(codeWord)
    printMatrix(originWord, "NoErrorMessage")

    codedWord[0, 0] = (codedWord[0, 0] + 1) % 2
    originWord = mallerOneFour.decode(codeWord)
    printMatrix(originWord, "OneErrorMessage")

    codedWord[0, 1] = (codedWord[0, 1] + 1) % 2
    originWord = mallerOneFour.decode(codeWord)
    printMatrix(originWord, "TwoErrorMessage")

    codedWord[0, 2] = (codedWord[0, 2] + 1) % 2
    originWord = mallerOneFour.decode(codeWord)
    printMatrix(originWord, "ThreeErrorMessage")

    codedWord[0, 3] = (codedWord[0, 3] + 1) % 2
    originWord = mallerOneFour.decode(codeWord)
    printMatrix(originWord, "FourErrorMessage")
    print("End")
