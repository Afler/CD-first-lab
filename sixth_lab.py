import numpy as np
from numpy.polynomial import Polynomial as polynom


def printMatrix(arr, name):
    print(name, "=")
    print(arr)


class CycleCode:
    g = 0
    n = 0
    maxPowerPolynom = 0
    dimension = 0
    G = 0

    def __init__(self, n, dimension, vector_g):
        self.n = n
        self.dimension = dimension
        self.g = vector_g
        self.maxPowerPolynom = n - dimension
        self.G = self.getG()

    def getG(self):

        for i in range(self.g.shape[1], self.n):
            self.g = np.hstack([self.g, np.zeros([1, 1], dtype=int)])
        G = np.mat(np.zeros([0, self.g.shape[1]]), dtype=int)
        G = np.vstack([G, self.g])
        g = self.g
        for i in range(self.dimension - 1):
            g = np.roll(g, 1, 1)
            G = np.vstack([G, g])
        return G

    def encode(self, a):
        codeWord = (a @ self.G) % 2
        return codeWord

    def getSyndromePolynom(self, recvPolynom, polynomg):
        ostatok = (recvPolynom % polynomg)
        for i in range(len(ostatok.coef)):
            ostatok.coef[i] = ostatok.coef[i] % 2
        return ostatok

    def getSyndromWeight(self, syndrome):
        return np.count_nonzero(syndrome.coef)

    def decode(self, recvWord):
        i, syndrome = self.getIndexWithSyndrome(recvWord)
        if i == 0 and syndrome == 0:
            print("Неисправимая ошибка")
            exit(0)
        error = syndrome * (polynom([0, 1]) ** (self.n - i))
        answer = np.mat((error + polynom(np.ravel(recvWord))).coef % 2, dtype=int)
        return answer

    def getIndexWithSyndrome(self, recvWord):
        recvPolynom = polynom(np.ravel(recvWord))
        gPolynom = polynom(np.ravel(self.g))
        originSyndromePolynom = self.getSyndromePolynom(recvPolynom, gPolynom)
        powerOfOriginSyndrome = self.getSyndromWeight(originSyndromePolynom)
        for i in range(1, self.n):
            shiftedPolynom = originSyndromePolynom * (polynom([0, 1]) ** i)
            syndrome = self.getSyndromePolynom(shiftedPolynom, gPolynom)
            if self.getSyndromWeight(syndrome) < powerOfOriginSyndrome:
                return i, syndrome
        return 0, 0


if __name__ == '__main__':
    # порождающий полином
    genPolynom = np.mat([[1, 1, 0, 1]], dtype=int)
    code = CycleCode(7, 4, genPolynom)

    # входное слово
    word = np.mat([[1, 0, 0, 1]], dtype=int)
    # кодируем
    codedWord = code.encode(word)
    printMatrix(codedWord, "codeWord: ")

    # вносим однократную ошибку
    e1 = np.mat([[0, 0, 0, 0, 1, 0, 0]])
    recvWord = (codedWord + e1) % 2

    # декодируем
    decodedWord = code.decode(recvWord)
    printMatrix(decodedWord, "decodedWord: ")
