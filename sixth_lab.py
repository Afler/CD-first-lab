import numpy as np
from numpy.polynomial import Polynomial as polynom


def printMatrix(arr, name):
    print(name, "=")
    print(arr)


class CycleCode:
    g = 0
    n = 0
    gPower = 0
    k = 0
    G = 0
    t = -1
    d = -1

    def __init__(self, n, k, vector_g):
        self.n = n
        self.k = k
        self.g = vector_g
        self.gPower = n - k
        self.G = self.getG()
        self.t = self.getT()

    def getG(self):
        g = np.ravel(self.g).copy()
        g.resize(self.n, refcheck=False)
        G = np.mat(np.zeros([0, self.n]), dtype=int)
        for i in range(self.k):
            G = np.vstack([G, g])
            g = np.roll(g, 1, 0)
        return G

    def getT(self):
        self.d = self.G.shape[1]
        # определение минимального кодового расстояния
        for i in range(self.G.shape[0]):
            for j in range(i + 1, self.G.shape[0]):
                self.d = min(np.count_nonzero((self.G[j] - self.G[i]) % 2), self.d)
        return (self.d - 1) // 2

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
        for i in range(1, self.n):
            shiftedPolynom = originSyndromePolynom * (polynom([0, 1]) ** i)
            syndrome = self.getSyndromePolynom(shiftedPolynom, gPolynom)
            if self.getSyndromWeight(syndrome) <= self.t:
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
    printMatrix(codedWord, "encodedWord: ")

    # вносим однократную ошибку
    e1 = np.mat([[0, 0, 0, 0, 1, 0, 0]])
    recvWord = (codedWord + e1) % 2

    # декодируем
    decodedWord = code.decode(recvWord)
    printMatrix(decodedWord, "decodedWord: ")

    code = CycleCode(15, 9, np.mat([[1, 1, 1, 1, 0, 0, 1]], dtype=int))
    word = np.mat([[1, 0, 0, 1,
                    0, 0, 0, 1,
                    1]], dtype=int)
    codedWord = code.encode(word)
    printMatrix(codedWord, "encodedWord: ")

    e1 = np.mat([[0, 0, 0, 0,
                  0, 0, 0, 0,
                  1, 0, 1, 0,
                  0, 0, 0]])
    recvWord = (codedWord + e1) % 2

    decodedWord = code.decode(recvWord)
    printMatrix(decodedWord, "decodedWord: ")
