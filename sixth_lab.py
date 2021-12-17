import numpy as np
from numpy.polynomial import Polynomial as polynom
import math


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
    E = 0

    def __init__(self, n, k, vector_g):
        self.n = n
        self.k = k
        self.g = vector_g
        self.gPower = n - k
        self.G = self.getG()
        self.t = self.getT()
        self.E = self.createE()

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
                self.d = min(np.count_nonzero((self.G[[j]] - self.G[[i]]) % 2), self.d)
        return math.floor((self.d - 1) / 2)

    def encode(self, a):
        codeWord = (a @ self.G) % 2
        return codeWord

    def getSyndromePolynom(self, recvPolynom, polynomg):
        ostatok = (recvPolynom % polynomg)
        for i in range(len(ostatok.coef)):
            ostatok.coef[i] = ostatok.coef[i] % 2
        return ostatok

    def getPolynomWeight(self, syndrome):
        return np.count_nonzero(syndrome.coef)

    def decode(self, recvWord):
        i, syndrome = self.getIndexWithSyndromePolynom(recvWord)
        if i == -1:
            print("Неисправимая ошибка")
            exit(0)
        if i == 0:
            i = self.n
        error = syndrome * (polynom([0, 1]) ** (self.n - i))
        answerPolynom = (error + polynom(np.ravel(recvWord)))
        answerPolynom.coef.resize(self.n)
        answer = np.mat(answerPolynom.coef % 2, dtype=int)
        return answer

    def getIndexWithSyndromePolynom(self, recvWord):
        recvPolynom = polynom(np.ravel(recvWord))
        gPolynom = polynom(np.ravel(self.g))
        originSyndromePolynom = self.getSyndromePolynom(recvPolynom, gPolynom)
        for i in range(self.n):
            shiftedPolynom = originSyndromePolynom * (polynom([0, 1]) ** i)
            syndromePolynom = self.getSyndromePolynom(shiftedPolynom, gPolynom)
            # task 6.1
            # if self.getPolynomWeight(syndromePolynom) <= self.t:
            #     return i, syndromePolynom

            # task 6.2
            if self.containsInE(syndromePolynom):
                return i, syndromePolynom
        return -1, 0

    def createE(self):
        E = np.mat(np.zeros([0, self.n]), dtype=int)
        # формируем шаблон ошибок кратности 3
        for k in range(2 ** (2)):
            binaryIStr = (bin(k)[2:])[::-1]
            str = "1" + binaryIStr
            for j in range(len(binaryIStr), self.n - 1):
                str += "0"
            E = np.vstack([E, np.array(list(str), dtype=int)])
        E1 = E.copy()
        # формируем пакет ошибок кратности 3 циклическими сдвигами шаблона
        for i in range(self.k):
            E = np.vstack([E, np.roll(E1, i + 1, 1)])
        return E

    def containsInE(self, syndromePolynom):
        for i in range(self.E.shape[0]):
            check = np.array(syndromePolynom.coef, dtype=int)
            check.resize(self.n, refcheck=False)
            if np.array_equal(np.ravel(self.E[[i]]), check):
                return True
        return False




if __name__ == '__main__':
    # порождающий полином
    genPolynom = np.mat([[1, 0, 1, 1]], dtype=int)
    code = CycleCode(7, 4, genPolynom)

    # входное слово
    word = np.mat([[1, 1, 0, 0]], dtype=int)
    # кодируем
    codedWord = code.encode(word)
    printMatrix(codedWord, "encodedWord: ")

    # вносим однократную ошибку
    e1 = np.mat([[0, 1, 0, 0, 0, 0, 0]])

    recvWord = (codedWord + e1) % 2
    printMatrix(recvWord, "recvWord with error: ")
    # декодируем
    decodedWord = code.decode(recvWord)
    printMatrix(decodedWord, "decodedWord: ")

    # вариант первый без пакетов, исправляет ошибки кратности 1, и исправляет неправильно ошибки кратности два и три


    code = CycleCode(15, 9, np.mat([[1, 0, 0, 1, 1, 1, 1]], dtype=int))
    word = np.mat([[1, 0, 0, 1,
                    0, 0, 0, 1,
                    1]], dtype=int)
    codedWord = code.encode(word)
    printMatrix(codedWord, "encodedWord: ")

    e1 = np.mat([[0, 0, 0, 0,
                  0, 0, 0, 0,
                  0, 0, 0, 0,
                  1, 1, 1]])
    recvWord = (codedWord + e1) % 2
    printMatrix(recvWord, "recvWord with error: ")
    decodedWord = code.decode(recvWord)
    printMatrix(decodedWord, "decodedWord: ")

    # вариант с пакетом ошибок работает так, что мы формируем пакет ошибок заданной кратности, например как в задании 3,
    # и затем, ошибки кратности 1 2 и 3 исправляются, а кратности 4 не исправляются. Сам шаблон трёхкратных ошибок от
    # кода не зависит, но вот пакет зависит, так как чем больше код, тем больше будет цикличеких сдвигов.
