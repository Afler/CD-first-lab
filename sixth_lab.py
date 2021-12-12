import numpy as np
from numpy.polynomial import Polynomial as polynom
# новая лаба

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

    def code(self, a):
        codeWord = (a @ self.G) % 2
        return codeWord

    def decode(self, recvWord):
        recvPolynom = polynom(np.ravel(recvWord))
        polynomg = polynom(np.ravel(self.g))
        sourceOstatok = self.getOstatok(recvPolynom, polynomg)
        recvPolynom = polynom([0, 1]
        for i in range(1, self.n, 1):
            ostatok = self.getOstatok(recvPolynom, polynomg)

        pass

    def getOstatok(self, recvPolynom, polynomg):
        ostatok = (recvPolynom % polynomg)
        for i in range(len(ostatok.coef)):
            ostatok.coef[i] = ostatok.coef[i] % 2
        return ostatok

if __name__ == '__main__':
    mat = np.mat([[1, 1, 0, 1]], dtype=int)
    code = CycleCode(7, 4, mat)
    a = np.zeros([1, code.dimension], dtype=int)
    a[0, a.shape[1] - 1] = 1
    a[0, 0] = 1
    codeWord = code.code(a)
    printMatrix(codeWord, "codeWord: ")
    e1 = np.mat([[0, 0, 0, 0, 1, 0, 0]])
    recvWord = (codeWord + e1) % 2
    sourceWord = code.decode(recvWord)

