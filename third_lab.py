import numpy as np


def printMatrix(arr,
                name):
    print(name, "=")
    print(arr)


class HammingCode:
    n = 0
    k = 0
    d = 3
    G = np.mat([[]], dtype=int)
    GSTAR = np.mat([[]], dtype=int)
    H = np.mat([[]], dtype=int)
    HSTAR = np.mat([[]], dtype=int)
    t = 0

    def __init__(self, r):
        self.n = 2 ** r - 1
        self.k = 2 ** r - r - 1
        self.H = np.mat(np.zeros([self.n, r]), dtype=int)
        self.H[self.n - r:self.n + 1][:] = np.mat(np.eye(r, r))
        addIndex = 0
        # task 3.1
        # init H
        for i in range(0, self.H.shape[0]):
            for j in range(0, self.H.shape[0]):
                row = (self.H[[i]] + self.H[[j]]) % 2
                isAppend = True
                for k in range(0, self.H.shape[0]):
                    if np.array_equal(self.H[[k]], row) or np.array_equal(row, np.zeros([1, r])):
                        isAppend = False
                        break
                if isAppend:
                    self.H[[addIndex]] = row
                    addIndex = addIndex + 1
        # init G
        self.G = np.mat(np.zeros([self.k, self.n]), dtype=int)
        t = self.G[:][0: self.n - r]
        self.G[:, 0: self.n - r] = np.mat(np.eye(self.k, self.k))
        self.G[:, self.n - r: self.n] = self.H[0: self.n - r, :]
        # syndromes
        syndromes = (np.mat(np.eye(self.n, dtype=int)) @ self.H) % 2
        printMatrix(syndromes, "syndromes")
        printMatrix(self.H, "H")
        printMatrix(self.G, "G")
        # task 3.2
        word = self.G[[0]]
        # Однократная ошибка
        word[0, 0] = (word[0, 0] + 1) % 2
        wordSyndrome = (word @ self.H) % 2
        printMatrix(wordSyndrome, "wordSyndrome1")
        # Двукратная ошибка
        word[0, 1] = (word[0, 1] + 1) % 2
        wordSyndrome = (word @ self.H) % 2
        printMatrix(wordSyndrome, "wordSyndrome2")
        # Трехкратная ошибка
        word[0, 2] = (word[0, 2] + 1) % 2
        wordSyndrome = (word @ self.H) % 2
        printMatrix(wordSyndrome, "wordSyndrome3")
        # Вывод: Код Хеминга действительно исправляет однократные,
        # обнаруживает двукратные и возникает неопределеность для трехкратных ошибок

        # task 3.3
        # расширенный
        self.GSTAR = np.mat(np.zeros([self.k, self.n + 1]), dtype=int)
        self.GSTAR[:, 0: self.n] = self.G.copy()
        for i in range(0, self.GSTAR.shape[0]):
            count = np.count_nonzero(self.GSTAR[i, :])
            if count % 2 == 1:
                self.GSTAR[i, self.GSTAR.shape[1] - 1] = 1
        printMatrix(self.GSTAR, "GSTAR")
        self.HSTAR = self.H.copy()
        self.HSTAR = np.vstack([self.HSTAR, np.zeros([1, self.HSTAR.shape[1]], dtype=int)])
        self.HSTAR = np.hstack([self.HSTAR, np.ones([self.HSTAR.shape[0], 1], dtype=int)])
        printMatrix(self.HSTAR, "HSTAR")
        syndromesExpanded = (np.mat(np.eye(self.n + 1, dtype=int)) @ self.HSTAR) % 2
        printMatrix(syndromesExpanded, "syndromesExpanded")

        word = self.GSTAR[[0]]
        # Однократная ошибка
        word[0, 0] = (word[0, 0] + 1) % 2
        wordSyndrome = (word @ self.HSTAR) % 2
        printMatrix(wordSyndrome, "wordSyndrome1")
        # Двукратная ошибка
        word[0, 1] = (word[0, 1] + 1) % 2
        wordSyndrome = (word @ self.HSTAR) % 2
        printMatrix(wordSyndrome, "wordSyndrome2")
        # Трехкратная ошибка
        word[0, 2] = (word[0, 2] + 1) % 2
        wordSyndrome = (word @ self.HSTAR) % 2
        printMatrix(wordSyndrome, "wordSyndrome3")
        # Расширенный код Хэмминга позволяет ОДНОВРЕМЕННО и исправлять ошибку однократной кратности и обнаруживать двукратную (если последний
        # элемент произведения полученного слова на проверочную матрицу равен 0). Трёхкратные рандом.

if __name__ == '__main__':
    hammingThreeOne = HammingCode(4)
    # hammingSevenFour = HammingCode(3)
    # hammingFifteenEleven = HammingCode(4)
    print("End")
