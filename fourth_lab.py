import numpy as np


def printMatrix(arr,
                name):
    print(name, "=")
    print(arr)


class GoleyaCode:
    n = 0
    k = 0
    d = 3
    G = np.mat([[]], dtype=int)
    B = np.mat([[]], dtype=int)
    I = np.mat([[]], dtype=int)
    GSTAR = np.mat([[]], dtype=int)
    H = np.mat([[]], dtype=int)
    HSTAR = np.mat([[]], dtype=int)
    t = 0

    def __init__(self, r):
        self.n = 24
        self.k = 24
        self.d = 8
        self.H = np.mat(np.zeros([self.n, r]), dtype=int)
        self.B = np.mat([[1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
                         [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
                         [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
                         [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
                         [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
                         [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
                         [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
                         [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
                         [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
                         [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
                         [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=int)
        E = np.mat(np.eye(int(self.n / 2), int(self.k / 2), dtype=int))
        self.G = E.copy()
        self.G = np.hstack([self.G, self.B])
        printMatrix(self.G, "G")
        self.H = E.copy()
        self.H = np.vstack([self.H, self.B])
        printMatrix(self.H, "H")
        # task 4.2
        word = self.G[[11]]
        printMatrix(word, "correctword start")
        # Однократная ошибка
        word[0, 1] = (word[0, 22] + 1) % 2
        word[0, 21] = (word[0, 21] + 1) % 2
        word[0, 20] = (word[0, 20] + 1) % 2
        word[0, 19] = (word[0, 19] + 1) % 2
        printMatrix(word, "word with error")
        wordSyndrome = (word @ self.H) % 2
        check = wordSyndrome @ np.linalg.matrix_power(self.H, -1)
        print(check)
        printMatrix(wordSyndrome, "wordSyndrome1")
        countErrors = np.count_nonzero(wordSyndrome)
        if countErrors <= 3:
            correctword1 = word.copy()
            wordSyndrome = np.hstack([wordSyndrome, np.mat(np.zeros([1, 12]), dtype= int)])
            correctword1 = (word + wordSyndrome) % 2
            printMatrix(correctword1, "correctword finish")
            print("errors: " + str(countErrors))
        if countErrors > 3:
            for i in range(self.B.shape[0]):
                if np.array_equal(self.B[[i]], wordSyndrome):
                    print("sdfgdf")


    # Двукратная ошибка
    # word[0, 1] = (word[0, 1] + 1) % 2
    # wordSyndrome = (word @ self.H) % 2
    # printMatrix(wordSyndrome, "wordSyndrome2")
    # # Трехкратная ошибка
    # word[0, 2] = (word[0, 2] + 1) % 2
    # wordSyndrome = (word @ self.H) % 2
    # printMatrix(wordSyndrome, "wordSyndrome3")
    # word[0, 3] = (word[0, 3] + 1) % 2
    # wordSyndrome = (word @ self.H) % 2
    # printMatrix(wordSyndrome, "wordSyndrome4")


if __name__ == '__main__':
    hammingThreeOne = GoleyaCode(4)
    # hammingSevenFour = HammingCode(3)
    # hammingFifteenEleven = HammingCode(4)
    print("End")
