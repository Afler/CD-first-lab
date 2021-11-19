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
        # пример для первого случая когда две ошибки слева одна справа у принятого слова
        # word = np.mat([[0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0]])


        printMatrix(word, "correctword  sending start")
        # Однократная ошибка или двукратную или трёхкратную с левой стороны все - первый случай у принятого слова
        # word[0, 0] = (word[0, 0] + 1) % 2
        # word[0, 1] = (word[0, 1] + 1) % 2
        # word[0, 2] = (word[0, 2] + 1) % 2

        # четвёртый случай когда все ошибки с правой стороны у принятого слова
        # word[0, 13] = (word[0, 13] + 1) % 2
        # word[0, 14] = (word[0, 14] + 1) % 2
        # word[0, 15] = (word[0, 15] + 1) % 2

        # последний случай когда одна ошибка слева две ошибки справа у принятого слова
        word[0, 1] = (word[0, 1] + 1) % 2
        word[0, 14] = (word[0, 14] + 1) % 2
        word[0, 13] = (word[0, 13] + 1) % 2


        printMatrix(word, "word with error")
        wordSyndrome = (word @ self.H) % 2
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
                check = (wordSyndrome + self.B[[i]]) % 2
                if np.count_nonzero(check) <= 2:
                    print("section 3) b with index", str(i))
                    e = np.zeros([1, 12], dtype= int)
                    e[0, i] = 1
                    check = np.hstack([check, e])
                    printMatrix(check, "correct decryption word section 3)")
                    break

        secondSyndrom = (wordSyndrome @ self.B) % 2
        printMatrix(secondSyndrom, "secondSyndrome")
        countErrors = np.count_nonzero(secondSyndrom)
        if countErrors <= 3:
            correctword1 = word.copy()
            secondSyndrom = np.hstack([np.mat(np.zeros([1, 12]), dtype= int), secondSyndrom])
            correctword1 = (word + secondSyndrom) % 2
            printMatrix(correctword1, "correctword finish section 4)")
            print("errors: " + str(countErrors))
        if countErrors > 3:
            for i in range(self.B.shape[0]):
                check = (secondSyndrom + self.B[[i]]) % 2
                if np.count_nonzero(check) <= 2:
                    print("section5) b with index", str(i))
                    e = np.zeros([1, 12], dtype= int)
                    e[0, i] = 1
                    check = np.hstack([e, check])
                    printMatrix(check, "correct decryption word section 5")
                    break




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
