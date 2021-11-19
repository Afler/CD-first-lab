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
    H = np.mat([[]], dtype=int)
    t = 0

    def __init__(self, r):
        self.n = 2 ** r - 1
        self.k = 2 ** r - r - 1
        self.H = np.mat(np.zeros([self.n, r]), dtype=int)
        self.H[self.n - r:self.n + 1][:] = np.mat(np.eye(r, r))


if __name__ == '__main__':
    hammingThreeOne = HammingCode(2)
    #hammingSevenFour = HammingCode(3)
    #hammingFifteenEleven = HammingCode(4)
    print("End")
