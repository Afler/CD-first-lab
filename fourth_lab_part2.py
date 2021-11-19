import numpy as np


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


if __name__ == '__main__':
    # sys.setrecursionlimit(1500)
    # hammingThreeOne = GoleyaCode(4)
    print(RidMaller(1, 3).getG())
    # hammingSevenFour = HammingCode(3)
    # hammingFifteenEleven = HammingCode(4)
    print("End")
