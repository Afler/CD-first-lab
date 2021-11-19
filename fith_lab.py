import numpy as np


def printMatrix(arr, name):
    print(name, "=")
    print(arr)


def getGForRM(r, m):
    if r == 0:
        return np.mat(np.ones([1, 2 ** m]), dtype=int)
    elif r == m:
        rm = getGForRM(m - 1, m)
        zeros = np.zeros([1, rm.shape[1]])
        zeros[0, zeros.shape[1] - 1] = 1
        return np.mat(np.vstack([rm, zeros]), dtype=int)
    else:
        GUpperLeft = getGForRM(r, m - 1)
        GUpperRight = getGForRM(r, m - 1)
        GLowerRight = getGForRM(r - 1, m - 1)
        GLowerRight = np.mat(np.ravel(np.vstack([np.zeros([1, GUpperLeft.shape[0]]), GLowerRight])), dtype=int)
        GRight = np.vstack([GUpperRight, GLowerRight])
        GLeft = np.vstack([GUpperLeft, np.mat(np.zeros([GLowerRight.shape[0], GUpperLeft.shape[1]]))])
        return np.mat(np.hstack([GRight, GLeft]), dtype=int)


class RMCode:
    r = 0
    m = 0
    G = np.mat([[]], dtype=int)

    def __init__(self, r, m):
        self.r = r
        self.m = 2 ** m
        self.G[[0]] = getGForRM(r, m)


if __name__ == '__main__':
    matrix = getGForRM(2, 2)
    printMatrix(matrix, "matrix")
