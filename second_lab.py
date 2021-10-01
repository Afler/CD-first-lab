import numpy as np

from first_lab import LinearCode

if __name__ == '__main__':
    # task 2.1
    G = np.mat([[1, 0, 0, 0, 0, 1, 1],
                [0, 1, 0, 0, 1, 1, 0],
                [0, 0, 1, 0, 1, 0, 1],
                [0, 0, 0, 1, 1, 1, 1]])
    n = G.shape[1]
    k = G.shape[0]
    d = 3

    # task 2.2
    B = np.mat([[0, 1, 1],
                [1, 1, 0],
                [1, 0, 1],
                [1, 1, 1],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
    H = np.mat([[]], dtype=int)
    y = 0

    H = np.zeros((n, n - k), dtype=int)

    for i in range(k, n):
        H[i][y] = 1
        y += 1

    for i in range(k):
        for j in range(n - k):
            H[i, j] = G[i, j + k]

    print((G @ H) % 2)
    # linearcode = LinearCode(S)
