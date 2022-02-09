import numpy as np

import symplectic as symp

class Stabilizer:

    def __init__(self, M, v):
        self.M = M
        self.v = v
        nn, self.k = np.shape(M)
        self.n = nn // 2
        self.GF = type(v)

    @staticmethod
    def std(n, k, GF):
        M = GF.Zeros((2 * n, k))
        for i in range(k):
            M[2 * i][i] = GF(1)
        v = GF.Zeros(2 * n)
        return Stabilizer(M, v)

    @staticmethod
    def random(n, k, GF):
        stab = Stabilizer.std(n, k, GF)
        s = symp.get_random_symplectic(n, GF)
        stab.M = s @ stab.M
        stab.v = GF.Random(2 * n)
        return stab


