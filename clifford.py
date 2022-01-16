import random as rand
import numpy as np

import symplectic as symp

class Weyl:
    def __init__(self, v):
        self.v = v
        self.n = len(v) // 2
        self.GF = type(v)

    @staticmethod
    def random(n, GF, weight=-1):
        if weight < 0:
            weight = n
        v = np.zeros(2 * n, dtype=int)
        for _ in range(weight):
            i = rand.randrange(0, 2 * n, 2)
            pq = np.random.randint(0, GF.order, 2, dtype=int)
            while pq[0] == 0 and pq[1] == 0:
                pq = np.random.randint(0, GF.order, 2, dtype=int)
            v[i:i+2] = pq
        return Weyl(v.view(GF))

    def __mul__(self, other):
        return Weyl(self.v + other.v)

    def get_weight(self):
        weight = 0
        for i in range(0, 2 * self.n, 2):
            p_i = self.v[i].view(np.ndarray)
            q_i = self.v[i+1].view(np.ndarray)
            if p_i != 0 or q_i != 0:
                weight += 1
        return weight

class Clifford:
    def __init__(self, s, w):
        self.symp = s
        self.weyl = w
        self.n = w.n
        self.GF = w.GF

    @staticmethod
    def random(n, GF):
        s = symp.get_random_symplectic(n, GF)
        w = Weyl.random(n, GF)
        return Clifford(s, w)

    def apply_to(self, weyl):
        return Weyl(self.symp @ weyl.v)

    def __mul__(self, other):
        s = self.symp @ other.symp
        w = self.weyl * self.apply_to(other.weyl)
        return Clifford(s, w)
    
