import random as rand
import numpy as np

import qstab.misc as misc
import qstab.symplectic as symp

# Implements the projective representation of a Weyl operator on n sites as an
# element of a 2n-dimensional (symplectic) vector space.
# Phase factors are ignored.
class Weyl:

    # Default constructor. Wraps a 2n-dimensional vector into a Weyl object.
    def __init__(self, v):
        if len(v) % 2 != 0:
            raise RuntimeError("The dimension of v has to be an even number.")
        self.v = v
        self.n = len(v) // 2
        self.GF = type(v)

    @staticmethod
    def zero(n, GF):
        return Weyl(GF.Zeros(2 * n))

    # Returns a randomly sampled Weyl operator of a given weight.
    # By default the weight is chosen randomly.
    @staticmethod
    def random(n, GF, weight=-1):
        if weight < 0:
            weight = rand.randint(0, n)
        v = np.zeros(2 * n, dtype=int)
        for _ in range(weight):
            i = rand.randrange(0, 2 * n, 2)
            pq = GF.Random(2)
            while pq[0] == 0 and pq[1] == 0:
                pq = GF.Random(2)
            v[i:i+2] = pq
        return Weyl(v.view(GF))

    # Allows forming a tensor product of Weyl operators by using "+".
    def __add__(self, other):
        n_add = self.n + other.n
        v_add = self.GF.Zeros(2 * n_add)
        v_add[0:2*self.n] = self.v
        v_add[2*self.n:2*n_add] = other.v
        return Weyl(v_add)

    # Allows multiplication of Weyl operators by using "*".
    def __mul__(self, other):
        return Weyl(self.v + other.v)

    # Returns the weight of the Weyl operator, i.e. the total number of
    # non-trivial Pauli operators on all sites.
    def get_weight(self):
        weight = 0
        for i in range(0, 2 * self.n, 2):
            p_i = self.v[i].view(np.ndarray)
            q_i = self.v[i+1].view(np.ndarray)
            if p_i != 0 or q_i != 0:
                weight += 1
        return weight

# Implements the projective representation of a Clifford operator on n sites as
# a symplectic matric on a 2n-dimensional (symplectic) vector space.
# Phase factors are ignored.
class Clifford:

    # Default constructor. Wraps a 2n x 2n symplectic matrix into a Clifford object.
    def __init__(self, s):
        self.symp = s
        self.n = len(s) // 2
        self.GF = type(s)

    # Returns the Clifford operator corresponding to the 2n x 2n identity operator.
    @staticmethod
    def identity(n, GF):
        s = GF.Identity(2 * n)
        return Clifford(s)

    # Returns a randomly sampled Clifford operator.
    @staticmethod
    def random(n, GF):
        s = symp.get_random_symplectic(n, GF)
        return Clifford(s)

    @staticmethod
    def random_k_site(n, k, GF):
        sites = misc.pick_random_sites(n, k)
        symp_sites = \
            list(map(lambda s: (symp.get_random_symplectic(k, GF), s), sites))
        symp_k_site = symp.embed_symplectics(n, GF, *symp_sites)
        return Clifford(symp_k_site)

    # Embeds a Clifford operator on n sites into one on n_embed sites, which
    # thus acts like the original Clifford on n <= n_embed specified subsites.
    def embed(self, n_embed, sites):
        s_embed = symp.embed_symplectics(n_embed, self.GF, (self.symp, sites))
        return Clifford(s_embed)

    # Applies the Clifford operator to a given Weyl operator, returning the
    # resulting Weyl operator. If n_clifford is smaller than n_weyl, an
    # embedded version of the Clifford is applied to the specified sites.
    def apply_to_weyl(self, weyl, sites = []):
        if self.n == weyl.n:
            return Weyl(self.symp @ weyl.v)
        elif self.n > weyl.n:
            raise RuntimeError \
                ("Dimension of Clifford can't be larger than the one of Weyl!")
        return self.embed(weyl.n, sites).apply_to(weyl)

    # Allows matrix multiplication of Clifford operators by using "*".
    def __mul__(self, other):
        s = self.symp @ other.symp
        return Clifford(s)
    
