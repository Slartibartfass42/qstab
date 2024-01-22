import numpy as np

import qstab.misc as misc
import qstab.symplectic as symp
from qstab.clifford import Weyl

class Stabilizer:

    # Constructs a new stabilizer object using a basis array of dimensions
    # k x 2*n and a phase vector v of length 2*n. k <= n.
    def __init__(self, array, v):
        self.array = array
        self.v = v
        self.span = None
        self.GF = type(v)
        self.n = len(v) // 2
        self.k = len(array)

    # Returns the basis array as a list of Weyl operators.
    def get_basis(self):
        return list(map(lambda v: Weyl(v), self.array))

    # Returns the space of all stabilizer elements as Weyl operators.
    def get_span(self):
        if self.span == None:
            self.span = misc.compute_span(self.array)
        return self.span

    # Returns a stabilizer object with a canonical basis array of dimension
    # k x n and zero phase vector v. k <= n.
    @staticmethod
    def std(n, k, GF):
        array_std = GF.Zeros((k, 2*n))
        for i in range(k):
            array_std[i, 2 * i] = GF(1)
        v_std = GF.Zeros(2 * n)
        return Stabilizer(array_std, v_std)

    # Returns a stabilizer object with a random basis array of dimension
    # k x n and random phase vector v. k <= n.
    @staticmethod
    def random(n, k, GF):
        array_rand = GF.Zeros((k, 2*n))
        s = symp.get_random_symplectic(n, GF).transpose()
        for i in range(k):
            array_rand[i] = s[2 * i]
        v_rand = GF.Random(2 * n)
        return Stabilizer(array_rand, v_rand)

    # Returns a stabilizer object representing two parties sharing n_party
    # maximally entangled qudits. The resulting basis array will have dimensions
    # 2*n_party x 4*n_party, where the left half of columns "belongs" to one
    # party and the right half "belongs" to the other. If random=True then a
    # random Clifford is applied to the right half of the basis array.
    @staticmethod
    def bell_state(n_party, GF, random=False):
        n_bell = 2 * n_party
        array_bell = GF.Zeros((n_bell, 2*n_bell))
        array_bell[0:n_bell,0:n_bell] = GF.Identity(n_bell)
        array_bell[0:n_bell-1:2, n_bell:2*n_bell-1:2] = -GF.Identity(n_party)
        array_bell[1:n_bell:2, n_bell+1:2*n_bell:2] = GF.Identity(n_party)

        if random:
            s = symp.get_random_symplectic(n_party, GF)
            s_embed = symp.embed_symplectics \
                (n_bell, GF, (s, range(n_party, n_bell)))
            array_bell = array_bell @ s_embed
            v_bell = GF.Random(2 * n_bell)
        else:
            v_bell = GF.Zeros(2 * n_bell)

        return Stabilizer(array_bell, v_bell)

    # Allows forming a tensor product of two stabilizer states using "+".
    def __add__(self, other):
        k_add = self.k + other.k
        n_add = self.n + other.n
        array_add = self.GF.Zeros((k_add, 2 * n_add))
        v_add = self.GF.Zeros(2 * n_add)
        array_add[0:self.k, 0:2*self.n] = self.array
        array_add[self.k:k_add, 2*self.n:2*n_add] = other.array
        v_add[0:2*self.n] = self.v
        v_add[2*self.n:2*n_add] = other.v
        return Stabilizer(array_add, v_add)

    # Returns a stabilizer object with its basis array in reduced-row echelon
    # form (RREF).
    def get_rref(self):
        array_rref = misc.compute_rref(self.array)
        return Stabilizer(array_rref, self.v)

    # Permutes the pairs of columns representing one qudit site according to the
    # permutation of [1, ..., n] provided in sites_perm. Note that n here must
    # be half of the number of columns of the basis array.
    def permute_sites(self, sites_perm):
        if len(sites_perm) != self.n:
            raise RuntimeError("All sites have to be permutated!")
        index_perm = [2 * s + i for s in sites_perm for i in [0, 1]]
        array_perm = self.array[:, index_perm]
        v_perm = self.v[index_perm]
        return Stabilizer(array_perm, v_perm)

    # Returns the stabilizer object with the provided sites (pairs of column
    # vectors) traced out. 
    def trace_out(self, *sites_traced):
        GF = self.GF
        n_traced = len(sites_traced)
        if n_traced == 0:
            return self
        sites_traced = list(sites_traced)
        sites_compl = [i for i in range(self.n) if i not in sites_traced]
        sites_perm = sites_traced + sites_compl
        stab_perm = self.permute_sites(sites_perm)

        array_rref = misc.compute_rref(stab_perm.array)
        array_traced = [misc.remove_sites(v, *range(n_traced)) \
            for v in array_rref if misc.has_zeros_at(v, *range(n_traced))]

        array_traced = np.array(array_traced, dtype=int).view(GF)

        v_traced = misc.remove_sites(self.v, *range(n_traced))

        return Stabilizer(array_traced, v_traced)

    # Returns the number of stabilizers in the span of the basis array.
    def get_cardinality(self):
        d = self.GF.order
        return d**self.k

    # Returns the (entanglement) entropy of the corresponding stabilizer state.
    def get_entropy(self):
        return self.n - self.k

    # Returns the operator weights (number of non-identity Pauli operators) of
    # the stabilizer basis.
    def get_weights(self):
        return list(map(lambda w: w.get_weight(), self.get_basis()))

    # Returns the mutual information between two sets of sites of the Stabilizer. 
    def get_mutual_information(self, sites_a, sites_b):
        sites_ab = sorted(set().union(sites_a, sites_b))
        sites_compl = [i for i in range(self.n) if i not in sites_ab]
        state_ab = self.trace_out(*sites_compl)
        
        state_a = self.trace_out(*sites_b, *sites_compl)
        state_b = self.trace_out(*sites_a, *sites_compl)

        mut_inf =  state_a.get_entropy() + state_b.get_entropy() \
            - state_ab.get_entropy()
        return mut_inf

    # Applies a Clifford operator to the stabilizer object and returns the
    # result. If specific sites are supplied, the Clifford will only be applied
    # to those (given that the dimensions match).
    def apply_clifford(self, cliff, sites = []):
        if self.n == cliff.n:
            v_applied = cliff.symp @ self.v
            array_applied = self.array @ cliff.symp
            return Stabilizer(array_applied, v_applied)
        elif self.n < cliff.n:
            raise RuntimeError \
                ("Dimension of Clifford can't be larger than the one of Stab!")
        return self.apply_clifford(cliff.embed(self.n, sites))

    