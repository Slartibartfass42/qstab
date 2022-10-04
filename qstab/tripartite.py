from math import log, cos, pi
import itertools as iter
import numpy as np

import qstab.misc as misc
from qstab.stabilizer import Stabilizer
import qstab.symplectic as symp

def compute_ghz_content(n_A, n_B, n_C, GF):
    n_ABC = n_A + n_B + n_C
    n_AB = n_A + n_B

    state_ABC = Stabilizer.random(n_ABC, n_ABC, GF)

    state_AB = state_ABC.trace_out(*range(n_AB, n_ABC))
    state_A = state_AB.trace_out(*range(n_A, n_AB))
    state_B = state_AB.trace_out(*range(0, n_A))
    state_C = state_ABC.trace_out(*range(0, n_AB))

    entropy_A = state_A.get_entropy()
    entropy_B = state_B.get_entropy()
    entropy_C = state_C.get_entropy()

    if entropy_A == 0 or entropy_B == 0 or entropy_C == 0:
        return 0

    d = GF.order
    trace = compute_part_transpose_trace(state_AB, n_A, n_B)
    g = entropy_A + entropy_B + entropy_C + round(log(trace, d))

    print(g)

    return g


def compute_part_transpose_trace(state_AB, n_A, n_B):
    n_AB = n_A + n_B
    GF = state_AB.GF
    d = GF.order
    k = state_AB.k

    array_A = state_AB.array[:, 0:2*n_A]
    basis_A_inner = GF.Zeros((k, k))

    for i in range(k):
        for j in range(k):
            basis_A_inner[i][j] = symp.eval_inner(array_A[i], array_A[j])

    if not np.any(basis_A_inner):
        return (state_AB.get_cardinality() / d**n_AB)**2

    coeffs = list(iter.product(*([GF.Elements()] * k)))
    card = d**k

    sum = 0
    for i in range(card):
        m_A = GF.Zeros(2 * n_A)
        for r in range(k):
            m_A += coeffs[i][r] * array_A[r]
        commutes = True
        for s in range(k):
            inner = symp.eval_inner(m_A, array_A[s])
            if inner != GF(0):
                commutes = False
                break
        if commutes:
            sum += 1

    return (sum * d**k) / d**(2 * n_AB)