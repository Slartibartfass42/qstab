import itertools as iter

import numpy as np
from qstab.stabilizer import Stabilizer
import random as rand

import qstab.symplectic as symp
from qstab.clifford import Weyl, Clifford

def generate_stepwise_k_scrambler(n, k, steps, GF):
    scrambler = Clifford.random_k_site(n, k, GF)
    for _ in range(steps - 1):
        single_step_scrambler = Clifford.random_k_site(n, k, GF)
        scrambler *= single_step_scrambler
    return scrambler

def generate_syk_scrambler(n, n0, k, layers, steps_layer, GF):
    symps = [Clifford.identity(n, GF)] * layers
    for i in range(layers):
        n_scramble = n0 + k**(i+1)
        scrambler = generate_stepwise_k_scrambler \
            (n_scramble, k, steps_layer, GF)
        symp_layer = scrambler.embed(n, range(n_scramble))
        symps[i] = symp_layer
    return symps

def extract_code_distance(bell_stab, n_aux, n_mc=0):
    n_party = (bell_stab.n - n_aux) // 2
    sites_a = list(range(n_party))
    sites_b = list(range(n_party, bell_stab.n))
    if n_mc <= 0:
        for d in range(1, n_aux):
            s_b_perms = [s for s in iter.permutations(sites_b, d) \
                if sorted(s) == list(s)]
            for s in s_b_perms:
                mut_inf = bell_stab.get_mutual_information(s, sites_a)
                if mut_inf > 0:
                    return d-1
        return -1
    else:
        d_max = n_aux // 2 + 1
        iter = n_mc
        while (iter > 0):
            perm_size = rand.randint(1, d_max)
            s_b_perm = rand.sample(sites_b, perm_size)
            mut_inf = bell_stab.get_mutual_information(sites_a, s_b_perm)
            if mut_inf > 0:
                d_max = perm_size - 1
                iter = n_mc + 1
                if d_max == 0:
                    return 0
            iter -= 1
        return d_max

def syk_bell_scrambling(n0, k, layers, steps_layer, GF):
    n_anc = k**layers
    n_syk = n0 + n_anc
    n_total = n_syk + n0

    bell_stab = Stabilizer.bell_state(n0, GF)
    anc_stab = Stabilizer.std(n_anc, n_anc, GF)

    syk_stab = bell_stab + anc_stab\

    symps = generate_syk_scrambler(n_syk, n0, k, layers, steps_layer, GF)

    for s in symps:
        syk_stab = \
            syk_stab.apply_clifford(s, list(range(n0, n_total)))

    return syk_stab

def get_syk_weights(n0, k, layers, steps_layer, GF):
    n = n0 + k**layers
    n1 = n - n0
    stab_0 = Stabilizer.std(n0, n0, GF)
    stab_1 = Stabilizer.std(n1, n1, GF)
    stab = stab_0 + stab_1

    weights = np.zeros((layers + 1, n))
    weights[0] = stab.get_weights()

    symps = generate_syk_scrambler(n, n0, k, layers, steps_layer, GF)
    for i in range(layers):
        stab = stab.apply_clifford(symps[i])
        weights[i+1] = stab.get_weights()

    return weights