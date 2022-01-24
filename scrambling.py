import numpy as np

from clifford import Weyl, Clifford

# Randomly chooses (n/k) arrays of k sites each out of n sites total, without
# overlap. Requires that n is divisible by k without remainer.
def pick_random_sites(n, k):
    if n % k != 0:
        raise RuntimeError("k has to be a divisor of n!")
    sites = np.arange(n)
    np.random.shuffle(sites)
    sites = sites.reshape((n // k, k))
    sites.sort()
    return sites
    
# Randomly chooses a weyl operator of weight 1, and then for each step applies
# (n/k) random Clifford gates to (n/k) randomly selected sets each containing 
# k sites, without overlap. The weights of the resulting Weyl operators for all
# steps are returned as an array.
def compute_k_site_scrambling_weights(n, k, GF, steps):
    weyl = Weyl.random(n, GF, 1)
    weights = np.zeros(steps + 1, dtype=int)
    weights[0] = 1
    for i in range(steps):
        sites = pick_random_sites(n, k)
        for s in sites:
            cliff = Clifford.random(k, GF)
            weyl = cliff.apply_to(weyl, s)
        weights[i+1] = weyl.get_weight()
    return weights





