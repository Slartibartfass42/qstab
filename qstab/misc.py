from math import log
import itertools as iter

from multiprocessing import Process, Queue

import numpy as np

def compute_rref(array):
    GF = type(array)
    lead = 0
    n_rows, n_columns = array.shape
    array_rref = array.copy()
    for r in range(n_rows):
        if n_columns <= lead:
            return array_rref
        i = r
        while array_rref[i, lead] == GF(0):
            i += 1
            if n_rows == i:
                i = r
                lead += 1
                if n_columns == lead:
                    return array_rref
        if i != r:
            temp = array_rref[i].copy()
            array_rref[i] = array_rref[r]
            array_rref[r] = temp
        array_rref[r] /= array_rref[r, lead]
        for i in range(n_rows):
            if i != r:
                array_rref[i] -= array_rref[i, lead] * array_rref[r]
        lead += 1
    return array_rref

def has_zeros_at(v, *sites):
    GF = type(v)
    has_zeros = True
    for i in sites:  
        if v[2 * i] != GF(0) or v[2 * i + 1] != GF(0):
            has_zeros = False
    return has_zeros

def vecs_equal(v, w):
    for i in range(len(v)):
        if v[i] != w[i]:
            return False
    return True

def is_vec_in_list(v, vecs):
    for w in vecs:
        if vecs_equal(v, w):
            return True
    return False

def remove_sites(v, *sites):
    GF = type(v)
    n = len(v) // 2
    v_reduced = GF([v[2 * i + k] for i in range(n) \
            if i not in sites for k in [0, 1]])
    return v_reduced

def find_basis(span):
    GF = type(span[0])
    d = GF.order
    n = len(span[0]) // 2
    card = len(span)
    k = int(log(card, d))
    
    if len(span) == 1:
        return span
    
    if vecs_equal(span[0], GF.Zeros(2 * n)):
        basis = [span[1]]
    else:
        basis = [span[0]]

    for v in span:
        if len(basis) == k:
            break
        b_span = compute_span(basis)
        if not is_vec_in_list(v, b_span):
            basis.append(v)

    return basis

def compute_span(basis):
    GF = type(basis[0])
    d = GF.order
    n = len(basis[0]) // 2
    k = len(basis)
    card = d**k

    span = list(GF.Zeros((card, 2 * n)))
    coeffs = list(iter.product(GF.Elements(), repeat=k))
    
    for i in range(card):
        for j in range(k):
            span[i] += coeffs[i][j] * basis[j]
 
    return span

# Randomly chooses (n/k) arrays of k sites each out of n sites total, without
# overlap. Requires that n is divisible by k without remainer.
def pick_random_sites(n, k):
    if n % k != 0:
        raise RuntimeError("k has to be a divisor of n!")
    sites = np.arange(n)
    np.random.shuffle(sites)
    sites = sites.reshape((n // k, k))
    sites.sort()
    return list(sites)

# Multiprocessor allowing multiple functions to be executed in parallel.
class Multiprocessor():
    def __init__(self):
        self.processes = []
        self.queue = Queue()

    @staticmethod
    def _wrapper(f, queue, args, kwargs):
        res = f(*args, **kwargs)
        queue.put(res)

    # Add a function to be run in parallel
    def run(self, f, *args, **kwargs):
        args_wrapper = [f, self.queue, args, kwargs]
        p = Process(target=self._wrapper, args=args_wrapper)
        self.processes.append(p)
        p.start()

    # Wait for the executions to finish.
    # Returns the results of the input functions.
    def wait(self):
        results = []
        for p in self.processes:
            res = self.queue.get()
            results.append(res)
        for p in self.processes:
            p.join()
        return results