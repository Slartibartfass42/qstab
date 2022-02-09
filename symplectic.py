import random as rand
import numpy as np

# Returns the kth pair (e_k, f_k) of the standard symplectic basis.
# k goes from 0 to n-1
def get_std_basis_pair(k, n, GF):
    e_k = GF.Zeros(2 * n)
    f_k = GF.Zeros(2 * n)
    e_k[k] = 1
    f_k[k+1] = 1
    return (e_k, f_k)

# Returns the (2n x 2n)-dimensional matrix defining the symplectic inner product.
def get_inner_matrix(n, GF):
    inner = GF.Zeros((2*n, 2*n))
    for i in range(0, 2*n, 2):
        inner[i + 1][i] = GF(0) - GF(1)
        inner[i][i+1] = 1
    return inner

# Evaluates the inner product between two 2*n-dimensional vectors
def eval_inner(v, w):
    GF = type(v)
    sum = GF(0)
    for i in range(0, len(v), 2):
        sum += v[i] * w[i+1] - v[i+1] * w[i]
    return sum

# Transvects a vector using a single transvection t = (a, h)
def transvect_once(v, t):
    a, h = t
    return v + a * eval_inner(v, h) * h

# Transvects a vector using a multiple transvections.
# ts is a list of transvections with them being applied first to last.
def transvect(v, ts):
    w = v
    for t in ts:
        w = transvect_once(w, t)
    return w

# Computes the transvection relating two vectors with non-vanishing symplectic
# product. Panics if this is not the case.
def find_single_transvection(v, w):
    GF = type(v)
    if eval_inner(v, w) != 0:
        a = GF(1) / eval_inner(v, w)
        h = v - w
        return (a, h)
    else:
        raise RuntimeError("Input vectors have vanishing symplectic product!")

# Computes the transvection(s) relating two arbitrary vectors.
def find_transvection(v, w):
    GF = type(v)
    n = len(v) // 2

    # Case 0: v = w
    if np.array_equal(v, w):
        return []

    # Case 1: v and w have a non-vanishing symplectic product.
    # Only one transvection necessary (second transvection is zero)
    if eval_inner(v, w) != 0:
        return [find_single_transvection(v, w)]

    # Case 2: v and w have a vanishing symplectic product.
    # v = \sum_i (a_i e_i + _i f_i)
    # w = \sum_i (c_i e_i + d_i f_i)
    # Find z such that <v,z> != 0 != <z,w>.
    # Chain transvections from v to z and z to w.

    # Case 2.1: a_i or b_i != 0, and c_i or d_i != 0
    # z = k e_i + l f_i
    for i in range(0, 2*n, 2):
        if (v[i] != 0 or v[i + 1] != 0) and (w[i] != 0 or w[i + 1] != 0):
            e_i, f_i = get_std_basis_pair(i, n, GF)
            for k in GF.Elements():
                for l in GF.Elements():
                    z = k * e_i + l * f_i
                    if eval_inner(v,z) != 0 and eval_inner(z,w) != 0:
                        t1 = find_single_transvection(v, z)
                        t2 = find_single_transvection(z, w)
                        return [t1, t2]

    # Case 2.2: a_i or b_i != 0 and c_i = 0 = d_i (or vice versa)
    # For i with a_i or b_i != 0 there exists j with c_j or d_j != 0 and i != j
    # z = k e_i + l f_i + m e_j + n f_j
    i = 0
    j = 0
    while(v[i] == 0 and v[i + 1] == 0):
        i = i + 2
    while(w[j] == 0 and w[j + 1] == 0):
        j = j + 2 
    
    e_i, f_i = get_std_basis_pair(i, n, GF)
    e_j, f_j = get_std_basis_pair(j, n, GF)
    for k in GF.Elements():
        for l in GF.Elements():
            for m in GF.Elements():
                for n in GF.Elements():
                    z = k * e_i + l * f_i + m * e_j + n * f_j
                    if eval_inner(v,z) != 0 and eval_inner(z,w) != 0:
                        t1 = find_single_transvection(v, z)
                        t2 = find_single_transvection(z, w)
                        return [t1, t2]

    raise RuntimeError("No transvection could be found!")

# Returns the number of elements of the symplelctic group Sym(n, GF).
def get_group_order(n, GF):
    g = GF.order
    order = g**(n**2)
    for i in range(1, n+1):
        order *= g**(2 * i) - 1
    return order

# Returns the radix representation of a given 0 <= k < |GF|^n as a list
# with length n. The index of the list corresponds to the power of |GF|.
def get_radix_repr(k, n, GF):
    v = GF.Zeros(n)
    rem = k
    for i in reversed(range(n)):
        ord = GF.order**i
        v[i] = GF(rem // ord)
        rem = rem % ord
    return v

# Computes a unique symplectic matrix for each 0 <= k < |Sym(n, GF)|.
def find_symplectic_matrix(k, n, GF): 
    s = GF.order**(2 * n) - 1
    u_seed = (k % s) + 1
    v_seed = (k // s) % GF.order**(2 * n - 1)
    std_basis = GF.Identity(2 * n)
    u = get_radix_repr(u_seed, 2 * n, GF)
    v_coeffs = get_radix_repr(v_seed, 2 * n - 1, GF)
    t1 = find_transvection(std_basis[0], u)
    
    if v_coeffs[0] != GF(0):
        v_coeffs[1:] /= v_coeffs[0]

    e = np.copy(std_basis[0]).view(GF)
    for i in range(2, 2 * n):
        e += v_coeffs[i-1] * std_basis[i]

    h0 = transvect(e, t1)

    if v_coeffs[0] != GF(0):
        t2 = [(-v_coeffs[0], h0)]
    else:
        t2 = [(-GF(1), h0), (GF(1), u)]

    t = t1 + t2

    if n == 1:
        v = transvect(std_basis[1], t)
        return np.column_stack((u, v))
    else:
        m = std_basis
        k_new = k // (s * GF.order**(2 * n - 1))
        m[2:,2:] = find_symplectic_matrix(k_new, n - 1, GF)
        m = m.transpose()
        for i in range(2 * n):
            m[i] = transvect(m[i], t)
        m = m.transpose()
        return m

# Returns random symplectic matrix.
def get_random_symplectic(n, GF):
    ord = get_group_order(n, GF)
    k = rand.randrange(ord)
    return find_symplectic_matrix(k, n, GF)

# For given tuples of symplectics of size 2n and sites of hyperbolic pairs in a
# 2n_embed dimensional space, this embeds the former into a symplectic of
# dimension 2n_embed with n_embed >= \sum n such that they each only act on the
# the specified sites. 
def embed_symplectics(n_embed, GF, *symp_sites):
    symp_embed = GF.Identity(2 * n_embed)
    n_free = n_embed
    for (symp, sites) in symp_sites:
        n = len(sites)
        if n != len(symp) // 2:
            raise RuntimeError \
                ("Number of sites has to match with n = len(symp)/2!")
        if n_free < n:
            raise RuntimeError \
                ("Not enough remaining sites to embed symplectic!")
        for i in range(n):
            i_embed = sites[i]
            for j in range(n):
                j_embed = sites[j]
                symp_embed[2*i_embed:2*i_embed+2, 2*j_embed:2*j_embed+2] \
                    = symp[2*i:2*i+2, 2*j:2*j+2]
        n_free -= n
    return symp_embed