"""(L, d)-generic version of the generalized cluster-state ED machinery.

Same physics and conventions as cluster_ed.py (which is pinned to the paper's
L = 10, d = 8): PBC, even-parity sector of P = prod sx_i, Hadamard-rotated
frame, H(t) = sum_a t_a H_a, labels from polynomial roots, JW free-fermion
reference with the alpha = 0 sign correction. Requires 1 <= d <= L (alpha
ranges over 0..d-1; endpoint sites of every string stay distinct).
"""

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import qmc, norm


def build_block_operators(L, D, parity=+1):
    DIM = 1 << L
    want_odd = (parity == -1)
    states = np.array([s for s in range(DIM)
                       if (bin(s).count("1") & 1) == want_odd], dtype=np.int64)
    index = -np.ones(DIM, dtype=np.int64)
    index[states] = np.arange(len(states))
    n = len(states)
    bits = ((states[:, None] >> np.arange(L)[None, :]) & 1)
    zval = 1.0 - 2.0 * bits

    ops = []
    for alpha in range(D):
        A = np.zeros((n, n))
        if alpha == 0:
            np.fill_diagonal(A, -zval.sum(axis=1))
        else:
            for i in range(L):
                j = (i + alpha) % L
                string = [(i + k) % L for k in range(1, alpha)]
                target = index[states ^ ((1 << i) | (1 << j))]
                sign = np.ones(n) if not string else zval[:, string].prod(axis=1)
                np.add.at(A, (target, np.arange(n)), -sign)
        ops.append(A)
    return ops, states


class BlockSolverG:
    def __init__(self, L, D, parity=+1):
        assert 1 <= D <= L
        self.L, self.D = L, D
        dense, self.states = build_block_operators(L, D, parity)
        csr = [sp.csr_matrix(A) for A in dense]
        pattern = sum(abs(c) for c in csr)
        pattern.sum_duplicates()
        pattern.data[:] = 1.0
        self.pattern = pattern.astype(np.float64)
        lut = pattern.copy()
        lut.data = np.arange(len(pattern.data), dtype=np.float64)
        datas = []
        for c in csr:
            d = np.zeros(len(pattern.data))
            ci = c.tocoo()
            pos = np.asarray(lut[ci.row, ci.col]).ravel().astype(np.int64)
            np.add.at(d, pos, ci.data)
            datas.append(d)
        self.datas = np.array(datas)
        self.dense_stack = np.array(dense) if (1 << (L - 1)) <= 2048 else None

    def h_csr(self, t):
        h = self.pattern.copy()
        h.data = t @ self.datas
        return h

    def solve_eigsh(self, t, k=2, tol=0.0, maxiter=10000):
        vals, vecs = spla.eigsh(self.h_csr(t), k=k, which="SA",
                                tol=tol, maxiter=maxiter)
        o = np.argsort(vals)
        return vals[o], vecs[:, o]

    def solve_dense(self, t, k=2):
        h = (np.tensordot(t, self.dense_stack, axes=1)
             if self.dense_stack is not None
             else self.h_csr(t).toarray())
        return sla.eigh(h, subset_by_index=[0, k - 1])


def winding_label_g(t, trim_tol=1e-13):
    t = np.asarray(t, dtype=np.float64)
    coeffs = t[::-1]
    nz = np.nonzero(np.abs(coeffs) > trim_tol * np.max(np.abs(t)))[0]
    coeffs = coeffs[nz[0]:]
    if len(coeffs) == 1:
        return 0, np.inf
    radii = np.abs(np.roots(coeffs))
    return int((radii < 1.0).sum()), float(np.min(np.abs(radii - 1.0)))


def winding_label_phys_g(t, trim_tol=1e-13):
    tt = np.array(t, dtype=np.float64)
    tt[0] = -tt[0]
    return winding_label_g(tt, trim_tol)


def majorana_matrix_g(t, L, sector="NS"):
    D = len(t)
    s = -1.0 if sector == "NS" else +1.0
    M = np.zeros((L, L))
    for alpha in range(D):
        c = t[0] if alpha == 0 else -t[alpha]
        for j in range(L):
            l = j + alpha
            M[j, l % L] += c * (s if l >= L else 1.0)
    return M


def free_fermion_even_E0_g(t, L):
    M = majorana_matrix_g(t, L, "NS")
    sigma = np.sort(sla.svdvals(M))
    E = -sigma.sum()
    if np.linalg.det(M) < 0:
        E += 2.0 * sigma[0]
    return E, sigma


def sobol_sphere_g(n, d, seed):
    eng = qmc.Sobol(d=d, scramble=True, seed=seed)
    u = np.clip(eng.random(n), 1e-15, 1 - 1e-15)
    g = norm.ppf(u)
    return g / np.linalg.norm(g, axis=1, keepdims=True)
