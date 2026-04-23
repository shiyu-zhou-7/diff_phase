import numpy as np
from scipy.linalg import eigh


class XXZhX_np:
    """XXZ + transverse field MPO (numpy). Mirrors hamiltonians.XXZhX exactly."""
    def __init__(self, L, delta, h, bc="finite"):
        assert bc in ["finite", "infinite"]
        self.bc = bc
        self.delta = delta
        self.d = 2
        self.L = L
        self.h = h

        sx = np.array([[0, 1],    [1, 0]],    dtype=complex)
        sy = np.array([[0, -1j],  [1j, 0]],   dtype=complex)
        sz = np.array([[1.0, 0.0],[0.0, -1.0]],dtype=complex)
        idn = np.eye(2, dtype=complex)

        self.Ws = []
        for _ in range(L):
            w = np.zeros((5, 5, 2, 2), dtype=complex)
            w[0, 0] = idn
            w[0, 1] = sx
            w[0, 2] = sy
            w[0, 3] = sz
            w[0, 4] = h * sx
            w[1, 4] = sx
            w[2, 4] = sy
            w[3, 4] = delta * sz
            w[4, 4] = idn
            self.Ws.append(w)


class MPS_np:
    """1D MPS (numpy). Mirrors mps.MPS exactly."""
    def __init__(self, Ss, Bs, bc="finite"):
        self.Ss = Ss
        self.Bs = Bs
        assert bc in ["finite", "infinite"]
        self.bc = bc
        self.num_bonds = len(Bs) - 1 if bc == "finite" else len(Bs)
        self.L = len(Bs)

    def get_theta(self, ind, k=2):
        assert ind <= self.num_bonds
        assert k in [1, 2]
        if k == 2:
            theta = np.tensordot(np.diag(self.Ss[ind]), self.Bs[ind], axes=(1, 0))
            theta = np.tensordot(theta, self.Bs[(ind + 1) % self.L], axes=(2, 0))
            return theta
        else:
            theta = np.tensordot(self.Bs[ind], np.diag(self.Ss[ind]), axes=(0, 1))
            return theta

    def get_site_exp_val(self, ops):
        """ops: iterable of (site, op) — mirrors MPS.get_site_exp_val."""
        exp_vals = []
        for site, op in ops:
            op_np = np.asarray(op)
            theta = self.get_theta(site, k=1)
            theta_ = np.conj(theta)
            exp_val = np.tensordot(theta, op_np, axes=(0, 1))
            exp_val = np.tensordot(exp_val, theta_, axes=([0, 2, 1], [1, 0, 2]))
            exp_vals.append(exp_val)
        return exp_vals


def split_and_truncate_np(theta, shape, chi_max, eps=1e-14):
    """SVD split + truncation. Mirrors mps.split_and_truncate exactly."""
    chiL, dL, dR, chiR = shape
    theta_matrix = theta.reshape((chiL * dL, chiR * dR))
    U, Sfull, V = np.linalg.svd(theta_matrix, full_matrices=False)

    chi_keep = int(np.sum(Sfull > eps))
    chi_keep = min(chi_keep, chi_max)

    A = U[:, :chi_keep]
    B = V[:chi_keep, :]
    S = Sfull[:chi_keep]
    S = S / np.linalg.norm(S)

    A = A.reshape([chiL, dL, chi_keep])
    B = B.reshape([chi_keep, dR, chiR])
    return A, S, B


def get_random_MPS_np(L, d, bond_dim=2, bc="finite", seed=0):
    """Random MPS initialisation. Mirrors mps.get_random_MPS exactly."""
    rng = np.random.default_rng(seed)
    Bs = []
    Ss = []
    for i in range(L):
        bond_left  = 1 if i == 0     else bond_dim
        bond_right = 1 if i == L - 1 else bond_dim
        B = rng.standard_normal((bond_left, d, bond_right))
        S = np.abs(rng.standard_normal((bond_right,))) + 1e-10
        Bs.append(B)
        Ss.append(S)
    return MPS_np(Ss, Bs, bc)


class H_eff_np:
    """Effective two-site Hamiltonian (numpy). Mirrors dmrg.H_eff exactly."""
    def __init__(self, lenv, renv, W1, W2):
        self.lenv = lenv
        self.renv = renv
        self.W1 = W1
        self.W2 = W2
        self.dtype = W1.dtype
        chiL, chiR = lenv.shape[0], renv.shape[0]
        d1, d2 = W1.shape[2], W2.shape[2]
        self.theta_shape = (chiL, d1, d2, chiR)
        self.shape = (chiL * d1 * d2 * chiR, chiL * d1 * d2 * chiR)

    def matvec(self, theta):
        state = theta.reshape(self.theta_shape)
        state = np.tensordot(self.lenv, state, axes=(0, 0))
        state = np.tensordot(state, self.W1, axes=[[0, 2], [0, 3]])
        state = np.tensordot(state, self.W2, axes=[[3, 1], [0, 3]])
        state = np.tensordot(state, self.renv, axes=[[1, 3], [0, 1]])
        return state.reshape(self.shape[0])

    def to_dense(self):
        size = self.shape[0]
        dense_matrix = np.zeros((size, size), dtype=self.dtype)
        for i in range(size):
            vec = np.zeros(size, dtype=self.dtype)
            vec[i] = 1.0
            dense_matrix[:, i] = self.matvec(vec)
        return dense_matrix


class DMRG_np:
    """Two-site finite DMRG (numpy). Mirrors dmrg.DMRG exactly."""
    def __init__(self, psi, MPO, chi_max, eps=1e-14):
        self.L = psi.L
        self.psi = psi
        self.MPO = MPO
        self.renvs = [None] * self.L
        self.lenvs = [None] * self.L
        self.chi_max = chi_max
        self.eps = eps

        chi = psi.Bs[0].shape[0]
        D   = self.MPO.Ws[0].shape[0]

        lenv = np.zeros((chi, D, chi))
        renv = np.zeros((chi, D, chi))
        lenv[:, 0,     :] = np.eye(chi)
        renv[:, D - 1, :] = np.eye(chi)

        self.lenvs[0]  = lenv
        self.renvs[-1] = renv

        for i in range(self.L - 1, 1, -1):
            self.update_renv(i)

    def sweep(self):
        for i in range(self.psi.num_bonds - 1):
            self.update_bond(i)
        for i in range(self.psi.num_bonds - 1, 0, -1):
            self.update_bond(i)

    def update_bond(self, i):
        j = (i + 1) % self.psi.L
        h_eff = H_eff_np(
            self.lenvs[i], self.renvs[j],
            self.MPO.Ws[i], self.MPO.Ws[j],
        )
        theta = self.psi.get_theta(i).reshape(h_eff.shape[0])

        h_eff_dense = h_eff.to_dense()
        evals, evecs = eigh(h_eff_dense)
        theta_new = evecs[:, np.argmin(evals)].reshape(h_eff.theta_shape)

        A, Sj, B = split_and_truncate_np(theta_new, h_eff.theta_shape, self.chi_max, self.eps)

        Si = self.psi.Ss[i]
        Bprev = np.tensordot(np.diag(1.0 / Si), A,            axes=(1, 0))
        Bprev = np.tensordot(Bprev,              np.diag(Sj), axes=(2, 0))
        self.psi.Ss[j] = Sj
        self.psi.Bs[i] = Bprev
        self.psi.Bs[j] = B

        self.update_lenv(i)
        self.update_renv((i + 1) % self.psi.L)

    def update_lenv(self, i):
        j      = (i + 1) % self.psi.L
        lenv_i = self.lenvs[i]
        W      = self.MPO.Ws[i]
        B      = self.psi.Bs[i]

        S    = np.diag(self.psi.Ss[i])
        Sinv = np.diag(1.0 / self.psi.Ss[j])

        G  = np.tensordot(S, B, axes=(1, 0))
        A  = np.tensordot(G, Sinv, axes=(2, 0))
        A_ = np.conj(A)

        lenv_new = np.tensordot(lenv_i, A,  axes=(0, 0))
        lenv_new = np.tensordot(lenv_new, W,  axes=[[0, 2], [0, 3]])
        lenv_new = np.tensordot(lenv_new, A_, axes=[[0, 3], [0, 1]])

        self.lenvs[j] = lenv_new

    def update_renv(self, i):
        renv_i = self.renvs[i]
        W      = self.MPO.Ws[i]
        B      = self.psi.Bs[i]
        B_     = np.conj(B)

        renv_new = np.tensordot(B,        renv_i, axes=(2, 0))
        renv_new = np.tensordot(renv_new, W,      axes=[[1, 2], [3, 1]])
        renv_new = np.tensordot(renv_new, B_,     axes=[[1, 3], [2, 1]])

        self.renvs[(i - 1) % self.L] = renv_new


def run_dmrg_numpy(L, delta, h, dmrg_cfg):
    """Run DMRG for data generation (no JAX). Returns MPS_np."""
    model = XXZhX_np(L, delta, h)
    psi   = get_random_MPS_np(L, d=2, bond_dim=1)
    dmrg  = DMRG_np(psi, model, chi_max=dmrg_cfg.max_bond)
    for _ in range(dmrg_cfg.sweeps):
        dmrg.sweep()
    return dmrg.psi
