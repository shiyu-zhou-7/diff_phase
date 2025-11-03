import jax
import jax.numpy as jnp
from dmrg.mps import *
from dmrg.hamiltonians import *
from jax.numpy.linalg import svd, eigh
from jax import jit

class H_eff:
    """ Class for an effective Hamiltonian to perform two-site DMRG """
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
        """ Returns the action of H_eff on a two-site state theta """
        state = theta.reshape(self.theta_shape)
        state = jnp.tensordot(self.lenv, state, axes=(0, 0))
        state = jnp.tensordot(state, self.W1, axes=[[0, 2], [0, 3]])
        state = jnp.tensordot(state, self.W2, axes=[[3, 1], [0, 3]])
        state = jnp.tensordot(state, self.renv, axes=[[1, 3], [0, 1]])

        return state.reshape(self.shape[0])

    def to_dense(self):
        """ Converts the H_eff operator into a dense matrix """
        # Create a dense version of the effective Hamiltonian
        size = self.shape[0]
        dense_matrix = jnp.zeros((size, size), dtype=self.dtype)

        for i in range(size):
            vec = jnp.zeros(size)
            vec = vec.at[i].set(1.0)
            dense_matrix = dense_matrix.at[:, i].set(self.matvec(vec))

        return dense_matrix

class DMRG:
    """ DMRG toycode class implemented in JAX. """
    def __init__(self, psi, MPO, chi_max, eps=1e-14, lanczos = True):
        self.L = psi.L
        self.psi = psi
        self.MPO = MPO
        self.renvs = [None] * self.L
        self.lenvs = [None] * self.L
        self.chi_max = chi_max
        self.eps = eps
        self.lanczos = lanczos
        
        chi = psi.Bs[0].shape[0]  # Bond dimension
        D = self.MPO.Ws[0].shape[0]  # MPO dimension

        lenv = jnp.zeros((chi, D, chi))
        renv = jnp.zeros((chi, D, chi))

        lenv = lenv.at[:, 0, :].set(jnp.eye(chi))
        renv = renv.at[:, D - 1, :].set(jnp.eye(chi))

        self.lenvs[0] = lenv
        self.renvs[-1] = renv  # This is vR

        for i in range(self.L - 1, 1, -1):
            self.update_renv(i)

    def sweep(self):
        for i in range(self.psi.num_bonds - 1):
            self.update_bond(i)
        for i in range(self.psi.num_bonds - 1, 0, -1):
            self.update_bond(i)

    def update_bond(self, i):
        j = (i + 1) % self.psi.L
        h_eff = H_eff(self.lenvs[i], self.renvs[j], self.MPO.Ws[i], self.MPO.Ws[j])
    
        theta = self.psi.get_theta(i).reshape(h_eff.shape[0])
    
        # Lanczos algorithm for eigenvalue computation
        if self.lanczos ==True:
            T, V = lanczos(h_eff.matvec, theta, k=4)  # Pass `matvec` method of h_eff
            evals, evecs = jnp.linalg.eigh(T)
            if V.shape[1] != evecs.shape[0]:
                raise ValueError(f"Mismatch: V.shape[1] = {V.shape[1]}, evecs.shape[0] = {evecs.shape[0]}")
    
            # Transform back to the original space
            theta_new = V @ evecs[:, jnp.argmin(evals)]
            theta_new = theta_new.reshape(h_eff.theta_shape)
        else:
            h_eff_dense = h_eff.to_dense()
            evals, evecs = eigh(h_eff_dense)
            theta_new = evecs[:, jnp.argmin(evals)].reshape(h_eff.theta_shape)
            #print(evals[0])
    
        # Debugging shapes
        #print("Shape of V:", V.shape)
        #print("Shape of evecs:", evecs.shape)
        #print("Shape of evecs[:, jnp.argmin(evals)]:", evecs[:, jnp.argmin(evals)].shape)
    
        # Fix mismatch by padding V if necessary

    
        A, Sj, B = split_and_truncate(theta_new, h_eff.theta_shape, self.chi_max, self.eps)
    
        # Return to right canonical form
        Si = self.psi.Ss[i]
        Bprev = jnp.tensordot(jnp.diag(1.0 / Si), A, axes=(1, 0))
        Bprev = jnp.tensordot(Bprev, jnp.diag(Sj), axes=(2, 0))
        self.psi.Ss[j] = Sj
        self.psi.Bs[i] = Bprev
        self.psi.Bs[j] = B
    
        self.update_lenv(i)
        self.update_renv((i + 1) % self.psi.L)

    def update_lenv(self, i):
        """ Updates the left environment with all tensors right of tensor i """
        j = (i + 1) % self.psi.L
        lenv_i = self.lenvs[i]
        W = self.MPO.Ws[i]
        B = self.psi.Bs[i]

        S = jnp.diag(self.psi.Ss[i])
        Sinv = jnp.diag(1.0 / self.psi.Ss[j])

        G = jnp.tensordot(S, B, axes=(1, 0))
        A = jnp.tensordot(G, Sinv, axes=(2, 0))
        A_ = jnp.conj(A)

        lenv_new = jnp.tensordot(lenv_i, A, axes=(0, 0))
        lenv_new = jnp.tensordot(lenv_new, W, axes=[[0, 2], [0, 3]])
        lenv_new = jnp.tensordot(lenv_new, A_, axes=[[0, 3], [0, 1]])

        self.lenvs[j] = lenv_new

    def update_renv(self, i):
        """ Updates the right environment with all tensors right of tensor i """
        renv_i = self.renvs[i]
        W = self.MPO.Ws[i]
        B = self.psi.Bs[i]
        B_ = jnp.conj(B)

        renv_new = jnp.tensordot(B, renv_i, axes=(2, 0))
        renv_new = jnp.tensordot(renv_new, W, axes=[[1, 2], [3, 1]])
        renv_new = jnp.tensordot(renv_new, B_, axes=[[1, 3], [2, 1]])

        self.renvs[(i - 1) % self.L] = renv_new

# @jax.jit(static_argnames=['k'])
def lanczos(A, v0, *, k):
    """
    Lanczos algorithm to approximate the smallest eigenvalue of a symmetric matrix `A`.

    Args:
        A: A callable function (matrix-vector product) or a dense/sparse matrix.
        v0: (n,) initial vector (should be normalized).
        k: Number of Lanczos iterations (static argument).

    Returns:
        T: (k, k) tridiagonal matrix.
        V: (n, k) matrix of Lanczos vectors, padded if convergence happens early.
    """
    n = v0.shape[0]
    v = v0 / jnp.linalg.norm(v0)
    V = [v]
    T = jnp.zeros((k, k))
    beta = 0

    for j in range(k):
        w = A(v)  # Use A as a matrix-vector multiplication function
        alpha = jnp.dot(v, w)
        w = w - alpha * v - beta * (V[-2] if j > 0 else 0)
        beta = jnp.linalg.norm(w)

        if beta < 1e-10:  # Convergence
            break

        v = w / beta
        V.append(v)
        T = T.at[j, j].set(alpha)
        if j < k - 1:
            T = T.at[j, j + 1].set(beta).at[j + 1, j].set(beta)

    V = jnp.stack(V, axis=1)

    # Ensure V has exactly k columns
    if V.shape[1] > k:
        V = V[:, :k]
    elif V.shape[1] < k:
        padding = jnp.zeros((n, k - V.shape[1]))
        V = jnp.concatenate([V, padding], axis=1)

    return T, V


def run_dmrg(L, model, dmrg_cfg):
    # Call the DMRG algorithm with the provided parameters
    psi = get_random_MPS(L, d=2, bond_dim=1)
    dmrg = DMRG(psi, model, chi_max=dmrg_cfg.max_bond, lanczos=dmrg_cfg.lanczos_bool)
    for _ in range(dmrg_cfg.sweeps):
        dmrg.sweep()
    return dmrg.psi

