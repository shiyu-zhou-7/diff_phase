"""
1D transverse-field Ising model on a length-N chain.
H = J * sum_i Sz_i Sz_{i+1} + h * sum_i Sx_i

Exposes:
  - `Sx`, `Sz`, `Id`            : single-site Pauli matrices (jax)
  - `prod(N, sites, ops)`       : tensor product of single-site ops on `sites`, identities elsewhere
  - `build_tfim_chain(N, J)`    : returns (ham_X, ham_ZZ) so H = h * ham_X + ham_ZZ
  - `H_tfim(h, ham_X, ham_ZZ)`  : assemble Hamiltonian at given h
  - `gd_solver_ed(h, ...)`      : exact-diag ground state (jit-compatible)
"""
import jax.numpy as jnp
from jax import jit


Id = jnp.eye(2)

Sz = jnp.zeros([2, 2])
Sz = Sz.at[0, 0].set(1.)
Sz = Sz.at[1, 1].set(-1.)

Sx = jnp.zeros([2, 2])
Sx = Sx.at[0, 1].set(1.)
Sx = Sx.at[1, 0].set(1.)


def prod(N, sites, ops):
    """
    N (int)     : chain length
    sites (list): integer site indices the operators act on
    ops   (list): single-site operators corresponding to `sites`
    Returns the (2**N, 2**N) operator with `ops` on `sites` and Id elsewhere.
    """
    arg = jnp.argsort(jnp.asarray(sites))
    sites = jnp.array(sites)[arg]
    ops = jnp.array(ops)[arg]
    cnt = 0
    Out = None
    if sites[cnt] == 0:
        Out = ops[cnt]
        cnt += 1
    else:
        Out = Id

    for i in range(1, N):
        if cnt == len(sites):
            Out = jnp.kron(Out, Id)
            continue

        if i == sites[cnt]:
            Out = jnp.kron(Out, ops[cnt])
            cnt += 1
        else:
            Out = jnp.kron(Out, Id)

    return Out


def build_tfim_chain(N, J=-1.0):
    """
    Returns (ham_X, ham_ZZ) so that H_tfim(h, ham_X, ham_ZZ) = h * ham_X + ham_ZZ.

    H = J * sum_i Sz_i Sz_{i+1} + h * sum_i Sx_i  (open boundaries).
    """
    ham_ZZ = prod(N, [0, 1], [Sz, Sz])
    ham_X = prod(N, [0], [Sx])
    for i in range(1, N - 1):
        ham_ZZ += prod(N, [i, i + 1], [Sz, Sz])
        ham_X += prod(N, [i], [Sx])
    ham_X += prod(N, [N - 1], [Sx])
    ham_ZZ *= J
    return ham_X, ham_ZZ


@jit
def H_tfim(h, ham_X, ham_ZZ):
    return ham_X * h + ham_ZZ


@jit
def gd_solver_ed(h, ham_X, ham_ZZ):
    """Exact-diag eigenpairs (e, v) of H_tfim. v[:, 0] is the ground state."""
    H = H_tfim(h, ham_X, ham_ZZ)
    e, v = jnp.linalg.eigh(H)
    return e, v
