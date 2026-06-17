"""
ED solver backend for the generalized cluster (stabilizer) chain.

    H(t) = - sum_{alpha=0}^{d-1} t_alpha * sum_i  Z_i X_{i+1}...X_{i+alpha-1} Z_{i+alpha}
           (+ optional kappa * sum_i Z_i Z_{i+2}, see `kappa` -- interacting,
            relinquishes the free-fermion analytic ground truth)

Design (conforms to the diffphase solver contract; mirrors the TFIM/xxz pattern
where term operators are constant and the Hamiltonian is LINEAR in the params):

  * Every term is a Pauli string of X's and Z's only (no Y) -> a signed
    permutation in the computational basis. H is real symmetric, ground state
    real. We work in real arithmetic.
  * H is LINEAR in t: build each term operator ONCE, projected into the frozen
    P-sector, as a constant matrix T_alpha. Then H_sector(t) = sum_alpha
    t_alpha * T_alpha and dH/dt_alpha = T_alpha (Hellmann-Feynman is exact).
    The minus sign in H is folded into T_alpha (T_alpha = - sum_i stabilizer_i).
  * We NEVER materialize a dense 2**L x 2**L operator (2 GB at L=14). Terms are
    built sparse, projected to the 2**(L-1) sector by the frozen isometry V
    (hamiltonians/projector.py), then densified to a small jnp matrix for eigh.
  * The differentiated forward pass uses dense jnp.linalg.eigh (correct analytic
    eigh adjoint for free); NO iterative Krylov on the autodiff tape.
  * State observables route through rho0 = |psi0><psi0| via the
    signed-permutation expectation <psi|O|psi> (a differentiable quadratic form),
    so the eigensolver's arbitrary sign cancels.
"""
from dataclasses import dataclass
from functools import partial

import numpy as np
import scipy.sparse as sp

import jax
import jax.numpy as jnp

from hamiltonians.projector import build_parity_sector, ParitySector


# ----------------------------------------------------------------------------
# Bit / Pauli-string helpers (numpy, build-time only -- not differentiated)
# ----------------------------------------------------------------------------
def bit_parity(v):
    """Parity (popcount mod 2) of each entry of an int64 array `v`. Vectorized
    fold; valid for up to 32-bit values (L <= 16 here)."""
    v = v.astype(np.int64)
    v = v ^ (v >> 16)
    v = v ^ (v >> 8)
    v = v ^ (v >> 4)
    v = v ^ (v >> 2)
    v = v ^ (v >> 1)
    return (v & 1).astype(np.int64)


def stabilizer_masks(L, alpha, i, bc):
    """(xmask, zmask) bitmasks for the alpha-stabilizer based at site i.

    alpha = 0 : X_i                          (transverse field; special-cased,
                                              NOT Z_i Z_i = I)
    alpha >=1 : Z_i, X_{i+1..i+alpha-1}, Z_{i+alpha}
    Sites wrap mod L for bc='pbc'. For bc='obc' the term is dropped (returns
    None) when it would wrap past the last site.
    Returns (xmask, zmask) or None if out of range under OBC.
    """
    if alpha == 0:
        return (1 << i), 0
    last = i + alpha
    if bc == 'obc' and last > L - 1:
        return None
    sites_z = [i % L, last % L]
    sites_x = [(i + j) % L for j in range(1, alpha)]
    zmask = 0
    for s in sites_z:
        zmask |= (1 << s)
    xmask = 0
    for s in sites_x:
        xmask |= (1 << s)
    return xmask, zmask


def kappa_masks(L, i, bc):
    """(xmask=0, zmask) for the interacting control term Z_i Z_{i+2}."""
    last = i + 2
    if bc == 'obc' and last > L - 1:
        return None
    return 0, (1 << (i % L)) | (1 << (last % L))


def _pauli_sparse(L, xmask, zmask):
    """Signed-permutation sparse matrix (2**L x 2**L) for the Pauli string with
    the given X / Z masks: O|s> = (-1)^popcount(s & zmask) |s ^ xmask|."""
    dim = 1 << L
    cols = np.arange(dim, dtype=np.int64)
    rows = cols ^ xmask
    signs = 1.0 - 2.0 * bit_parity(cols & zmask)  # (-1)^parity
    return sp.csr_matrix((signs, (rows, cols)), shape=(dim, dim))


def _term_full(L, alpha, bc):
    """Full-space sparse operator  - sum_i stabilizer_{alpha,i}  (minus folded
    in so H = sum_alpha t_alpha * T_alpha)."""
    dim = 1 << L
    T = sp.csr_matrix((dim, dim))
    n_sites = L if bc == 'pbc' else L  # iterate all i; OBC drops wrapped terms
    for i in range(n_sites):
        m = stabilizer_masks(L, alpha, i, bc)
        if m is None:
            continue
        T = T + _pauli_sparse(L, m[0], m[1])
    return -T


def _kappa_full(L, bc):
    """Full-space sparse operator  sum_i Z_i Z_{i+2}  (no folded minus -- enters
    H with coefficient +kappa)."""
    dim = 1 << L
    T = sp.csr_matrix((dim, dim))
    for i in range(L):
        m = kappa_masks(L, i, bc)
        if m is None:
            continue
        T = T + _pauli_sparse(L, m[0], m[1])
    return T


# ----------------------------------------------------------------------------
# Model bundle
# ----------------------------------------------------------------------------
@dataclass
class ClusterModel:
    L: int
    d: int
    bc: str
    sector: int
    kappa: float
    ps: ParitySector
    T_sector: jnp.ndarray      # (d, dim_sector, dim_sector) constant terms
    Tkappa_sector: jnp.ndarray  # (dim_sector, dim_sector) constant
    # frozen lift (sector -> full) as JAX arrays (see projector.py)
    lift_rows: jnp.ndarray
    lift_cols: jnp.ndarray
    lift_vals: jnp.ndarray


def build_cluster_model(L=12, d=3, bc='pbc', sector=+1, kappa=0.0):
    """Build the constant, frozen operators for the cluster chain.

    Returns a ClusterModel. The projector V and all T_alpha are constructed
    here, OUTSIDE any differentiated region, and treated as constants downstream.
    """
    if bc not in ('pbc', 'obc'):
        raise ValueError(f"bc must be 'pbc' or 'obc', got {bc!r}")
    ps = build_parity_sector(L, sector)
    V = ps.V

    T_list = []
    for alpha in range(d):
        Tfull = _term_full(L, alpha, bc)
        Tsec = (V.T @ Tfull @ V).toarray()        # (dim_sector, dim_sector)
        # symmetrize away tiny asymmetry from float roundoff before eigh
        Tsec = 0.5 * (Tsec + Tsec.T)
        T_list.append(np.asarray(Tsec, dtype=np.float64))
    T_sector = jnp.asarray(np.stack(T_list, axis=0))

    Tkfull = _kappa_full(L, bc)
    Tk = (V.T @ Tkfull @ V).toarray()
    Tk = 0.5 * (Tk + Tk.T)
    Tkappa_sector = jnp.asarray(np.asarray(Tk, dtype=np.float64))

    return ClusterModel(
        L=L, d=d, bc=bc, sector=sector, kappa=float(kappa), ps=ps,
        T_sector=T_sector, Tkappa_sector=Tkappa_sector,
        lift_rows=jnp.asarray(ps.lift_rows),
        lift_cols=jnp.asarray(ps.lift_cols),
        lift_vals=jnp.asarray(ps.lift_vals),
    )


# ----------------------------------------------------------------------------
# Differentiable forward pass
# ----------------------------------------------------------------------------
def H_sector(t, model: ClusterModel):
    """Assemble H_sector(t) = sum_alpha t_alpha T_alpha + kappa * Tkappa.

    Linear in t; T_alpha constant -> dH/dt_alpha = T_alpha (Hellmann-Feynman).
    """
    t = jnp.asarray(t, dtype=model.T_sector.dtype)
    H = jnp.tensordot(t, model.T_sector, axes=(0, 0))
    if model.kappa != 0.0:
        H = H + model.kappa * model.Tkappa_sector
    return H


def gd_solver_ed(t, model: ClusterModel):
    """Dense eigh in the frozen sector. Returns (e, psi0_sector) with psi0 the
    real ground-state vector IN THE SECTOR (length dim_sector)."""
    H = H_sector(t, model)
    e, v = jnp.linalg.eigh(H)
    return e, v[:, 0]


@partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def ground_state_vector(t, model: ClusterModel, eta=0.0):
    """Ground-state vector psi0 (in the sector) with a HAND-WRITTEN eigenvector
    adjoint -- the differentiable entry point for every state-dependent loss.

    Why not differentiate jnp.linalg.eigh directly: the stabilizer Hamiltonian has
    a massively degenerate EXCITED spectrum, and eigh's eigenvector jvp builds the
    full 1/(e_i - e_j) matrix over ALL pairs, so excited-excited degeneracies make
    it NaN even though only the ground vector is used. First-order perturbation
    theory needs only the GROUND-vs-excited resolvent:
        |dpsi0> = sum_{j != 0} |v_j> <v_j|dH|psi0> / (e0 - e_j),
    which is well defined whenever the GROUND state is non-degenerate (the
    in-sector gap is O(1) in cell interiors). Excited degeneracies never enter.

    `eta` is an optional resolvent (Lorentzian) shift: 1/(e0-e_j) ->
    (e0-e_j)/((e0-e_j)^2 + eta^2). eta=0 gives the exact adjoint (gradient
    diverges AT a boundary by design -- the detector signal); eta>0 keeps it
    finite at the closing. Report the eta you use.
    """
    H = H_sector(t, model)
    _, V = jnp.linalg.eigh(H)
    return V[:, 0]


@ground_state_vector.defjvp
def _ground_state_vector_jvp(model, eta, primals, tangents):
    (t,), (dt,) = primals, tangents
    H = H_sector(t, model)
    e, V = jnp.linalg.eigh(H)
    v0 = V[:, 0]
    e0 = e[0]

    # dH = sum_alpha dt_alpha T_alpha (the kappa term is t-independent -> no contribution)
    dH = jnp.tensordot(dt, model.T_sector, axes=(0, 0))
    proj = V.T @ (dH @ v0)                 # <v_j | dH | psi0>
    denom = e0 - e                         # 0 at j=0 (and any ground-degenerate j)
    reg = denom / (denom ** 2 + eta ** 2)  # 1/denom for eta=0; finite at closing
    # exclude j=0 and guard accidental exact ground-degeneracy (denom ~ 0)
    reg = jnp.where(jnp.abs(denom) < 1e-12, 0.0, reg)
    dv0 = V @ (proj * reg)
    return v0, dv0


def lift_to_full(psi_sector, model: ClusterModel):
    """Lift a sector state vector to the full 2**L space via the frozen
    isometry (differentiable scatter, no dense V)."""
    psi_full = jnp.zeros(model.ps.dim_full, dtype=psi_sector.dtype)
    contrib = model.lift_vals * psi_sector[model.lift_cols]
    return psi_full.at[model.lift_rows].add(contrib)


# ----------------------------------------------------------------------------
# rho0-routed observables (differentiable quadratic forms)
# ----------------------------------------------------------------------------
def pauli_expval_data(L, xmask, zmask):
    """Precompute the constant (src_idx, signs) for the Pauli-string expectation
    <psi|O|psi> = sum_s psi[s] * signs[s] * psi[src_idx[s]]."""
    idx = np.arange(1 << L, dtype=np.int64)
    src = idx ^ xmask
    signs = 1.0 - 2.0 * bit_parity(src & zmask)
    return jnp.asarray(src), jnp.asarray(signs, dtype=jnp.float64)


def expval_full(psi_full, src_idx, signs):
    """<psi|O|psi> for a Pauli string given its precomputed (src_idx, signs).
    psi_full real; routes through rho0 so the eigensolver sign cancels."""
    return jnp.sum(psi_full * signs * psi_full[src_idx])
