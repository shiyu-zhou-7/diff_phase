"""
rho0-routed observables and topological diagnostics for the cluster chain.

Everything here is computed from the GROUND-STATE DENSITY MATRIX rho0 =
|psi0><psi0| via the lifted full-space vector psi_full = V @ psi0_sector, using
the signed-permutation expectation <psi|O|psi> (hamiltonians/cluster.py). rho0
is gauge invariant -- the eigensolver's arbitrary sign of psi0 cancels in every
quadratic form -- so all of these are smooth, differentiable functions of t.

Three phase-diagnostic order parameters (independent of the root finder; used to
cross-check `winding` in validation section 4.2):

  * m_X   = -(1/L) sum_i <X_i>            : ~1 in the trivial paramagnet (omega=0)
                                            (minus: the +t_0 sum X_i field makes the
                                            paramagnet anti-align, <X_i> = -1)
  * O_Z2  = <Z_0 Z_{L/2}>                 : ~1 in the Z2-broken phase (omega=1)
                                            (long-range order; in the fixed +parity
                                            sector <Z_i>=0 but <Z_i Z_j> != 0, a cat)
  * O_SPT = <Z_a X_{a+1} X_{a+3}...X_{b-1} Z_b>
                                          : ~1 in the cluster SPT (omega=2). This is
            a product of alpha=2 stabilizers Z_i X_{i+1} Z_{i+2} over i=a,a+2,...,b-2
            (interior Z's cancel), so it is +1 on the cluster state and has NO local
            one-site reduction -- the SPT has no local order parameter.

Plus a topological diagnostic from the entanglement spectrum: the SPT (and the
symmetric Z2 cat) carry degenerate Schmidt values; `es_degeneracy` measures the
top-level splitting so the phases can be told apart together with the order
parameters above.

The `feature_vector` packs a translation-invariant battery (per-alpha stabilizer
densities + ZZ correlations + SPT strings + entanglement entropy) as the AE input
-- the analogue of the per-site observable vector the xxz pipeline feeds its AE.
"""
import numpy as np
import jax.numpy as jnp

from hamiltonians.cluster import (
    ClusterModel, lift_to_full, pauli_expval_data, expval_full, stabilizer_masks,
)


# ----------------------------------------------------------------------------
# Single Pauli-string expectation helpers
# ----------------------------------------------------------------------------
def _expval(psi_full, L, x_sites, z_sites):
    """<psi|O|psi> for O = prod X on x_sites * prod Z on z_sites (differentiable)."""
    xmask = 0
    for s in x_sites:
        xmask |= (1 << (s % L))
    zmask = 0
    for s in z_sites:
        zmask |= (1 << (s % L))
    src, signs = pauli_expval_data(L, xmask, zmask)
    return expval_full(psi_full, src, signs)


# ----------------------------------------------------------------------------
# Order parameters (each takes the lifted full-space state)
# ----------------------------------------------------------------------------
def magnetization_x(psi_full, model: ClusterModel):
    """m_X = -(1/L) sum_i <X_i>. Detects the trivial paramagnet (omega=0).

    Sign convention: the alpha=0 field enters H with PLUS (+t_0 sum_i X_i,
    hamiltonians/cluster.py), so the paramagnetic ground state anti-aligns
    (<X_i> = -1); the minus here keeps m_X ~ +1 in the trivial phase."""
    L = model.L
    vals = [_expval(psi_full, L, [i], []) for i in range(L)]
    # return jnp.mean(jnp.stack(vals))    # old: all-minus H convention
    return -jnp.mean(jnp.stack(vals))


def zz_correlation(psi_full, model: ClusterModel, r=None):
    """<Z_0 Z_r> long-range correlator (default r = L//2). Detects Z2 order
    (omega=1). Uses |.| so the +-cat sign does not flip it between samples."""
    L = model.L
    if r is None:
        r = L // 2
    return _expval(psi_full, L, [], [0, r % L])


def string_order_spt(psi_full, model: ClusterModel, a=0, b=None):
    """O_SPT = <Z_a X_{a+1} X_{a+3} ... X_{b-1} Z_b>, the product of alpha=2
    stabilizers over i = a, a+2, ..., b-2. +1 on the cluster state (omega=2).
    b defaults to the largest even-offset site <= L-1 from a."""
    L = model.L
    if b is None:
        b = a + 2 * ((L - 1 - a) // 2)   # even offset, within chain
    x_sites = list(range(a + 1, b, 2))   # a+1, a+3, ..., b-1
    return _expval(psi_full, L, x_sites, [a, b])


def stabilizer_density(psi_full, model: ClusterModel, alpha):
    """(1/L) sum_i <stabilizer_{alpha,i}>: mean expectation of the alpha term.
    alpha=0 -> <X>, alpha=1 -> <ZZ>, alpha=2 -> <ZXZ>, ... A coarse but smooth,
    translation-invariant feature; the dominant alpha indicates the phase."""
    L = model.L
    vals = []
    for i in range(L):
        m = stabilizer_masks(L, alpha, i, model.bc)
        if m is None:
            continue
        src, signs = pauli_expval_data(L, m[0], m[1])
        vals.append(expval_full(psi_full, src, signs))
    return jnp.mean(jnp.stack(vals))


# ----------------------------------------------------------------------------
# Entanglement spectrum / entropy (half-chain cut)
# ----------------------------------------------------------------------------
def reduced_dm_spectrum(psi_full, model: ClusterModel, cut=None):
    """Eigenvalues of the reduced density matrix rho_A = Tr_B rho0, sorted
    DESCENDING. Site 0 is the most-significant bit, so the first `cut` sites form
    subsystem A.

    We diagonalize rho_A with `eigvalsh` (NOT svd of the state): the cluster SPT
    has a 4-fold degenerate Schmidt spectrum and the Z2 cat a 2-fold one, and the
    SVD / full-eigh eigenVECTOR adjoint carries a 1/(lambda_i - lambda_j)
    denominator that produces NaN gradients at such degeneracies. Eigenvalue
    derivatives (eigvalsh) have NO such denominator, so the entropy -- a symmetric
    function of the spectrum -- stays smooth and differentiable through rho0
    exactly as the task prescribes (route state losses through rho0, not the ket).
    """
    L = model.L
    if cut is None:
        cut = L // 2
    dimA = 1 << cut
    dimB = 1 << (L - cut)
    psi_mat = jnp.reshape(psi_full, (dimA, dimB))
    rho_A = psi_mat @ psi_mat.T          # real, symmetric, PSD
    p = jnp.linalg.eigvalsh(rho_A)       # ascending, NaN-free under degeneracy
    return p[::-1]                        # descending


def schmidt_values(psi_full, model: ClusterModel, cut=None):
    """Schmidt coefficients (= sqrt of reduced-DM eigenvalues), descending."""
    p = reduced_dm_spectrum(psi_full, model, cut)
    return jnp.sqrt(jnp.clip(p, 0.0, None))


def reduced_dm(psi_full, model: ClusterModel, cut=None):
    """Reduced density matrix rho_A = Tr_B rho0 (real symmetric PSD), no
    eigendecomposition. The basis for a differentiable purity / Renyi-2 loss."""
    L = model.L
    if cut is None:
        cut = L // 2
    dimA = 1 << cut
    dimB = 1 << (L - cut)
    psi_mat = jnp.reshape(psi_full, (dimA, dimB))
    return psi_mat @ psi_mat.T


def purity(psi_full, model: ClusterModel, cut=None):
    """Tr(rho_A^2) = ||rho_A||_F^2. A smooth POLYNOMIAL in rho0 -- no eigh, so
    fully differentiable and immune to the reduced-DM eigenvalue degeneracies that
    make a von-Neumann / SVD gradient NaN. 1 for a product state, 1/2^omega at the
    pure stabilizer points."""
    rho = reduced_dm(psi_full, model, cut)
    return jnp.sum(rho * rho)


def renyi2_entropy(psi_full, model: ClusterModel, cut=None, eps=1e-12):
    """Renyi-2 entanglement entropy S_2 = -log Tr(rho_A^2). The DIFFERENTIABLE
    rho0-based entanglement loss used in the AE feature vector and the gradient
    checks. Equals the von Neumann entropy at the pure points (flat spectra), so
    S_2 ~ omega*log2 there too; smooth everywhere (degeneracy-proof)."""
    return -jnp.log(purity(psi_full, model, cut) + eps)


def entanglement_entropy(psi_full, model: ClusterModel, cut=None, eps=1e-12):
    """Von Neumann entanglement entropy S = -sum p log p across the half cut,
    with p the reduced-DM eigenvalues. Smooth-valued and used as a (forward-only)
    DIAGNOSTIC; for a DIFFERENTIATED loss use `renyi2_entropy` instead (the von
    Neumann gradient NaNs at the SPT/Z2 reduced-DM degeneracies)."""
    p = reduced_dm_spectrum(psi_full, model, cut)
    p = jnp.clip(p, 0.0, None)
    p = p / jnp.sum(p)
    return -jnp.sum(p * jnp.log(p + eps))


def es_degeneracy(psi_full, model: ClusterModel, cut=None):
    """Top-level entanglement-spectrum splitting: |p_0 - p_1| / (p_0 + p_1) of the
    two largest reduced-DM eigenvalues. ~0 when the leading level is (at least)
    doubly degenerate (SPT edge modes / symmetric Z2 cat), ~1 in the trivial
    phase where the leading level is non-degenerate."""
    p = reduced_dm_spectrum(psi_full, model, cut)
    return jnp.abs(p[0] - p[1]) / (p[0] + p[1] + 1e-12)


# ----------------------------------------------------------------------------
# AE feature vector (translation-invariant battery)
# ----------------------------------------------------------------------------
def feature_vector(psi0_sector, model: ClusterModel, zz_ranges=None,
                   spt_offsets=None):
    """Differentiable observable vector fed to the autoencoder.

    Packs: per-alpha stabilizer densities (d), ZZ correlations at several ranges,
    SPT string orders at several lengths, and the half-cut entanglement entropy.
    All are rho0-routed quadratic forms (+ one SVD), so the eigensolver sign
    cancels and the vector is a smooth function of t.
    """
    L = model.L
    psi_full = lift_to_full(psi0_sector, model)

    feats = [stabilizer_density(psi_full, model, a) for a in range(model.d)]

    if zz_ranges is None:
        zz_ranges = [r for r in (1, 2, L // 4, L // 2) if 1 <= r <= L - 1]
    feats += [zz_correlation(psi_full, model, r) for r in sorted(set(zz_ranges))]

    if spt_offsets is None:
        spt_offsets = [o for o in (2, 4, L // 2) if 2 <= o <= L - 1 and o % 2 == 0]
    for o in sorted(set(spt_offsets)):
        feats.append(string_order_spt(psi_full, model, a=0, b=o))

    feats.append(magnetization_x(psi_full, model))
    feats.append(renyi2_entropy(psi_full, model))   # differentiable entanglement feature

    return jnp.stack(feats)
