"""
Parameter sampling and feature-data generation for the cluster-chain pipeline.

Each generated row is the gauge-invariant feature vector of an ED ground state at
a sampled coupling t. Every ED solve is one SOLVER QUERY -- the workflow counts
them (queries vs phases is the headline benchmark axis, section 5).
"""
import numpy as np
import jax
import jax.numpy as jnp

from hamiltonians.cluster import gd_solver_ed, lift_to_full
from hamiltonians.observables import feature_vector


def normalize_rows(t):
    """Project each row onto the unit sphere S^{d-1} (phase diagram is a cell
    decomposition of the sphere; t and lambda*t are the same phase)."""
    n = np.linalg.norm(t, axis=-1, keepdims=True)
    return t / np.where(n > 0, n, 1.0)


def sample_params(center_t, sigma, n, key, normalize=True):
    """Sample n d-vectors ~ Normal(center_t, sigma^2 I); optionally normalize to
    the sphere. Returns (n, d) numpy array."""
    center_t = np.asarray(center_t, dtype=float)
    d = center_t.shape[0]
    noise = sigma * np.asarray(jax.random.normal(key, (n, d)))
    ts = center_t[None, :] + noise
    if normalize:
        ts = normalize_rows(ts)
    return ts


def generate_features(ts, model):
    """ED-solve each t in ts (forward only) and stack feature vectors.

    Returns (X, n_queries) where X is (len(ts), D) and n_queries = len(ts).
    """
    rows = []
    for t in ts:
        _, psi = gd_solver_ed(jnp.asarray(t, dtype=jnp.float64), model)
        rows.append(np.asarray(feature_vector(psi, model)))
    X = jnp.asarray(np.stack(rows, axis=0))
    return X, len(ts)


def _sign_fix_np(psi):
    """Gauge convention: make the largest-|amplitude| component positive so the
    same physical state maps to one vector (removes eigh's arbitrary overall sign)."""
    k = int(np.argmax(np.abs(psi)))
    s = np.sign(psi[k])
    return psi * (s if s != 0 else 1.0)


def generate_wavefunctions(ts, model):
    """ED-solve each t in the P=+1 sector (forward only), lift to the full 2^L space,
    and sign-fix -> one gauge-fixed full wavefunction per t.

    The AE-input analogue of generate_features, but the raw (sign-fixed) wavefunction
    instead of the rho0-routed observable vector. Consistent with the sign convention
    used inside the differentiable inner loss (training/optim_t.py). Returns
    (X, n_queries) with X of shape (len(ts), 2^L)."""
    rows = []
    for t in ts:
        _, psi_sec = gd_solver_ed(jnp.asarray(t, dtype=jnp.float64), model)
        psi_full = np.real(np.asarray(lift_to_full(psi_sec, model))).ravel()
        rows.append(_sign_fix_np(psi_full))
    X = jnp.asarray(np.stack(rows, axis=0))
    return X, len(ts)


def feature_dim(model):
    """Length D of the feature vector for this model (one cheap probe solve)."""
    import numpy as _np
    t0 = _np.zeros(model.d); t0[0] = 1.0
    _, psi = gd_solver_ed(jnp.asarray(t0), model)
    return int(feature_vector(psi, model).shape[0])
