"""
Bit/state utilities and magnetization observables for the TFIM chain.
Lifted verbatim (modulo namespacing) from the original tfim scripts.
"""
import numpy as np
import jax.numpy as jnp


def hslabeltoocc(hslabel, N):
    """Hilbert-space integer label -> length-N spin-occupation array (1=up, 0=down)."""
    return np.array(list(np.binary_repr(hslabel, N)), dtype=int)


def occtohslabel(occ, N):
    """Length-N spin-occupation array -> Hilbert-space integer label."""
    return int(''.join(map(str, occ)), 2)


def cal_m(v):
    """
    |<m>| of a single state vector v (length 2**N), per-site, in [0, 1].
    """
    m_tot = 0
    N = int(np.log2(len(v)))
    for i in range(len(v)):
        state = hslabeltoocc(i, N)
        num_up = jnp.count_nonzero(state)
        m = abs(2 * num_up - N)
        m_tot += m * (jnp.conj(v[i]) * v[i])
    return m_tot / N


def cal_m_reconstructed(x):
    """Average |<m>| over a batch of reconstructed wavefunctions x (shape [B, 2**N])."""
    m_list = [cal_m(v) for v in x]
    return m_list
