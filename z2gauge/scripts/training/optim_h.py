"""
Gradient-descent over the transverse-field strength h for z2gauge.

Mirrors tfim_reorg/scripts/training/optim_h.py, with the ED ground-state
solver replaced by ITE (Z2 has degenerate ground subspace where `jnp.linalg.eigh`
produces NaN gradients — see project memory `project_z2_ite.md`).

Twin convention: the ITE init-state key is pinned to PRNGKey(0) so the same
H(h) always converges to the same Z2 twin. Both this module and bootstrap.py
share `_TWIN_KEY = PRNGKey(0)`, so training latents and optim-time latents
live on a single, consistent twin branch.
"""
import jax
import jax.numpy as jnp

from hamiltonian.z2ham import hamiltonian
from hamiltonian.ite import ite_ground_state
from models.autoencoder import fetch_latent


_NORM_EPS = 1e-8
_TWIN_KEY = jax.random.PRNGKey(0)
_ITE_STEPS = 150
_ITE_DT = 1e-2


def _ae_latent_loss(h, j_a, star_ops, trans_ops, latent_target, params, drop_p, rng_key):
    """Negative sqrt-MSE between encoded ITE ground state and target latent centroid.

    The encoded ground state is soft-normalized (z / sqrt(‖z‖² + eps)) so it
    lives on the unit hypersphere — matching the geometry the AE was trained
    on, and the geometry of the saved centroid. The leading -1 turns
    minimization into maximization of latent distance from the centroid.
    """
    h32 = jnp.asarray(h, dtype=jnp.float32)
    j32 = jnp.asarray(j_a, dtype=jnp.float32)
    H = hamiltonian(j32, h32, star_ops, trans_ops).astype(jnp.float32)
    v, _ = ite_ground_state(H, n_steps=_ITE_STEPS, dt=_ITE_DT, key=_TWIN_KEY)
    z = fetch_latent(
        params, v,
        rng_key=rng_key,
        drop_p=drop_p,
        normalize=True, eps=_NORM_EPS,
    )
    l = (z - latent_target) ** 2
    l_sum = jnp.sum(l)
    return -1.0 * jnp.sqrt(l_sum) / l.shape[-1]
