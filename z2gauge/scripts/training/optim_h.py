"""
Gradient-descent over the transverse-field strength h for z2gauge.

Mirrors tfim_reorg/scripts/training/optim_h.py, with the ED ground-state
solver replaced by ITE (Z2 has degenerate ground subspace where `jnp.linalg.eigh`
produces NaN gradients — see project memory `project_z2_ite.md`).

Twin convention: the ITE init-state key selects which Z2 twin H(h) converges
to. It is passed in as `twin_key` so the caller can make twin selection
depend on the run seed (see active_phase_discovery.run_active_phase_discovery,
which derives one twin_key per run from ham_cfg.seed). The default
`_TWIN_KEY = PRNGKey(0)` preserves the old pinned-twin behavior for any
standalone caller. Within a single run twin_key is constant, so the same H(h)
always converges to the same twin and the autodiff path is unaffected.
"""
import jax
import jax.numpy as jnp

from hamiltonian.z2ham import hamiltonian
from hamiltonian.ite import ite_ground_state
from models.autoencoder import fetch_latent


_NORM_EPS = 1e-8
_TWIN_KEY = jax.random.PRNGKey(0)  # default twin; run-time twin is passed in
_ITE_STEPS = 150
_ITE_DT = 1e-2


def _ae_latent_loss(h, j_a, star_ops, trans_ops, latent_target, params, drop_p,
                    rng_key, twin_key=_TWIN_KEY):
    """Negative sqrt-MSE between encoded ITE ground state and target latent centroid.

    The encoded ground state is soft-normalized (z / sqrt(‖z‖² + eps)) so it
    lives on the unit hypersphere — matching the geometry the AE was trained
    on, and the geometry of the saved centroid. The leading -1 turns
    minimization into maximization of latent distance from the centroid.
    """
    h32 = jnp.asarray(h, dtype=jnp.float32)
    j32 = jnp.asarray(j_a, dtype=jnp.float32)
    H = hamiltonian(j32, h32, star_ops, trans_ops).astype(jnp.float32)
    v, _ = ite_ground_state(H, n_steps=_ITE_STEPS, dt=_ITE_DT, key=twin_key)
    z = fetch_latent(
        params, v,
        rng_key=rng_key,
        drop_p=drop_p,
        normalize=True, eps=_NORM_EPS,
    )
    l = (z - latent_target) ** 2
    l_sum = jnp.sum(l)
    return -1.0 * jnp.sqrt(l_sum) / l.shape[-1]
