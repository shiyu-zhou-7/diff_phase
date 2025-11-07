# training/dmrg_optimize.py
from venv import logger
import jax
import jax.numpy as jnp
from typing import Tuple

import optax

from dmrg.dmrg import run_dmrg 
from models.autoencoder import fetch_latent

from dmrg.hamiltonians import XXZhX

def _latent_loss(delta_h: jnp.ndarray,            # shape (2,) = (Δ, h)
                 ae_params,
                 ferro_centroid: jnp.ndarray,
                 observables_list,  # list of (site, op) tuples
                 dmrg_cfg,
                 L: int, 
                 ) -> jnp.ndarray:
    """
    Differentiable loss: run DMRG → get ψ → encode with AE → ||z - z_ferro||^2
    """
    delta, h = delta_h[0], delta_h[1]

    # run_dmrg must be JAX-differentiable w.r.t. (delta, h)
    model = XXZhX(L, delta, h)
    psi = run_dmrg(L, model, dmrg_cfg)  # psi: (2**N,) real/complex
    # psi = jax.lax.stop_gradient(psi)  # stop gradient through DMRG

    # Compute observables, force real
    obs_list = psi.get_site_exp_val(observables_list)
    obs = jnp.stack(
            [jnp.real(jnp.asarray(x)).reshape(()) for x in obs_list],
            axis=0
            ) # shape (len(observables_list),)

    # default_device context will handle placement

    # AE expects a batch
    z = fetch_latent(ae_params, obs[None, :], jax.random.PRNGKey(0))[0]
    diff = z - ferro_centroid
    return jnp.sum(diff * diff)


def make_opt_step(opt, L, dmrg_cfg, observables_list):
    """
    Returns a single (Δ,h) optimizer step using JAX value_and_grad on the DMRG+AE loss.
    """
    # We jit the step; dmrg_cfg / dmrg_state are passed through but treated as pytrees.

    sweeps    = int(dmrg_cfg.sweeps)
    max_bond  = int(dmrg_cfg.max_bond)

    # @jax.jit
    def opt_step(delta_h: jnp.ndarray,
                  ae_params,
                  ferro_centroid: jnp.ndarray,
                  opt_state,
                  ):
        loss, grads = jax.value_and_grad(_latent_loss)(
            delta_h, ae_params, ferro_centroid, observables_list, dmrg_cfg, L
        )
        updates, opt_state = opt.update(grads, opt_state, delta_h)
        delta_h = optax.apply_updates(delta_h, updates)
        gnorm = jnp.linalg.norm(grads)
        return delta_h, opt_state, loss, gnorm
    return opt_step
