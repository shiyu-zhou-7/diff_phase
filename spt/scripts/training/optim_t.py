"""
Differentiable inner loss over the coupling vector t.

The discovery loss MAXIMIZES the AE-latent distance from the current phase's
centroid (drives t out of the known phase, toward a boundary / a new phase) --
the single-centroid escape loss, the cluster-chain analogue of the TFIM
`_ae_latent_loss`. At a phase boundary the in-sector gap closes, the ground-state
derivative (custom-jvp eigenvector adjoint) diverges, and the gradient blows up /
NaNs -- which the workflow's NaN handler turns into a jump ACROSS the boundary.
That divergence is the detector signal, by design.

The whole loss is routed through rho0 (feature_vector is built from
expectation values), so the eigensolver's arbitrary sign never reaches the AE.
"""
import jax
import jax.numpy as jnp
from jax import value_and_grad, jit
import optax

from hamiltonians.cluster import ground_state_vector
from hamiltonians.observables import feature_vector
from models.autoencoder import fetch_latent


def latent_distance_loss(t, ae_params, centroid, model, eta):
    """-||z(t) - centroid||^2 : minimizing it MAXIMIZES latent distance."""
    psi = ground_state_vector(t, model, eta)          # custom-jvp eigenvector
    feats = feature_vector(psi, model)                # rho0-routed, gauge invariant
    z = fetch_latent(ae_params, feats[None, :], jax.random.PRNGKey(0))[0]
    diff = z - centroid
    return -jnp.sum(diff * diff)


def make_opt_step(opt, model, eta):
    """Returns a single Adam step over t on the latent-distance loss."""
    def opt_step(t, ae_params, centroid, opt_state):
        loss, grads = value_and_grad(latent_distance_loss)(
            t, ae_params, centroid, model, eta
        )
        updates, opt_state = opt.update(grads, opt_state, t)
        t = optax.apply_updates(t, updates)
        gnorm = jnp.linalg.norm(grads)
        return t, opt_state, loss, gnorm
    return opt_step
