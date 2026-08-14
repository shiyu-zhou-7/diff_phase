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

from hamiltonians.cluster import ground_state_vector, lift_to_full
from hamiltonians.observables import feature_vector  # kept for the commented revert path
from models.autoencoder import fetch_latent


def latent_distance_loss(t, ae_params, centroid, model, eta):
    """-||z(t) - centroid||^2 : minimizing it MAXIMIZES latent distance.

    AE input is the sign-fixed FULL wavefunction: the P=+1 sector ground state
    (custom-jvp) lifted to the full 2^L space. The sign-fix (largest-|amplitude|
    positive) supplies the gauge convention the non-stoquastic cluster chain lacks;
    its multiplier s = +/-1 is locally constant, so autodiff sees d(psi_fixed) =
    s * d(psi) and the gradient is well-behaved (it still diverges AT a boundary,
    the detector signal). The largest amplitude is O(1) for a normalized state, so
    sign() is never 0.
    """
    psi_sec = ground_state_vector(t, model, eta)      # custom-jvp sector eigenvector
    psi_full = lift_to_full(psi_sec, model)           # differentiable lift to 2^L
    k = jnp.argmax(jnp.abs(psi_full))
    psi_fixed = psi_full * jnp.sign(psi_full[k])      # sign-fix (see docstring)
    z = fetch_latent(ae_params, psi_fixed[None, :], jax.random.PRNGKey(0))[0]
    # --- old rho0-routed observable input (gauge invariant), kept for easy revert:
    # feats = feature_vector(psi_sec, model)
    # z = fetch_latent(ae_params, feats[None, :], jax.random.PRNGKey(0))[0]
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
    # return opt_step    # eager: ~1500s/step at L=12 (op-by-op dispatch); OK at L=10
    return jit(opt_step)
