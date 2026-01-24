import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, value_and_grad
from hamiltonian.z2ham import *
from models.autoencoder import fetch_latent
from hamiltonian.ite import ite_ground_state_from_params, ite_ground_state

# @jit
# def ham_loss(ham_param, star_ops, trans_ops, ae_params, latent_target):
#     h = ham_param[0]  # Extract scalar from array
#     H = hamiltonian(-1.0, h, star_ops, trans_ops)
#     # Add small regularization to break degeneracies and stabilize gradients
#     # This helps prevent NaN gradients when eigenvalues are too close
#     # Use a very small diagonal perturbation based on index to make eigenvalues distinct
#     # The perturbation needs to be large enough to create gaps > numerical precision (~1e-15)
#     # but small enough to not affect physics. Using 1e-8 to ensure stable gradients.
#     reg_strength = 1e-8
#     diag_indices = jnp.arange(H.shape[0], dtype=H.dtype)
#     H_reg = H + reg_strength * jnp.diag(diag_indices)
#     eigenvalues, eigenvectors = jnp.linalg.eigh(H_reg)
#     ground_state = eigenvectors[:, 0]
#     ground_state = jnp.real(ground_state)
#     # Add batch dimension: (D,) -> (1, D)
#     ground_state = ground_state[None, :]
#     z = fetch_latent(ae_params, ground_state, jax.random.PRNGKey(0))
#     diff = z[0] - latent_target
#     loss = -jnp.sqrt(jnp.sum(diff**2) + 1e-12) / z.shape[-1]  # Add small epsilon for numerical stability
#     return loss

# @jit
def ham_loss(ham_param, star_ops, trans_ops, ae_params, latent_target):
    ## ite version
    h = ham_param[0]
    H = hamiltonian(-1.0, h, star_ops, trans_ops)

    # ITE instead of eigh
    v, _ = ite_ground_state(H, n_steps=150, dt=1e-2, key=jax.random.PRNGKey(0))
    v = v[None, :]  # add batch dim

    z = fetch_latent(ae_params, v, jax.random.PRNGKey(0))
    diff = z[0] - latent_target
    loss = -jnp.sqrt(jnp.sum(diff**2) + 1e-12) / z.shape[-1]
    return loss

# @jit
def ham_update(ham_param, star_ops, trans_ops, ae_params, latent_target, opt_state, opt):
    val, grads = value_and_grad(ham_loss)(ham_param, star_ops, trans_ops, ae_params, latent_target)
    updates, opt_state = opt.update(grads, opt_state, ham_param)
    ham_param = optax.apply_updates(ham_param, updates)
    grad_norm = jnp.linalg.norm(jnp.concatenate([grads.reshape(-1)]))
    return ham_param, opt_state, val, grad_norm


def make_ham_step(opt):
    # @jit
    def step(ham_param, star_ops, trans_ops, ae_params, latent_target, opt_state):
        val, grads = value_and_grad(ham_loss)(ham_param, star_ops, trans_ops, ae_params, latent_target)
        grad_is_finite = jnp.all(jnp.isfinite(grads))
        # print(f"grad_is_finite: {grad_is_finite}")
        
        # Clip gradients to prevent explosion (max gradient norm of 10.0)
        # If gradient is NaN/Inf, replace with zeros to skip update
        grads_safe = jax.tree_util.tree_map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads)
        grad_norm = jnp.linalg.norm(jnp.ravel(grads_safe))
        max_grad_norm = 10.0
        grads_clipped = jax.tree_util.tree_map(lambda g: g * jnp.minimum(1.0, max_grad_norm / (grad_norm + 1e-12)), grads_safe)
        updates, opt_state = opt.update(grads_clipped, opt_state, ham_param)
        ham_param = optax.apply_updates(ham_param, updates)
        # Clip parameter to reasonable range
        ham_param = jnp.clip(ham_param, -5.0, 5.0)
        # Return NaN grad_norm if original gradient was invalid
        grad_norm = jnp.where(grad_is_finite, jnp.linalg.norm(jnp.ravel(grads)), jnp.nan)
        return ham_param, opt_state, val, grad_norm
    return step
