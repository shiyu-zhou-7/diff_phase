import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, value_and_grad
from hamiltonian.z2ham import *
from models.autoencoder import fetch_latent

@jit
def ham_loss(ham_param, star_ops, trans_ops, ae_params, latent_target):
    h = ham_param[0]  # Extract scalar from array
    H = hamiltonian(-1.0, h, star_ops, trans_ops)
    eigenvalues, eigenvectors = jnp.linalg.eigh(H)
    ground_state = eigenvectors[:, 0]
    ground_state = jnp.real(ground_state)
    # Add batch dimension: (D,) -> (1, D)
    ground_state = ground_state[None, :]
    z = fetch_latent(ae_params, ground_state, jax.random.PRNGKey(0))
    diff = z[0] - latent_target
    return -jnp.sqrt(jnp.sum(diff**2)) / z.shape[-1]

@jit
def ham_update(ham_param, star_ops, trans_ops, ae_params, latent_target, opt_state, opt):
    val, grads = value_and_grad(ham_loss)(ham_param, star_ops, trans_ops, ae_params, latent_target)
    updates, opt_state = opt.update(grads, opt_state, ham_param)
    ham_param = optax.apply_updates(ham_param, updates)
    grad_norm = jnp.linalg.norm(jnp.concatenate([grads.reshape(-1)]))
    return ham_param, opt_state, val, grad_norm


def make_ham_step(opt):
    @jit
    def step(ham_param, star_ops, trans_ops, ae_params, latent_target, opt_state):
        val, grads = value_and_grad(ham_loss)(ham_param, star_ops, trans_ops, ae_params, latent_target)
        updates, opt_state = opt.update(grads, opt_state, ham_param)
        ham_param = optax.apply_updates(ham_param, updates)
        grad_norm = jnp.linalg.norm(jnp.ravel(grads))
        return ham_param, opt_state, val, grad_norm
    return step