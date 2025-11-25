import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, value_and_grad
from physics.hamiltonian import H_xxzh, eigh_ground_state
from models.autoencoder import fetch_latent

@jit
def ham_loss(ham_param, ham_ops, ae_params, latent_target):
    delta, h = ham_param
    ham_xx, ham_yy, ham_zz, ham_x = ham_ops
    H = H_xxzh(delta, h, ham_xx, ham_yy, ham_zz, ham_x)
    _, v0 = eigh_ground_state(H)
    x = v0.real[None, :]
    z = fetch_latent(ae_params, x, jax.random.PRNGKey(0))
    diff = z[0] - latent_target
    return -jnp.sqrt(jnp.sum(diff**2)) / z.shape[-1]

@jit
def ham_update(ham_param, ham_ops, ae_params, latent_target, opt_state, opt):
    val, grads = value_and_grad(ham_loss)(ham_param, ham_ops, ae_params, latent_target)
    updates, opt_state = opt.update(grads, opt_state, ham_param)
    ham_param = optax.apply_updates(ham_param, updates)
    grad_norm = jnp.linalg.norm(jnp.concatenate([grads.reshape(-1)]))
    return ham_param, opt_state, val, grad_norm


def make_ham_step(opt):
    @jit
    def step(ham_param, ham_ops, ae_params, latent_target, opt_state):
        val, grads = value_and_grad(ham_loss)(ham_param, ham_ops, ae_params, latent_target)
        updates, opt_state = opt.update(grads, opt_state, ham_param)
        ham_param = optax.apply_updates(ham_param, updates)
        grad_norm = jnp.linalg.norm(jnp.ravel(grads))
        return ham_param, opt_state, val, grad_norm
    return step