"""
Gradient-descent over the transverse-field strength h.

Two flavors:
  - `optimize_h_ed` : loss is a magnetization observable on the ED ground state.
                      Used by `main_optim_h_ed.py` (no AE involved).
  - `optimize_h_ae` : loss is the squared distance in the AE latent space
                      between the encoded ground state and a target centroid.
                      Used by `main_optim_h_ae.py`.
"""
import jax
import jax.numpy as jnp
from jax import value_and_grad, jit
import optax

from hamiltonians.tfim import gd_solver_ed
from models.autoencoder import fetch_latent
from utils.bits import cal_m


def _ed_mag_loss(h, ham_X, ham_ZZ, target_phase):
    """target_phase: 'para' -> minimize m**2 (drive away from ssb).
                     'ssb'  -> minimize (1-m)**2 (drive toward ssb).
    """
    _, v = gd_solver_ed(h, ham_X, ham_ZZ)
    m = cal_m(v[:, 0])
    if target_phase == 'para':
        return m ** 2
    elif target_phase == 'ssb':
        return (1.0 - m) ** 2
    raise ValueError(f"target_phase must be 'para' or 'ssb', got {target_phase!r}")


def optimize_h_ed(h_init, ham_X, ham_ZZ, *, epochs=500, lr=0.3,
                  target_phase='para', log_every=25):
    """
    Mirrors the body of the original `tfim_autoDiff.py`: jit the forward loss,
    take `value_and_grad` per iteration (no outer jit — the magnetization
    observable unrolls 2**N basis states at trace time, so a global jit blows
    up XLA compile memory).
    """
    h = jnp.asarray(h_init, dtype=jnp.float64)

    @jit
    def loss_fn(h):
        return _ed_mag_loss(h, ham_X, ham_ZZ, target_phase)

    h_list = []
    loss_list = []
    for i in range(epochs):
        value, gradient = value_and_grad(loss_fn, argnums=0)(h)
        h = h - lr * gradient
        loss_list.append(float(value))
        h_list.append(float(h))
        if i % log_every == 0:
            print(f'epoch={i} loss={float(value):.6e} grad={float(gradient):+.6e} h={float(h):+.6f}')
    return h_list, loss_list


_NORM_EPS = 1e-8


def _ae_latent_loss(h, ham_X, ham_ZZ, latent_target, params, drop_p, rng_key):
    """
    Negative sqrt-MSE between encoded ground state and target latent centroid.
    The encoded ground state is soft-normalized (z / sqrt(‖z‖² + eps)) so it
    lives on the unit hypersphere — matching the geometry the AE was trained
    on, and the geometry of the saved centroid. The leading -1 turns
    minimization into maximization of latent distance from the centroid.
    """
    _, v = gd_solver_ed(h, ham_X, ham_ZZ)
    z = fetch_latent(params, v[:, 0], drop_p, rng_key,
                     normalize=True, eps=_NORM_EPS)
    l = (z - latent_target) ** 2
    l_sum = jnp.sum(l)
    return -1.0 * jnp.sqrt(l_sum) / l.shape[-1]


def optimize_h_ae(h_init, ham_X, ham_ZZ, *, latent_target, params,
                  epochs=1000, lr=0.1, drop_p=0.1, log_every=25, seed=0):
    """
    Adam over h with the AE-latent loss. Mirrors `tfim_autoDiff_autoEncoder.py`.
    """
    h = jnp.asarray(h_init, dtype=jnp.float64)
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(h)
    key = jax.random.PRNGKey(seed)

    @jit
    def update(h, opt_state, rng_key):
        value, grads = value_and_grad(_ae_latent_loss, argnums=0)(
            h, ham_X, ham_ZZ, latent_target, params, drop_p, rng_key
        )
        updates, opt_state = opt.update(grads, opt_state)
        h = optax.apply_updates(h, updates)
        return h, opt_state, value

    h_list = []
    loss_list = []
    for i in range(epochs):
        key, sub = jax.random.split(key)
        h, opt_state, value = update(h, opt_state, sub)
        loss_list.append(float(value))
        h_list.append(float(h))
        if i % log_every == 0:
            print(f'epoch={i} loss={float(value):+.6e} h={float(h):+.6f}')
    return h_list, loss_list
