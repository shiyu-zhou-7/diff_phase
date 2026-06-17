"""
JAX feed-forward autoencoder over the cluster-chain feature vector.

Adapted from xxz_dmrg/scripts/models/autoencoder.py: the encoder L2-normalizes the
latent onto a hypersphere, the loss is MSE reconstruction with a small latent-
variance penalty. The AE input is the gauge-invariant observable feature vector
(hamiltonians/observables.feature_vector), NOT a bare wavefunction -- so the
eigensolver's arbitrary sign never reaches the network.
"""
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import optax


def init_params(layer_widths, key, scale=1e-2):
    params = []
    keys = jax.random.split(key, len(layer_widths) - 1)
    for (din, dout), k in zip(zip(layer_widths[:-1], layer_widths[1:]), keys):
        wk, bk = jax.random.split(k)
        W = scale * jax.random.normal(wk, shape=(din, dout))
        b = scale * jax.random.normal(bk, shape=(dout,))
        params.append([W, b])
    return params


def dropout(x, drop_p, key):
    keep_prob = 1.0 - drop_p
    mask = jax.random.bernoulli(key, keep_prob, shape=x.shape)
    scale = jnp.where(keep_prob > 0, 1.0 / keep_prob, 0.0)
    return x * mask * scale


@jit
def encoder(params, x, drop_p, key):
    h = x
    for W, b in params[:-1]:
        key, sub = jax.random.split(key)
        h = jax.nn.relu(h @ W + b)
        h = dropout(h, drop_p, sub)
    W, b = params[-1]
    z = h @ W + b
    z = z / (jnp.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    return z


@jit
def decoder(params, z, drop_p, key):
    h = z
    for W, b in params[:-1]:
        key, sub = jax.random.split(key)
        h = jax.nn.relu(h @ W + b)
        h = dropout(h, drop_p, sub)
    W, b = params[-1]
    return h @ W + b


@jit
def autoencoder(params, x, drop_p, key):
    mid = len(params) // 2
    z = encoder(params[:mid], x, drop_p, key)
    return decoder(params[mid:], z, drop_p, key)


@jit
def ae_loss(params, x, drop_p, key, center_coeff=1e-3):
    """MSE reconstruction + small latent-variance penalty."""
    x = jnp.asarray(x)
    x_hat = autoencoder(params, x, drop_p, key)
    recon = jnp.mean((x_hat - x) ** 2)
    mid = len(params) // 2
    z = encoder(params[:mid], x, 0.0, key)
    var_pen = jnp.mean(jnp.var(z, axis=0))
    return recon + center_coeff * var_pen


def make_ae_step(opt):
    @jit
    def step(params, x, opt_state, drop_p, key, center_coeff=1e-3):
        val, grads = value_and_grad(ae_loss)(params, x, drop_p, key, center_coeff)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, val
    return step


def fetch_latent(params, x, key):
    """Encode-only path (no dropout). x is a batch (B, D)."""
    mid = len(params) // 2
    return encoder(params[:mid], x, 0.0, key)
