import jax
import jax.numpy as jnp
import optax
from jax import jit, value_and_grad, device_put


def init_params(layer_widths, key, scale=1e-2):
    params = []
    keys = jax.random.split(key, len(layer_widths) - 1)
    for (din, dout), k in zip(zip(layer_widths[:-1], layer_widths[1:]), keys):
        wk, bk = jax.random.split(k)
        W = scale * jax.random.normal(wk, shape=(din, dout))
        b = scale * jax.random.normal(bk, shape=(dout,))
        # Ensure GPU placement
        W = device_put(W)
        b = device_put(b)
        params.append([W, b])
    return params


# def dropout(x, drop_p, key):
#     if drop_p <= 0.0:
#         return x
#     mask = jax.random.bernoulli(key, 1.0 - drop_p, shape=x.shape)
#     return x * mask


def dropout(x, drop_p, key):
    # drop_p can be a traced scalar; do everything with JAX ops
    keep_prob = 1.0 - drop_p
    # sample mask (same shape as x)
    mask = jax.random.bernoulli(key, keep_prob, shape=x.shape)
    # (optional) inverted dropout scaling so expectation is preserved
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
    x_hat = h @ W + b
    return x_hat


@jit
def autoencoder(params, x, drop_p, key):
    mid = len(params) // 2
    z = encoder(params[:mid], x, drop_p, key)
    x_hat = decoder(params[mid:], z, drop_p, key)
    return x_hat


# Fidelity-like AE loss with tiny latent variance penalty
# @jit
# def ae_loss(params, x, drop_p, key, center_coeff=1e-3):
#     x = jnp.asarray(x)
#     center_coeff = jnp.asarray(center_coeff)
#     x = x / (jnp.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
#     x_hat = autoencoder(params, x, drop_p, key)
#     x_hat = x_hat / (jnp.linalg.norm(x_hat, axis=1, keepdims=True) + 1e-12)
#     overlap = jnp.sum(x * x_hat, axis=1)
#     recon = 1.0 - jnp.mean(overlap**2)
#     mid = len(params) // 2
#     z = encoder(params[:mid], x, 0.0, key)
#     var_pen = jnp.mean(jnp.var(z, axis=0))
#     return recon + center_coeff * var_pen

@jit
def ae_loss(params, x, drop_p, key, center_coeff=1e-3, normalize=False):
    """MSE reconstruction loss + small latent variance penalty."""
    eps = 1e-12
    x = jnp.asarray(x)

    # (Optional) enforce unit-norm wavefunctions so MSE isn't driven by scale
    if normalize:
        x = x / (jnp.linalg.norm(x, axis=1, keepdims=True) + eps)

    x_hat = autoencoder(params, x, drop_p, key)
    if normalize:
        x_hat = x_hat / (jnp.linalg.norm(x_hat, axis=1, keepdims=True) + eps)

    # ----- MSE choices -----
    # 1) Per-element MSE (recommended: scale ~ O(1))
    recon = jnp.mean((x_hat - x) ** 2)

    # 2) If you prefer per-sample L2^2 averaged across batch, uncomment:
    # recon = jnp.mean(jnp.sum((x_hat - x) ** 2, axis=1))

    # Latent variance penalty (keeps latent from collapsing/exploding slightly)
    # mid = len(params) // 2
    # z = encoder(params[:mid], x, 0.0, key)         # no dropout in eval
    # var_pen = jnp.mean(jnp.var(z, axis=0))

    var_pen = 0

    return recon + center_coeff * var_pen

@jit
def ae_update(params, x, opt_state, opt, drop_p, key, center_coeff=1e-3):
    val, grads = value_and_grad(ae_loss)(params, x, drop_p, key, center_coeff)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, val


def make_ae_step(opt):
    @jit
    def step(params, x, opt_state, drop_p, key, center_coef=1e-3):
        val, grads = value_and_grad(ae_loss)(params, x, drop_p, key, center_coef)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, val
    return step


def fetch_latent(params, x, key):
    x = x / (jnp.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    mid = len(params) // 2
    return encoder(params[:mid], x, 0.0, key)


def cal_ave_m(xz_val, L):
    # first half is Z, second half is X
    if xz_val.ndim == 1:
        return jnp.mean(xz_val[:L])
    else:
        return jnp.mean(xz_val[:, :L], axis=1)