"""
JAX feed-forward autoencoder for z2gauge ground-state wavefunctions.

Encoder soft-normalizes the latent vector to (approximately) unit-L2 via
`z / sqrt(‖z‖² + eps)`, so the gradient stays bounded as ‖z‖ → 0. This is
the same convention as tfim_reorg/scripts/models/autoencoder.py.

Decoder reconstructs the wavefunction; output is L2-normalized.

All tensor reductions use `axis=-1` so the same module accepts both 1D ([D])
ground-state vectors (used by the h-optim path) and 2D ([B, D]) batches (used
during training).
"""
import jax
import jax.numpy as jnp
import optax
from jax import jit, value_and_grad


def init_params(layer_widths, key, scale=1e-2):
    """Symmetric MLP init. layer_widths is a tuple/list, e.g. (4096, 500, 10, 500, 4096)."""
    params = []
    keys = jax.random.split(key, len(layer_widths) - 1)
    for (din, dout), k in zip(zip(layer_widths[:-1], layer_widths[1:]), keys):
        wk, bk = jax.random.split(k)
        W = scale * jax.random.normal(wk, shape=(din, dout))
        b = scale * jax.random.normal(bk, shape=(dout,))
        params.append([W, b])
    return params


def dropout(x, drop_p, key):
    # drop_p can be a traced scalar; do everything with JAX ops
    keep_prob = 1.0 - drop_p
    mask = jax.random.bernoulli(key, keep_prob, shape=x.shape)
    scale = jnp.where(keep_prob > 0, 1.0 / keep_prob, 0.0)
    return x * mask * scale


def encoder(params, x, drop_p, key, *, eps=1e-12):
    """Returns soft-normalized latent `z / sqrt(‖z‖² + eps)`.

    Pass eps > 0 (e.g. 1e-8) for the h-optim path where ‖z‖ can be driven
    toward zero by Adam updates near phase boundaries — soft-normalize keeps
    the gradient bounded. The training-time default eps=1e-12 is
    indistinguishable from strict L2-normalize for non-pathological z.
    """
    h = x
    for W, b in params[:-1]:
        key, sub = jax.random.split(key)
        h = jax.nn.relu(h @ W + b)
        h = dropout(h, drop_p, sub)
    W, b = params[-1]
    z = h @ W + b
    norm2 = jnp.sum(z ** 2, axis=-1, keepdims=True)
    return z / jnp.sqrt(norm2 + eps)


def _encoder_raw(params, x, drop_p, key):
    """Encoder without the final normalization. Used by the optional un-normalized
    fetch_latent path (kept for parity with TFIM API)."""
    h = x
    for W, b in params[:-1]:
        key, sub = jax.random.split(key)
        h = jax.nn.relu(h @ W + b)
        h = dropout(h, drop_p, sub)
    W, b = params[-1]
    return h @ W + b


def decoder(params, z, drop_p, key):
    h = z
    for W, b in params[:-1]:
        key, sub = jax.random.split(key)
        h = jax.nn.relu(h @ W + b)
        h = dropout(h, drop_p, sub)
    W, b = params[-1]
    return h @ W + b


def _normalize_output(x):
    """L2-normalize along the last axis. Works for both [B, D] and [D]."""
    sq = jnp.sum(x ** 2, axis=-1, keepdims=True)
    return x / jnp.sqrt(sq + 1e-24)


def autoencoder(params, x, drop_p, key):
    """End-to-end AE: x -> latent (soft-normalized) -> reconstruction (L2-normalized)."""
    mid = len(params) // 2
    key, sub = jax.random.split(key)
    z = encoder(params[:mid], x, drop_p, sub)
    key, sub = jax.random.split(key)
    x_hat = decoder(params[mid:], z, drop_p, sub)
    return _normalize_output(x_hat)


@jit
def ae_loss(params, x, drop_p, key, center_coeff=1e-3):
    """Quantum-fidelity reconstruction loss + latent-variance penalty.

    Reconstruction (fidelity):
        1 - mean_batch ( ⟨ψ|ψ̂⟩² ),  ψ and ψ̂ unit-normalized.
    Latent-variance penalty:
        center_coeff * mean_dims ( Var_batch(z_i) ).

    Squaring the overlap removes the unphysical |ψ⟩ vs −|ψ⟩ sign indifference
    (same quantum state up to a global phase). The variance penalty keeps the
    encoded latents from spreading out as a cheap reconstruction shortcut.
    """
    x = jnp.asarray(x)
    center_coeff = jnp.asarray(center_coeff)

    eps = 1e-12
    x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)

    key, sub_ae = jax.random.split(key)
    x_hat = autoencoder(params, x, drop_p, sub_ae)
    overlap = jnp.sum(x * x_hat, axis=-1)
    recon = 1.0 - jnp.mean(overlap ** 2)

    key, sub_enc = jax.random.split(key)
    mid = len(params) // 2
    z = encoder(params[:mid], x, 0.0, sub_enc)
    var_pen = jnp.mean(jnp.var(z, axis=0))

    return recon + center_coeff * var_pen


def make_ae_step(opt):
    @jit
    def step(params, x, opt_state, drop_p, key, center_coef=1e-3):
        val, grads = value_and_grad(ae_loss)(params, x, drop_p, key, center_coef)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, val
    return step


def fetch_latent(params, x, rng_key=None, *, drop_p=0.0, normalize=True, eps=1e-8):
    """Encode-only path, no dropout by default.

    Backwards-compatible positional signature: callers that pass `key` as the
    third positional arg (the prior z2gauge convention) keep working. New
    callers should use kwargs for `normalize` / `eps`.

    `normalize=True` (default) returns the soft-normalized latent
    `z / sqrt(‖z‖² + eps)`. `normalize=False` returns the raw pre-norm latent
    (TFIM-API parity, rarely needed for z2gauge).
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    mid = len(params) // 2
    if normalize:
        return encoder(params[:mid], x, drop_p, rng_key, eps=eps)
    return _encoder_raw(params[:mid], x, drop_p, rng_key)
