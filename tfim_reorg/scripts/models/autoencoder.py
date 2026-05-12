"""
JAX feed-forward autoencoder for TFIM ground-state wavefunctions.

Encoder normalizes the latent vector to unit-L2 (so latent space lives on a
hypersphere). Decoder reconstructs the wavefunction; output is also L2-normalized.
"""
import jax
import jax.numpy as jnp
from jax import jit


def init_params(layer_widths, parent_key, scale=0.01):
    """Symmetric MLP init. layer_widths is a tuple/list, e.g. (1024, 500, 20, 500, 1024)."""
    params = []
    keys = jax.random.split(parent_key, num=(len(layer_widths) - 1))
    for in_w, out_w, key in zip(layer_widths[:-1], layer_widths[1:], keys):
        wkey, bkey = jax.random.split(key)
        params.append([
            scale * jax.random.normal(wkey, shape=(in_w, out_w)),
            scale * jax.random.normal(bkey, shape=(out_w,)),
        ])
    return params


def dropout(x, drop_p, rng_key):
    keep_prob = 1.0 - drop_p
    mask = jax.random.bernoulli(rng_key, keep_prob, shape=x.shape)
    return x * mask


def encoder(params, x, drop_p, rng_key, eps=0.0):
    """Returns L2-normalized latent. With `eps > 0`, uses soft-normalize
    `z / sqrt(‖z‖² + eps)` so the gradient stays bounded at `‖z‖ → 0`."""
    activation = x
    for w, b in params[0:-1]:
        activation = jax.nn.relu(jnp.dot(activation, w) + b)
        rng_key, subkey = jax.random.split(rng_key)
        activation = dropout(activation, drop_p, subkey)
    w, b = params[-1]
    z = jnp.dot(activation, w) + b
    norm2 = jnp.sum(z ** 2, axis=-1, keepdims=True)
    return z / jnp.sqrt(norm2 + eps)


def decoder(params, z, drop_p, rng_key):
    activation = z
    for w, b in params[0:-1]:
        activation = jax.nn.relu(jnp.dot(activation, w) + b)
        rng_key, subkey = jax.random.split(rng_key)
        activation = dropout(activation, drop_p, subkey)
    return jnp.dot(activation, params[-1][0]) + params[-1][1]


def _normalize_output(x):
    """L2-normalize along the last axis. Works for shape [B, D] or [D]."""
    sq = x ** 2
    if x.ndim == 1:
        denom = jnp.sqrt(jnp.sum(sq))
        return x / denom
    denom = jnp.sqrt(jnp.sum(sq, axis=-1, keepdims=True))
    return x / denom


def autoencoder(params, x, drop_p, rng_key):
    """End-to-end AE: x -> latent (normalized) -> reconstruction (normalized)."""
    mid = len(params) // 2
    rng_key, sub = jax.random.split(rng_key)
    z = encoder(params[0:mid], x, drop_p, sub)
    rng_key, sub = jax.random.split(rng_key)
    x_out = decoder(params[mid:], z, drop_p, sub)
    return _normalize_output(x_out)


def _encoder_raw(params, x, drop_p, rng_key):
    """Encoder without the L2 normalization on the latent. Used by the
    h-optimization path to match the un-normalized fetch in the original
    `tfim_autoDiff_autoEncoder.py`."""
    activation = x
    for w, b in params[0:-1]:
        activation = jax.nn.relu(jnp.dot(activation, w) + b)
        rng_key, subkey = jax.random.split(rng_key)
        activation = dropout(activation, drop_p, subkey)
    w, b = params[-1]
    return jnp.dot(activation, w) + b


def fetch_latent(params, x, drop_p=0.0, rng_key=None, normalize=False, eps=0.0):
    """Encode-only path, no dropout by default.

    `normalize=False` (default) returns the raw pre-norm latent — this matches
    the original `tfim_autoDiff_autoEncoder.py` behavior. `normalize=True`
    returns the unit-norm vector that the AE was trained with; pass `eps > 0`
    to use the soft-normalize `z / sqrt(‖z‖² + eps)` variant (recommended for
    the h-optim path where ‖z‖ can be driven toward zero by Adam updates).
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    mid = len(params) // 2
    if normalize:
        return encoder(params[0:mid], x, drop_p, rng_key, eps=eps)
    return _encoder_raw(params[0:mid], x, drop_p, rng_key)


def ae_loss(params, x, drop_p, rng_key, center_coeff=1e-3):
    """Quantum-fidelity reconstruction loss + latent-variance penalty.

    Reconstruction (fidelity):
        1 - mean_batch ( ⟨ψ|ψ̂⟩² ),  ψ and ψ̂ unit-normalised.
    Latent-variance penalty:
        center_coeff * mean_dims ( Var_batch(z_i) ).

    Squaring the overlap removes the unphysical |ψ⟩ vs −|ψ⟩ sign indifference
    (same quantum state up to a global phase). The variance penalty keeps the
    encoded latents from spreading out as a cheap reconstruction shortcut.
    """
    eps = 1e-12
    x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)

    rng_key, sub_ae = jax.random.split(rng_key)
    x_hat = autoencoder(params, x, drop_p, sub_ae)
    overlap = jnp.sum(x * x_hat, axis=-1)
    recon = 1.0 - jnp.mean(overlap ** 2)

    rng_key, sub_enc = jax.random.split(rng_key)
    mid = len(params) // 2
    z = encoder(params[:mid], x, 0.0, sub_enc)
    var_pen = jnp.mean(jnp.var(z, axis=0))

    return recon + center_coeff * var_pen
