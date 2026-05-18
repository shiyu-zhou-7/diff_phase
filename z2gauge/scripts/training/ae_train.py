"""
Autoencoder training loop for z2gauge ground-state wavefunctions.

Adam optimizer, MSE-style loss (see models.autoencoder.ae_loss).

Mirrors tfim_reorg/scripts/training/ae_train.py: returns a dict with the
trained params plus per-epoch history, so callers can pull
`hist['params']` and inspect `hist['loss_list']`.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import value_and_grad, jit
import optax

from models.autoencoder import ae_loss, autoencoder, encoder


def _evaluate_latent(params, x_a, x_b, rng_key):
    """Run the encoder on two phase-labeled batches; returns (latent_a, latent_b)."""
    mid = len(params) // 2
    z_a = encoder(params[0:mid], x_a, 0.0, rng_key)
    z_b = encoder(params[0:mid], x_b, 0.0, rng_key)
    return z_a, z_b


def train_autoencoder(
    params,
    x_train,
    *,
    x_test=None,
    x_para_eval=None,
    epochs=1000,
    lr=1e-4,
    drop_p=0.1,
    log_every=100,
    eval_every=20,
    seed=0,
    center_coeff=1e-3,
):
    """
    Trains the AE in-place-style (returns new params + history).

    Returns a dict with:
      - 'params'                : trained parameters
      - 'loss_list'             : training loss per epoch
      - 'val_loss_list'         : [(val_loss, epoch), ...] every `log_every`
      - 'latent_a_list'         : [(latent_z_a, epoch), ...] every `eval_every`
      - 'latent_b_list'         : [(latent_z_b, epoch), ...] every `eval_every`
    """
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(params)
    key = jax.random.PRNGKey(seed)

    @jit
    def update(params, x, opt_state, rng_key):
        value, grads = value_and_grad(ae_loss, argnums=0)(
            params, x, drop_p, rng_key, center_coeff,
        )
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value

    loss_list = []
    val_loss_list = []
    latent_a_list = []
    latent_b_list = []

    for i in range(epochs):
        key, sub = jax.random.split(key)
        params, opt_state, loss_val = update(params, x_train, opt_state, sub)
        loss_list.append(float(loss_val))

        if i % log_every == 0:
            if x_test is not None:
                val_loss = float(ae_loss(params, x_test, 0.0, key, center_coeff))
                val_loss_list.append((val_loss, i))
                print(f'  AE epoch {i+1:5d}/{epochs} loss={float(loss_val):.6e} val_loss={val_loss:.6e}')
            else:
                print(f'  AE epoch {i+1:5d}/{epochs} loss={float(loss_val):.6e}')

        if i % eval_every == 0 and x_test is not None and x_para_eval is not None:
            z_a, z_b = _evaluate_latent(params, x_test, x_para_eval, key)
            latent_a_list.append((np.asarray(z_a), i))
            latent_b_list.append((np.asarray(z_b), i))

    return {
        'params': params,
        'loss_list': loss_list,
        'val_loss_list': val_loss_list,
        'latent_a_list': latent_a_list,
        'latent_b_list': latent_b_list,
    }
