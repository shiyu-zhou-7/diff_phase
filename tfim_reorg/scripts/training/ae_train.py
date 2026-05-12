"""
Autoencoder training loop for TFIM. Adam optimizer, MSE-style loss
(see models.autoencoder.ae_loss).

Mirrors the body of the original `tfim_autoEncoder.py` training loop, with
the latent-space snapshots collected as a side effect (used downstream for
PCA / scatter plots).
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import value_and_grad, jit
import optax

from models.autoencoder import ae_loss, autoencoder, encoder
from utils.bits import cal_m_reconstructed


def _evaluate_latent(params, x_ssb, x_para, rng_key):
    """Run the encoder on two phase-labeled batches; returns (latent_ssb, latent_para)."""
    mid = len(params) // 2
    z_ssb = encoder(params[0:mid], x_ssb, 0.0, rng_key)
    z_para = encoder(params[0:mid], x_para, 0.0, rng_key)
    return z_ssb, z_para


def train_autoencoder(
    params,
    x_train,
    *,
    x_test=None,
    x_para_eval=None,
    epochs=1000,
    lr=0.1,
    drop_p=0.1,
    log_every=100,
    eval_every=20,
    seed=0,
    center_coeff=1e-3,
):
    """
    Trains the AE in-place-style (returns new params + history).

    Returns a dict with:
      - 'params'             : trained parameters
      - 'loss_list'          : training loss per epoch
      - 'latent_ssb_list'    : [(latent_z_ssb, epoch), ...] every `eval_every`
      - 'latent_para_list'   : [(latent_z_para, epoch), ...] every `eval_every`
      - 'val_loss_list'      : [(val_loss, epoch), ...] every `log_every`
      - 'mag_list'           : [(<m>_reconstructed, epoch), ...] every `log_every`
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
    mag_list = []
    latent_ssb_list = []
    latent_para_list = []

    for i in range(epochs):
        key, sub = jax.random.split(key)
        params, opt_state, loss_val = update(params, x_train, opt_state, sub)
        loss_list.append(float(loss_val))

        if i % log_every == 0:
            val_loss = None
            if x_test is not None:
                val_loss = float(ae_loss(params, x_test, 0.0, key, center_coeff))
                val_loss_list.append((val_loss, i))
                m_recon = cal_m_reconstructed(autoencoder(params, x_test, 0.0, key))
                m_avg = float(sum(m_recon) / len(m_recon))
                mag_list.append((m_avg, i))
                print(f'epoch={i} loss={float(loss_val):.6f} val_loss={val_loss:.6f} <m>={m_avg:.4f}')
            else:
                print(f'epoch={i} loss={float(loss_val):.6f}')

        if i % eval_every == 0 and x_test is not None and x_para_eval is not None:
            z_ssb, z_para = _evaluate_latent(params, x_test, x_para_eval, key)
            # materialize to numpy at append time so the snapshot list stays
            # small and pickling later doesn't trigger a giant async-evaluation burst
            latent_ssb_list.append((np.asarray(z_ssb), i))
            latent_para_list.append((np.asarray(z_para), i))

    return {
        'params': params,
        'loss_list': loss_list,
        'val_loss_list': val_loss_list,
        'mag_list': mag_list,
        'latent_ssb_list': latent_ssb_list,
        'latent_para_list': latent_para_list,
    }
