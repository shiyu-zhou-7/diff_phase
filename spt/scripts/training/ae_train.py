"""
Autoencoder training loop for the cluster-chain feature vectors. AdamW + MSE loss
(models.autoencoder.ae_loss). Mirrors xxz_dmrg/scripts/training/ae_train.py.
"""
import jax
import jax.numpy as jnp
import optax

from models.autoencoder import make_ae_step, make_ae_step_fidelity


def train_autoencoder(init_params, x_train, *, epochs=4000, lr=1e-3,
                      weight_decay=0.0, drop_p=0.0, center_coeff=1e-3,
                      seed=83948, log_every=1000, loss='mse'):
    """Train the AE on x_train (B, D); returns trained params.

    loss='mse' (default): MSE reconstruction -- for the feature-vector AEs.
    loss='fidelity': 1 - fidelity -- for normalized wavefunction inputs."""
    key = jax.random.PRNGKey(seed)
    x_train = jnp.asarray(x_train)

    opt = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    if loss == 'fidelity':
        step = make_ae_step_fidelity(opt)
    elif loss == 'mse':
        step = make_ae_step(opt)
    else:
        raise ValueError(f"loss must be 'mse' or 'fidelity', got {loss!r}")
    opt_state = opt.init(init_params)

    params = init_params
    for i in range(epochs):
        key, sub = jax.random.split(key)
        params, opt_state, loss_val = step(params, x_train, opt_state, drop_p, sub, center_coeff)
        if log_every and ((i + 1) % log_every == 0 or i == 0):
            print(f'    AE epoch {i+1:5d}/{epochs}  loss={float(loss_val):.6e}')
    return params
