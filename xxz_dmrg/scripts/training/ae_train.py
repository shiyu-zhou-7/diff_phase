import jax
import jax.numpy as jnp
import numpy as np
import optax
from models.autoencoder import init_params, ae_update, make_ae_step, autoencoder


def train_autoencoder(init_params_list, x_train, epochs=1000, lr=1e-4, weight_decay=1e-4, drop_p=0.05, center_coeff=1e-3, seed=45348497, X_test=None):
    key = jax.random.PRNGKey(seed)

    # Convert to JAX arrays (device will be selected by default_device)
    x_train = jnp.array(x_train)
    if X_test is not None:
        X_test = jnp.array(X_test)

    opt = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    step = make_ae_step(opt)              # capture opt in closure (not traced)
    opt_state = opt.init(init_params_list)

    params = init_params_list
    for i in range(epochs):
        key, sub = jax.random.split(key)
        # params, opt_state, _ = ae_update(params, x_train, opt_state, opt, drop_p, sub, center_coeff)
        params, opt_state, loss_val = step(params, x_train, opt_state, drop_p, sub, center_coeff)

        if (i+1) % 500 == 0 or i == 0:
            if X_test is not None:
                key, sub = jax.random.split(key)
                _, _, loss_val_test = step(params, X_test, opt_state, 0.0, sub, center_coeff)
                print(f'  AE Epoch {i+1:4d} / {epochs}, Train Loss: {loss_val:.6e}, Test Loss: {loss_val_test:.6e}')
            else:
                print(f'  AE Epoch {i+1:4d} / {epochs}, Loss: {loss_val:.6e}')


    return params