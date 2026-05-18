"""
Bootstrap routine for the z2gauge active-phase-discovery workflow.

Three pure functions, no disk I/O:
  - sample_circle(h_center, radius, num_samples, j_a, star_ops, trans_ops, rng_key, ite_cfg)
      Uniformly sample h-values around `h_center`, ITE-solve each, stack
      ground states into a training set.
  - train_ae_and_centroid(x_train, ae_cfg, rng_key, eps_norm=1e-8)
      Initialize and train a fresh AE on x_train; compute the latent
      centroid as the mean of soft-normalized latents (matches the geometry
      of `_ae_latent_loss` in training/optim_h.py).
  - bootstrap_ae(...)
      Single-circle convenience wrapper composing the two above.
  - retrain_with_history(...)
      History-aware retrain: re-samples one circle per historical center,
      concatenates them, then trains a fresh AE from scratch.

Twin-selection convention: ITE is initialized from a FIXED PRNGKey(0) so
the same H(h) always converges to the same Z2 twin in the degenerate ground
subspace. This matches both the data-gen path (generate_confined_data.py)
and the optim path (training/optim_h.py).
"""
import jax
import jax.numpy as jnp

from hamiltonian.ite import ite_ground_state_from_params, ite_ground_state_batched
from models.autoencoder import init_params, fetch_latent
from training.ae_train import train_autoencoder


# Pin the ITE init-state key so all generations share a single twin convention.
_TWIN_KEY = jax.random.PRNGKey(0)


def sample_circle(
    h_center, radius, num_samples,
    j_a, star_ops, trans_ops,
    rng_key,
    ite_n_steps=150, ite_dt=1e-2,
):
    """Sample `num_samples` h-values uniformly in [h_center - radius,
    h_center + radius], ITE-solve each, stack ground states into x_train.

    Returns:
        {'h_samples': (num_samples,) jnp array,
         'x_train':   (num_samples, 2**(2*Lx*Ly)) jnp array of ground-state amplitudes}
    """
    h_samples = jax.random.uniform(
        rng_key,
        shape=(num_samples,),
        minval=h_center - radius,
        maxval=h_center + radius,
    )

    # Batched ITE path: single JAX dispatch; star/trans ops are shared across
    # all B samples so we never materialize the per-sample (D, D) Hamiltonian.
    x_train, _ = ite_ground_state_batched(
        j_a, h_samples,
        star_ops, trans_ops,
        n_steps=ite_n_steps, dt=ite_dt,
        key=_TWIN_KEY,
    )

    # --- Legacy single-sample loop (kept commented for easy revert) ---------
    # x_list = []
    # for h in h_samples:
    #     v, _ = ite_ground_state_from_params(
    #         j_a, float(h),
    #         star_ops, trans_ops,
    #         n_steps=ite_n_steps, dt=ite_dt,
    #         key=_TWIN_KEY,
    #     )
    #     x_list.append(v)
    # x_train = jnp.stack(x_list, axis=0)
    # ------------------------------------------------------------------------

    return {'h_samples': h_samples, 'x_train': x_train}


def train_ae_and_centroid(x_train, ae_cfg, rng_key, eps_norm=1e-8):
    """Init + train a fresh AE on x_train; centroid is the mean of
    soft-normalized latents.

    Returns:
        {'ae_params':  trained AE parameter list,
         'centroid':   (latent_dim,) jnp array (in the unit ball),
         'ae_history': history dict from train_autoencoder}
    """
    init_key, train_key = jax.random.split(rng_key)

    params = init_params(list(ae_cfg.layer_widths), init_key, scale=ae_cfg.init_scale)

    # train_autoencoder expects an int seed; derive one from the JAX subkey.
    train_seed = int(jax.random.randint(train_key, (), 0, 2**31 - 1))

    hist = train_autoencoder(
        params,
        x_train,
        x_test=None,
        x_para_eval=None,
        epochs=ae_cfg.epochs,
        lr=ae_cfg.lr,
        drop_p=ae_cfg.dropout_p,
        log_every=ae_cfg.log_every,
        eval_every=ae_cfg.eval_every,
        seed=train_seed,
        center_coeff=ae_cfg.center_coeff,
    )
    trained_params = hist['params']

    z = fetch_latent(
        trained_params, x_train,
        rng_key=jax.random.PRNGKey(0),
        drop_p=0.0,
        normalize=True, eps=eps_norm,
    )
    centroid = jnp.mean(z, axis=0)

    return {'ae_params': trained_params, 'centroid': centroid, 'ae_history': hist}


def bootstrap_ae(
    h_center, radius, num_samples,
    j_a, star_ops, trans_ops,
    ae_cfg, rng_key,
    ite_n_steps=150, ite_dt=1e-2,
):
    """Single-circle bootstrap: sample one circle around `h_center`, then
    train a fresh AE and compute its centroid.

    Returns the merged dict from sample_circle + train_ae_and_centroid.
    """
    sample_key, train_key = jax.random.split(rng_key)
    circle = sample_circle(
        h_center, radius, num_samples,
        j_a, star_ops, trans_ops,
        sample_key,
        ite_n_steps=ite_n_steps, ite_dt=ite_dt,
    )
    ae = train_ae_and_centroid(circle['x_train'], ae_cfg, train_key)
    return {**circle, **ae}


def retrain_with_history(
    bootstrap_history,
    radius,
    num_samples_per_circle,
    j_a, star_ops, trans_ops,
    ae_cfg,
    rng_key,
    eps_norm=1e-8,
    ite_n_steps=150, ite_dt=1e-2,
):
    """History-aware AE retrain. For each h_center in `bootstrap_history`,
    freshly sample a circle of `num_samples_per_circle` h-values with the same
    `radius`, ITE-solve each, stack ground states. Concatenate all per-circle
    x_trains into x_train_all, then train a fresh AE from scratch on the
    combined set and compute the centroid as the mean of soft-normalized
    latents.

    Returns:
        {'h_samples_per_circle': list of (num_samples,) arrays, one per center,
         'x_train_per_circle':   list of (num_samples, 2**(2*Lx*Ly)) arrays,
         'h_samples_all':        (N_hist*num_samples,) concatenated,
         'x_train_all':          (N_hist*num_samples, 2**(2*Lx*Ly)) concatenated,
         'ae_params':            trained AE params,
         'centroid':             mean of normalized latents over x_train_all,
         'ae_history':           training history dict}
    """
    if len(bootstrap_history) == 0:
        raise ValueError('retrain_with_history requires a non-empty bootstrap_history')

    keys = jax.random.split(rng_key, len(bootstrap_history) + 1)
    sample_keys = keys[:-1]
    train_key = keys[-1]

    h_per_circle = []
    x_per_circle = []
    for h_center, k in zip(bootstrap_history, sample_keys):
        c = sample_circle(
            h_center, radius, num_samples_per_circle,
            j_a, star_ops, trans_ops,
            k,
            ite_n_steps=ite_n_steps, ite_dt=ite_dt,
        )
        h_per_circle.append(c['h_samples'])
        x_per_circle.append(c['x_train'])

    x_train_all = jnp.concatenate(x_per_circle, axis=0)
    h_samples_all = jnp.concatenate(h_per_circle, axis=0)

    ae = train_ae_and_centroid(x_train_all, ae_cfg, train_key, eps_norm=eps_norm)
    return {
        'h_samples_per_circle': h_per_circle,
        'x_train_per_circle': x_per_circle,
        'h_samples_all': h_samples_all,
        'x_train_all': x_train_all,
        **ae,
    }
