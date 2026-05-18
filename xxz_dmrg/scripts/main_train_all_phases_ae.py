"""
Train an autoencoder on the all-phases training set produced by
main_generate_all_phases_data.py.

Architecture exactly matches the active-phase workflow
(main_resume_with_history.py:128):  [D, 20, latent_dim, 20, D].
"""

import os
import glob
import numpy as np

import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

from configs.config import AEConfig, ActiveConfig
from models.autoencoder import init_params as init_ae
from training.ae_train import train_autoencoder
from utils.io import save_pickle, load_pickle


DATA_PATH = os.environ.get('DATA_PATH') or sorted(
    glob.glob('../data/all_phases_latent_data_*.pkl')
)[-1]
print(f"Loading: {DATA_PATH}")

bundle = load_pickle(DATA_PATH)
X = jnp.asarray(bundle['train_obs'])
D = X.shape[1]
print(f"Training set: {X.shape}  (D={D})")

ae_cfg  = AEConfig()
act_cfg = ActiveConfig()
layers  = [D, 20, ae_cfg.latent_dim, 20, D]
print(f"AE layers: {layers}")

key = jax.random.PRNGKey(ae_cfg.seed)
key, sub = jax.random.split(key)
ae_params = init_ae(layers, sub)

print(f"Training {ae_cfg.epochs} epochs on {X.shape[0]} samples...")
ae_params = train_autoencoder(
    ae_params, X,
    epochs=ae_cfg.epochs,
    lr=ae_cfg.lr,
    weight_decay=ae_cfg.weight_decay,
    drop_p=ae_cfg.dropout_p,
    center_coeff=act_cfg.center_coeff,
    seed=ae_cfg.seed,
)

# centroid is unused for visualization; kept so the pickle shape matches the
# active-phase code's expectation of {'params', 'centroid'}.
centroid = np.zeros(ae_cfg.latent_dim)

OUT_PATH = (
    f'../models/xxzhdmrg_autoencoder_params_all_phases_latent'
    f'{ae_cfg.latent_dim}_L20.pkl'
)
save_pickle({'params': ae_params, 'centroid': centroid}, OUT_PATH)
print(f"Saved: {OUT_PATH}")
