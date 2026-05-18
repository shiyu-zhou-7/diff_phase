"""
Plot the 437956 trajectory in the latent space of the all-phases AE.

Pipeline:
  1. Load the data pickle (training obs + trajectory obs) and the trained AE.
  2. Encode both sets via fetch_latent.
  3. Fit PCA(2) on training latents, project both.
  4. Scatter training points colored by phase, overlay trajectory (orange
     colormap, start/end markers) matching plot_combined_trajectory.py style.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from sklearn.decomposition import PCA

from models.autoencoder import fetch_latent
from utils.io import load_pickle


DATA_PATH = os.environ.get('DATA_PATH') or sorted(
    glob.glob('../data/all_phases_latent_data_*.pkl')
)[-1]
AE_PATH = os.environ.get('AE_PATH') or sorted(
    glob.glob('../models/xxzhdmrg_autoencoder_params_all_phases_latent*_L20.pkl')
)[-1]
print(f"Data: {DATA_PATH}")
print(f"AE  : {AE_PATH}")

bundle    = load_pickle(DATA_PATH)
ae_pkl    = load_pickle(AE_PATH)
ae_params = ae_pkl['params']

train_obs   = jnp.asarray(bundle['train_obs'])
train_phase = np.asarray(bundle['train_phase'])
traj_obs    = jnp.asarray(bundle['traj_obs'])

key     = jax.random.PRNGKey(0)
Z_train = np.array(fetch_latent(ae_params, train_obs, key))
Z_traj  = np.array(fetch_latent(ae_params, traj_obs,  key))
print(f"Z_train: {Z_train.shape}, Z_traj: {Z_traj.shape}")

pca     = PCA(n_components=2)
P_train = pca.fit_transform(Z_train)
P_traj  = pca.transform(Z_traj)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# ── Plot ───────────────────────────────────────────────────────────────────
PHASE_ORDER   = ['Neel_Z', 'FM_Z', 'Neel_Y', 'PM', 'Liquid']
PHASE_COLORS  = {
    'Neel_Z':  '#4477AA',
    'FM_Z':    '#EE6677',
    'Neel_Y':  '#228833',
    'PM':      '#888888',
    'Liquid':  'crimson',
}
PHASE_MARKERS = {
    'Neel_Z':  '^',
    'FM_Z':    'v',
    'Neel_Y':  's',
    'PM':      'o',
    'Liquid':  'D',
}

fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)

for ph in PHASE_ORDER:
    mask = (train_phase == ph)
    if not mask.any():
        continue
    ax.scatter(P_train[mask, 0], P_train[mask, 1],
               c=PHASE_COLORS[ph], marker=PHASE_MARKERS[ph],
               s=18, alpha=0.55, edgecolors='none',
               label=ph, zorder=1)

n_traj = P_traj.shape[0]
shades = plt.cm.Oranges(np.linspace(0.3, 1.0, n_traj))
ax.plot(P_traj[:, 0], P_traj[:, 1], lw=0.8, color='0.5', alpha=0.4, zorder=2)
ax.scatter(P_traj[:, 0], P_traj[:, 1],
           c=shades, s=10, edgecolors='none', zorder=3)
ax.scatter(P_traj[0, 0],  P_traj[0, 1],  s=70,
           color=shades[0],  edgecolors='k', zorder=5, label='Start')
ax.scatter(P_traj[-1, 0], P_traj[-1, 1], s=70,
           color=shades[-1], edgecolors='k', zorder=5, label='End')

ax.set_xlabel('PC1', fontsize=15)
ax.set_ylabel('PC2', fontsize=18)
ax.tick_params(labelsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc='best')
fig.tight_layout()

OUT = '../figures/trajectory_in_latent_437956.pdf'
fig.savefig(OUT, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {OUT}")
