"""
Scatter the varied-seed z2gauge dataset in the first two principal components
of the varied-seed AE's latent space.

Loads:
    ../models/z2gauge_ite_autoencoder_variedseed_Lx2Ly3.pkl
    ../data/data_ite_confined_2x3_h-1.5_to_-0.7_n500_variedseed.pkl
    ../data/data_ite_deconfined_2x3_h-0.15_to_-0.1_n500_variedseed.pkl

Encodes every sample through the AE, fits a 2-component PCA on the latents via
numpy SVD (sklearn is not available in this venv), and scatters PC1 vs PC2
colored by phase (confined vs deconfined).

Unlike the production pinned-twin datasets, these samples draw v_0
independently per sample, so they scatter across the Z2 4-fold ground
manifold. This plot shows what that looks like in the AE's latent geometry.

Usage:
    cd z2gauge/scripts && python plot_variedseed_pca.py
"""
import os, sys, pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import jax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.autoencoder import fetch_latent


Lx, Ly = 2, 3

ae_path  = f'../models/z2gauge_ite_autoencoder_variedseed_Lx{Lx}Ly{Ly}.pkl'
conf_pkl = f'../data/data_ite_confined_{Lx}x{Ly}_h-1.5_to_-0.7_n500_variedseed.pkl'
dec_pkl  = f'../data/data_ite_deconfined_{Lx}x{Ly}_h-0.15_to_-0.1_n500_variedseed.pkl'
print(f'AE:           {ae_path}')
print(f'Confined:     {conf_pkl}')
print(f'Deconfined:   {dec_pkl}')

ae = pickle.load(open(ae_path, 'rb'))
ae_params = ae['params']

data_conf = pickle.load(open(conf_pkl, 'rb'))
data_dec  = pickle.load(open(dec_pkl, 'rb'))

X_all = np.stack([d['v'].real for d in (list(data_conf) + list(data_dec))]).astype(np.float32)
phase_all = np.array(['confined']*len(data_conf) + ['deconfined']*len(data_dec))
h_all     = np.array([float(d['h']) for d in (list(data_conf) + list(data_dec))])
print(f'X_all:        {X_all.shape}')
print(f'  confined   h ∈ [{h_all[phase_all=="confined"].min():+.4f}, {h_all[phase_all=="confined"].max():+.4f}]')
print(f'  deconfined h ∈ [{h_all[phase_all=="deconfined"].min():+.4f}, {h_all[phase_all=="deconfined"].max():+.4f}]')

# Encode through the AE
Z_all = np.asarray(fetch_latent(ae_params, X_all, jax.random.PRNGKey(0)))
print(f'Z_all:        {Z_all.shape}  (mean ‖z‖ = {np.linalg.norm(Z_all, axis=1).mean():.3f})')

# 2-component PCA via numpy SVD
mu = Z_all.mean(axis=0, keepdims=True)
Zc = Z_all - mu
U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
P = Vt[:2]
var_ratio = (S[:2] ** 2) / (S ** 2).sum()
print(f'PCA var ratios: PC1={var_ratio[0]:.2%}, PC2={var_ratio[1]:.2%}')

Z2_all = (Z_all - mu) @ P.T

# ── plot ──────────────────────────────────────────────────────────────────────
out_path = f'../figures/latent_pca_variedseed_Lx{Lx}Ly{Ly}.pdf'
fig, ax = plt.subplots(figsize=(7.5, 6), dpi=150)

for p, color in [('confined', 'tab:red'), ('deconfined', 'tab:blue')]:
    mask = phase_all == p
    ax.scatter(Z2_all[mask, 0], Z2_all[mask, 1],
               c=color, s=18, alpha=0.6, edgecolors='none',
               label=f'{p} (n={mask.sum()})', zorder=2)

ax.set_xlabel(f'PC1 ({var_ratio[0]:.0%})', fontsize=14)
ax.set_ylabel(f'PC2 ({var_ratio[1]:.0%})', fontsize=14)
ax.set_title(f'Varied-seed dataset in varied-seed AE latent PCA (Lx{Lx}Ly{Ly})',
             fontsize=13)
ax.tick_params(axis='both', labelsize=11)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=11, framealpha=0.92)

fig.savefig(out_path, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out_path}')
