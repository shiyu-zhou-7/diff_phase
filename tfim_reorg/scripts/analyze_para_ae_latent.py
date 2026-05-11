"""
Latent-space analysis for the para-trained autoencoder.

Loads the latest `para_ae_params_*.pkl`, encodes both the para training set
(`data_para.pkl`) and the SSB set (`data_ssb_debiased.pkl`) through the
encoder, and produces:

  1. PCA scatter (2D) of the two phase clouds in para-AE latent space.
     Marks the para centroid.
  2. Histogram of ||z|| per phase — diagnoses whether out-of-distribution
     SSB inputs blow up the un-normalized latent (which is what's driving
     the deep-para divergence in main_optim_h_ae.py).
  3. Per-h scatter of the first PCA component, to see how the latent moves
     through h.

Reads:
    ../models/para_ae_params_<ts>.pkl
    ../data/data_para.pkl
    ../data/data_ssb_debiased.pkl

Writes (with timestamp suffix <ts>):
    ../figures/para_ae_latent_pca_<ts>.pdf
    ../figures/para_ae_latent_norm_hist_<ts>.pdf
    ../figures/para_ae_latent_pc1_vs_h_<ts>.pdf

Usage:
    cd tfim_reorg/scripts && python analyze_para_ae_latent.py
"""
import sys, os, glob
import numpy as np
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.autoencoder import encoder, _encoder_raw
from utils.io import load_pickle, timestamp


def _latest(pattern):
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f'No file matched pattern: {pattern}')
    return matches[-1]


# ── inputs ───────────────────────────────────────────────────────────────────
ae_params_path = os.environ.get('AE_PARAMS') or _latest('../models/para_ae_params_*.pkl')
para_data_path = '../data/data_para.pkl'
ssb_data_path  = '../data/data_ssb_debiased.pkl'

ts = timestamp()
print(f'Timestamp:     {ts}')
print(f'AE params:     {ae_params_path}')
print(f'para data:     {para_data_path}')
print(f'ssb data:      {ssb_data_path}')

ae_blob = load_pickle(ae_params_path)
params  = ae_blob['params']
mid     = len(params) // 2
half    = params[0:mid]

para_data = load_pickle(para_data_path)
ssb_data  = load_pickle(ssb_data_path)

x_para = np.array([d['v'] for d in para_data])
h_para = np.array([d['h'] for d in para_data])
x_ssb  = np.array([d['v'] for d in ssb_data])
h_ssb  = np.array([d['h'] for d in ssb_data])
print(f'para shape: {x_para.shape}, h ∈ [{h_para.min():.3f}, {h_para.max():.3f}]  '
      f'(file convention; equiv. negative-h under H(h)=-ZZ+hX)')
print(f'ssb shape:  {x_ssb.shape},  h ∈ [{h_ssb.min():.3f}, {h_ssb.max():.3f}]')

# ── encode (both normalized and un-normalized variants) ──────────────────────
key = jax.random.PRNGKey(0)
z_para_norm = np.asarray(encoder(half, jnp.asarray(x_para), 0.0, key))
z_ssb_norm  = np.asarray(encoder(half, jnp.asarray(x_ssb),  0.0, key))
z_para_raw  = np.asarray(_encoder_raw(half, jnp.asarray(x_para), 0.0, key))
z_ssb_raw   = np.asarray(_encoder_raw(half, jnp.asarray(x_ssb),  0.0, key))
print(f'normalized z shapes: para {z_para_norm.shape}, ssb {z_ssb_norm.shape}')

centroid_para_norm = z_para_norm.mean(axis=0)

# ── PCA on combined normalized latents ───────────────────────────────────────
Z = np.concatenate([z_para_norm, z_ssb_norm], axis=0)
Zc = Z - Z.mean(axis=0, keepdims=True)
U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
P = Vt[:2]                               # 2 principal directions
proj_para = (z_para_norm - Z.mean(axis=0)) @ P.T
proj_ssb  = (z_ssb_norm  - Z.mean(axis=0)) @ P.T
proj_centroid_para = (centroid_para_norm - Z.mean(axis=0)) @ P.T
print(f'PCA singular values (top 5): {S[:5]}')

# ── plot 1: PCA scatter, color-coded by h (codebase convention) ─────────────
# Convert the file's h field into the codebase's negative-h convention.
h_para_codebase = -h_para
h_ssb_codebase  = -h_ssb

fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
sc_para = ax.scatter(proj_para[:, 0], proj_para[:, 1], s=12, alpha=0.7,
                     c=h_para_codebase, cmap='Oranges_r', marker='o',
                     edgecolors='none')
sc_ssb  = ax.scatter(proj_ssb[:, 0], proj_ssb[:, 1], s=12, alpha=0.7,
                     c=h_ssb_codebase, cmap='Blues_r', marker='o',
                     edgecolors='none')
ax.scatter(proj_centroid_para[0], proj_centroid_para[1], s=200,
           marker='*', color='red', edgecolors='black', linewidths=0.8,
           zorder=10, label='centroid_para')

cbar_para = fig.colorbar(sc_para, ax=ax, pad=0.02, shrink=0.85,
                         label='para  $h$  (codebase, $-2 \\to -1$)')
cbar_ssb  = fig.colorbar(sc_ssb,  ax=ax, pad=0.06, shrink=0.85,
                         label='ssb  $h$  (codebase, $-1 \\to 0$)')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('para-AE latent space (normalized z, PCA), colored by $h$')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
pca_path = f'../figures/para_ae_latent_pca_{ts}.pdf'
fig.savefig(pca_path, bbox_inches='tight')
plt.close(fig)
print(f'Saved {pca_path}')

# ── plot 2: histogram of ||z|| per phase (un-normalized z) ───────────────────
norm_para = np.linalg.norm(z_para_raw, axis=1)
norm_ssb  = np.linalg.norm(z_ssb_raw,  axis=1)
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
ax.hist(norm_para, bins=40, alpha=0.55, color='tab:orange', label='para')
ax.hist(norm_ssb,  bins=40, alpha=0.55, color='tab:blue',   label='ssb')
ax.set_xlabel(r'$\|\,z_{\mathrm{raw}}\,\|$  (un-normalized latent magnitude)')
ax.set_ylabel('count')
ax.set_title('Latent magnitude distribution per phase (para-AE)')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
norm_hist_path = f'../figures/para_ae_latent_norm_hist_{ts}.pdf'
fig.savefig(norm_hist_path, bbox_inches='tight')
plt.close(fig)
print(f'Saved {norm_hist_path}')
print(f'||z_raw|| stats:  para mean={norm_para.mean():.3f} std={norm_para.std():.3f}   '
      f'ssb mean={norm_ssb.mean():.3f} std={norm_ssb.std():.3f}')

# ── plot 3: PC1 vs h — see how latent flows along the parameter axis ────────
# Convert the h field into "this codebase's negative-h convention" for plotting
h_para_codebase = -h_para   # data_para.pkl stored positive h ≡ negative-h here
h_ssb_codebase  = -h_ssb    # data_ssb_debiased.pkl stored positive h ≡ negative-h here
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
ax.scatter(h_para_codebase, proj_para[:, 0], s=8, alpha=0.5,
           color='tab:orange', label='para')
ax.scatter(h_ssb_codebase, proj_ssb[:, 0], s=8, alpha=0.5,
           color='tab:blue', label='ssb')
ax.axvline(-1.0, color='gray', ls='--', lw=1, label='phase boundary  $|h|=1$')
ax.set_xlabel('h  (codebase convention, $H=-ZZ+hX$)')
ax.set_ylabel('PC1 of normalized latent')
ax.set_title('PC1 vs h — latent flow across the phase diagram')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
pc1_h_path = f'../figures/para_ae_latent_pc1_vs_h_{ts}.pdf'
fig.savefig(pc1_h_path, bbox_inches='tight')
plt.close(fig)
print(f'Saved {pc1_h_path}')

print('\nDone.')
