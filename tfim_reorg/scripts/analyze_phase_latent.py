"""
Latent-space analysis: train a fresh AE on 500 SSB + 500 PARA samples and
inspect what it learned.

Pipeline:
1. Sample 500 SSB h-values (uniform in [-0.95, -0.05]) and 500 PARA
   (uniform in [-3.0, -1.05]). ED-solve each.
2. Train an AE on the 1000 ground states (AEConfig defaults; override
   EPOCHS env var to change).
3. Encode the training set; fit PCA(10) on the latents.
4. Plot histograms of PC1..PC10, colored by phase (500 vs 500).
5. Sample 100 new SSB + 100 new PARA test points, encode, project onto
   PC1-PC2, and overlay on the training scatter.

Usage:
    python analyze_phase_latent.py
    EPOCHS=10000 python analyze_phase_latent.py    # quicker

Outputs:
    ../figures/phase_latent_pc_histograms_<ts>.pdf
    ../figures/phase_latent_pc12_scatter_<ts>.pdf
"""
import sys, os
from dataclasses import replace, asdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import jax
from jax import config
config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import AEConfig, HamConfig
from hamiltonians.tfim import build_tfim_chain
from training.bootstrap import sample_circle, train_ae_and_centroid
from models.autoencoder import fetch_latent
from utils.io import timestamp


# ── config ────────────────────────────────────────────────────────────────────
ham_cfg = HamConfig()
ae_cfg = AEConfig()
if 'EPOCHS' in os.environ:
    ae_cfg = replace(ae_cfg, epochs=int(os.environ['EPOCHS']))
if 'LATENT_DIM' in os.environ:
    d = int(os.environ['LATENT_DIM'])
    input_dim = 2 ** ham_cfg.N
    ae_cfg = replace(ae_cfg, layer_widths=(input_dim, 500, d, 500, input_dim))

ts = timestamp()
print(f'Timestamp: {ts}')
print(f'ae_cfg.epochs={ae_cfg.epochs}, lr={ae_cfg.lr}, layer_widths={ae_cfg.layer_widths}')

# ── phase h ranges (consistent with plot_active_phase_pca_trajectory.py) ────
# SSB: h in [-0.95, -0.05]  → center -0.5, radius 0.45
# PARA: h in [-3.0, -1.05]  → center -2.025, radius 0.975
SSB_CENTER, SSB_RADIUS = -0.5, 0.45
PARA_CENTER, PARA_RADIUS = -2.025, 0.975

# ── build Hamiltonian ────────────────────────────────────────────────────────
ham_X, ham_ZZ = build_tfim_chain(ham_cfg.N, ham_cfg.J)

key = jax.random.PRNGKey(42)

print('Sampling 500 SSB training points...')
key, sub = jax.random.split(key)
ssb_train = sample_circle(SSB_CENTER, SSB_RADIUS, 500, ham_X, ham_ZZ, sub)

print('Sampling 500 PARA training points...')
key, sub = jax.random.split(key)
para_train = sample_circle(PARA_CENTER, PARA_RADIUS, 500, ham_X, ham_ZZ, sub)

print('Sampling 100 SSB test points...')
key, sub = jax.random.split(key)
ssb_test = sample_circle(SSB_CENTER, SSB_RADIUS, 100, ham_X, ham_ZZ, sub)

print('Sampling 100 PARA test points...')
key, sub = jax.random.split(key)
para_test = sample_circle(PARA_CENTER, PARA_RADIUS, 100, ham_X, ham_ZZ, sub)

x_train = np.concatenate(
    [np.asarray(ssb_train['x_train']), np.asarray(para_train['x_train'])],
    axis=0,
)  # (1000, 1024), first 500 = SSB, next 500 = PARA

# ── train AE ──────────────────────────────────────────────────────────────────
print(f'\nTraining AE on 1000 samples (epochs={ae_cfg.epochs}, lr={ae_cfg.lr})...')
key, sub = jax.random.split(key)
ae_result = train_ae_and_centroid(x_train, ae_cfg, sub)
trained_params = ae_result['ae_params']

# ── encode training set ──────────────────────────────────────────────────────
print('Encoding training latents...')
z_train = np.asarray(fetch_latent(
    trained_params, x_train,
    drop_p=0.0, rng_key=None, normalize=True, eps=1e-8,
))                                              # (1000, latent_dim)

# ── PCA on training latents via numpy SVD ────────────────────────────────────
latent_dim = z_train.shape[-1]
n_pcs = min(10, latent_dim)
print(f'Computing PCA({n_pcs}) on training latents (latent_dim={latent_dim})...')
z_mean = z_train.mean(axis=0, keepdims=True)
_, S, Vt = np.linalg.svd(z_train - z_mean, full_matrices=False)
pca_components = Vt[:n_pcs]                       # (n_pcs, latent_dim)
var_ratio = (S ** 2) / (S ** 2).sum()             # explained variance fraction
print(f'PC variance ratios (top {n_pcs}): {[f"{v:.2%}" for v in var_ratio[:n_pcs]]}')


def pca_transform(Y, n=None):
    if n is None:
        n = n_pcs
    return (Y - z_mean) @ pca_components[:n].T


z_train_pca = pca_transform(z_train)               # (1000, n_pcs)

# ── encode test sets ─────────────────────────────────────────────────────────
print('Encoding test latents...')
z_test_ssb = np.asarray(fetch_latent(
    trained_params, np.asarray(ssb_test['x_train']),
    drop_p=0.0, rng_key=None, normalize=True, eps=1e-8,
))
z_test_para = np.asarray(fetch_latent(
    trained_params, np.asarray(para_test['x_train']),
    drop_p=0.0, rng_key=None, normalize=True, eps=1e-8,
))
z_test_ssb_pca = pca_transform(z_test_ssb, n=2)
z_test_para_pca = pca_transform(z_test_para, n=2)

# ── plot 1: histograms of PC1..PC{n_pcs} by phase ────────────────────────────
print(f'\nPlotting PC1-PC{n_pcs} histograms...')
# layout: at most 5 columns; rows = ceil(n_pcs / 5)
ncols = min(5, n_pcs)
nrows = (n_pcs + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), dpi=300,
                         squeeze=False)
for i in range(nrows * ncols):
    ax = axes.flat[i]
    if i >= n_pcs:
        ax.axis('off')
        continue
    pc_ssb  = z_train_pca[:500, i]
    pc_para = z_train_pca[500:, i]
    lo = min(pc_ssb.min(), pc_para.min())
    hi = max(pc_ssb.max(), pc_para.max())
    bins = np.linspace(lo, hi, 30)
    ax.hist(pc_ssb,  bins=bins, alpha=0.55, color='tab:blue', label='SSB')
    ax.hist(pc_para, bins=bins, alpha=0.55, color='tab:red',  label='PARA')
    ax.set_title(f'PC{i+1} ({var_ratio[i]:.1%})', fontsize=11)
    ax.tick_params(labelsize=9)
    if i == 0:
        ax.legend(fontsize=9, loc='best')
fig.suptitle(f'Histograms of PC1–PC{n_pcs} by phase (500 SSB + 500 PARA training samples)',
             fontsize=13)
fig.tight_layout()
hist_path = f'../figures/phase_latent_pc_histograms_{ts}.pdf'
fig.savefig(hist_path, bbox_inches='tight')
print(f'Saved: {hist_path}')

# ── plot 2: PC1 vs PC2 scatter, dots colored by h ─────────────────────────────
print('Plotting PC1-PC2 scatter colored by h...')
h_train_ssb  = np.asarray(ssb_train['h_samples'])
h_train_para = np.asarray(para_train['h_samples'])
h_train_all  = np.concatenate([h_train_ssb, h_train_para])

h_test_ssb  = np.asarray(ssb_test['h_samples'])
h_test_para = np.asarray(para_test['h_samples'])
h_test_all  = np.concatenate([h_test_ssb, h_test_para])
z_test_all  = np.vstack([z_test_ssb_pca, z_test_para_pca])

# Single colormap spanning the full h range so train and test share it
vmin = float(min(h_train_all.min(), h_test_all.min()))
vmax = float(max(h_train_all.max(), h_test_all.max()))

fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=300)
sc_train = ax.scatter(
    z_train_pca[:, 0], z_train_pca[:, 1],
    c=h_train_all, cmap='viridis', vmin=vmin, vmax=vmax,
    alpha=0.55, s=18, label='train (1000)',
)
sc_test = ax.scatter(
    z_test_all[:, 0], z_test_all[:, 1],
    c=h_test_all, cmap='viridis', vmin=vmin, vmax=vmax,
    marker='X', s=55, edgecolor='black', lw=0.6, label='test (200)',
)
cb = fig.colorbar(sc_train, ax=ax)
cb.set_label('h', fontsize=12)
cb.ax.tick_params(labelsize=10)
# mark the phase boundary at h = -1 on the colorbar
cb.ax.axhline(-1.0, color='red', lw=1.5)
cb.ax.text(1.05, -1.0, ' h=−1 (boundary)', transform=cb.ax.get_yaxis_transform(),
           color='red', fontsize=9, va='center')

ax.set_xlabel(f'PC1 ({var_ratio[0]:.1%})', fontsize=14)
ax.set_ylabel(f'PC2 ({var_ratio[1]:.1%})', fontsize=14)
ax.tick_params(labelsize=12)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10)
scatter_path = f'../figures/phase_latent_pc12_scatter_{ts}.pdf'
fig.savefig(scatter_path, bbox_inches='tight')
print(f'Saved: {scatter_path}')

print('\nDone.')
