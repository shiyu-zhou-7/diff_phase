"""
Train a 'combined' AE on the pre-computed wavefunction data files
(data_ssb_debiased.pkl + data_para.pkl) and visualize its latent space:
    1. PCA on a 500 SSB + 500 PARA subsample of the trained latents.
    2. Histograms of the first 5 PCs by phase.
    3. PC1-PC2 scatter (clusters by phase).
    4. (Optional) overlay one or more active-phase-discovery trajectories
       by ED-solving at each trajectory h, encoding v[:, 0] with THIS
       combined AE (NOT the bundle's own AE), and projecting onto the
       same PC basis.

The active phase search pipeline is not touched; this is analysis-only.

Usage:
    python analyze_phase_latent_from_data.py [bundle1.pkl bundle2.pkl ...]

Env-vars:
    EPOCHS=<N>          override AE training epochs (smoke override)
    REUSE_AE=<path>     skip training; load AE+PCA from a previously saved
                        phase_latent_data_ae_*.pkl

Outputs (timestamped together):
    ../figures/phase_latent_data_pc_histograms_<ts>.pdf
    ../figures/phase_latent_data_pc12_scatter_<ts>.pdf
    ../data/phase_latent_data_ae_<ts>.pkl
"""
import sys
import os
import re
from dataclasses import replace, asdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap

import jax
from jax import config
config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import AEConfig, HamConfig
from hamiltonians.tfim import build_tfim_chain, gd_solver_ed
from models.autoencoder import fetch_latent
from training.bootstrap import train_ae_and_centroid
from utils.io import load_pickle, save_pickle, timestamp


# ── configs ──────────────────────────────────────────────────────────────────
ham_cfg = HamConfig()
ae_cfg = AEConfig()
if 'EPOCHS' in os.environ:
    ae_cfg = replace(ae_cfg, epochs=int(os.environ['EPOCHS']))

ts = timestamp()
print(f'Timestamp: {ts}')
print(f'ae_cfg.epochs={ae_cfg.epochs}, lr={ae_cfg.lr}, layer_widths={ae_cfg.layer_widths}')

# ── parse positional bundle args (trajectories to overlay) ───────────────────
# Each arg is either `path` (use all steps) or `path:N` (use first N steps).
def _parse_bundle_arg(arg):
    """Return (path, max_steps_or_None). `:N` suffix overrides max_steps."""
    if ':' in arg and arg.rsplit(':', 1)[1].isdigit():
        path, max_n = arg.rsplit(':', 1)
        return path, int(max_n)
    return arg, None


bundle_specs = [_parse_bundle_arg(a) for a in sys.argv[1:]]
bundle_paths = [p for p, _ in bundle_specs]  # kept for backward compatibility
if bundle_specs:
    print(f'Overlay trajectories from {len(bundle_specs)} bundle(s):')
    for p, n in bundle_specs:
        suffix = f' (first {n} steps)' if n is not None else ''
        print(f'  - {p}{suffix}')
else:
    print('No bundle paths given; figure will show only the SSB/PARA clusters.')

# ── load data files ──────────────────────────────────────────────────────────
SSB_PATH = '../data/data_ssb_debiased.pkl'
PARA_PATH = '../data/data_para.pkl'

print(f'\nLoading {SSB_PATH} and {PARA_PATH}...')
data_ssb = load_pickle(SSB_PATH)
data_para = load_pickle(PARA_PATH)
x_ssb = np.stack([np.asarray(d['v']) for d in data_ssb])    # (1000, 1024)
x_para = np.stack([np.asarray(d['v']) for d in data_para])  # (1000, 1024)
# Stored h is positive but corresponds to the codebase's negative-h Hamiltonian
# (sign convention memory: project_tfim_data_para_sign.md). Convert here.
h_ssb_codebase = -np.array([float(d['h']) for d in data_ssb])    # in [-0.999, 0]
h_para_codebase = -np.array([float(d['h']) for d in data_para])  # in [-2.0, -1.001]
h_all = np.concatenate([h_ssb_codebase, h_para_codebase])        # (2000,) in codebase units
print(f'  SSB samples: {x_ssb.shape}, PARA samples: {x_para.shape}')
print(f'  h range (codebase): [{h_all.min():.3f}, {h_all.max():.3f}]')

x_train = np.concatenate([x_ssb, x_para], axis=0)           # (2000, 1024)
n_ssb, n_para = x_ssb.shape[0], x_para.shape[0]
print(f'  Combined x_train: {x_train.shape}  (first {n_ssb} = SSB, next {n_para} = PARA)')

# ── train or load the combined AE ────────────────────────────────────────────
reuse_ae_path = os.environ.get('REUSE_AE')
if reuse_ae_path:
    print(f'\nREUSE_AE set; loading {reuse_ae_path}...')
    cached = load_pickle(reuse_ae_path)
    trained_params = cached['ae_params']
    z_all = np.asarray(cached['z_all'])
    z_mean = np.asarray(cached['z_mean'])
    pca_components = np.asarray(cached['pca_components'])
    var_ratio = np.asarray(cached['var_ratio'])
    idx_sub = np.asarray(cached['idx_sub'])
    actual_epochs = int(cached['cfg_snapshot']['ae_cfg']['epochs'])
    print(f'  Loaded ae_params (latent_dim={z_all.shape[-1]}, '
          f'trained on {actual_epochs} epochs); skipping training.')
else:
    print(f'\nTraining combined AE on {x_train.shape[0]} samples '
          f'(epochs={ae_cfg.epochs}, lr={ae_cfg.lr})...')
    rng_key = jax.random.PRNGKey(ae_cfg.seed)
    ae_result = train_ae_and_centroid(x_train, ae_cfg, rng_key)
    trained_params = ae_result['ae_params']

    print('Encoding all 2000 latents...')
    z_all = np.asarray(fetch_latent(
        trained_params, x_train,
        drop_p=0.0, rng_key=None, normalize=True, eps=1e-8,
    ))  # (2000, 10)

    # 500 SSB + 500 PARA subsample for PCA / histograms / scatter
    rng = np.random.default_rng(0)
    idx_ssb = rng.choice(n_ssb, size=500, replace=False)
    idx_para = rng.choice(n_para, size=500, replace=False) + n_ssb
    idx_sub = np.concatenate([idx_ssb, idx_para])

    z_sub = z_all[idx_sub]
    z_mean = z_sub.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(z_sub - z_mean, full_matrices=False)
    pca_components = Vt
    var_ratio = (S ** 2) / (S ** 2).sum()
    actual_epochs = int(ae_cfg.epochs)

# ── (always) compute the 500+500 PCA projection from cached or freshly-fit basis ──
z_sub = z_all[idx_sub]
z_sub_pca = (z_sub - z_mean) @ pca_components.T

latent_dim = z_all.shape[-1]
n_pcs_hist = min(5, latent_dim)
print(f'\nLatent dim = {latent_dim}; PC variance ratios (top {n_pcs_hist}):'
      f' {[f"{v:.2%}" for v in var_ratio[:n_pcs_hist]]}')

# ── plot 1: histograms of PC1..PC5 by phase ──────────────────────────────────
print(f'\nPlotting PC1-PC{n_pcs_hist} histograms (500 SSB + 500 PARA)...')
fig, axes = plt.subplots(1, n_pcs_hist, figsize=(3 * n_pcs_hist, 3), dpi=300, squeeze=False)
for i in range(n_pcs_hist):
    ax = axes[0, i]
    pc_ssb = z_sub_pca[:500, i]
    pc_para = z_sub_pca[500:, i]
    lo = min(pc_ssb.min(), pc_para.min())
    hi = max(pc_ssb.max(), pc_para.max())
    bins = np.linspace(lo, hi, 30)
    ax.hist(pc_ssb, bins=bins, alpha=0.55, color='tab:blue', label='SSB')
    ax.hist(pc_para, bins=bins, alpha=0.55, color='tab:red', label='PARA')
    ax.set_title(f'PC{i+1} ({var_ratio[i]:.1%})', fontsize=11)
    ax.tick_params(labelsize=9)
    if i == 0:
        ax.legend(fontsize=9, loc='best')
fig.suptitle(f'Histograms of PC1-PC{n_pcs_hist} by phase '
             f'(500 SSB + 500 PARA, combined-AE latents)', fontsize=12)
fig.tight_layout()
hist_path = f'../figures/phase_latent_data_pc_histograms_ep{actual_epochs}_{ts}.pdf'
fig.savefig(hist_path, bbox_inches='tight')
plt.close(fig)
print(f'  saved: {hist_path}')

# ── plot 2: PC1-PC2 scatter ──────────────────────────────────────────────────
# No bundles → single figure with only SSB/PARA clusters.
# N bundles    → N per-trajectory figures, each with a time-graded green trajectory
#                (start/end squares colored to match the trajectory ends, plus
#                direction arrows every ARROW_STRIDE kept points).

TRAJ_CMAP = LinearSegmentedColormap.from_list(
    'Oranges_trunc', plt.cm.Oranges(np.linspace(0.3, 1.0, 256)),
)
DATA_CMAP = LinearSegmentedColormap.from_list(
    'Blues_trunc', plt.cm.Blues(np.linspace(0.25, 1.0, 256)),
)

h_sub = h_all[idx_sub]                     # (1000,) — h for each subsample point
H_VMIN = float(h_all.min())                # full data range for consistent colorbar
H_VMAX = float(h_all.max())


def _draw_background(ax, fig):
    """Plot the 1000 sub-samples colored by h (codebase convention); add an h colorbar."""
    sc = ax.scatter(z_sub_pca[:, 0], z_sub_pca[:, 1],
                    c=h_sub, cmap=DATA_CMAP,
                    vmin=H_VMIN, vmax=H_VMAX,
                    s=22, alpha=0.7, zorder=1)
    ax.set_xlabel(f'PC1 ({var_ratio[0]:.1%})', fontsize=14)
    ax.set_ylabel(f'PC2 ({var_ratio[1]:.1%})', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax, orientation='vertical', fraction=0.04, pad=0.02)
    cbar.set_label(r'$h$', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    return sc


if not bundle_paths:
    print('\nPlotting PC1-PC2 scatter (no trajectory)...')
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=300)
    _draw_background(ax, fig)
    ax.legend(loc='best', fontsize=10, frameon=True)
    scatter_path = f'../figures/phase_latent_data_pc12_scatter_ep{actual_epochs}_{ts}.pdf'
    fig.savefig(scatter_path, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved: {scatter_path}')
else:
    ham_X, ham_ZZ = build_tfim_chain(ham_cfg.N, ham_cfg.J)
    arrow_stride = int(os.environ.get('ARROW_STRIDE', '5'))
    MAX_POINTS = 40

    def encode_h(h):
        """ED-solve at h; encode v[:, 0] with the combined AE."""
        _, v = gd_solver_ed(float(h), ham_X, ham_ZZ)
        z = fetch_latent(
            trained_params, v[:, 0],
            drop_p=0.0, rng_key=None, normalize=True, eps=1e-8,
        )
        return np.asarray(z)

    for b_idx, (path, max_steps) in enumerate(bundle_specs):
        suffix_msg = f' (first {max_steps} steps)' if max_steps is not None else ''
        print(f'\nPlotting PC1-PC2 scatter for bundle {b_idx}: {path}{suffix_msg}')
        bundle = load_pickle(path)
        h_per_step = bundle['history']['h_per_step']
        event_per_step = bundle['history']['event_per_step']
        if max_steps is not None:
            h_per_step = h_per_step[:max_steps]
            event_per_step = event_per_step[:max_steps]
        h_init = bundle['cfg_snapshot']['h_init']
        h_tag = f'{h_init:+.2f}'.replace('+', 'p').replace('-', 'm')
        # Append the bundle's own timestamp so two runs at the same h_init
        # don't collide in the output filename.
        basename = os.path.basename(path)
        m = re.search(r'(\d{8}_\d{6})', basename)
        if m:
            h_tag = f'{h_tag}_{m.group(1)}'
        if max_steps is not None:
            h_tag = f'{h_tag}_first{max_steps}'

        # Step-index filter + stride subsample (same as before).
        real_idx = [i for i, e in enumerate(event_per_step) if e in ('normal', 'nan_kick')]
        nan_idx_within_real = [k for k, i in enumerate(real_idx) if event_per_step[i] == 'nan_kick']
        stride = max(1, len(real_idx) // MAX_POINTS)
        keep = set(range(0, len(real_idx), stride))
        keep.update(nan_idx_within_real)
        if real_idx:
            keep.add(len(real_idx) - 1)
        keep = sorted(keep)

        traj_h_values = [float(h_per_step[real_idx[k]]) for k in keep]
        traj_event_tags = [event_per_step[real_idx[k]] for k in keep]
        print(f'  trajectory: {len(real_idx)} real entries -> {len(traj_h_values)} kept (stride={stride})')

        print(f'  encoding trajectory points through combined AE...')
        traj_latents = np.stack([encode_h(h) for h in traj_h_values])
        traj_pca = (traj_latents - z_mean) @ pca_components[:2].T
        n_traj = len(traj_pca)

        # Time-graded green colors. Start = lighter green, end = darker green.
        t_norm = np.linspace(0, 1, n_traj) if n_traj > 1 else np.array([0.5])
        colors_along = TRAJ_CMAP(t_norm)

        fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=300)
        _draw_background(ax, fig)

        # Line: LineCollection so each segment can carry its own color.
        if n_traj > 1:
            pts = traj_pca[:, np.newaxis, :]
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(segs, colors=colors_along[:-1], linewidths=2.5, zorder=3)
            ax.add_collection(lc)

        # Per-step dots, colored by time.
        ax.scatter(traj_pca[:, 0], traj_pca[:, 1],
                   c=colors_along, s=45, edgecolor='black', lw=0.4, zorder=4)

        # Start square — color matches the trajectory color at t=0.
        ax.scatter([traj_pca[0, 0]], [traj_pca[0, 1]],
                   c=[colors_along[0]], marker='s', s=110,
                   edgecolor='black', lw=1.0, zorder=10, label='start')
        # End square — color matches the trajectory color at t=1.
        ax.scatter([traj_pca[-1, 0]], [traj_pca[-1, 1]],
                   c=[colors_along[-1]], marker='s', s=110,
                   edgecolor='black', lw=1.0, zorder=10, label='end')

        # Direction arrows every ARROW_STRIDE kept points.
        for i in range(0, n_traj - 1, arrow_stride):
            j = min(i + 1, n_traj - 1)
            ax.annotate(
                '', xy=traj_pca[j], xytext=traj_pca[i],
                arrowprops=dict(arrowstyle='-|>',
                                color=colors_along[i],
                                lw=2.4, mutation_scale=32),
                zorder=5,
            )

        # NaN kicks (rare; emit red Xs if any).
        nan_traj_indices = [k for k, tag in enumerate(traj_event_tags) if tag == 'nan_kick']
        if nan_traj_indices:
            nx = [traj_pca[k, 0] for k in nan_traj_indices]
            ny = [traj_pca[k, 1] for k in nan_traj_indices]
            ax.scatter(nx, ny, c='red', marker='X', s=100, zorder=6, label='NaN kick')

        ax.set_title(f'$g_0 = {h_init:+.2f}$', fontsize=13)
        ax.legend(loc='best', fontsize=10, frameon=True)

        # Boxed annotation in the lower-right: initial + (truncated) final g.
        final_h = float(h_per_step[-1])
        g_text = f'$g_0 = {h_init:+.2f}$\n$g_{{\\mathrm{{final}}}} = {final_h:+.2f}$'
        ax.text(0.97, 0.03, g_text, transform=ax.transAxes,
                fontsize=11, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='white', edgecolor='black', lw=0.8),
                zorder=15)

        # Time colorbar at the bottom.
        sm = ScalarMappable(norm=Normalize(0, 1), cmap=TRAJ_CMAP)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='horizontal',
                            fraction=0.04, pad=0.12)
        cbar.set_label('step (0 → final)', fontsize=13, labelpad=-2)
        cbar.set_ticks([0, 1])
        cbar.ax.tick_params(labelsize=10)

        scatter_path = f'../figures/phase_latent_data_pc12_scatter_h{h_tag}_ep{actual_epochs}_{ts}.pdf'
        fig.savefig(scatter_path, bbox_inches='tight')
        plt.close(fig)
        print(f'  saved: {scatter_path}')

# ── save combined-AE artifact (for REUSE_AE on later runs) ───────────────────
if not reuse_ae_path:
    ae_out_path = f'../data/phase_latent_data_ae_ep{actual_epochs}_{ts}.pkl'
    save_pickle({
        'ae_params': trained_params,
        'z_all': z_all,
        'z_mean': z_mean,
        'pca_components': pca_components,
        'var_ratio': var_ratio,
        'idx_sub': idx_sub,
        'cfg_snapshot': {
            'ae_cfg': asdict(ae_cfg),
            'ham_cfg': asdict(ham_cfg),
            'n_ssb': n_ssb,
            'n_para': n_para,
        },
    }, ae_out_path)
    print(f'  saved: {ae_out_path}')

print('\nDone.')
