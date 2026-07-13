"""
Plot an active-phase-discovery trajectory in the LATENT space of an autoencoder
trained on data from ALL phases of the cluster chain.

This is the latent-space companion to plot_trajectory.py (which plots the trajectory
in t-space). Pipeline, mirroring xxz_dmrg/scripts/plot_trajectory_in_latent.py:

  1. Build an all-phases dataset: sample t uniformly on the sphere S^{d-1}, label
     each by the ANALYTIC winding (free-fermion -- NO ED, so labelling a huge pool
     is free), then keep a balanced set per realized phase and ED-solve the feature
     vector only for the kept points.
  2. Train a fresh global AE on those features (separate from any active-discovery
     AE -- this one sees every phase).
  3. Load a single-start trajectory bundle (data/spt_active_*.pkl), re-solve the
     feature vector along its t_per_step trail, and encode it with the same AE.
  4. Fit PCA(2) on the TRAINING latents (numpy SVD -- no sklearn, per repo
     convention), project both, and plot: training points colored by winding phase
     with the trajectory overlaid (start / end / NaN-jump markers).

The dataset and trained AE are cached in ../data/ so reruns only redo the plot.

Usage (from spt/scripts/):
    python plot_trajectory_in_latent.py                       # newest spt_active_d8_*.pkl
    BUNDLE=../data/spt_active_d8_L10_pbc_XXXX.pkl python plot_trajectory_in_latent.py
    TARGET_N=300 EPOCHS=6000 REBUILD=1 python plot_trajectory_in_latent.py
"""
import sys, os, glob

import numpy as np
import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import replace

from configs.config import AEConfig
from hamiltonians.cluster import build_cluster_model
from training.dataset import generate_features, normalize_rows
from models.autoencoder import init_params, fetch_latent
from training.ae_train import train_autoencoder
from analytic.cluster_exact import winding, is_gapless
from utils.io import load_pickle, save_pickle, timestamp

# Consistent with plot_trajectory.py.
PHASE_COLORS = {0: '#1f77b4', 1: '#d62728', 2: '#2ca02c', 3: '#9467bd',
                4: '#ff7f0e', 5: '#8c564b', 6: '#e377c2', 7: '#7f7f7f'}


def _envint(name, default):
    return int(os.environ[name]) if name in os.environ else default


# ---------------------------------------------------------------------------
# PCA via numpy SVD (no sklearn -- repo convention)
# ---------------------------------------------------------------------------
def pca_fit(X, k=2):
    """Fit PCA(k) on X. Returns (proj, components (k,D), mean (1,D), evr (k,))."""
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comp = Vt[:k]
    var = (S ** 2) / max(len(X) - 1, 1)
    evr = var[:k] / var.sum()
    return Xc @ comp.T, comp, mean, evr


def pca_transform(X, comp, mean):
    return (np.asarray(X) - mean) @ comp.T


# ---------------------------------------------------------------------------
# All-phases dataset: balanced over realized winding phases (analytic labels)
# ---------------------------------------------------------------------------
def build_all_phases_dataset(model, d, target_n, pool, seed):
    """Sample `pool` points on S^{d-1}, label by analytic winding, keep up to
    `target_n` non-gapless points per realized phase, then ED-solve their
    feature vectors. Returns (X (N,D), t (N,d), phase (N,))."""
    rng = np.random.default_rng(seed)
    ts = normalize_rows(rng.standard_normal((pool, d)))
    # analytic labels for the whole pool -- cheap, no ED
    gap = np.array([is_gapless(t) for t in ts])
    lab = np.array([(-1 if g else int(winding(t))) for t, g in zip(ts, gap)])

    keep_t, keep_phase = [], []
    for w in sorted(set(lab) - {-1}):
        idx = np.where(lab == w)[0]
        take = idx[:target_n]
        keep_t.append(ts[take])
        keep_phase.append(np.full(len(take), w))
        if len(take) < target_n:
            print(f'  phase {w}: only {len(take)}/{target_n} found in pool '
                  f'(rare phase -- kept all available)')
        else:
            print(f'  phase {w}: {len(take)}')
    t_all = np.concatenate(keep_t, axis=0)
    phase_all = np.concatenate(keep_phase, axis=0)

    print(f'  ED-solving features for {len(t_all)} selected points ...')
    X, _ = generate_features(t_all, model)
    return np.asarray(X), t_all, phase_all


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    bundle_path = os.environ.get('BUNDLE')
    if not bundle_path:
        cands = sorted(glob.glob('../data/spt_active_d8_*.pkl')) or \
                sorted(glob.glob('../data/spt_active_*.pkl'))
        if not cands:
            raise SystemExit('no ../data/spt_active_*.pkl found; run main_active_phase.py first')
        bundle_path = cands[-1]
    print(f'trajectory bundle: {bundle_path}')
    b = load_pickle(bundle_path)

    cc = b['cfg_snapshot']['cluster_cfg']
    d, L, bc = cc['d'], cc['L'], cc['bc']
    model = build_cluster_model(L=L, d=d, bc=bc, sector=cc['sector'], kappa=cc['kappa'])

    target_n = _envint('TARGET_N', 200)
    pool = _envint('POOL', 40000)
    seed = _envint('SEED', 0)
    ae_cfg = AEConfig()
    ae_cfg = replace(ae_cfg, latent_dim=_envint('LATENT_DIM', ae_cfg.latent_dim))
    epochs = _envint('EPOCHS', ae_cfg.epochs)
    rebuild = bool(os.environ.get('REBUILD'))

    tag = f'd{d}_L{L}_{bc}_n{target_n}'
    data_cache = f'../data/spt_all_phases_{tag}.pkl'
    ae_cache = f'../data/spt_combined_ae_{tag}_z{ae_cfg.latent_dim}.pkl'

    # 1. all-phases dataset (cached)
    if os.path.exists(data_cache) and not rebuild:
        print(f'loading cached dataset {data_cache}')
        ds = load_pickle(data_cache)
        train_X, train_t, train_phase = ds['X'], ds['t'], ds['phase']
    else:
        print(f'building all-phases dataset (d={d}, L={L}, target_n={target_n}, pool={pool})')
        train_X, train_t, train_phase = build_all_phases_dataset(model, d, target_n, pool, seed)
        save_pickle({'X': train_X, 't': train_t, 'phase': train_phase}, data_cache)
        print(f'  dataset -> {data_cache}')

    D = train_X.shape[1]
    print(f'dataset: {train_X.shape[0]} points, feature dim {D}, '
          f'phases {sorted(set(int(p) for p in train_phase))}')

    # 2. train a fresh global AE (cached)
    if os.path.exists(ae_cache) and not rebuild:
        print(f'loading cached AE {ae_cache}')
        ae_params = load_pickle(ae_cache)['params']
    else:
        layers = [D, ae_cfg.hidden, ae_cfg.latent_dim, ae_cfg.hidden, D]
        print(f'training combined AE  layers={layers}  epochs={epochs}')
        params0 = init_params(layers, jax.random.PRNGKey(ae_cfg.seed), scale=ae_cfg.init_scale)
        ae_params = train_autoencoder(
            params0, jnp.asarray(train_X), epochs=epochs, lr=ae_cfg.lr,
            weight_decay=ae_cfg.weight_decay, drop_p=ae_cfg.dropout_p,
            center_coeff=ae_cfg.center_coeff, seed=ae_cfg.seed,
        )
        save_pickle({'params': ae_params, 'layers': layers, 'tag': tag}, ae_cache)
        print(f'  AE -> {ae_cache}')

    # 3. trajectory features (re-solve along the saved t-trail) + encode
    traj_t = np.asarray(b['history']['t_per_step'], dtype=float)
    traj_omega = np.asarray(b['history']['omega_per_step'], dtype=int)
    events = b['history']['event_per_step']
    print(f'encoding trajectory: {len(traj_t)} steps')
    traj_X, _ = generate_features(traj_t, model)

    key = jax.random.PRNGKey(0)
    train_Z = np.asarray(fetch_latent(ae_params, jnp.asarray(train_X), key))
    traj_Z = np.asarray(fetch_latent(ae_params, jnp.asarray(traj_X), key))

    # 4. PCA(2) fit on training latents, project both
    train_2d, comp, mean, evr = pca_fit(train_Z, k=2)
    traj_2d = pca_transform(traj_Z, comp, mean)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    for w in sorted(set(int(p) for p in train_phase)):
        m = train_phase == w
        ax.scatter(train_2d[m, 0], train_2d[m, 1], s=10, alpha=0.30,
                   color=PHASE_COLORS.get(w, '#333'), linewidths=0,
                   label=f'$\\omega$={w}')

    # trajectory polyline + per-step phase coloring
    ax.plot(traj_2d[:, 0], traj_2d[:, 1], '-', color='0.3', lw=1.1, alpha=0.85, zorder=4)
    ax.scatter(traj_2d[:, 0], traj_2d[:, 1], s=16, zorder=5, linewidths=0.3,
               edgecolor='k',
               color=[PHASE_COLORS.get(int(w), '#333') for w in traj_omega])
    ax.scatter(*traj_2d[0], marker='o', s=150, facecolor='none', edgecolor='lime',
               linewidths=2.4, zorder=7, label='start')
    ax.scatter(*traj_2d[-1], marker='X', s=150, color='k', zorder=7, label='end')
    nan_idx = [i for i, e in enumerate(events) if e == 'nan_jump']
    if nan_idx:
        ax.scatter(traj_2d[nan_idx, 0], traj_2d[nan_idx, 1], marker='x', s=70,
                   color='magenta', zorder=6, label='NaN jump')

    ax.set_xlabel(f'latent PC1 ({evr[0]*100:.0f}% var)')
    ax.set_ylabel(f'latent PC2 ({evr[1]*100:.0f}% var)')
    ax.set_title(f'trajectory in all-phases AE latent space  (d={d}, L={L})')
    ax.legend(loc='best', fontsize=8, framealpha=0.9, ncol=2)

    ts = timestamp()
    out = f'../figures/spt_trajectory_in_latent_{tag}_{ts}.png'
    os.makedirs('../figures', exist_ok=True)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.93, bottom=0.11)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'trajectory phases: {sorted(set(int(w) for w in traj_omega))}')
    print(f'PCA explained variance (PC1,PC2): {evr[0]:.3f}, {evr[1]:.3f}')
    print(f'figure -> {out}')


if __name__ == '__main__':
    main()
