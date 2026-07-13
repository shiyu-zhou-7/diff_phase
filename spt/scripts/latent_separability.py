"""
Quantify how well the all-phases AE latent space separates the winding phases.

The PCA(2) scatter in plot_trajectory_in_latent.py can hide separation that lives
along the latent directions PCA drops (it maximizes variance, not class margin).
This script measures separation in the FULL latent space and contrasts it with a
supervised LDA(2) view.

Metrics (phase label = analytic winding; all numpy, no sklearn per repo convention):
  * k-NN classification accuracy (k=1, k=5) on a held-out split -- the headline
    cluster-separability number (chance = 1/n_phases).
  * silhouette score (mean over points; intra- vs nearest inter-cluster distance).
  * confusion matrix (which phases get mixed up).
  * the SAME metrics on the RAW feature vectors, as a baseline -- does the AE add
    separation over the inputs it was built from?

Plot: LDA(2) projection of the training latents colored by phase (the trajectory
from the bundle overlaid in the same frame), saved next to the PCA figure.

Usage (from spt/scripts/):
    python latent_separability.py
    BUNDLE=../data/spt_active_d8_L10_pbc_XXXX.pkl TARGET_N=200 python latent_separability.py
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
from training.dataset import generate_features
from models.autoencoder import fetch_latent
from utils.io import load_pickle, timestamp

PHASE_COLORS = {0: '#1f77b4', 1: '#d62728', 2: '#2ca02c', 3: '#9467bd',
                4: '#ff7f0e', 5: '#8c564b', 6: '#e377c2', 7: '#7f7f7f'}


def _envint(name, default):
    return int(os.environ[name]) if name in os.environ else default


# ---------------------------------------------------------------------------
# Metrics (numpy only)
# ---------------------------------------------------------------------------
def _pairwise_sq_dists(A, B):
    """(|a-b|^2) matrix for rows of A (n,D) vs B (m,D)."""
    a2 = np.sum(A * A, axis=1, keepdims=True)
    b2 = np.sum(B * B, axis=1, keepdims=True).T
    return np.maximum(a2 + b2 - 2.0 * A @ B.T, 0.0)


def knn_accuracy(X, y, seed=0, test_frac=0.3, ks=(1, 5)):
    """Stratified train/test split; k-NN (Euclidean) accuracy for each k.
    Returns (acc_dict, y_test, pred_for_max_k) so a confusion matrix can be built."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        ncut = int(round(len(idx) * (1 - test_frac)))
        tr.append(idx[:ncut]); te.append(idx[ncut:])
    tr = np.concatenate(tr); te = np.concatenate(te)
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    d2 = _pairwise_sq_dists(Xte, Xtr)            # (n_test, n_train)
    order = np.argsort(d2, axis=1)
    acc, pred_max = {}, None
    for k in ks:
        nn = ytr[order[:, :k]]                    # (n_test, k)
        # majority vote (ties -> smallest label)
        pred = np.array([np.bincount(row, minlength=int(y.max()) + 1).argmax()
                         for row in nn])
        acc[k] = float(np.mean(pred == yte))
        if k == max(ks):
            pred_max = pred
    return acc, yte, pred_max


def silhouette(X, y):
    """Mean silhouette over all points (labels = clusters)."""
    D = np.sqrt(_pairwise_sq_dists(X, X))
    classes = np.unique(y)
    s = np.zeros(len(X))
    for i in range(len(X)):
        same = (y == y[i]); same[i] = False
        a = D[i, same].mean() if same.any() else 0.0
        b = np.inf
        for c in classes:
            if c == y[i]:
                continue
            m = (y == c)
            if m.any():
                b = min(b, D[i, m].mean())
        s[i] = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
    return float(s.mean())


def confusion(y_true, y_pred, n):
    M = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[int(t), int(p)] += 1
    return M


# ---------------------------------------------------------------------------
# LDA(2) (numpy generalized eigenproblem)
# ---------------------------------------------------------------------------
def lda_fit(X, y, k=2, reg=1e-6):
    """Fisher LDA. Returns (W (D,k), mean (1,D)). Projects to directions that
    maximize between- vs within-class scatter."""
    X = np.asarray(X)
    mu = X.mean(axis=0, keepdims=True)
    D = X.shape[1]
    Sw = np.zeros((D, D)); Sb = np.zeros((D, D))
    for c in np.unique(y):
        Xc = X[y == c]
        muc = Xc.mean(axis=0, keepdims=True)
        Sw += (Xc - muc).T @ (Xc - muc)
        diff = (muc - mu)
        Sb += len(Xc) * (diff.T @ diff)
    A = np.linalg.solve(Sw + reg * np.eye(D), Sb)
    evals, evecs = np.linalg.eig(A)
    evals, evecs = np.real(evals), np.real(evecs)
    order = np.argsort(evals)[::-1][:k]
    W = evecs[:, order]
    W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-12
    return W, mu


def main():
    bundle_path = os.environ.get('BUNDLE')
    if not bundle_path:
        cands = sorted(glob.glob('../data/spt_active_d8_*.pkl')) or \
                sorted(glob.glob('../data/spt_active_*.pkl'))
        bundle_path = cands[-1]
    b = load_pickle(bundle_path)
    cc = b['cfg_snapshot']['cluster_cfg']
    d, L, bc = cc['d'], cc['L'], cc['bc']
    model = build_cluster_model(L=L, d=d, bc=bc, sector=cc['sector'], kappa=cc['kappa'])

    ae_cfg = AEConfig()
    ae_cfg = replace(ae_cfg, latent_dim=_envint('LATENT_DIM', ae_cfg.latent_dim))
    target_n = _envint('TARGET_N', 200)
    tag = f'd{d}_L{L}_{bc}_n{target_n}'
    ds = load_pickle(f'../data/spt_all_phases_{tag}.pkl')
    ae_params = load_pickle(f'../data/spt_combined_ae_{tag}_z{ae_cfg.latent_dim}.pkl')['params']

    X_raw = np.asarray(ds['X'])
    phase = np.asarray(ds['phase']).astype(int)
    n_phases = len(np.unique(phase))
    key = jax.random.PRNGKey(0)
    Z = np.asarray(fetch_latent(ae_params, jnp.asarray(X_raw), key))

    # ---- metrics: latent vs raw baseline ----
    print(f'dataset: {len(phase)} points, {n_phases} phases, '
          f'latent_dim={Z.shape[1]}, raw_dim={X_raw.shape[1]}  (chance={1/n_phases:.3f})')
    print('\n=== separation metrics (full space) ===')
    print(f'{"space":8s} {"kNN k=1":>9s} {"kNN k=5":>9s} {"silhouette":>11s}')
    z_acc, z_yte, z_pred = knn_accuracy(Z, phase)
    r_acc, _, _ = knn_accuracy(X_raw, phase)
    z_sil = silhouette(Z, phase)
    r_sil = silhouette(X_raw, phase)
    print(f'{"latent":8s} {z_acc[1]:9.3f} {z_acc[5]:9.3f} {z_sil:11.3f}')
    print(f'{"raw":8s} {r_acc[1]:9.3f} {r_acc[5]:9.3f} {r_sil:11.3f}')

    print('\n=== confusion matrix (latent, k=5; rows=true, cols=pred) ===')
    M = confusion(z_yte, z_pred, n_phases)
    hdr = '      ' + ' '.join(f'{j:4d}' for j in range(n_phases))
    print(hdr)
    for i in range(n_phases):
        print(f'  {i:2d} |' + ' '.join(f'{M[i, j]:4d}' for j in range(n_phases)))

    # ---- LDA(2) plot ----
    W, mu = lda_fit(Z, phase, k=2)
    Z2 = (Z - mu) @ W

    traj_t = np.asarray(b['history']['t_per_step'], dtype=float)
    traj_omega = np.asarray(b['history']['omega_per_step'], dtype=int)
    traj_X, _ = generate_features(traj_t, model)
    traj_Z = np.asarray(fetch_latent(ae_params, jnp.asarray(traj_X), key))
    T2 = (traj_Z - mu) @ W

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    for w in sorted(np.unique(phase)):
        m = phase == w
        ax.scatter(Z2[m, 0], Z2[m, 1], s=10, alpha=0.35,
                   color=PHASE_COLORS.get(int(w), '#333'), linewidths=0,
                   label=f'$\\omega$={w}')
    ax.plot(T2[:, 0], T2[:, 1], '-', color='0.3', lw=1.1, alpha=0.85, zorder=4)
    ax.scatter(T2[:, 0], T2[:, 1], s=16, zorder=5, linewidths=0.3, edgecolor='k',
               color=[PHASE_COLORS.get(int(w), '#333') for w in traj_omega])
    ax.scatter(*T2[0], marker='o', s=150, facecolor='none', edgecolor='lime',
               linewidths=2.4, zorder=7, label='start')
    ax.scatter(*T2[-1], marker='X', s=150, color='k', zorder=7, label='end')

    ax.set_xlabel('LDA1'); ax.set_ylabel('LDA2')
    ax.set_title(f'all-phases AE latent, LDA(2) supervised view  (d={d}, L={L})')
    ax.legend(loc='best', fontsize=8, framealpha=0.9, ncol=2)

    ts = timestamp()
    out = f'../figures/spt_latent_lda_{tag}_{ts}.png'
    os.makedirs('../figures', exist_ok=True)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.93, bottom=0.11)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'\nLDA figure -> {out}')


if __name__ == '__main__':
    main()
