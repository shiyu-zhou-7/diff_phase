"""
Incremental phase-separation sweep on the full-Hilbert wavefunction AE.

Question: starting with the two phases {w=0, w=1}, can the AE latent separate them
well? Then add w=2, then w=3, ... -- at how many phases does separation break down?

For each cumulative subset {0, 1, ..., k-1} (k = 2..d) we train a FRESH AE with the
SAME layers [1024, 512, 128, 512, 1024], 1-fidelity loss, center_coeff=0, on only
those phases' wavefunctions, then measure how well the latent separates them
(kNN k=1/k=5 accuracy, silhouette) against the raw-wavefunction baseline. Chance
accuracy is 1/k, so we report accuracy AND chance per subset.

Reuses the cached FULL-Hilbert wavefunctions produced by classify_wavefn_full.py
(../data/spt_wavefn_full_<tag>.pkl) -- no new ED solves.

Usage (from spt/scripts/):
    D=8 L=10 python phase_separation_sweep.py
    EPOCHS=6000 REBUILD=1 python phase_separation_sweep.py
env knobs: D L BC TARGET_N EPOCHS LATENT_DIM HIDDEN CENTER_COEFF REBUILD
"""
import sys, os

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
from models.autoencoder import init_params, fetch_latent
from training.ae_train import train_autoencoder
from utils.io import load_pickle, save_pickle, timestamp
from latent_separability import knn_accuracy, silhouette, confusion, _envint, PHASE_COLORS
from plot_trajectory_in_latent import pca_fit


def _envfloat(name, default):
    return float(os.environ[name]) if name in os.environ else default


def train_ae_on(Xsub, ae_cfg, epochs, cache):
    """Train (or load) a fresh AE on the subset wavefunctions Xsub (N, Dim)."""
    if os.path.exists(cache) and not bool(os.environ.get('REBUILD')):
        print(f'    loading cached AE {cache}')
        return load_pickle(cache)['params']
    Dim = Xsub.shape[1]
    layers = [Dim, ae_cfg.hidden, ae_cfg.latent_dim, ae_cfg.hidden, Dim]
    params0 = init_params(layers, jax.random.PRNGKey(ae_cfg.seed), scale=ae_cfg.init_scale)
    params = train_autoencoder(
        params0, jnp.asarray(Xsub), epochs=epochs, lr=ae_cfg.lr,
        weight_decay=ae_cfg.weight_decay, drop_p=ae_cfg.dropout_p,
        center_coeff=ae_cfg.center_coeff, seed=ae_cfg.seed, loss='fidelity',
        log_every=0,
    )
    save_pickle({'params': params, 'layers': layers}, cache)
    return params


def main():
    d = _envint('D', 8)
    L = _envint('L', 10)
    bc = os.environ.get('BC', 'pbc')
    target_n = _envint('TARGET_N', 200)

    ae_cfg = AEConfig()
    ae_cfg = replace(
        ae_cfg,
        hidden=_envint('HIDDEN', 512),
        latent_dim=_envint('LATENT_DIM', 128),
        center_coeff=_envfloat('CENTER_COEFF', 0.0),
    )
    epochs = _envint('EPOCHS', ae_cfg.epochs)

    # SECTOR=1: use the P=+1-sector ground state lifted to full 2^L (unique, smooth);
    # default: arbitrary full-space eigh. Must match classify_wavefn_full.py's cache.
    kind = 'sectorlift' if bool(os.environ.get('SECTOR')) else 'full'
    t_sign = os.environ.get('T_SIGN', 'free')   # must match classify_wavefn_full.py
    sign_tag = '' if t_sign == 'free' else f'_{t_sign}t'
    dom_tag = '_dom' if bool(os.environ.get('DOMINANT')) else ''

    tag = f'd{d}_L{L}_{bc}_n{target_n}{sign_tag}{dom_tag}'
    wf_cache = f'../data/spt_wavefn_{kind}_{tag}.pkl'
    if not os.path.exists(wf_cache):
        raise SystemExit(f'missing {wf_cache}; run classify_wavefn_full.py '
                         f'(SECTOR={"1" if kind == "sectorlift" else "0"}) first to '
                         f'build the {kind} wavefunctions')
    ds = load_pickle(wf_cache)
    Xall = np.asarray(ds['X'])
    phase = np.asarray(ds['phase']).astype(int)
    all_phases = sorted(np.unique(phase).tolist())
    print(f'{kind} wavefunction sweep  (d={d}, L={L}, bc={bc}, '
          f'input dim {Xall.shape[1]}, latent {ae_cfg.latent_dim}, '
          f'loss=1-fidelity, center_coeff={ae_cfg.center_coeff})')
    print(f'available phases: {all_phases}\n')

    key = jax.random.PRNGKey(0)
    rows = []  # (k, chance, lat_k1, lat_k5, lat_sil, raw_k1, raw_k5, raw_sil)
    confusions = {}
    panels = []  # (k, keep, Z_2d, yk, evr) for the per-run PCA(2) grid
    for k in range(2, len(all_phases) + 1):
        keep = all_phases[:k]
        m = np.isin(phase, keep)
        Xk, yk = Xall[m], phase[m]
        chance = 1.0 / k
        print(f'--- phases {keep}  (N={len(yk)}, chance={chance:.3f}) ---')

        cache = f'../data/spt_ae_wavefn_{kind}_sweep_k{k}_{tag}_z{ae_cfg.latent_dim}_fid.pkl'
        params = train_ae_on(Xk, ae_cfg, epochs, cache)
        Z = np.asarray(fetch_latent(params, jnp.asarray(Xk), key))

        z_acc, z_yte, z_pred = knn_accuracy(Z, yk)
        r_acc, _, _ = knn_accuracy(Xk, yk)
        z_sil, r_sil = silhouette(Z, yk), silhouette(Xk, yk)
        rows.append((k, chance, z_acc[1], z_acc[5], z_sil, r_acc[1], r_acc[5], r_sil))
        confusions[k] = (confusion(z_yte, z_pred, max(all_phases) + 1), keep)
        Z2d, _, _, evr = pca_fit(Z, k=2)   # PCA(2) on this run's latent (numpy SVD)
        panels.append((k, keep, Z2d, yk, evr))
        print(f'    latent  kNN1={z_acc[1]:.3f}  kNN5={z_acc[5]:.3f}  sil={z_sil:+.3f}')
        print(f'    raw     kNN1={r_acc[1]:.3f}  kNN5={r_acc[5]:.3f}  sil={r_sil:+.3f}\n')

    # ---- per-run PCA(2) latent scatter grid ----
    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    figp, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (k, keep, Z2d, yk, evr) in zip(axes, panels):
        for w in keep:
            mm = yk == w
            ax.scatter(Z2d[mm, 0], Z2d[mm, 1], s=8, alpha=0.45, linewidths=0,
                       color=PHASE_COLORS.get(int(w), '#333'), label=f'$\\omega$={w}')
        ax.set_title(f'{len(keep)} phases {keep}', fontsize=9)
        ax.set_xlabel(f'PC1 ({evr[0]*100:.0f}%)', fontsize=8)
        ax.set_ylabel(f'PC2 ({evr[1]*100:.0f}%)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(loc='best', fontsize=6, framealpha=0.85, ncol=2)
    for ax in axes[len(panels):]:
        ax.axis('off')
    figp.suptitle(f'latent PCA(2) per cumulative-phase run  '
                  f'({kind} wavefn, d={d}, L={L})', fontsize=11)
    ts = timestamp()
    outp = f'../figures/spt_phase_sweep_pca_panels_{kind}_{tag}_z{ae_cfg.latent_dim}_{ts}.png'
    os.makedirs('../figures', exist_ok=True)
    figp.tight_layout(rect=(0, 0, 1, 0.97))
    figp.savefig(outp, dpi=150)
    plt.close(figp)
    print(f'PCA panel grid -> {outp}\n')

    # ---- summary table ----
    print('=== summary: separation vs number of phases (latent / raw) ===')
    print(f'{"#phases":>7s} {"chance":>7s} {"L-kNN1":>7s} {"L-kNN5":>7s} {"L-sil":>7s} '
          f'{"R-kNN1":>7s} {"R-kNN5":>7s} {"R-sil":>7s}')
    for (k, ch, zk1, zk5, zs, rk1, rk5, rs) in rows:
        print(f'{k:7d} {ch:7.3f} {zk1:7.3f} {zk5:7.3f} {zs:+7.3f} '
              f'{rk1:7.3f} {rk5:7.3f} {rs:+7.3f}')

    # ---- plot: kNN k=5 accuracy vs #phases (latent vs raw vs chance) ----
    ks = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(ks, [r[3] for r in rows], 'o-', color='#1f77b4', label='AE latent (kNN k=5)')
    ax.plot(ks, [r[6] for r in rows], 's--', color='#888', label='raw wavefn (kNN k=5)')
    ax.plot(ks, [r[1] for r in rows], ':', color='#d62728', label='chance = 1/k')
    ax.set_xlabel('number of cumulative phases  {0,...,k-1}')
    ax.set_ylabel('kNN (k=5) classification accuracy')
    ax.set_title(f'phase-separation breakdown  ({kind} wavefn, d={d}, L={L})')
    ax.set_ylim(0, 1.02); ax.set_xticks(ks)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    ts = timestamp()
    out = f'../figures/spt_phase_separation_sweep_{kind}_{tag}_z{ae_cfg.latent_dim}_{ts}.png'
    os.makedirs('../figures', exist_ok=True)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.93, bottom=0.12)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'\nfigure -> {out}')


if __name__ == '__main__':
    main()
