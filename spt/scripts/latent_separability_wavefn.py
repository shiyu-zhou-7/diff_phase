"""
Separability experiment: feed the RAW (gauge-fixed) ED wavefunction to the AE
instead of the 15-observable feature vector, and measure phase classification.

Motivation: the production pipeline feeds gauge-invariant observables, NOT the bare
eigenvector, because (1) eigh's sign of psi0 is arbitrary, (2) the sector vector is
2^{L-1}-dim (exponential in L), (3) the observables are smooth in t for the
differentiable discovery loop. This script is a CLASSIFICATION-ONLY probe (no
discovery loop) asking: does the full wavefunction separate phases better than the
15 observables? To neutralize reason (1) we GAUGE-FIX the sign of each ground state
(make its largest-magnitude amplitude positive) before feeding it in.

Reuses the SAME t-points / phase labels as the 15-observable run (the cached
spt_all_phases_*.pkl) so the comparison is head-to-head. Metric code is imported
from latent_separability.py.

Usage (from spt/scripts/):
    python latent_separability_wavefn.py
    LATENT_DIM=30 EPOCHS=6000 REBUILD=1 python latent_separability_wavefn.py
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
from hamiltonians.cluster import build_cluster_model, gd_solver_ed
from models.autoencoder import init_params, fetch_latent
from training.ae_train import train_autoencoder
from utils.io import load_pickle, save_pickle, timestamp
# reuse the metric + plotting helpers from the observable-space diagnostics
from latent_separability import (
    knn_accuracy, silhouette, confusion, lda_fit, PHASE_COLORS, _envint,
)


def gauge_fixed_wavefn(ts, model):
    """ED-solve each t and return the sign-gauge-fixed real ground state.

    The eigensolver's overall sign is arbitrary; we fix it deterministically by
    making the largest-magnitude amplitude positive so that the same physical
    state always maps to the same vector. Returns (N, 2^{L-1})."""
    rows = []
    for t in ts:
        _, psi = gd_solver_ed(jnp.asarray(t, dtype=jnp.float64), model)
        psi = np.real(np.asarray(psi)).ravel()
        k = int(np.argmax(np.abs(psi)))
        s = np.sign(psi[k])
        rows.append(psi * (s if s != 0 else 1.0))
    return np.stack(rows, axis=0)


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
    epochs = _envint('EPOCHS', ae_cfg.epochs)
    rebuild = bool(os.environ.get('REBUILD'))

    tag = f'd{d}_L{L}_{bc}_n{target_n}'
    # reuse the SAME t-points / labels as the observable-space run
    ds = load_pickle(f'../data/spt_all_phases_{tag}.pkl')
    train_t = np.asarray(ds['t'])
    phase = np.asarray(ds['phase']).astype(int)
    n_phases = len(np.unique(phase))

    # gauge-fixed wavefunction dataset (cached -- ED solving 1600 states is the cost)
    wf_cache = f'../data/spt_all_phases_wavefn_{tag}.pkl'
    if os.path.exists(wf_cache) and not rebuild:
        print(f'loading cached wavefunctions {wf_cache}')
        Xwf = load_pickle(wf_cache)['X']
    else:
        print(f'ED-solving {len(train_t)} gauge-fixed ground states (dim {2**(L-1)}) ...')
        Xwf = gauge_fixed_wavefn(train_t, model)
        save_pickle({'X': Xwf}, wf_cache)
        print(f'  wavefunctions -> {wf_cache}')
    D = Xwf.shape[1]
    print(f'wavefunction input dim = {D}  (= 2^(L-1)),  N = {len(phase)},  '
          f'latent_dim = {ae_cfg.latent_dim}  (chance = {1/n_phases:.3f})')

    # train AE on the wavefunctions (cached)
    ae_cache = f'../data/spt_combined_ae_wavefn_{tag}_z{ae_cfg.latent_dim}.pkl'
    if os.path.exists(ae_cache) and not rebuild:
        print(f'loading cached AE {ae_cache}')
        ae_params = load_pickle(ae_cache)['params']
    else:
        layers = [D, ae_cfg.hidden, ae_cfg.latent_dim, ae_cfg.hidden, D]
        print(f'training wavefunction AE  layers={layers}  epochs={epochs}')
        params0 = init_params(layers, jax.random.PRNGKey(ae_cfg.seed), scale=ae_cfg.init_scale)
        ae_params = train_autoencoder(
            params0, jnp.asarray(Xwf), epochs=epochs, lr=ae_cfg.lr,
            weight_decay=ae_cfg.weight_decay, drop_p=ae_cfg.dropout_p,
            center_coeff=ae_cfg.center_coeff, seed=ae_cfg.seed,
        )
        save_pickle({'params': ae_params, 'layers': layers, 'tag': tag}, ae_cache)
        print(f'  AE -> {ae_cache}')

    key = jax.random.PRNGKey(0)
    Z = np.asarray(fetch_latent(ae_params, jnp.asarray(Xwf), key))

    # ---- metrics: wavefunction-latent vs raw-wavefunction baseline ----
    print('\n=== separation metrics (gauge-fixed wavefunction input) ===')
    print(f'{"space":16s} {"kNN k=1":>9s} {"kNN k=5":>9s} {"silhouette":>11s}')
    z_acc, z_yte, z_pred = knn_accuracy(Z, phase)
    r_acc, _, _ = knn_accuracy(Xwf, phase)
    print(f'{"wavefn latent":16s} {z_acc[1]:9.3f} {z_acc[5]:9.3f} {silhouette(Z, phase):11.3f}')
    print(f'{"raw wavefn":16s} {r_acc[1]:9.3f} {r_acc[5]:9.3f} {silhouette(Xwf, phase):11.3f}')

    print('\n=== confusion matrix (wavefn latent, k=5; rows=true, cols=pred) ===')
    M = confusion(z_yte, z_pred, n_phases)
    print('      ' + ' '.join(f'{j:4d}' for j in range(n_phases)))
    for i in range(n_phases):
        print(f'  {i:2d} |' + ' '.join(f'{M[i, j]:4d}' for j in range(n_phases)))

    # ---- LDA(2) plot with the trajectory overlaid ----
    W, mu = lda_fit(Z, phase, k=2)
    Z2 = (Z - mu) @ W
    traj_t = np.asarray(b['history']['t_per_step'], dtype=float)
    traj_omega = np.asarray(b['history']['omega_per_step'], dtype=int)
    traj_Z = np.asarray(fetch_latent(ae_params, jnp.asarray(gauge_fixed_wavefn(traj_t, model)), key))
    T2 = (traj_Z - mu) @ W

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    for w in sorted(np.unique(phase)):
        m = phase == w
        ax.scatter(Z2[m, 0], Z2[m, 1], s=10, alpha=0.35,
                   color=PHASE_COLORS.get(int(w), '#333'), linewidths=0, label=f'$\\omega$={w}')
    ax.plot(T2[:, 0], T2[:, 1], '-', color='0.3', lw=1.1, alpha=0.85, zorder=4)
    ax.scatter(T2[:, 0], T2[:, 1], s=16, zorder=5, linewidths=0.3, edgecolor='k',
               color=[PHASE_COLORS.get(int(w), '#333') for w in traj_omega])
    ax.scatter(*T2[0], marker='o', s=150, facecolor='none', edgecolor='lime',
               linewidths=2.4, zorder=7, label='start')
    ax.scatter(*T2[-1], marker='X', s=150, color='k', zorder=7, label='end')
    ax.set_xlabel('LDA1'); ax.set_ylabel('LDA2')
    ax.set_title(f'gauge-fixed wavefunction AE latent, LDA(2)  (d={d}, L={L}, z={ae_cfg.latent_dim})')
    ax.legend(loc='best', fontsize=8, framealpha=0.9, ncol=2)

    ts = timestamp()
    out = f'../figures/spt_latent_lda_wavefn_{tag}_z{ae_cfg.latent_dim}_{ts}.png'
    os.makedirs('../figures', exist_ok=True)
    fig.subplots_adjust(left=0.11, right=0.97, top=0.93, bottom=0.11)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'\nLDA figure -> {out}')


if __name__ == '__main__':
    main()
