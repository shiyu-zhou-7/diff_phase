"""
Manuscript-styled PCA trajectory plot: the active-discovery trajectory in the
PCA(2) latent space of an AE trained on ALL-PHASE WAVEFUNCTIONS.

Same embedding pipeline as plot_trajectory_in_latent_wavefn.py (reuses its dataset
and AE caches), but rendered to match the other PCA trajectory figures in
manuscript/figures/ (trajectory_in_latent_437956.pdf, overlay_traj_*_phaselatent_*):

    figsize (6.5, 4.5), fixed margins, no tight bbox
    axes 'PC1 (xx.x%)' / 'PC2 (yy.y%)' with explained variance
    ticks 12 pt, x-label 15 pt, y-label 18 pt, legend 10 pt, no title
    phase-colored background scatter + legend; trajectory as a path with direction
    arrows; start = open square, end = filled triangle; param info-box lower-right.

Usage (from spt/scripts/):
    BUNDLE=../data/spt_active_d8_L10_pbc_20260714_085646.pkl TARGET_N=300 \
        python plot_trajectory_in_latent_manuscript.py
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
from training.dataset import generate_wavefunctions
from models.autoencoder import init_params, fetch_latent
from training.ae_train import train_autoencoder
from utils.io import load_pickle, save_pickle
from plot_trajectory_in_latent import pca_fit, pca_transform, PHASE_COLORS, _envint
from plot_trajectory_in_latent_wavefn import build_all_phases_wavefn_dataset, _envfloat

# manuscript house style (matches trajectory_in_latent_437956.pdf)
FS_TICK, FS_XLABEL, FS_YLABEL, FS_LEGEND, FS_BOX = 12, 15, 18, 10, 9
# shared axes-box margins used across manuscript/figures (identical frame everywhere)
MANUSCRIPT_MARGINS = dict(left=0.135, right=0.965, bottom=0.135, top=0.965)


def _medoid(P, cap=400):
    """Representative point of a cluster: the member minimizing total distance to
    the rest (subsampled to `cap` for speed). More robust than the mean when the
    points are scattered."""
    P = np.asarray(P)
    if len(P) > cap:
        P = P[np.linspace(0, len(P) - 1, cap).astype(int)]
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)).sum(1)
    return P[int(D.argmin())]


def lda_fit(Z, labels, k=2, reg=1e-3):
    """Supervised 2-D projection: top-k Fisher discriminants of Z given `labels`
    (numpy generalized eig, no sklearn). Returns (proj_2d, W, mean, evr) mirroring
    pca_fit; evr = share of between-class separation in each discriminant."""
    Z = np.asarray(Z, dtype=float); labels = np.asarray(labels)
    mean = Z.mean(0)
    D = Z.shape[1]
    Sw = np.zeros((D, D)); Sb = np.zeros((D, D))
    for c in np.unique(labels):
        Zc = Z[labels == c]
        mc = Zc.mean(0)
        Sw += (Zc - mc).T @ (Zc - mc)
        dc = (mc - mean).reshape(-1, 1)
        Sb += len(Zc) * (dc @ dc.T)
    Sw += reg * (np.trace(Sw) / D) * np.eye(D)          # ridge (latent is on a sphere)
    evals, evecs = np.linalg.eig(np.linalg.solve(Sw, Sb))
    evals, evecs = evals.real, evecs.real
    order = np.argsort(evals)[::-1]
    W = evecs[:, order[:k]]
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    evr = np.clip(evals[order], 0, None)
    evr = evr[:k] / evr.sum()
    return (Z - mean) @ W, W, mean, evr


def main():
    bundle_path = os.environ.get('BUNDLE')
    if not bundle_path:
        cands = [c for c in sorted(glob.glob('../data/spt_active_d8_*.pkl'))
                 if not c.endswith('_ckpt.pkl')] or \
                [c for c in sorted(glob.glob('../data/spt_active_*.pkl'))
                 if not c.endswith('_ckpt.pkl')]
        if not cands:
            raise SystemExit('no ../data/spt_active_*.pkl found')
        bundle_path = cands[-1]
    print(f'trajectory bundle: {bundle_path}')
    b = load_pickle(bundle_path)

    cc = b['cfg_snapshot']['cluster_cfg']
    d, L, bc = cc['d'], cc['L'], cc['bc']
    model = build_cluster_model(L=L, d=d, bc=bc, sector=cc['sector'], kappa=cc['kappa'])

    target_n = _envint('TARGET_N', 300)
    pool = _envint('POOL', 40000)
    seed = _envint('SEED', 0)
    ae_cfg = replace(AEConfig(), hidden=_envint('HIDDEN', 512),
                     latent_dim=_envint('LATENT_DIM', 128),
                     center_coeff=_envfloat('CENTER_COEFF', 0.0))
    epochs = _envint('EPOCHS', ae_cfg.epochs)
    rebuild = bool(os.environ.get('REBUILD'))

    tag = f'd{d}_L{L}_{bc}_n{target_n}'
    data_cache = f'../data/spt_all_phases_wavefn_disc_{tag}.pkl'
    ae_cache = f'../data/spt_combined_ae_wavefn_disc_{tag}_z{ae_cfg.latent_dim}_fid.pkl'

    # 1. all-phases wavefunction dataset (reuse cache from plot_trajectory_in_latent_wavefn)
    if os.path.exists(data_cache) and not rebuild:
        print(f'loading cached dataset {data_cache}')
        ds = load_pickle(data_cache)
        train_X, train_phase = ds['X'], ds['phase']
    else:
        print(f'building all-phases wavefunction dataset (d={d}, L={L}, target_n={target_n})')
        train_X, _, train_phase = build_all_phases_wavefn_dataset(model, d, target_n, pool, seed)
        save_pickle({'X': train_X, 't': _, 'phase': train_phase}, data_cache)
    Dim = train_X.shape[1]
    print(f'dataset: {train_X.shape[0]} points, phases '
          f'{sorted(set(int(p) for p in train_phase))}')

    # 2. all-phases AE (fidelity loss), reuse cache
    if os.path.exists(ae_cache) and not rebuild:
        print(f'loading cached AE {ae_cache}')
        ae_params = load_pickle(ae_cache)['params']
    else:
        layers = [Dim, ae_cfg.hidden, ae_cfg.latent_dim, ae_cfg.hidden, Dim]
        print(f'training AE layers={layers} epochs={epochs} loss=1-fidelity')
        params0 = init_params(layers, jax.random.PRNGKey(ae_cfg.seed), scale=ae_cfg.init_scale)
        ae_params = train_autoencoder(
            params0, jnp.asarray(train_X), epochs=epochs, lr=ae_cfg.lr,
            weight_decay=ae_cfg.weight_decay, drop_p=ae_cfg.dropout_p,
            center_coeff=ae_cfg.center_coeff, seed=ae_cfg.seed, loss='fidelity')
        save_pickle({'params': ae_params, 'layers': layers, 'tag': tag}, ae_cache)

    # 3. trajectory wavefunctions + encode
    traj_t = np.asarray(b['history']['t_per_step'], dtype=float)
    traj_omega = np.asarray(b['history']['omega_per_step'], dtype=int)
    print(f'encoding trajectory: {len(traj_t)} steps')
    traj_X, _ = generate_wavefunctions(traj_t, model)
    key = jax.random.PRNGKey(0)
    train_Z = np.asarray(fetch_latent(ae_params, jnp.asarray(train_X), key))
    traj_Z = np.asarray(fetch_latent(ae_params, jnp.asarray(traj_X), key))

    # 4. 2-D projection (PCA by default; LDA if PROJ=lda), fit on training latents
    proj = os.environ.get('PROJ', 'pca').lower()
    if proj == 'lda':
        train_2d, W, mean, evr = lda_fit(train_Z, train_phase, k=2)
        traj_2d = (traj_Z - mean) @ W
        axname = 'LD'
    else:
        train_2d, comp, mean, evr = pca_fit(train_Z, k=2)
        traj_2d = pca_transform(traj_Z, comp, mean)
        axname = 'PC'

    # ---- manuscript-styled figure ----
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fig.subplots_adjust(**MANUSCRIPT_MARGINS)

    phases = sorted(set(int(p) for p in train_phase))
    for w in phases:                                      # phase-colored background
        m = train_phase == w
        ax.scatter(train_2d[m, 0], train_2d[m, 1], s=10, alpha=0.30,
                   color=PHASE_COLORS.get(w, '#333'), linewidths=0,
                   label=f'$\\omega$={w}')

    # trajectory as a NUMBERED phase tour: one representative node per winding
    # (its medoid among the trajectory's points in that phase), linked in the
    # order the run first reached each phase so all visited windings appear.
    # Abstracts away the within-phase jitter that made the full path a web.
    order = []
    for w in traj_omega:
        if int(w) not in order:
            order.append(int(w))
    # one representative node per winding (its medoid), placed in discovery order;
    # if a node would land on an already-placed one, re-pick the same-phase point
    # farthest from the placed nodes so the markers don't overlap (e.g. omega=4
    # otherwise sits on the omega=2 end node)
    thr = 0.05 * float(np.hypot(train_2d[:, 0].ptp(), train_2d[:, 1].ptp()))
    nodes, placed = {}, []
    for w in order:
        P = traj_2d[traj_omega == w]
        cand = _medoid(P)
        if placed and min(float(np.hypot(*(cand - q))) for q in placed) < thr:
            Psub = P if len(P) <= 400 else P[np.linspace(0, len(P) - 1, 400).astype(int)]
            dd = np.array([min(float(np.hypot(*(p - q))) for q in placed) for p in Psub])
            cand = Psub[int(dd.argmax())]
        nodes[w] = cand
        placed.append(cand)
    end_w = int(traj_omega[-1])
    # simple path start -> ... -> end: visit phases in discovery order but move the
    # run's end phase to the terminus, so every dot has exactly one incoming and one
    # outgoing arrow (start: out only; end: in only) -- no node reused, no return arrow.
    path_order = [w for w in order if w != end_w] + [end_w]
    ppts = np.array([nodes[w] for w in path_order])
    _arrow = dict(arrowstyle='-|>', color='0.25', lw=1.3, alpha=0.85, mutation_scale=13)
    for j in range(len(path_order) - 1):
        ax.annotate('', xy=ppts[j + 1], xytext=ppts[j], zorder=5, arrowprops=_arrow)
    for j, w in enumerate(path_order):                    # phase nodes (no labels)
        if j == 0:
            mk, s, lab = 's', 130, 'start'
        elif j == len(path_order) - 1:
            mk, s, lab = '^', 150, 'end'
        else:
            mk, s, lab = 'o', 90, None
        ax.scatter(*nodes[w], marker=mk, s=s, zorder=7, linewidths=0.8,
                   facecolor=PHASE_COLORS.get(w, '#333'), edgecolor='k', label=lab)

    ax.set_xlabel(f'{axname}1 ({evr[0]*100:.1f}%)', fontsize=FS_XLABEL)
    ax.set_ylabel(f'{axname}2 ({evr[1]*100:.1f}%)', fontsize=FS_YLABEL)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.legend(loc='lower right', fontsize=FS_LEGEND, framealpha=0.9,
              handletextpad=0.3, labelspacing=0.3, borderpad=0.4)

    stem = (bundle_path.replace('../data/', '../figures/')
                       .replace('.pkl', f'_traj_in_latent_{proj}_manuscript'))
    os.makedirs('../figures', exist_ok=True)
    fig.savefig(stem + '.png', dpi=200)
    fig.savefig(stem + '.pdf')
    plt.close(fig)
    print(f'trajectory phases: {sorted(set(int(w) for w in traj_omega))}')
    print(f'PCA explained variance (PC1,PC2): {evr[0]*100:.1f}%, {evr[1]*100:.1f}%')
    print(f'figure -> {stem}.png / {stem}.pdf')


if __name__ == '__main__':
    main()
