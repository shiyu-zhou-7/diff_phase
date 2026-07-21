"""
Manuscript-styled single-panel t-space (parameter-space) trajectory plot.

Same content as the LEFT panel of plot_trajectory.py -- the active-discovery
t-trajectory projected with the top-2 PCs of the d pure-point corners e_alpha
(numpy SVD, no sklearn), over a faint winding-colored sphere-sample backdrop,
corners drawn as labeled stars, trajectory points colored by their winding,
start = green open circle, end = black X -- but rendered as ONE panel at the
paper's figure/font style (matches manuscript/figures/trajectory_in_latent_437956):

    figsize (6.5, 4.5), fixed margins, no tight bbox, no title
    ticks 12 pt, x-label 15 pt, y-label 18 pt, legend 10 pt.

Usage (from spt/scripts/):
    BUNDLE=../data/spt_active_d8_L10_pbc_20260714_085646.pkl \
        python plot_trajectory_tspace_manuscript.py
"""
import sys, os, glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytic.cluster_exact import winding, is_gapless
from utils.io import load_pickle
from plot_trajectory import corner_basis, project, PHASE_COLORS, PHASE_NAME

# manuscript house style (matches trajectory_in_latent_437956.pdf)
FS_TICK, FS_XLABEL, FS_YLABEL, FS_LEGEND = 12, 15, 18, 10


def main():
    bundle_path = os.environ.get('BUNDLE')
    if not bundle_path:
        cands = [c for c in sorted(glob.glob('../data/spt_active_*.pkl'))
                 if not c.endswith('_ckpt.pkl')]
        if not cands:
            raise SystemExit('no ../data/spt_active_*.pkl found')
        bundle_path = cands[-1]
    print(f'loading {bundle_path}')
    b = load_pickle(bundle_path)

    traj = np.asarray(b['history']['t_per_step'], dtype=float)
    omega = np.asarray(b['history']['omega_per_step'], dtype=int)
    events = b['history']['event_per_step']
    d = traj.shape[1]
    L = b['cfg_snapshot']['cluster_cfg']['L']
    mean, basis = corner_basis(d)
    # explained-variance ratio of the corner-simplex PCA (the basis these axes are
    # fit on), so the labels carry a percentage like the manuscript latent-PCA
    # figures. For the regular d-corner simplex all d-1 PCs are equal (1/(d-1)).
    _S = np.linalg.svd(np.eye(d) - np.full((d, d), 1.0 / d), compute_uv=False)
    cevr = _S ** 2 / (_S ** 2).sum()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fig.subplots_adjust(left=0.155, right=0.975, top=0.975, bottom=0.135)

    # faint winding-colored sphere-sample backdrop
    rng = np.random.default_rng(0)
    bg = rng.standard_normal((6000, d))
    bg /= np.linalg.norm(bg, axis=1, keepdims=True)
    bg_w = np.array([(-1 if is_gapless(t) else winding(t)) for t in bg])
    bg2 = project(bg, mean, basis)
    for w in sorted(set(bg_w) - {-1}):
        m = bg_w == w
        ax.scatter(bg2[m, 0], bg2[m, 1], s=6, alpha=0.12,
                   color=PHASE_COLORS.get(w, '#333'), linewidths=0)

    # corners e_alpha as labeled stars -- omitted from the paper PCA figure (the
    # manuscript trajectory figures show no corner/bootstrap markers). Kept
    # commented for easy revert.
    # corners2 = project(np.eye(d), mean, basis)
    # for a in range(d):
    #     ax.scatter(*corners2[a], marker='*', s=220, edgecolor='k',
    #                facecolor=PHASE_COLORS.get(a, '#333'), zorder=5)
    #     ax.annotate(f'$e_{a}$', corners2[a], textcoords='offset points',
    #                 xytext=(6, 6), fontsize=11, fontweight='bold')

    # trajectory: faint path + points colored by winding
    tr2 = project(traj, mean, basis)
    ax.plot(tr2[:, 0], tr2[:, 1], '-', color='0.35', lw=1.0, alpha=0.8, zorder=3)
    for w in sorted(set(omega)):
        m = omega == w
        ax.scatter(tr2[m, 0], tr2[m, 1], s=14, color=PHASE_COLORS.get(w, '#333'),
                   zorder=4, label=PHASE_NAME.get(w, f'$\\omega$={w}'))
    ax.scatter(*tr2[0], marker='s', s=110,
               facecolor=PHASE_COLORS.get(int(omega[0]), '#333'), edgecolor='k',
               linewidths=0.8, zorder=7, label='start')
    ax.scatter(*tr2[-1], marker='^', s=140,
               facecolor=PHASE_COLORS.get(int(omega[-1]), '#333'), edgecolor='k',
               linewidths=0.8, zorder=7, label='end')
    nan_idx = [i for i, e in enumerate(events) if e == 'nan_jump']
    if nan_idx:
        ax.scatter(tr2[nan_idx, 0], tr2[nan_idx, 1], marker='x', s=70,
                   color='magenta', zorder=6, label='NaN jump')

    ax.set_xlabel(f'PC1 ({cevr[0]*100:.1f}%)', fontsize=FS_XLABEL)
    ax.set_ylabel(f'PC2 ({cevr[1]*100:.1f}%)', fontsize=FS_YLABEL)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.set_xlim(-1.05, 1.05)
    ax.legend(loc='lower right', fontsize=FS_LEGEND, framealpha=0.9,
              handletextpad=0.3, labelspacing=0.3, borderpad=0.4)

    stem = (bundle_path.replace('../data/', '../figures/')
                       .replace('.pkl', '_tspace_manuscript'))
    os.makedirs('../figures', exist_ok=True)
    fig.savefig(stem + '.png', dpi=200)
    fig.savefig(stem + '.pdf')
    plt.close(fig)
    print(f'distinct phases: {sorted(set(int(w) for w in omega))}  (d={d}, L={L})')
    print(f'figure -> {stem}.png / {stem}.pdf')


if __name__ == '__main__':
    main()
