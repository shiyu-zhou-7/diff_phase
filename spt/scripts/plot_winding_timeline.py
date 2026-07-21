"""
Standalone winding-timeline figure for an active-phase-discovery run.

A single clean panel: winding omega vs (normalized) discovery step, colored by
phase, with block/retrain boundaries marked. Styled to match the manuscript
figure `manuscript/figures/combined_active_phase_trajectory_*.pdf`:

    figsize = (6.5, 4.5)     tick labels = 12 pt
    x-label = 15 pt          y-label = 18 pt          legend = 10 pt
    start = square, retrain = star, end = triangle; light grid; no tight bbox.

Block / retrain boundaries are recovered from query_per_step (a retrain injects a
batch of bootstrap solves with no optimization step, so query_per_step jumps by
>1 there; a normal step adds exactly 1). numpy + matplotlib only.

Usage (from spt/scripts/):
    python plot_winding_timeline.py                         # newest bundle
    python plot_winding_timeline.py ../data/spt_active_d8_....pkl
    BUNDLE=../data/spt_active_d6_....pkl python plot_winding_timeline.py
"""
import sys, os, glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.io import load_pickle

PHASE_COLORS = {0: '#1f77b4', 1: '#d62728', 2: '#2ca02c', 3: '#9467bd',
                4: '#ff7f0e', 5: '#8c564b', 6: '#e377c2', 7: '#7f7f7f',
                8: '#17becf', 9: '#bcbd22'}

# fonts matched to the manuscript combined-trajectory figure
FS_TICK, FS_XLABEL, FS_YLABEL, FS_LEGEND = 12, 15, 18, 10
# shared axes-box margins used across manuscript/figures (identical frame everywhere)
MANUSCRIPT_MARGINS = dict(left=0.135, right=0.965, bottom=0.135, top=0.965)


def _pcolor(w):
    return PHASE_COLORS.get(int(w), '#333333')


def block_boundaries(query_per_step):
    """Indices where a new inner block starts (0, then each retrain gap)."""
    q = np.asarray(query_per_step)
    return [0] + [i for i in range(1, len(q)) if q[i] - q[i - 1] > 1]


def omega_segments(omega):
    """Contiguous [i0, i1, w] runs of constant winding, for background shading."""
    omega = np.asarray(omega, dtype=int)
    segs, i0 = [], 0
    for i in range(1, len(omega)):
        if omega[i] != omega[i0]:
            segs.append((i0, i, int(omega[i0])))
            i0 = i
    segs.append((i0, len(omega), int(omega[i0])))
    return segs


def plot_winding_timeline(bundle, out_path=None):
    if isinstance(bundle, str):
        print(f'loading {bundle}')
        path = bundle
        b = load_pickle(bundle)
    else:
        b, path = bundle, out_path or '../figures/spt_winding_timeline.png'

    h = b['history']
    omega = np.asarray(h['omega_per_step'], dtype=int)
    N = len(omega)
    starts = block_boundaries(h['query_per_step'])
    retrain_idx = starts[1:]                          # block starts after the 1st
    phases_seen = sorted(set(omega.tolist()))
    x = np.arange(N) / max(1, N - 1)                  # normalized step in [0, 1]
    segs = omega_segments(omega)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fig.subplots_adjust(**MANUSCRIPT_MARGINS)

    # faint phase shading + light grid
    for i0, i1, w in segs:
        ax.axvspan(x[i0], x[min(i1, N - 1)], color=_pcolor(w), alpha=0.10, lw=0)
    ax.grid(True, color='0.85', lw=0.6, zorder=0)

    # the omega step trace + per-step colored points
    ax.plot(x, omega, '-', color='0.55', lw=1.0, drawstyle='steps-post', zorder=2)
    sub = np.arange(0, N, max(1, N // 500))
    for w in phases_seen:
        m = omega[sub] == w
        ax.scatter(x[sub][m], omega[sub][m], s=12, color=_pcolor(w), zorder=3)

    # start (square), retrain (stars), end (triangle) -- matches the manuscript fig
    ax.scatter(x[0], omega[0], marker='s', s=110, color=_pcolor(omega[0]),
               edgecolor='k', linewidths=1.0, zorder=5)
    ax.scatter(x[retrain_idx], omega[retrain_idx], marker='*', s=160,
               color='white', edgecolor='k', linewidths=1.1, zorder=5)
    ax.scatter(x[-1], omega[-1], marker='^', s=130, color=_pcolor(omega[-1]),
               edgecolor='k', linewidths=1.0, zorder=5)

    ax.set_xlabel('step (normalized)', fontsize=FS_XLABEL)
    ax.set_ylabel(r'$\omega$', fontsize=FS_YLABEL)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.set_yticks(phases_seen)
    ax.set_ylim(min(phases_seen) - 0.5, max(phases_seen) + 0.5)
    ax.set_xlim(-0.02, 1.02)

    handles = [
        Line2D([], [], marker='s', ls='', color='0.5', mec='k', ms=9,
               label='start'),
        Line2D([], [], marker='*', ls='', color='white', mec='k', ms=13,
               label='retrain'),
        Line2D([], [], marker='^', ls='', color='0.5', mec='k', ms=10,
               label='end'),
    ]
    ax.legend(handles=handles, fontsize=FS_LEGEND, loc='upper right',
              framealpha=0.9)

    if path.endswith('.pkl'):
        stem = (path.replace('../data/', '../figures/')
                    .replace('.pkl', '_winding_timeline'))
    else:
        stem = '../figures/spt_winding_timeline'
    os.makedirs(os.path.dirname(stem) or '.', exist_ok=True)
    fig.savefig(stem + '.png', dpi=150)          # no tight bbox (manuscript std)
    fig.savefig(stem + '.pdf')
    plt.close(fig)
    print(f'distinct phases: {phases_seen}  blocks: {len(starts)}  '
          f'retrains: {len(retrain_idx)}')
    print(f'figure -> {stem}.png / {stem}.pdf')
    return stem + '.png'


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    elif os.environ.get('BUNDLE'):
        path = os.environ['BUNDLE']
    else:
        cands = [c for c in sorted(glob.glob('../data/spt_active_*.pkl'))
                 if not c.endswith('_ckpt.pkl')]
        if not cands:
            raise SystemExit('no ../data/spt_active_*.pkl found')
        path = cands[-1]
    plot_winding_timeline(path)


if __name__ == '__main__':
    main()
