"""
Plot an active-phase-discovery trajectory for the cluster chain.

Loads a single-start bundle (data/spt_active_*.pkl from main_active_phase.py) and
renders two panels into ../figures/:

  (left)  the t-trajectory in a phase-diagram backdrop. t lives on the sphere
          S^{d-1}; we project to 2D with the top-2 PCs of the d pure-point corners
          e_alpha (numpy SVD -- no sklearn, per repo convention), so the corners
          spread out maximally and the same frame holds the trajectory and a
          background scatter of sphere samples colored by the analytic winding
          label. (winding(-t) = winding(t), so the linear projection is antipodally
          consistent -- no color ambiguity.)
  (right) winding omega and the loss-gradient norm vs optimization step, with NaN
          jump events marked.

Usage (from spt/scripts/):
    python plot_trajectory.py                       # newest data/spt_active_*.pkl
    python plot_trajectory.py ../data/spt_active_d3_L10_pbc_XXXX.pkl
"""
import sys, os, glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analytic.cluster_exact import winding, is_gapless
from utils.io import load_pickle

PHASE_COLORS = {0: '#1f77b4', 1: '#d62728', 2: '#2ca02c', 3: '#9467bd',
                4: '#ff7f0e', 5: '#8c564b', 6: '#e377c2', 7: '#7f7f7f'}
PHASE_NAME = {0: r'$\omega$=0 trivial', 1: r'$\omega$=1 Z2', 2: r'$\omega$=2 SPT'}


def corner_basis(d):
    """Top-2 PCs of the d pure-point corners e_alpha (centered)."""
    E = np.eye(d)
    Ec = E - E.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Ec, full_matrices=False)
    return Ec.mean(axis=0) * 0 + E.mean(axis=0), Vt[:2].T   # (mean, basis dx2)


def project(ts, mean, basis):
    return (np.asarray(ts) - mean) @ basis


def plot_trajectory(bundle, out_path=None, max_points_right=100):
    """Render the 2-panel trajectory figure.

    bundle : a result dict (from run_active_phase_discovery) or a path to a
             pickled one. out_path defaults to the bundle path with
             '_trajectory.png'. The right panel is thinned to <= max_points_right
             steps so the omega / gradient traces stay readable on long runs.
    Returns the output figure path.
    """
    if isinstance(bundle, str):
        print(f'loading {bundle}')
        path = bundle
        b = load_pickle(bundle)
    else:
        b = bundle
        path = out_path or '../figures/spt_active_trajectory.png'

    traj = np.asarray(b['history']['t_per_step'], dtype=float)
    omega = np.asarray(b['history']['omega_per_step'], dtype=int)
    grad = np.asarray(b['history']['grad_per_step'], dtype=float)
    events = b['history']['event_per_step']
    d = traj.shape[1]
    L = b['cfg_snapshot']['cluster_cfg']['L']

    mean, basis = corner_basis(d)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # ---- left: phase-diagram backdrop + trajectory ----
    rng = np.random.default_rng(0)
    bg = rng.standard_normal((6000, d))
    bg /= np.linalg.norm(bg, axis=1, keepdims=True)
    bg_w = np.array([(-1 if is_gapless(t) else winding(t)) for t in bg])
    bg2 = project(bg, mean, basis)
    for w in sorted(set(bg_w) - {-1}):
        m = bg_w == w
        axL.scatter(bg2[m, 0], bg2[m, 1], s=6, alpha=0.12,
                    color=PHASE_COLORS.get(w, '#333'), linewidths=0)

    # corners
    corners2 = project(np.eye(d), mean, basis)
    for a in range(d):
        axL.scatter(*corners2[a], marker='*', s=220, edgecolor='k',
                    facecolor=PHASE_COLORS.get(a, '#333'), zorder=5)
        axL.annotate(f'$e_{a}$', corners2[a], textcoords='offset points',
                     xytext=(6, 6), fontsize=11, fontweight='bold')

    tr2 = project(traj, mean, basis)
    axL.plot(tr2[:, 0], tr2[:, 1], '-', color='0.35', lw=1.0, alpha=0.8, zorder=3)
    for w in sorted(set(omega)):
        m = omega == w
        axL.scatter(tr2[m, 0], tr2[m, 1], s=14, color=PHASE_COLORS.get(w, '#333'),
                    zorder=4, label=PHASE_NAME.get(w, f'$\\omega$={w}'))
    # start / end / nan jumps
    axL.scatter(*tr2[0], marker='o', s=120, facecolor='none', edgecolor='lime',
                linewidths=2.2, zorder=6, label='start')
    axL.scatter(*tr2[-1], marker='X', s=120, color='k', zorder=6, label='end')
    nan_idx = [i for i, e in enumerate(events) if e == 'nan_jump']
    if nan_idx:
        axL.scatter(tr2[nan_idx, 0], tr2[nan_idx, 1], marker='x', s=80,
                    color='magenta', zorder=6, label='NaN jump')

    axL.set_title(f'trajectory in t-space  (d={d}, L={L})')
    axL.set_xlabel('PC1 of corners'); axL.set_ylabel('PC2 of corners')
    axL.legend(loc='best', fontsize=8, framealpha=0.9)
    axL.set_aspect('equal', 'datalim')

    # ---- right: omega and |grad| vs step (downsampled, transitions kept) ----
    N = len(omega)
    steps = np.arange(N)
    stride = max(1, N // max_points_right)
    # clean strided subsample (+ endpoints and NaN events); do NOT force every
    # phase transition -- near a boundary omega flips rapidly and keeping all of
    # them just re-fills the panel. A strided read shows the broad structure.
    keep = set(range(0, N, stride)) | {0, N - 1} | set(nan_idx)
    kidx = np.array(sorted(keep))

    axR.plot(steps[kidx], omega[kidx], '-', color='0.5', lw=1.0,
             drawstyle='steps-post', zorder=1)
    for w in sorted(set(omega)):
        m = omega[kidx] == w
        axR.scatter(steps[kidx][m], omega[kidx][m], s=22,
                    color=PHASE_COLORS.get(w, '#333'), zorder=3,
                    label=PHASE_NAME.get(w, f'$\\omega$={w}'))
    axR.set_ylabel(r'winding $\omega$'); axR.set_xlabel('optimization step')
    axR.set_yticks(sorted(set(omega)))
    axR.set_ylim(min(omega) - 0.4, max(omega) + 0.4)
    axR.set_title(f'phase label & gradient vs step ({N} steps, {len(kidx)} shown)')
    axR.legend(loc='center right', fontsize=8, framealpha=0.9)

    axG = axR.twinx()
    with np.errstate(invalid='ignore'):
        axG.semilogy(steps[kidx], grad[kidx], color='#888', lw=0.9, alpha=0.7)
    axG.set_ylabel('loss-grad norm', color='0.5')
    axG.tick_params(axis='y', colors='0.5')
    for i in nan_idx:
        axR.axvline(i, color='magenta', lw=0.7, alpha=0.5)

    fig.subplots_adjust(left=0.07, right=0.93, top=0.92, bottom=0.12, wspace=0.28)
    if path.endswith('.pkl'):
        out = path.replace('../data/', '../figures/').replace('.pkl', '_trajectory.png')
    else:
        out = out_path or '../figures/spt_active_trajectory.png'
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'distinct phases: {sorted(set(omega))}  queries: {b.get("total_queries", "?")}')
    print(f'figure -> {out}')
    return out


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        cands = sorted(glob.glob('../data/spt_active_*.pkl'))
        if not cands:
            raise SystemExit('no ../data/spt_active_*.pkl found; run main_active_phase.py first')
        path = cands[-1]
    plot_trajectory(path)


if __name__ == '__main__':
    main()
