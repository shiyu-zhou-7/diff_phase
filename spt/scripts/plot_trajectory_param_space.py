"""
Clean parameter-space views of an active-phase-discovery trajectory.

The t-trajectory lives on the sphere S^{d-1} in d couplings (d=8 here), so any 2D
scatter (latent PCA, corner PCA) throws away most of the structure. Instead of
projecting, this script presents the trajectory *honestly* in three exact views,
combined into one figure:

  (top-left)     WINDING TIMELINE   -- omega vs discovery step, colored by phase,
                 with retrain / block boundaries marked. The tour order, no loss.
  (bottom-left)  PHASE-TRANSITION GRAPH -- windings 0..d-1 on a ring, directed
                 edges = observed block-to-block transitions, numbered by order.
                 A compact 'map' of which phases the walk connected.
  (right)        PER-COORDINATE SMALL MULTIPLES -- d stacked panels, t_alpha vs
                 step, background shaded by the current winding. Shows exactly
                 which coupling drives each transition; no dimensionality loss.

Block / retrain boundaries are recovered from query_per_step: a retrain injects a
batch of bootstrap solves with no optimization step, so query_per_step jumps by
>1 there (a normal step adds exactly 1). No sklearn; numpy + matplotlib only,
per repo convention.

Usage (from spt/scripts/):
    python plot_trajectory_param_space.py                        # newest bundle
    python plot_trajectory_param_space.py ../data/spt_active_d8_....pkl
    BUNDLE=../data/spt_active_d6_....pkl python plot_trajectory_param_space.py
"""
import sys, os, glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.io import load_pickle

PHASE_COLORS = {0: '#1f77b4', 1: '#d62728', 2: '#2ca02c', 3: '#9467bd',
                4: '#ff7f0e', 5: '#8c564b', 6: '#e377c2', 7: '#7f7f7f',
                8: '#17becf', 9: '#bcbd22'}


def _pcolor(w):
    return PHASE_COLORS.get(int(w), '#333333')


def block_boundaries(query_per_step):
    """Indices where a new inner block starts (0, then each retrain gap)."""
    q = np.asarray(query_per_step)
    N = len(q)
    starts = [0] + [i for i in range(1, N) if q[i] - q[i - 1] > 1]
    return starts


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


def plot_param_space(bundle, out_path=None):
    if isinstance(bundle, str):
        print(f'loading {bundle}')
        path = bundle
        b = load_pickle(bundle)
    else:
        b, path = bundle, out_path or '../figures/spt_active_param_space.png'

    h = b['history']
    traj = np.asarray(h['t_per_step'], dtype=float)          # (N, d)
    omega = np.asarray(h['omega_per_step'], dtype=int)        # (N,)
    events = h['event_per_step']
    N, d = traj.shape
    L = b['cfg_snapshot']['cluster_cfg']['L']

    starts = block_boundaries(h['query_per_step'])           # block start indices
    ends = [(starts[k + 1] - 1) if k + 1 < len(starts) else N - 1
            for k in range(len(starts))]
    nan_idx = [i for i, e in enumerate(events) if e == 'nan_jump']
    steps = np.arange(N)
    phases_seen = sorted(set(omega.tolist()))
    segs = omega_segments(omega)

    # settled-phase tour: start corner omega, then each block's end omega
    tour = [int(omega[0])] + [int(omega[e]) for e in ends]

    # ---- figure: left column (timeline / graph), right column (small multiples)
    fig = plt.figure(figsize=(15.0, 0.95 * d + 1.5))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.16,
                             left=0.055, right=0.985, top=0.945, bottom=0.075)
    gsL = outer[0].subgridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.30)
    gsR = outer[1].subgridspec(d, 1, hspace=0.0)

    axT = fig.add_subplot(gsL[0])   # timeline
    axG = fig.add_subplot(gsL[1])   # transition graph

    # ================= (1) WINDING TIMELINE =================
    for i0, i1, w in segs:
        axT.axvspan(i0, i1, color=_pcolor(w), alpha=0.10, linewidth=0)
    axT.plot(steps, omega, '-', color='0.55', lw=0.8,
             drawstyle='steps-post', zorder=1)
    sub = np.arange(0, N, max(1, N // 400))
    for w in phases_seen:
        m = omega[sub] == w
        axT.scatter(steps[sub][m], omega[sub][m], s=10, color=_pcolor(w),
                    zorder=3, label=f'$\\omega$={w}')
    for k, s in enumerate(starts[1:], start=1):             # retrain lines
        axT.axvline(s, color='0.3', ls='--', lw=0.8, alpha=0.7, zorder=2)
        axT.text(s, max(phases_seen) + 0.55, f'R{k}', ha='center', va='bottom',
                 fontsize=8, color='0.3')
    for i in nan_idx:
        axT.axvline(i, color='magenta', lw=0.6, alpha=0.5, zorder=2)
    axT.scatter([0], [omega[0]], marker='o', s=90, facecolor='none',
                edgecolor='lime', linewidths=2.0, zorder=5)
    axT.scatter([N - 1], [omega[-1]], marker='X', s=90, color='k', zorder=5)
    axT.set_ylabel(r'winding $\omega$')
    axT.set_xlabel('discovery step')
    axT.set_yticks(phases_seen)
    axT.set_ylim(min(phases_seen) - 0.5, max(phases_seen) + 1.1)
    axT.set_xlim(-0.02 * N, 1.02 * N)
    axT.set_title(f'winding timeline  ({N} steps, {len(starts)} blocks, '
                  f'tour {"→".join(str(x) for x in tour)})', fontsize=10)
    axT.legend(loc='center left', bbox_to_anchor=(1.005, 0.5), fontsize=7.5,
               framealpha=0.9, handletextpad=0.3, borderpad=0.3)

    # ================= (2) PHASE-TRANSITION GRAPH =================
    ang = {w: 2 * np.pi * i / len(phases_seen) for i, w in enumerate(phases_seen)}
    pos = {w: np.array([np.cos(a), np.sin(a)]) for w, a in ang.items()}
    # nodes
    for w in phases_seen:
        axG.scatter(*pos[w], s=560, color=_pcolor(w), edgecolor='k',
                    linewidths=1.2, zorder=4)
        axG.text(*pos[w], str(w), ha='center', va='center', color='w',
                 fontsize=11, fontweight='bold', zorder=5)
    # edges = consecutive settled-phase transitions, numbered by order
    seen_pairs = {}
    for order in range(1, len(tour)):
        a, c = tour[order - 1], tour[order]
        if a == c:                                          # self-loop (re-settle)
            p = pos[a] * 1.16
            axG.text(p[0], p[1], f'↺{order}', ha='center', va='center',
                     fontsize=8, color='0.25', zorder=6)
            continue
        key = (a, c)
        seen_pairs[key] = seen_pairs.get(key, 0) + 1
        rad = 0.18 + 0.16 * (seen_pairs[key] - 1)
        # opposite direction already drawn -> curve the other way to avoid overlap
        if (c, a) in seen_pairs:
            rad = -rad
        p0 = pos[a] + 0.12 * (pos[c] - pos[a])
        p1 = pos[c] + 0.12 * (pos[a] - pos[c])
        arr = FancyArrowPatch(p0, p1, connectionstyle=f'arc3,rad={rad}',
                              arrowstyle='-|>', mutation_scale=14,
                              color='0.35', lw=1.3, zorder=3, alpha=0.9)
        axG.add_patch(arr)
        mid = 0.5 * (p0 + p1) + rad * 0.6 * np.array(
            [-(p1 - p0)[1], (p1 - p0)[0]])
        axG.text(mid[0], mid[1], str(order), ha='center', va='center',
                 fontsize=8.5, color='k', zorder=6,
                 bbox=dict(boxstyle='circle,pad=0.12', fc='white',
                           ec='0.5', lw=0.6))
    axG.scatter(*pos[tour[0]] * 1.28, marker='o', s=90, facecolor='none',
                edgecolor='lime', linewidths=2.0, zorder=6)
    axG.text(*(pos[tour[0]] * 1.42), 'start', ha='center', va='center',
             fontsize=8, color='green')
    axG.scatter(*pos[tour[-1]] * 1.28, marker='X', s=90, color='k', zorder=6)
    axG.text(*(pos[tour[-1]] * 1.42), 'end', ha='center', va='center',
             fontsize=8, color='k')
    axG.set_title('phase-transition graph  (edges numbered by block order)',
                  fontsize=10)
    axG.set_aspect('equal')
    axG.set_xlim(-1.6, 1.6)
    axG.set_ylim(-1.6, 1.6)
    axG.axis('off')

    # ================= (3) PER-COORDINATE SMALL MULTIPLES =================
    tmin, tmax = float(traj.min()), float(traj.max())
    pad = 0.08 * (tmax - tmin + 1e-9)
    axes = []
    for a in range(d):
        ax = fig.add_subplot(gsR[a], sharex=axes[0] if axes else None)
        axes.append(ax)
        for i0, i1, w in segs:
            ax.axvspan(i0, i1, color=_pcolor(w), alpha=0.10, linewidth=0)
        ax.axhline(0, color='0.7', lw=0.5, zorder=1)
        ax.plot(steps, traj[:, a], '-', color='0.15', lw=0.8, zorder=3)
        for s in starts[1:]:
            ax.axvline(s, color='0.3', ls='--', lw=0.7, alpha=0.6, zorder=2)
        ax.set_ylim(tmin - pad, tmax + pad)
        ax.set_ylabel(f'$t_{a}$', rotation=0, labelpad=12, va='center',
                      fontsize=10)
        ax.set_yticks([round(tmin, 1), 0, round(tmax, 1)])
        ax.tick_params(labelsize=7)
        if a < d - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
    axes[0].set_title(f'per-coupling trajectory  (d={d}, L={L})  '
                      '— bg shaded by $\\omega$', fontsize=10)
    axes[-1].set_xlabel('discovery step')

    if path.endswith('.pkl'):
        out = (path.replace('../data/', '../figures/')
                   .replace('.pkl', '_param_space.png'))
    else:
        out = out_path or '../figures/spt_active_param_space.png'
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'distinct phases: {phases_seen}  blocks: {len(starts)}  '
          f'tour: {tour}')
    print(f'figure -> {out}')
    return out


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    elif os.environ.get('BUNDLE'):
        path = os.environ['BUNDLE']
    else:
        cands = sorted(glob.glob('../data/spt_active_*.pkl'))
        cands = [c for c in cands if not c.endswith('_ckpt.pkl')]
        if not cands:
            raise SystemExit('no ../data/spt_active_*.pkl found')
        path = cands[-1]
    plot_param_space(path)


if __name__ == '__main__':
    main()
