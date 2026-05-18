"""
Combined h-trajectory figure for four active-phase-discovery runs.

Each trajectory is x-rescaled to [0, 1] so all runs share the right edge of
the plot regardless of their actual step count. h_init is prepended as the
first point, then the requested slice of h_per_step is concatenated.

Output: ../figures/combined_active_phase_trajectory_<ts>.pdf
"""
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.io import load_pickle, timestamp


RUNS = [
    {
        'path': '../data/active_phase_hm0.50_20260513_210852.pkl',
        'max_steps': 300,
        'color': 'tab:blue',
        'label': r'$g_0 = -0.5$',
    },
    {
        'path': '../data/active_phase_hm0.30_20260511_155044.pkl',
        'max_steps': None,
        'color': 'tab:orange',
        'label': r'$g_0 = -0.3$',
    },
    {
        'path': '../data/active_phase_hm1.50_20260511_133336.pkl',
        'max_steps': None,
        'color': 'tab:green',
        'label': r'$g_0 = -1.5$ (run A)',
    },
    {
        'path': '../data/active_phase_hm1.50_20260512_101827.pkl',
        'max_steps': None,
        'color': 'tab:red',
        'label': r'$g_0 = -1.5$ (run B)',
    },
]


fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)

for run in RUNS:
    bundle = load_pickle(run['path'])
    h_init = float(bundle['cfg_snapshot']['h_init'])
    h_per_step = list(bundle['history']['h_per_step'])
    event_per_step = list(bundle['history']['event_per_step'])
    if run['max_steps'] is not None:
        h_per_step = h_per_step[: run['max_steps']]
        event_per_step = event_per_step[: run['max_steps']]

    # Prepend h_init as the trajectory's starting point.
    h_full = [h_init] + h_per_step
    events_full = ['start'] + event_per_step
    n = len(h_full)
    x = np.linspace(0.0, 1.0, n)

    # Trajectory body: dots only, no connecting line.
    ax.scatter(x, h_full, color=run['color'], s=3, label=run['label'], zorder=3)

    # Retrain markers (synthetic 'retrain' events from the workflow), matched
    # to the trajectory color so each run's retrains read as that run's events.
    retrain_idx = [i for i, e in enumerate(events_full) if e == 'retrain']
    if retrain_idx:
        rx = [x[i] for i in retrain_idx]
        ry = [h_full[i] for i in retrain_idx]
        ax.scatter(rx, ry, color=run['color'], marker='*', s=110,
                   edgecolor='black', lw=0.6, zorder=5)

    # Start = filled square, end = filled triangle, both in the run color.
    ax.scatter([x[0]], [h_full[0]], color=run['color'], marker='s', s=110,
               edgecolor='black', lw=0.9, zorder=6)
    ax.scatter([x[-1]], [h_full[-1]], color=run['color'], marker='^', s=140,
               edgecolor='black', lw=0.9, zorder=6)

# Phase boundary line (no text annotation).
ax.axhline(-1.0, color='gray', ls='--', lw=1.2, zorder=0)

# SSB / PARA on the right side of the plot. Both ha='left' at the same x so
# the two labels' left edges are aligned (the right edges differ because the
# words have different widths).
LABEL_X = 0.91
ax.text(LABEL_X, -0.40, 'SSB',  color='mediumorchid', fontsize=13, ha='left', va='center')
ax.text(LABEL_X, -1.45, 'PARA', color='mediumorchid', fontsize=13, ha='left', va='center')

ax.set_xlim(-0.025, 1.035)
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlabel('step (normalized)', fontsize=15)
ax.set_ylabel(r'$g$', fontsize=18)
ax.tick_params(labelsize=12)
ax.grid(True, alpha=0.3)

# Legend: four per-run color entries + a universal 'retrain' glyph (5 total),
# laid out in 3 columns at lower-left.
handles, labels = ax.get_legend_handles_labels()
handles.append(Line2D([0], [0], marker='*', markerfacecolor='lightgray',
                      markeredgecolor='black', color='black', lw=0, markersize=12,
                      label='retrain'))
ax.legend(handles=handles, loc='lower left', fontsize=10, frameon=True, ncol=3,
          columnspacing=0.8, handletextpad=0.4)

ts = timestamp()
out = f'../figures/combined_active_phase_trajectory_{ts}.pdf'
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out}')

# Diagnostic summary.
for run in RUNS:
    bundle = load_pickle(run['path'])
    h_init = float(bundle['cfg_snapshot']['h_init'])
    h_full_n = len(bundle['history']['h_per_step'])
    used_n = run['max_steps'] if run['max_steps'] is not None else h_full_n
    print(f'  {run["label"]}: h_init={h_init:.2f}, {used_n}/{h_full_n} steps used')
