"""
Plot the h-trajectory of a single z2gauge active-phase-discovery run from its
bundle pickle.

Paper-style figure with phase boundary at |h| = 0.3, mediumorchid
deconfined/confined region labels, and per-event markers on the trajectory:
  - 'normal'    -> tab:blue dot
  - 'retrain'   -> tab:orange dot (larger)
  - 'nan_kick'  -> red 'X'

Usage:
    python plot_active_phase_trajectory.py [bundle_path]

No argument:
    Glob the newest ../data/active_phase_*.pkl and plot it.

Output:
    ../figures/active_phase_trajectory_h{tag}_<ts>.pdf
"""
import sys, os, glob, re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.io import load_pickle


# z2gauge phase boundary (per project_z2gauge_phase_boundary memory):
# deconfined when |h| < 0.3, confined when |h| > 0.3
PHASE_BOUNDARY = -0.3


# ── locate bundle ─────────────────────────────────────────────────────────────
if len(sys.argv) >= 2:
    bundle_path = sys.argv[1]
else:
    candidates = sorted(glob.glob('../data/active_phase_*.pkl'))
    candidates = [p for p in candidates if not p.endswith('_ckpt.pkl')]
    if not candidates:
        raise SystemExit('No active_phase_*.pkl bundle found in ../data/')
    bundle_path = candidates[-1]
    print(f'No path given; using latest bundle: {bundle_path}')

bundle = load_pickle(bundle_path)
h_per_step    = bundle['history']['h_per_step']
event_per_step = bundle['history']['event_per_step']
bootstrap_history = bundle['bootstrap_history']
h_init = bundle['cfg_snapshot']['h_init']
final_h = bundle['final_h']

n = len(h_per_step)
steps = np.arange(n)
h_arr = np.asarray(h_per_step)
ev_arr = np.asarray(event_per_step)

# Parse the bundle filename to reuse its <ts> in the output filename
basename = os.path.basename(bundle_path)
m = re.match(r'active_phase_h([mp]\d+\.\d+)_(\d{8}_\d{6})\.pkl$', basename)
if m:
    h_tag, ts = m.group(1), m.group(2)
else:
    h_tag = f'{h_init:+.2f}'.replace('+', 'p').replace('-', 'm')
    ts = basename.rstrip('.pkl').split('_')[-1]
out_path = f'../figures/active_phase_trajectory_h{h_tag}_{ts}.pdf'

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)

ax.plot(steps, h_arr, '-', color='tab:blue', lw=1.4, zorder=2)

normal_mask = ev_arr == 'normal'
retrain_mask = ev_arr == 'retrain'
nan_mask = ev_arr == 'nan_kick'

ax.plot(steps[normal_mask], h_arr[normal_mask], 'o', color='tab:blue',
        markersize=2.5, lw=0, zorder=3, label='Adam step')
if retrain_mask.any():
    ax.plot(steps[retrain_mask], h_arr[retrain_mask], 'o', color='tab:orange',
            markersize=7, lw=0, zorder=5, label='retrain')
if nan_mask.any():
    ax.plot(steps[nan_mask], h_arr[nan_mask], 'X', color='red',
            markersize=8, lw=0, zorder=6, label='NaN kick')

# phase boundary at h = -0.3 (negative-h convention)
ax.axhline(PHASE_BOUNDARY, color='gray', ls='--', lw=1.2, zorder=0)
ax.text(n - 0.5, PHASE_BOUNDARY + 0.02, 'phase boundary',
        color='gray', fontsize=12, ha='right', va='bottom')

# deconfined / confined labels (mediumorchid, right-side aligned)
x_label = n * 0.93
ymin, ymax = float(h_arr.min()), float(h_arr.max())
y_pad = max(0.05, 0.05 * (ymax - ymin))
ax.set_ylim(min(ymin - y_pad, PHASE_BOUNDARY - 0.1),
            max(ymax + y_pad, -0.05))
ylim_top = ax.get_ylim()[1]
ylim_bot = ax.get_ylim()[0]
gap = 0.05

# deconfined is the |h| < 0.3 region (closer to h=0); confined is |h| > 0.3
ax.text(x_label, max(ylim_top - gap, PHASE_BOUNDARY + gap), 'deconfined',
        color='mediumorchid', fontsize=13, ha='left', va='center')
ax.text(x_label, PHASE_BOUNDARY - gap, 'confined',
        color='mediumorchid', fontsize=13, ha='left', va='center')

ax.set_xlabel('step', fontsize=16)
ax.set_ylabel(r'$h$', fontsize=18)
ax.tick_params(axis='both', labelsize=13)
ax.grid(True, alpha=0.3)

legend_loc = 'lower left' if final_h < (ylim_top + ylim_bot) / 2 else 'lower right'
ax.legend(loc=legend_loc, fontsize=11, frameon=True)

fig.savefig(out_path, bbox_inches='tight')
print(f'Saved: {out_path}')
print(f'  bundle: {bundle_path}')
print(f'  h_init = {h_init:+.4f}, final_h = {final_h:+.4f}')
print(f'  total steps = {n}, retrains = {int(retrain_mask.sum())}, NaN kicks = {int(nan_mask.sum())}')
print(f'  bootstrap_history = {[round(x, 4) for x in bootstrap_history]}')
