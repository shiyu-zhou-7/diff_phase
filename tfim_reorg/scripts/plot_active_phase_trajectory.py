"""
Plot the h-trajectory of a single active-phase-discovery run from its bundle pickle.

Paper-style figure with phase boundary at g = -1, mediumorchid SSB/PARA region
labels, and per-event markers on the trajectory:
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

# trajectory line
ax.plot(steps, h_arr, '-', color='tab:blue', lw=1.4, zorder=2)

# per-event markers
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

# phase boundary
ax.axhline(-1.0, color='gray', ls='--', lw=1.2, zorder=0)
ax.text(n - 0.5, -0.98, 'phase boundary',
        color='gray', fontsize=12, ha='right', va='bottom')

# SSB / PARA labels (mediumorchid, right-side aligned)
x_label = n * 0.93
ymin, ymax = float(h_arr.min()), float(h_arr.max())
y_pad_top = max(0.05, 0.05 * (ymax - ymin))
y_pad_bot = max(0.05, 0.05 * (ymax - ymin))
ax.set_ylim(ymin - y_pad_bot, max(ymax + y_pad_top, -0.8))
ylim_top = ax.get_ylim()[1]
ylim_bot = ax.get_ylim()[0]
gap = 0.20

ax.text(x_label, ylim_top - gap, 'SSB', color='mediumorchid',
        fontsize=13, ha='left', va='center')
ax.text(x_label, -1.0 - gap, 'PARA', color='mediumorchid',
        fontsize=13, ha='left', va='center')

# axes
ax.set_xlabel('step', fontsize=16)
ax.set_ylabel(r'$g$', fontsize=18)
ax.tick_params(axis='both', labelsize=13)
ax.grid(True, alpha=0.3)

# legend in lower-left (paper-figure style); fall back to lower-right if the
# trajectory ends below mid-axis (the typical case is h drops over time, so
# lower-left is usually safer to avoid the line)
legend_loc = 'lower left' if final_h < (ylim_top + ylim_bot) / 2 else 'lower right'
ax.legend(loc=legend_loc, fontsize=11, frameon=True)

fig.savefig(out_path, bbox_inches='tight')
print(f'Saved: {out_path}')
print(f'  bundle: {bundle_path}')
print(f'  h_init = {h_init:+.4f}, final_h = {final_h:+.4f}')
print(f'  total steps = {n}, retrains = {int(retrain_mask.sum())}, NaN kicks = {int(nan_mask.sum())}')
print(f'  bootstrap_history = {[round(x, 4) for x in bootstrap_history]}')
