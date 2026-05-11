"""
Plot combined h trajectories (first 21 epochs, 0..20) for three optim_h_ae
runs at h_init = -0.10, -0.30, -1.20, with a phase-boundary line at h = -1
and purple SSB/PARA region labels.

Output: ../figures/combined_h_trajectory_first20_<ts>.pdf

Usage:
    cd tfim_reorg/scripts && python plot_combined_h_trajectory.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.io import load_pickle, timestamp


# NOTE: ham_cfg['h_init'] in the pickles is stale (-0.4, the script default) —
# the env-var-set h_init didn't propagate. True h_init taken from the filenames.
PICKLES = [
    (-0.10, '../data/optim_h_ae_h-0.10_e50_20260508_112927.pkl', 'tab:blue',   r'$g_0 = -0.10$'),
    (-0.30, '../data/optim_h_ae_h-0.30_e50_20260508_112934.pkl', 'tab:orange', r'$g_0 = -0.30$'),
    (-1.20, '../data/optim_h_ae_hm1.20_e50_20260508_140204.pkl', 'tab:green',  r'$g_0 = -1.20$'),
]
N_EPOCHS = 21  # first 21 points -> epochs 0..20


def load_trajectory(path, h_init):
    blob = load_pickle(path)
    h_full = [h_init] + list(blob['h_list'])
    return h_full[:N_EPOCHS]


fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)

epochs = list(range(N_EPOCHS))
for h_init, path, color, label in PICKLES:
    ax.plot(epochs, load_trajectory(path, h_init), '-o',
            color=color, markersize=4.5, lw=1.6, label=label)

ax.axhline(-1.0, color='gray', ls='--', lw=1.2, zorder=0)
ax.text(20.0, -0.98, 'phase boundary',
        color='gray', fontsize=12, ha='right', va='bottom')

ax.text(17.5, -0.15, 'SSB',  color='mediumorchid', fontsize=13, ha='left', va='center')
ax.text(17.5, -1.20, 'PARA', color='mediumorchid', fontsize=13, ha='left', va='center')

ax.set_xlim(-0.5, 20.5)
ax.set_ylim(-2.30, 0.05)
ax.set_xticks([0, 4, 8, 12, 16, 20])
ax.set_xlabel('epoch', fontsize=16)
ax.set_ylabel(r'$g$', fontsize=18)
ax.tick_params(labelsize=13)
ax.grid(alpha=0.3)
ax.legend(loc='lower left', fontsize=12)

ts = timestamp()
out = f'../figures/combined_h_trajectory_first20_{ts}.pdf'
fig.savefig(out, bbox_inches='tight')
print(f'Saved: {out}')
