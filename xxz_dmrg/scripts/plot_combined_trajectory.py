import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# --- load trajectories ---
with open('../models/active_phase_discovery_checkpoint_latest.pkl', 'rb') as f:
    prev = pickle.load(f)
prev_params = np.array(prev['hist']['ham_params'])   # (1141, 2)

with open('../models/active_phase_ham_params_history_20260423_174514.pkl', 'rb') as f:
    run2 = pickle.load(f)
run2_params = np.array(run2['ham_params'])           # (356, 2)

# --- NaN recovery kick origins in job 427681 (last valid point before each jump) ---
# idx 94: kick 1 origin (steps ~91-104 of log), idx 126: kick 2 origin (steps ~127-149)
nan_kick_origins = np.array([
    run2_params[94],   # (Δ=-0.904, h=-0.876)
    run2_params[126],  # (Δ=-0.261, h=0.005)
])

# --- AE retraining coordinates from job 427681 log ---
# (delta, h) at each event: 1 bootstrap + 6 outer-loop convergences
ae_spots = np.array([
    [-1.00325, -0.98680],   # bootstrap (resume point)
    [-0.97344, -0.02486],   # after outer 1
    [-0.99010, -0.06372],   # after outer 2
    [-0.99019, -0.06365],   # after outer 3
    [-0.99027, -0.06355],   # after outer 4
    [-0.99035, -0.06349],   # after outer 5
    [-0.99042, -0.06346],   # after outer 6
])

n1 = len(prev_params)
n2 = len(run2_params)

hs1,    ds1    = prev_params[:, 1], prev_params[:, 0]
hs2,    ds2    = run2_params[:, 1], run2_params[:, 0]
ae_h,   ae_d   = ae_spots[:, 1],   ae_spots[:, 0]

# --- colour ramps (match existing style: Blues from 0.3→1.0) ---
shades1 = plt.cm.Blues(np.linspace(0.3, 1.0, n1))
shades2 = plt.cm.Oranges(np.linspace(0.3, 1.0, n2))

fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

# connecting lines
ax.plot(hs1, ds1, lw=1.0, color='0.6', alpha=0.4, zorder=1)
ax.plot(hs2, ds2, lw=1.0, color='0.6', alpha=0.4, zorder=1)

# scatter points
ax.scatter(hs1, ds1, c=shades1, s=14, edgecolors='none', zorder=2)
ax.scatter(hs2, ds2, c=shades2, s=14, edgecolors='none', zorder=2)

# direction arrows — one set per run, spacing matches existing (n//12)
def add_arrows(ax, hs, ds, color):
    n = len(hs)
    step = max(1, n // 12)
    for i in range(0, n - 1, step):
        ann = ax.annotate(
            '', xy=(hs[i + 1], ds[i + 1]), xytext=(hs[i], ds[i]),
            arrowprops=dict(arrowstyle='-|>', color=color, lw=2, mutation_scale=18),
            zorder=3,
        )
        ann.arrow_patch.set_path_effects(
            [pe.withStroke(linewidth=3, foreground='white'), pe.Normal()]
        )

add_arrows(ax, hs1, ds1, 'tab:blue')
add_arrows(ax, hs2, ds2, 'tab:orange')

# start / handoff / end markers
ax.scatter(hs1[0],  ds1[0],  s=60, color=shades1[0],  edgecolors='k', zorder=4, label='Start')
ax.scatter(hs1[-1], ds1[-1], s=70, color='0.5', marker='D', edgecolors='k', zorder=4, label='Handoff (427681 resume)')
ax.scatter(hs2[-1], ds2[-1], s=60, color=shades2[-1], edgecolors='k', zorder=4, label='End (427681)')

# endpoint label
ax.annotate(
    fr"$h={hs2[-1]:.4f}$" + "\n" + fr"$\Delta={ds2[-1]:.4f}$",
    xy=(hs2[-1], ds2[-1]), xycoords='data',
    xytext=(10, 10), textcoords='offset points',
    ha='left', va='bottom',
    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.4', alpha=0.95),
    arrowprops=dict(arrowstyle='-|>', color='black', lw=1.2, mutation_scale=18),
)

# NaN recovery kick markers
ax.scatter(nan_kick_origins[:, 1], nan_kick_origins[:, 0], s=80, color='mediumpurple',
           marker='X', edgecolors='indigo', linewidths=0.8, zorder=10, label='NaN kick origin (427681)')

# AE retraining markers
ax.scatter(ae_h, ae_d, s=120, color='lightpink', marker='*',
           edgecolors='lightpink', linewidths=0.5, zorder=10, label='AE retrained (427681)')

ax.set_xlim(0.2, -1.2)
ax.set_ylim(-2, 0)
ax.set_xlabel(r'$h$')
ax.set_ylabel(r'$\Delta$')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig('../figures/combined_trajectory_427681_plus_prev.png', bbox_inches='tight', dpi=300)
plt.close(fig)
print('Saved: ../figures/combined_trajectory_427681_plus_prev.png')
