"""Plot exported numerical data; no models, pickle, JAX, or solvers required."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=Path(globals().get('OUTPUT', ROOT/'figures'))
OUTPUT.mkdir(parents=True,exist_ok=True)
D=np.load(ROOT/'data/tfim_latent.npz', allow_pickle=False)
META=json.loads((ROOT/'data/tfim_latent.json').read_text())
z_sub_pca=D['train_pca']; traj_pca=D['traj_pca']; var_ratio=D['var_ratio']; traj_event_tags=D['traj_event_tags']
n_traj=len(traj_pca); arrow_stride=5
def _draw_background(ax, fig):
    """Plot the 1000 sub-samples as phase-shaped scatters and label each cluster."""
    ax.scatter(z_sub_pca[:500, 0], z_sub_pca[:500, 1],
               color='tab:blue', marker='v', s=49, alpha=0.55,
               zorder=1, label='SSB')
    ax.scatter(z_sub_pca[500:, 0], z_sub_pca[500:, 1],
               color='gray', marker='o', s=42, alpha=0.55,
               zorder=1, label='PARA')
    ax.set_xlabel(f'PC1 ({var_ratio[0]:.1%})', fontsize=24)
    ax.set_ylabel(f'PC2 ({var_ratio[1]:.1%})', fontsize=24)
    ax.tick_params(labelsize=20)
    ax.grid(True, alpha=0.3)




traj_color = 'tab:orange'
start_color = plt.cm.Oranges(0.45)
end_color = plt.cm.Oranges(0.90)

fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
_draw_background(ax, fig)
# Narrower y-ticks (1 decimal, 0.5 spacing) so the rotated PC2 label fits
# within the fixed left margin without moving the axes box.
ax.set_yticks(np.arange(-1.0, 1.01, 0.5))

# Grey trajectory path, matching the XXZ latent-space panel and the
# 1.3-pt path weight used in the SPT latent-space panel.
if n_traj > 1:
    pts = traj_pca[:, np.newaxis, :]
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, colors='0.5', linewidths=1.3, zorder=3)
    ax.add_collection(lc)

# Small per-step dots, colored by time as in the XXZ panel.
ax.scatter(traj_pca[:, 0], traj_pca[:, 1],
           color=traj_color, s=10, edgecolors='none', zorder=4)

# Start = large circle, color = trajectory color at t=0.
# start = filled square, end = filled triangle (size matching Hamiltonian-space figures)
# ax.scatter([traj_pca[0, 0]], [traj_pca[0, 1]],
#            c=[colors_along[0]], marker='o', s=140,
#            edgecolor='black', lw=1.0, zorder=10, label='start')  # old: circle
ax.scatter([traj_pca[0, 0]], [traj_pca[0, 1]],
           color=start_color, marker='s', s=110,
           edgecolor='black', lw=1.0, zorder=10, label='start')
# End = large circle, color = trajectory color at t=1.
# ax.scatter([traj_pca[-1, 0]], [traj_pca[-1, 1]],
#            c=[colors_along[-1]], marker='o', s=140,
#            edgecolor='black', lw=1.0, zorder=10, label='end')  # old: circle
ax.scatter([traj_pca[-1, 0]], [traj_pca[-1, 1]],
           color=end_color, marker='^', s=140,
           edgecolor='black', lw=1.0, zorder=10, label='end')

# Direction triangles (no tail/shaft) every ARROW_STRIDE kept points.
# Each triangle marker is a 3-sided polygon rotated to point in the
# direction of motion from point i to point i+1.
for i in range(0, n_traj - 1, arrow_stride):
    j = min(i + 1, n_traj - 1)
    dx = traj_pca[j, 0] - traj_pca[i, 0]
    dy = traj_pca[j, 1] - traj_pca[i, 1]
    if dx == 0.0 and dy == 0.0:
        continue
    # matplotlib's (3, 0, angle) marker has a default vertex pointing
    # "up"; rotate by (direction - 90°) to point in (dx, dy).
    angle = float(np.degrees(np.arctan2(dy, dx)) - 90.0)
    mx = (traj_pca[i, 0] + traj_pca[j, 0]) / 2.0
    my = (traj_pca[i, 1] + traj_pca[j, 1]) / 2.0
    ax.scatter([mx], [my], marker=(3, 0, angle), s=320,
               color=traj_color,
               edgecolor='white', lw=1.5, zorder=5)

# NaN kicks (rare; emit red Xs if any).
nan_traj_indices = [k for k, tag in enumerate(traj_event_tags) if tag == 'nan_kick']
if nan_traj_indices:
    nx = [traj_pca[k, 0] for k in nan_traj_indices]
    ny = [traj_pca[k, 1] for k in nan_traj_indices]
    ax.scatter(nx, ny, c='red', marker='X', s=100, zorder=6, label='NaN kick')

legend = ax.legend(loc='upper right', fontsize=16, frameon=True,
                   framealpha=0.9, handletextpad=0.4,
                   labelspacing=0.3, borderpad=0.4)
for text in legend.get_texts():
    if text.get_text().lower() in ('start', 'end'):
        text.set_fontsize(16)
legend_handles = getattr(legend, 'legend_handles', getattr(legend, 'legendHandles', []))
for handle, text in zip(legend_handles, legend.get_texts()):
    if text.get_text().lower() not in ('start', 'end'):
        if hasattr(handle, 'set_sizes'):
            handle.set_sizes([40])
        elif hasattr(handle, 'set_markersize'):
            handle.set_markersize(7)

# --- uniform axes box across all manuscript figures (shared margins) ---
MANUSCRIPT_MARGINS = dict(left=0.23, right=0.965, bottom=0.20, top=0.965)
fig.subplots_adjust(**MANUSCRIPT_MARGINS)

fig.savefig(OUTPUT/'tfim_latent.pdf')
fig.savefig(OUTPUT/'tfim_latent.png',dpi=160)
plt.close(fig)
