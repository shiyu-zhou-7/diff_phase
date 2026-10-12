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
D=np.load(ROOT/'data/z2_latent.npz', allow_pickle=False)
META=json.loads((ROOT/'data/z2_latent.json').read_text())
Z2_conf=D['train_pca'][D['train_phase']=='confined']; Z2_dec=D['train_pca'][D['train_phase']=='deconfined']
traj_pca=D['traj_pca']; var_ratio=D['var_ratio']; n_traj=len(traj_pca); ARROW_STRIDE=5
traj_color = 'tab:green'
start_color = plt.cm.Greens(0.45)
end_color = plt.cm.Greens(0.90)

fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)

# Confined red matches the XXZ latent-space Liquid red; deconfined blue matches
# the TFIM latent-space SSB blue.
ax.scatter(Z2_conf[:, 0], Z2_conf[:, 1], color='gray', marker='o', s=42,
           alpha=0.55, zorder=1, label='confined')
ax.scatter(Z2_dec[:, 0], Z2_dec[:, 1], color='crimson', marker='v', s=49,
           alpha=0.55, zorder=1, label='deconfined')

if n_traj > 1:
    pts = traj_pca[:, np.newaxis, :]
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    ax.add_collection(LineCollection(segs, colors='0.5',
                                     linewidths=1.3, zorder=3))
ax.scatter(traj_pca[:, 0], traj_pca[:, 1], color=traj_color, s=10,
           edgecolors='none', zorder=4)

# start = filled square, end = filled triangle (size matching Hamiltonian-space figures)
# ax.scatter([traj_pca[0, 0]], [traj_pca[0, 1]], c=[colors_along[0]], marker='o',
#            s=140, edgecolor='black', lw=1.0, zorder=10, label='start')  # old: circle
# ax.scatter([traj_pca[-1, 0]], [traj_pca[-1, 1]], c=[colors_along[-1]], marker='o',
#            s=140, edgecolor='black', lw=1.0, zorder=10, label='end')  # old: circle
ax.scatter([traj_pca[0, 0]], [traj_pca[0, 1]], color=start_color, marker='s',
           s=110, edgecolor='black', lw=1.0, zorder=10, label='start')
ax.scatter([traj_pca[-1, 0]], [traj_pca[-1, 1]], color=end_color, marker='^',
           s=140, edgecolor='black', lw=1.0, zorder=10, label='end')

for i in range(0, n_traj - 1, ARROW_STRIDE):
    j = min(i + 1, n_traj - 1)
    dx, dy = traj_pca[j] - traj_pca[i]
    if dx == 0.0 and dy == 0.0:
        continue
    angle = float(np.degrees(np.arctan2(dy, dx)) - 90.0)
    mx, my = (traj_pca[i] + traj_pca[j]) / 2.0
    ax.scatter([mx], [my], marker=(3, 0, angle), s=320, color=traj_color,
               edgecolor='white', lw=1.5, zorder=5)

ax.set_xlabel(f'PC1 ({var_ratio[0]:.1%})', fontsize=24)
ax.set_ylabel(f'PC2 ({var_ratio[1]:.1%})', fontsize=24)
ax.tick_params(labelsize=20)
ax.grid(True, alpha=0.3)
legend = ax.legend(loc='best', fontsize=16, frameon=True, framealpha=0.9,
                   handletextpad=0.4, labelspacing=0.3, borderpad=0.4)
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

fig.savefig(OUTPUT/'z2_latent.pdf')
fig.savefig(OUTPUT/'z2_latent.png',dpi=160)
plt.close(fig)
