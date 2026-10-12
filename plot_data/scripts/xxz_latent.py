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
D=np.load(ROOT/'data/xxz_latent.npz', allow_pickle=False)
META=json.loads((ROOT/'data/xxz_latent.json').read_text())
P_train=D['train_pca']; P_traj=D['traj_pca']; train_phase=D['train_phase']; pca_evr=D['var_ratio']
PHASE_ORDER   = ['Neel_Z', 'FM_Z', 'Neel_Y', 'PM', 'Liquid']
PHASE_COLORS  = {
    'Neel_Z':  '#AA3377',
    'FM_Z':    '#4477AA',
    'Neel_Y':  '#228833',
    'PM':      '#888888',
    'Liquid':  'crimson',
}
PHASE_MARKERS = {
    'Neel_Z':  '^',
    'FM_Z':    'v',
    'Neel_Y':  's',
    'PM':      'o',
    'Liquid':  'D',
}

fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)

for ph in PHASE_ORDER:
    mask = (train_phase == ph)
    if not mask.any():
        continue
    ax.scatter(P_train[mask, 0], P_train[mask, 1],
               c=PHASE_COLORS[ph], marker=PHASE_MARKERS[ph],
               s=38, alpha=0.55, edgecolors='none',
               label=ph, zorder=1)

n_traj = P_traj.shape[0]
shades = plt.cm.Oranges(np.linspace(0.3, 1.0, n_traj))
ax.plot(P_traj[:, 0], P_traj[:, 1], lw=1.3, color='0.5', zorder=2)
ax.scatter(P_traj[:, 0], P_traj[:, 1],
           c=shades, s=10, edgecolors='none', zorder=3)

# Triangle tip arrows: (a) the K largest jumps in latent PCA space, plus
# (b) the steps closest in destination to the PM cluster centroid — so the
# trajectory's entry into the paramagnet shows up explicitly.
N_KEY     = 6
N_PM      = 2
steps     = np.diff(P_traj, axis=0)
jump_mag  = np.linalg.norm(steps, axis=1)

pm_mask   = (train_phase == 'PM')
pm_centroid = P_train[pm_mask].mean(axis=0)
dist_to_pm = np.linalg.norm(P_traj[1:] - pm_centroid, axis=1)

arrow_idx = set(np.argsort(jump_mag)[-N_KEY:].tolist())
arrow_idx |= set(np.argsort(dist_to_pm)[:N_PM].tolist())

# One more arrow at the sharpest direction reversal among PM-region steps —
# the "turning point" deep in para. Score = -cos(angle between consecutive
# step vectors), restricted to steps whose endpoint sits in the half of the
# trajectory closest to the PM centroid.
step_dirs  = steps / (jump_mag[:, None] + 1e-12)
cos_angles = np.sum(step_dirs[:-1] * step_dirs[1:], axis=1)   # len n_traj - 2
pm_cut     = np.median(dist_to_pm)
candidate  = np.where(dist_to_pm[1:] < pm_cut)[0]              # indices into cos_angles
if len(candidate):
    turn_step = int(candidate[np.argmin(cos_angles[candidate])])
    # Use the *outgoing* step from the turning point so the arrow points in
    # the post-turn (opposite) direction.
    arrow_idx.add(turn_step + 1)
for i in sorted(arrow_idx):
    direction = steps[i] / (jump_mag[i] + 1e-12)
    head      = P_traj[i + 1]
    tail      = head - 1e-3 * direction   # tiny offset → arrow has direction but no visible tail
    ann = ax.annotate(
        '',
        xy=tuple(head), xytext=tuple(tail),
        arrowprops=dict(arrowstyle='-|>', color='tab:orange',
                        lw=0, mutation_scale=30),
        zorder=4,
    )
    ann.arrow_patch.set_path_effects(
        [pe.withStroke(linewidth=2, foreground='white'), pe.Normal()]
    )

# start = filled square, end = filled triangle (size matching Hamiltonian-space figures)
# ax.scatter(P_traj[0, 0],  P_traj[0, 1],  s=70,
#            color=shades[0],  edgecolors='k', zorder=5, label='Start')  # old: circle
# ax.scatter(P_traj[-1, 0], P_traj[-1, 1], s=70,
#            color=shades[-1], edgecolors='k', zorder=5, label='End')  # old: circle
ax.scatter(P_traj[0, 0],  P_traj[0, 1],  marker='s', s=110,
           color=shades[0],  edgecolors='k', zorder=5, label='Start')
ax.scatter(P_traj[-1, 0], P_traj[-1, 1], marker='^', s=140,
           color=shades[-1], edgecolors='k', zorder=5, label='End')

ax.set_xlabel(f'PC1 ({100*pca_evr[0]:.1f}%)', fontsize=24)
ax.set_ylabel(f'PC2 ({100*pca_evr[1]:.1f}%)', fontsize=24)
ax.tick_params(labelsize=20)
ax.grid(True, alpha=0.3)
legend = ax.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.74, 1.0),
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
# fig.tight_layout()  # old: per-figure auto-margins -> axes box varies between figures
MANUSCRIPT_MARGINS = dict(left=0.23, right=0.965, bottom=0.20, top=0.965)
fig.subplots_adjust(**MANUSCRIPT_MARGINS)


fig.savefig(OUTPUT/'xxz_latent.pdf')
fig.savefig(OUTPUT/'xxz_latent.png',dpi=160)
plt.close(fig)
