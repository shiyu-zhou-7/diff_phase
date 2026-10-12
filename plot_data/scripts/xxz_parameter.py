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
D=np.load(ROOT/'data/xxz_parameter.npz', allow_pickle=False)
META=json.loads((ROOT/'data/xxz_parameter.json').read_text())
from phase_diagram import draw_phase_diagram
prev_params=D['prev_params']; run2_params=D['run2_params']
nan_kick_origins=D['nan_kick_origins']; ae_spots=D['ae_spots']
JOB='437956'
cfg=dict(xlim=(-2,2),ylim=(-2,2),end_label='End',kick_label='NaN kick',ae_label='Bootstrapping')

n1 = len(prev_params)
n2 = len(run2_params)

hs1, ds1 = prev_params[:, 1], prev_params[:, 0]
hs2, ds2 = run2_params[:, 1], run2_params[:, 0]
ae_h, ae_d = (ae_spots[:, 1], ae_spots[:, 0]) if len(ae_spots) else (np.array([]), np.array([]))

shades1 = plt.cm.Oranges(np.linspace(0.3, 1.0, n1))
shades2 = plt.cm.Oranges(np.linspace(0.3, 1.0, n2))

fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)

h_range = (min(cfg['xlim']), max(cfg['xlim']))
d_range = (min(cfg['ylim']), max(cfg['ylim']))
draw_phase_diagram(ax, h_range=h_range, delta_range=d_range,
                   show_regions=True, label_regions=True)

ax.plot(hs1, ds1, lw=1.0, color='0.6', alpha=0.4, zorder=1)
ax.plot(hs2, ds2, lw=1.0, color='0.6', alpha=0.4, zorder=1)

ax.scatter(hs1, ds1, c=shades1, s=14, edgecolors='none', zorder=2)
ax.scatter(hs2, ds2, c=shades2, s=14, edgecolors='none', zorder=2)


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


add_arrows(ax, hs1, ds1, 'tab:orange')
add_arrows(ax, hs2, ds2, 'tab:orange')

# start = filled square, end = filled triangle (matching combined_active_phase_trajectory figures)
# ax.scatter(hs1[0],  ds1[0],  s=60, color=shades1[0], edgecolors='k', zorder=4, label='Start')  # old: circle
# ax.scatter(hs2[-1], ds2[-1], s=60, color=shades2[-1], edgecolors='k', zorder=4, label=cfg['end_label'])  # old: circle
ax.scatter(hs1[0],  ds1[0],  marker='s', s=110, color=shades1[0],  edgecolors='k', zorder=4, label='Start')
ax.scatter(hs2[-1], ds2[-1], marker='^', s=140, color=shades2[-1], edgecolors='k', zorder=4, label=cfg['end_label'])

# ax.annotate(
#     fr"$h={hs2[-1]:.2f}$" + "\n" + fr"$\Delta={ds2[-1]:.2f}$",
#     xy=(hs2[-1], ds2[-1]), xycoords='data',
#     xytext=(60, 15), textcoords='offset points',
#     ha='left', va='top',
#     bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.4', alpha=0.95),
#     arrowprops=dict(arrowstyle='-|>', color='black', lw=1.2, mutation_scale=18),
# )

# ax.annotate(
#     fr"$h={hs1[0]:.2f}$" + "\n" + fr"$\Delta={ds1[0]:.2f}$",
#     xy=(hs1[0], ds1[0]), xycoords='data',
#     xytext=(-30, 5), textcoords='offset points',
#     ha='right', va='top',
#     bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.4', alpha=0.95),
#     arrowprops=dict(arrowstyle='-|>', color='black', lw=1.2, mutation_scale=18),
# )

if len(nan_kick_origins):
    ax.scatter(nan_kick_origins[:, 1], nan_kick_origins[:, 0], s=120, color='mediumpurple',
               marker='X', edgecolors='indigo', linewidths=0.8, zorder=10, label=cfg['kick_label'])

if len(ae_spots):
    ax.scatter(ae_h, ae_d, s=247.5, color='deepskyblue', marker='*',
               edgecolors='black', linewidths=0.6, zorder=10, label=cfg['ae_label'])

ax.set_xlim(*cfg['xlim'])
ax.set_ylim(*cfg['ylim'])
if JOB == '437956':
    manuscript_ticks = [-2, -1, 0, 1, 2]
    ax.set_xticks(manuscript_ticks)
    ax.set_yticks(manuscript_ticks)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.set_xlabel(r'$h$', fontsize=24)
ax.set_ylabel(r'$\Delta$', fontsize=24)
ax.tick_params(labelsize=20)
ax.grid(True, alpha=0.3)
legend = ax.legend(fontsize=16, loc='lower right', framealpha=0.9,
                   handletextpad=0.4, labelspacing=0.3, borderpad=0.4)
for text in legend.get_texts():
    if text.get_text().lower().startswith(('start', 'end')):
        text.set_fontsize(16)
    elif text.get_text() == 'NaN kick':
        text.set_fontsize(16)
legend_handles = getattr(legend, 'legend_handles', getattr(legend, 'legendHandles', []))
for handle, text in zip(legend_handles, legend.get_texts()):
    if not text.get_text().lower().startswith(('start', 'end')):
        if hasattr(handle, 'set_sizes'):
            if text.get_text() == 'Bootstrapping':
                handle.set_sizes([165])
            elif text.get_text() == 'NaN kick':
                handle.set_sizes([90])
            else:
                handle.set_sizes([40])
        elif hasattr(handle, 'set_markersize'):
            if text.get_text() == 'Bootstrapping':
                handle.set_markersize(np.sqrt(165))
            elif text.get_text() == 'NaN kick':
                handle.set_markersize(15.75)
            else:
                handle.set_markersize(7)
# --- uniform axes box across all manuscript figures (shared margins) ---
# fig.tight_layout()  # old: per-figure auto-margins -> axes box varies between figures
MANUSCRIPT_MARGINS = dict(left=0.23, right=0.965, bottom=0.20, top=0.965)
fig.subplots_adjust(**MANUSCRIPT_MARGINS)

fig.savefig(OUTPUT/'xxz_parameter.pdf')
fig.savefig(OUTPUT/'xxz_parameter.png',dpi=160)
plt.close(fig)
