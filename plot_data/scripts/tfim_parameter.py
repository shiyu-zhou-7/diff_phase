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
D=np.load(ROOT/'data/tfim_parameter.npz', allow_pickle=False)
META=json.loads((ROOT/'data/tfim_parameter.json').read_text())
def load_run(i):
    return {'cfg_snapshot':{'h_init':D[f'run{i}_values'][0]},
            'history':{'h_per_step':D[f'run{i}_values'][1:].tolist(),
                       'event_per_step':D[f'run{i}_event'][1:].tolist()}}
PHASE_BOUNDARY=0.329

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
        'path': '../data/active_phase_hm1.50_20260512_101827.pkl',
        'max_steps': None,
        'color': 'tab:green',
        'label': r'$g_0 = -1.5$',
    },
]


fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)

for run in RUNS:
    bundle = load_run(RUNS.index(run))
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
        ax.scatter(rx, ry, color=run['color'], marker='*', s=165,
                   edgecolor='black', lw=0.6, zorder=5)

    # Start = filled square, end = filled triangle, both in the run color.
    ax.scatter([x[0]], [h_full[0]], color=run['color'], marker='s', s=110,
               edgecolor='black', lw=0.9, zorder=6)
    ax.scatter([x[-1]], [h_full[-1]], color=run['color'], marker='^', s=140,
               edgecolor='black', lw=0.9, zorder=6)

# Phase backgrounds, using the same low-opacity phase tinting as the XXZ and
# SPT manuscript panels.  In the negative-g convention used here, g < -1 is
# paramagnetic and g > -1 is symmetry-broken.  Deriving the span bounds from
# the trajectory first keeps the shading from changing the axes limits.
ymin, ymax = ax.get_ylim()
ax.axhspan(ymin, -1.0, color='#999999', alpha=0.10, linewidth=0, zorder=0)
ax.axhspan(-1.0, ymax, color='#4477AA', alpha=0.10, linewidth=0, zorder=0)
ax.set_ylim(ymin, ymax)

# Phase boundary line (no text annotation), matching the solid black XXZ
# critical lines.
ax.axhline(-1.0, color='black', ls='-', lw=1.8, zorder=2)


ax.set_xlim(-0.025, 1.035)
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlabel('step (normalized)', fontsize=24)
ax.set_ylabel(r'$g$', fontsize=24)
ax.tick_params(labelsize=20)
ax.grid(True, alpha=0.3)

# Legend: four per-run color entries + a universal 'retrain' glyph (5 total),
# laid out in 3 columns at lower-left.
handles, labels = ax.get_legend_handles_labels()
handles.append(Line2D([0], [0], marker='*', markerfacecolor='lightgray',
                      markeredgecolor='black', color='black', lw=0, markersize=7,
                      label='Bootstrapping'))
legend = ax.legend(handles=handles, loc='lower left', fontsize=16, frameon=True,
                   framealpha=0.9, ncol=2, columnspacing=0.8,
                   handletextpad=0.4, labelspacing=0.3, borderpad=0.4)
legend_handles = getattr(legend, 'legend_handles', getattr(legend, 'legendHandles', []))
for handle, text in zip(legend_handles, legend.get_texts()):
    if hasattr(handle, 'set_sizes'):
        handle.set_sizes([165 if text.get_text() == 'Bootstrapping' else 40])
    elif hasattr(handle, 'set_markersize'):
        handle.set_markersize(np.sqrt(165) if text.get_text() == 'Bootstrapping' else 7)


fig.subplots_adjust(left=0.23,right=0.965,bottom=0.20,top=0.965)

fig.savefig(OUTPUT/'tfim_parameter.pdf')
fig.savefig(OUTPUT/'tfim_parameter.png',dpi=160)
plt.close(fig)
