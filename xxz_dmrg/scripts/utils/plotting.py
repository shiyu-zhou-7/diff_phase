import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

def plot_trajectory(deltas, hs, out_pdf):
    n = len(hs)
    cmap = plt.cm.Blues
    shades = cmap(np.linspace(0.3, 1.0, n))

    fig, ax = plt.subplots(figsize=(6,5), dpi=150)
    ax.plot(hs, deltas, lw=1.0, color='0.6', alpha=0.4, zorder=1)
    ax.scatter(hs, deltas, c=shades, s=14, edgecolors='none', zorder=2)

    # arrows every ~n/12 points
    step = max(1, n // 12)
    for i in range(0, n-1, step):
        ann = ax.annotate(
            '', xy=(hs[i+1], deltas[i+1]), xytext=(hs[i], deltas[i]),
            arrowprops=dict(arrowstyle='-|>', color='tab:blue', lw=2, mutation_scale=18),
            zorder=3
        )
        ann.arrow_patch.set_path_effects([pe.withStroke(linewidth=3, foreground='white'), pe.Normal()])

    # start/end
    ax.scatter(hs[0], deltas[0], s=60, color=shades[0],  edgecolors='k', zorder=4, label='Start')
    ax.scatter(hs[-1], deltas[-1], s=60, color=shades[-1], edgecolors='k', zorder=4, label='End')

    # endpoint two-line label adjacent
    hx, dx = hs[-1], deltas[-1]
    ax.annotate(fr"$h={hx:.5f}$"+"\n"+fr"$\Delta={dx:.5f}$",
                xy=(hx, dx), xycoords='data',
                xytext=(10, 10), textcoords='offset points',
                ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.4', alpha=0.95),
                arrowprops=dict(arrowstyle='-|>', color='black', lw=1.2, mutation_scale=18))

    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"$\Delta$")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches='tight', dpi=300)
    plt.close(fig)
