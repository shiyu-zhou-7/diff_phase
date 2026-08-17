"""
Manuscript figures for the L=12 scaling experiment (appendix app:scaling).

Reads spt/scaling/data/summary.json (always) and data/histories.json (optional,
produced by extract_histories.py -- needed for the random-walk figure). Writes
PDFs (+PNG previews) into the paper repo's img/:

  scaling_cost.pdf          queries to full coverage vs d (uncovered = censored)
  scaling_coverage_map.pdf  visited windings per run: contiguous mid-ladder bands
  scaling_walk.pdf          crossings afforded vs (d/2)^2 required; band width vs sqrt(N)

Style follows the spt manuscript plots (plot_winding_timeline.py): fixed figsize,
fixed margins via subplots_adjust, tick 12 / label 15 pt, no tight bbox.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_steps import winding

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
IMG = os.path.abspath(os.path.join(
    HERE, '..', '..', '..', 'diffphase_paper_git', 'diffphase_manuscript', 'img'))

FS_TICK, FS_LABEL, FS_LEGEND = 12, 15, 10
BLUE, VERM = '#0072B2', '#D55E00'          # Okabe-Ito (CVD-safe)
DS = (4, 6, 8, 10, 12)


def t_init_for(seed, d):
    """Reproduce main_active_phase.py's RANDOM_INIT draw exactly."""
    t = np.random.default_rng(seed).standard_normal(d)
    return t / np.linalg.norm(t)


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(IMG, f'{name}.{ext}'),
                    dpi=(200 if ext == 'png' else None))
    plt.close(fig)
    print(f'  -> img/{name}.pdf')


# ---------------------------------------------------------------- figure 1
def fig_cost(runs):
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.92, bottom=0.12)
    for d in DS:
        rs = [r for r in runs if r['d'] == d]
        cov = [r['total'] for r in rs if r['covered']]
        unc = [r['total'] for r in rs if not r['covered']]
        jc = rng.uniform(-0.35, 0.35, len(cov))
        ju = rng.uniform(-0.35, 0.35, len(unc))
        ax.scatter(d + jc, cov, s=42, color=BLUE, zorder=3,
                   label='covered all $d$ phases' if d == DS[0] else None)
        ax.scatter(d + ju, unc, s=46, facecolors='none', edgecolors=VERM,
                   marker='^', linewidths=1.4, zorder=3,
                   label='budget exhausted (censored)' if unc and d == 6 else None)
        if cov:
            ax.errorbar(d, np.mean(cov), yerr=np.std(cov), fmt='_', ms=22,
                        color='k', elinewidth=1.6, capsize=6, zorder=4)
        ax.annotate(f'{len(cov)}/8', (d, 9750), ha='center',
                    fontsize=FS_TICK, color='k')
    ax.set_xlabel('number of couplings $d$', fontsize=FS_LABEL)
    ax.set_ylabel('solver queries', fontsize=FS_LABEL)
    ax.set_xticks(DS)
    ax.set_ylim(0, 10400)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(axis='y', alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=FS_LEGEND, loc='lower right', framealpha=0.9)
    save(fig, 'scaling_cost')


# ---------------------------------------------------------------- figure 2
def fig_coverage_map(runs):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.97, bottom=0.12)
    gap, xpos = 2, 0
    centers = []
    for d in DS:
        rs = sorted((r for r in runs if r['d'] == d), key=lambda r: r['seed'])
        for i, r in enumerate(rs):
            x = xpos + i
            visited = set(r['visited'])
            w0 = winding(t_init_for(r['seed'], d))
            for w in range(d):
                if w in visited:
                    ax.scatter(x, w, marker='s', s=34, color=BLUE, zorder=3)
                else:
                    ax.scatter(x, w, marker='s', s=34, facecolors='none',
                               edgecolors='0.75', linewidths=0.8, zorder=2)
            ax.scatter(x, w0, marker='*', s=60, color='k', zorder=4)
        centers.append(xpos + (len(rs) - 1) / 2)
        xpos += len(rs) + gap
    ax.set_xticks(centers)
    ax.set_xticklabels([f'$d={d}$' for d in DS], fontsize=FS_TICK + 1)
    ax.set_yticks(range(0, 12, 2))
    ax.set_ylabel(r'winding $\omega$', fontsize=FS_LABEL)
    ax.set_ylim(-0.7, 11.7)
    ax.tick_params(labelsize=FS_TICK)
    # legend proxies
    h = [plt.Line2D([], [], marker='s', ls='', color=BLUE, label='visited'),
         plt.Line2D([], [], marker='s', ls='', markerfacecolor='none',
                    markeredgecolor='0.6', label='missed'),
         plt.Line2D([], [], marker='*', ls='', color='k', ms=10, label='start')]
    ax.legend(handles=h, fontsize=FS_LEGEND, loc='upper left', framealpha=0.9)
    save(fig, 'scaling_coverage_map')


# ---------------------------------------------------------------- figure 3
def crossings_and_width(rec):
    """Boundary crossings N (omega changes along the trail incl. the start) and
    visited-band width W."""
    w0 = winding(np.asarray(rec['t_init']))
    om = [w0] + rec['omega_per_step']
    n_cross = int(np.sum(np.diff(om) != 0))
    return n_cross, int(max(om) - min(om) + 1)


def fig_walk(hist):
    cmap = plt.get_cmap('viridis')
    cols = {d: cmap(i / (len(DS) - 1) * 0.85) for i, d in enumerate(DS)}
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.5, 3.1))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.90, bottom=0.19, wspace=0.30)
    rng = np.random.default_rng(1)

    # (a) crossings afforded per run vs (d/2)^2 required to reach one extreme
    for d in DS:
        ns = [crossings_and_width(r)[0] for r in hist if r['d'] == d]
        ax = axa
        ax.scatter(d + rng.uniform(-0.3, 0.3, len(ns)), ns, s=30,
                   color=cols[d], zorder=3)
        ax.errorbar(d, np.mean(ns), yerr=np.std(ns), fmt='_', color='k',
                    ms=16, capsize=4, zorder=4)
    dd = np.linspace(3.5, 12.5, 100)
    axa.plot(dd, (dd / 2) ** 2, 'k--', lw=1.4,
             label=r'$(d/2)^2$ (required)')
    axa.set_xlabel('$d$', fontsize=FS_LABEL)
    axa.set_ylabel('boundary crossings $N$', fontsize=FS_LABEL)
    axa.set_xticks(DS)
    axa.tick_params(labelsize=FS_TICK)
    axa.legend(fontsize=FS_LEGEND, loc='upper left', framealpha=0.9)
    axa.set_title('(a)', fontsize=FS_TICK, loc='left')

    # (b) visited-band width vs sqrt(N) diffusive guide
    for d in DS:
        nw = [crossings_and_width(r) for r in hist if r['d'] == d]
        axb.scatter([n for n, _ in nw], [w for _, w in nw], s=30,
                    color=cols[d], zorder=3, label=f'$d={d}$')
    nn = np.linspace(1, max(crossings_and_width(r)[0] for r in hist) + 2, 100)
    axb.plot(nn, 1 + np.sqrt(nn), 'k--', lw=1.4, label=r'$1+\sqrt{N}$')
    axb.set_xlabel('boundary crossings $N$', fontsize=FS_LABEL)
    axb.set_ylabel(r'band width $W$', fontsize=FS_LABEL)
    axb.tick_params(labelsize=FS_TICK)
    axb.legend(fontsize=FS_LEGEND - 1, loc='lower right', framealpha=0.9,
               ncol=2, columnspacing=0.8, handletextpad=0.3)
    axb.set_title('(b)', fontsize=FS_TICK, loc='left')
    save(fig, 'scaling_walk')


if __name__ == '__main__':
    runs = json.load(open(os.path.join(DATA, 'summary.json')))
    os.makedirs(IMG, exist_ok=True)
    fig_cost(runs)
    fig_coverage_map(runs)
    hpath = os.path.join(DATA, 'histories.json')
    if os.path.exists(hpath):
        fig_walk(json.load(open(hpath)))
    else:
        print('  (histories.json not found -- skipping scaling_walk; '
              'run extract_histories.py on the cluster first)')
