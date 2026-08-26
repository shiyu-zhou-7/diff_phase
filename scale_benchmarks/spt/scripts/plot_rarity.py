"""Rarity figure: the uniform measure over winding sectors, and how far each
method reaches into its tails. Small multiples over d (same encoding in each).
Colours: categorical slots 1-3, same entity->hue map as fig_main."""
import json, glob, collections
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
AGENT, RAND, UNIF = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
mpl.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.linewidth": .8,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.labelcolor": INK2, "figure.facecolor": "white",
                     "axes.facecolor": "white", "font.family": "DejaVu Sans"})
H = collections.defaultdict(lambda: None); NT = collections.defaultdict(int)
for f in glob.glob("hist_res/h_d*_r*.json"):
    j = json.load(open(f)); d = j["d"]; h = np.array(j["hist"], dtype=np.int64)
    H[d] = h if H[d] is None else H[d] + h; NT[d] += j["N"]
W = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob("which_res/w_d*.json"):
    j = json.load(open(f)); W[j["d"]][j["start"]].append(set(j["phases"]))

DS = [d for d in (60, 100, 140) if d in H and d in W]
fig, ax = plt.subplots(1, len(DS), figsize=(4.0 * len(DS), 3.5), sharey=True)
for a, d in zip(np.atleast_1d(ax), DS):
    h = H[d][:d]; N = NT[d]; p = h / N; s = np.flatnonzero(h)
    a.axhspan(2e-10, 1 / N, color=GRID, alpha=.55, lw=0, zorder=0)
    a.text(d * .97, 1.5 / N, f"below the {N/1e6:.0f}M-draw detection limit", fontsize=6.4,
           color=MUTED, ha="right", va="bottom")
    a.plot(s, p[s], "-", color=UNIF, lw=1.5, zorder=3)
    a.fill_between(s, 2e-10, p[s], color=UNIF, alpha=.13, lw=0, zorder=2)
    for st, c, y, lab in (("rand", RAND, 1.1e-8, "agent, random start"),
                          ("e0", AGENT, 1.6e-9, "agent, $e_0$ start")):
        S = sorted(set().union(*W[d][st]))
        S = [w for w in S if not (st == "e0" and w == 0)]
        a.plot(S, [y] * len(S), "|", color=c, ms=7, mew=1.4, zorder=4,
               label=lab if d == DS[0] else None)
        ext = [w for w in S if h[w] == 0]
        a.plot(ext, [y] * len(ext), "o", color=c, ms=3.4, zorder=5)
    a.plot(s, [7.5e-8] * len(s), "|", color=UNIF, ms=7, mew=1.4, zorder=4,
           label="uniform, 4M draws" if d == DS[0] else None)
    a.set_yscale("log"); a.set_ylim(4e-10, .8); a.set_xlim(-2, d + 2)
    a.set_xlabel("winding sector $\\omega$")
    a.set_title(f"$d = {d}$", loc="left", fontsize=9.5, color=INK, pad=6)
    a.grid(True, color=GRID, lw=.7); a.set_axisbelow(True)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
np.atleast_1d(ax)[0].set_ylabel("$p_\\omega$  under uniform sampling")
np.atleast_1d(ax)[0].legend(frameon=False, fontsize=7.4, loc="upper left",
                            labelcolor=INK2, handlelength=1.2, ncol=1)
fig.suptitle("The agent reaches sectors that four million uniform draws never hit",
             fontsize=11, color=INK, y=1.02, x=.008, ha="left")
fig.text(.008, -.05, "green curve = measured $p_\\omega$ (4$\\times$10$^6$ draws); ticks = sectors reached; "
         "filled dots = sectors never seen in 4M draws.  Agent budget 10$^5$ solves, union over 4 seeds; "
         "$\\omega=0$ excluded from the $e_0$ row (it is the start, not a discovery).",
         fontsize=7.2, color=MUTED, ha="left")
fig.tight_layout(rect=[0, .01, 1, .96])
fig.savefig("fig_rarity.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig("fig_rarity.pdf", bbox_inches="tight", facecolor="white")
print("written")
