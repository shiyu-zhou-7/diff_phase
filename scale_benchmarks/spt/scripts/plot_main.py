"""Main figure. Palette = dataviz reference instance.
 panel a: budget is an ORDERED magnitude -> single-hue blue ordinal ramp
          (steps 250/400/550/700; light-mode ordinal floor = step 250 respected);
          agent/uniform carried by linestyle, not colour.
 panels b,c: method is IDENTITY -> categorical slots 1-3 (blue/orange/aqua),
          documented all-pairs CVD-safe in light mode; same hue = same entity in both."""
import json, glob, collections
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt

RAMP = {10000: "#86b6ef", 30000: "#3987e5", 100000: "#1c5cab", 300000: "#0d366b"}
AGENT, RAND, UNIF = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
mpl.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.linewidth": .8,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.labelcolor": INK2, "figure.facecolor": "white",
                     "axes.facecolor": "white", "font.family": "DejaVu Sans",
                     "legend.title_fontsize": 7.6})

A = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob("final_res/f_*.json"):
    j = json.load(open(f)); A[j["budget"]][j["d"]].append(j["cov"])
U = {int(b): {int(d): v for d, v in c.items()} for b, c in json.load(open("unif_all.json")).items()}
W = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob("which_res/w_d*.json"):
    j = json.load(open(f)); W[j["d"]][j["start"]].append(j["cov"])

BUD = [10000, 30000, 100000, 300000]
LAB = {10000: "10k", 30000: "30k", 100000: "100k", 300000: "300k"}
fig, ax = plt.subplots(1, 3, figsize=(12.0, 3.8))

def band(a, xs, vals, c, ls, mk, lw=1.6, alpha=.15, ms=4.5, **kw):
    a.fill_between(xs, [min(vals[k]) for k in vals], [max(vals[k]) for k in vals],
                   color=c, alpha=alpha, lw=0)
    a.plot(xs, [np.median(vals[k]) for k in vals], ls, color=c, lw=lw,
           marker=mk, ms=ms, mfc="white", mew=1.3, **kw)

# ---- a  coverage across the grid ---------------------------------------
a = ax[0]
for b in BUD:
    ds = sorted(A[b])
    band(a, ds, {d: A[b][d] for d in ds}, RAMP[b], "-", "o", label=LAB[b])
    band(a, ds, {d: U[b][d] for d in ds}, RAMP[b], "--", "s", lw=1.2, alpha=.09)
lg = a.legend(frameon=False, fontsize=7.6, loc="upper left", labelcolor=INK2,
              handlelength=1.9, title="query budget", ncol=2, columnspacing=1.1)
lg.get_title().set_color(INK2)
a.set_xlabel("$d$  (number of phases present)"); a.set_ylabel("phases discovered")
a.set_title("a   Coverage across the grid", loc="left", fontsize=10, color=INK, pad=8)
a.annotate("agent", (200, 96), (186, 105), color=INK2, fontsize=8.5, ha="center")
a.annotate("uniform", (200, 71), (194, 58), color=INK2, fontsize=8.5, ha="center")

# ---- b  efficiency at fixed d ------------------------------------------
a = ax[1]; D0 = 200
band(a, BUD, {b: A[b][D0] for b in BUD}, AGENT, "-", "o", label="agent ($e_0$ start)")
band(a, BUD, {b: U[b][D0] for b in BUD}, UNIF, "--", "s", label="uniform sampling")
y0 = np.median(A[10000][D0])
a.axhline(y0, color=AGENT, lw=.9, ls=(0, (2, 3)), zorder=0)
a.annotate("", (10000, y0 - 2.6), (300000, y0 - 2.6),
           arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.1))
a.text(5.5e4, y0 - 7.4, "uniform still short at 30$\\times$ the budget\n(log fit: $\\sim$200$\\times$ to match)",
       fontsize=7.6, color=INK2, ha="center")
a.set_xscale("log"); a.set_xlabel("query budget  (ground-state solves)")
a.set_ylabel(f"phases discovered  ($d={D0}$)")
a.set_title("b   Sampling efficiency", loc="left", fontsize=10, color=INK, pad=8)
a.legend(frameon=False, fontsize=7.6, loc="lower right", labelcolor=INK2, handlelength=1.9, title="method").get_title().set_color(INK2)
a.set_xticks(BUD); a.set_xticklabels([LAB[b] for b in BUD], fontsize=8)
a.set_ylim(53, 103)

# ---- c  where the margin comes from ------------------------------------
a = ax[2]; ds = sorted(W)
for k, lab, c, ls, mk in (("e0", "agent, $e_0$ start", AGENT, "-", "o"),
                          ("rand", "agent, random start", RAND, "-", "^"),
                          ("unif", "uniform sampling", UNIF, "--", "s")):
    v = {d: (U[100000][d] if k == "unif" else W[d][k]) for d in ds}
    band(a, ds, v, c, ls, mk, label=lab)
e, r, u = (np.median(W[200]["e0"]), np.median(W[200]["rand"]), np.median(U[100000][200]))
for lo, hi, c, txt in ((u, r, RAND, f"+{r-u:.0f}\nbasis-free"), (r, e, AGENT, f"+{e-r:.0f}\nfrom $e_0$")):
    a.annotate("", (209, lo), (209, hi), arrowprops=dict(arrowstyle="<->", color=c, lw=1.1))
    a.text(213, (lo + hi) / 2, txt, fontsize=7.4, color=c, va="center")
a.set_xlim(50, 250); a.set_xlabel("$d$"); a.set_ylabel("phases discovered")
a.set_title("c   Where the margin comes from  (100k)", loc="left", fontsize=10, color=INK, pad=8)
a.legend(frameon=False, fontsize=7.6, loc="upper left", labelcolor=INK2, handlelength=1.9, title="method").get_title().set_color(INK2)

for a in ax:
    a.grid(True, color=GRID, lw=.7); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
fig.suptitle("Label-free phase discovery beats uniform sampling by $\\sim\\!30\\times$ in queries, and the gap widens with $d$",
             fontsize=11, color=INK, y=1.005, x=.010, ha="left")
fig.text(.010, -.045, "long-range Kitaev chain (class BDI), $L=500$, lazy factor $k=30$;  "
         "agent: 5 seeds (panel c: 4);  uniform: 5 independent realisations;  bands = min-max over seeds/realisations",
         fontsize=7.6, color=MUTED, ha="left")
fig.tight_layout(rect=[0, .01, 1, .97])
fig.savefig("fig_main.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig("fig_main.pdf", bbox_inches="tight", facecolor="white")
print("written")
