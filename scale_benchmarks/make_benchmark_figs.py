"""Benchmark figures for Sec. IV (scale_v2.tex) and the appendix.

Data lives outside this repo (grid_baseline of the benchmark campaign); the
rendered PDFs are committed under img/ like every other data figure. Style
follows the existing results figures: boxed axes, light grid, framed legends,
tab10 colors, DejaVu Sans.

Outputs:
  img/results_figure_scale_2x1.pdf   main Fig: (a) BDI coverage, (b) Chern budget curve
  img/results_figure_chern_volume.pdf main Fig: rare-sector volume bins
"""
import json, glob, collections, statistics as st
import numpy as np, matplotlib as mpl, matplotlib.pyplot as plt

G = "./"   # data root: point at scale_benchmarks/ (see README)
BLUE, RED, ORANGE = "#1f77b4", "#d62728", "#ff7f0e"
mpl.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans",
                     "axes.linewidth": .8, "legend.framealpha": .9,
                     "legend.edgecolor": "#cccccc", "figure.facecolor": "white"})

def house(ax):
    ax.grid(True, color="#dddddd", lw=.6, alpha=.8); ax.set_axisbelow(True)

def band(ax, xs, vals, c, ls, mk, lab=None, lw=1.6, ms=4):
    ax.fill_between(xs, [min(vals[k]) for k in xs], [max(vals[k]) for k in xs],
                    color=c, alpha=.13, lw=0)
    ax.plot(xs, [st.median(vals[k]) for k in xs], ls, color=c, lw=lw,
            marker=mk, ms=ms, mfc="white", mew=1.1, label=lab)

# ---------------- BDI data ----------------
A = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(G + "final_res/f_*.json"):
    j = json.load(open(f)); A[j["budget"]][j["d"]].append(j["cov"])
U = {int(b): {int(d): v for d, v in c.items()}
     for b, c in json.load(open(G + "unif_all.json")).items()}
W = collections.defaultdict(lambda: collections.defaultdict(list))
WS = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(G + "which_res/w_d*.json"):
    j = json.load(open(f))
    W[j["d"]][j["start"]].append(j["cov"])
    WS[j["d"]][j["start"]].append(set(j["phases"]))
H = {}; NT = collections.defaultdict(int)
for f in glob.glob(G + "hist_res/h_d*_r*.json"):
    j = json.load(open(f)); d = j["d"]; h = np.array(j["hist"], dtype=np.int64)
    H[d] = h if d not in H else H[d] + h; NT[d] += j["N"]
BUD = [10000, 30000, 100000, 300000]
RAMP  = {10000: "#a6c8e8", 30000: "#6ba3d6", 100000: "#3d7fbf", 300000: "#1f5f99"}
RAMPR = {10000: "#f4b8b8", 30000: "#e88686", 100000: "#d64d4d", 300000: "#b71c1c"}
LABK = {10000: "10k", 30000: "30k", 100000: "100k", 300000: "300k"}

# ---------------- Chern data ----------------
SEEDS = [200, 201, 202, 203, 204]
cum = {s: {"agent": set(), "uniform": set()} for s in SEEDS}
bs = []; AM, UM, ALO, AHI, ULO, UHI = [], [], [], [], [], []
for j in range(10):
    for s in SEEDS:
        dd = json.load(open(G + f"chern_res_M768/relab_s{s}_c{j}.json"))
        for t in ("agent", "uniform"): cum[s][t] |= set(dd[t]["set_conv"])
    Av = [len(cum[s]["agent"]) for s in SEEDS]
    Uv = [len(cum[s]["uniform"]) for s in SEEDS]
    bs.append((j+1)*30000)
    AM.append(st.median(Av)); UM.append(st.median(Uv))
    ALO.append(min(Av)); AHI.append(max(Av)); ULO.append(min(Uv)); UHI.append(max(Uv))
vol = {int(k): v for k, v in json.load(open(G + "sector_volumes_M768.json")).items()}
BINS = [(1e-2, 1e-1, "1–10%"), (1e-3, 1e-2, "0.1–1%"),
        (1e-4, 1e-3, "0.01–0.1%"), (1e-5, 1e-4, "$10^{-5}$–$10^{-4}$"),
        (0, 1e-5, "$<10^{-5}$"), (-1, 0, "never seen in\n$10^6$ samples")]
def binof(c):
    v = vol.get(c)
    if v is None: return len(BINS) - 1
    for i, (lo, hi, _) in enumerate(BINS[:-1]):
        if lo < v <= hi: return i
    return 0
AC = collections.Counter(); UC = collections.Counter()
for s in SEEDS:
    for c in cum[s]["agent"]: AC[binof(c)] += 1
    for c in cum[s]["uniform"]: UC[binof(c)] += 1
n = len(SEEDS)
av = [AC[i]/n for i in range(len(BINS))]; uv = [UC[i]/n for i in range(len(BINS))]

# ============ Main Fig: (a) BDI coverage, (b) Chern budget curve ============
fig, axs = plt.subplots(2, 1, figsize=(3.5, 5.5))
a = axs[0]
for b in BUD:
    ds = sorted(A[b])
    band(a, ds, A[b], RAMP[b], "-", "o")
    band(a, ds, U[b], RAMPR[b], "--", "s", lw=1.1)
a.plot([], [], "-", color=RAMP[300000], lw=1.6, marker="o", ms=4, mfc="white",
       mew=1.1, label="autonomous search")
a.plot([], [], "--", color=RAMPR[300000], lw=1.1, marker="s", ms=4, mfc="white",
       mew=1.1, label="random sampling")
a.set_xlabel("$d$  (number of phases present)")
a.set_ylabel("phases discovered")
a.legend(fontsize=7, loc="upper left")
a.text(0.985, 0.06, "light to dark: $10^4$, $3{\\times}10^4$, $10^5$, $3{\\times}10^5$ queries",
       transform=a.transAxes, ha="right", fontsize=6.4, color="#555555")
a.set_title("(a)  cluster chain,  up to 200 phases", loc="left", fontsize=9)

a = axs[1]
a.fill_between(bs, ALO, AHI, color=BLUE, alpha=.13, lw=0)
a.fill_between(bs, ULO, UHI, color=RED, alpha=.13, lw=0)
a.plot(bs, AM, "-o", color=BLUE, lw=1.6, ms=4, mfc="white", mew=1.1,
       label="autonomous search")
a.plot(bs, UM, "--s", color=RED, lw=1.4, ms=4, mfc="white", mew=1.1,
       label="random sampling")
for i in (0, len(bs)-1):
    a.annotate(f"+{AM[i]-UM[i]:.0f}", xy=(bs[i], AM[i]), xytext=(0, 7),
               textcoords="offset points", ha="center", fontsize=8,
               color=BLUE, fontweight="bold")
a.set_xscale("log")
a.set_xlabel("budget (ground states evaluated)")
a.set_ylabel("distinct Chern sectors")
a.legend(fontsize=7, loc="upper left")
a.set_title("(b)  Chern insulator,  50 couplings", loc="left", fontsize=9)
for a in axs: house(a)
fig.tight_layout(h_pad=1.7)
fig.savefig("img/results_figure_scale_2x1.pdf", bbox_inches="tight")

# ============ Main Fig: rare-sector volume bins ============
fig, a = plt.subplots(figsize=(3.5, 3.0))
y = np.arange(len(BINS)); hgt = 0.36
a.barh(y - hgt/2, av, hgt, color=BLUE, zorder=3, label="autonomous search")
a.barh(y + hgt/2, uv, hgt, color=RED, zorder=3, label="random sampling")
for i, (x1, x2) in enumerate(zip(av, uv)):
    a.annotate(f"{x1:.0f}", xy=(x1, i-hgt/2), xytext=(3, 0),
               textcoords="offset points", va="center", fontsize=7, color=BLUE)
    a.annotate(f"{x2:.0f}", xy=(x2, i+hgt/2), xytext=(3, 0),
               textcoords="offset points", va="center", fontsize=7, color=RED)
    if x2 > 0.3 and x1/x2 > 1.3:
        a.annotate(f"{x1/x2:.1f}$\\times$", xy=(max(x1, x2), i), xytext=(16, 0),
                   textcoords="offset points", va="center", fontsize=8.5,
                   color="#333333", fontweight="bold")
a.set_yticks(y); a.set_yticklabels([b[2] for b in BINS], fontsize=7)
a.invert_yaxis(); a.set_xlim(0, 80)
a.set_xlabel("sectors found at $3\\times10^5$ (mean of 5 seeds)")
a.legend(fontsize=6.5, loc="upper right")
house(a)
fig.tight_layout()
fig.savefig("img/results_figure_chern_volume.pdf", bbox_inches="tight")

print("written 2 figures")
