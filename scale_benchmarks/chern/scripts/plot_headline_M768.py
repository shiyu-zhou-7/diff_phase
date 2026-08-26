"""Main figure, converged labels (M=768).

(a) budget curve built from nested chunk prefixes -- 10 points at 3e4 spacing,
    paired by construction.
(b) rarity breakdown against the M=768 volume reference.
"""
import json, collections, statistics as st
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, MUTED, TEXT, ORANGE = "#2a78d6", "#8a8a80", "#1a1a19", "#eb6834"
SEEDS = [200, 201, 202, 203, 204]

# ---- (a) cumulative over nested chunks ----------------------------------
cum = {s: {"agent": set(), "uniform": set()} for s in SEEDS}
bs, am, um, alo, ahi, ulo, uhi = [], [], [], [], [], [], []
for j in range(10):
    for s in SEEDS:
        d = json.load(open(f"chern_res_M768/relab_s{s}_c{j}.json"))
        for t in ("agent", "uniform"):
            cum[s][t] |= set(d[t]["set_conv"])
    A = [len(cum[s]["agent"]) for s in SEEDS]
    U = [len(cum[s]["uniform"]) for s in SEEDS]
    bs.append((j+1)*30000)
    am.append(st.median(A)); um.append(st.median(U))
    alo.append(min(A)); ahi.append(max(A)); ulo.append(min(U)); uhi.append(max(U))

# ---- (b) rarity ---------------------------------------------------------
vol = {int(k): v for k, v in json.load(open("sector_volumes_M768.json")).items()}
BINS = [(1e-2,1e-1,"1–10%"), (1e-3,1e-2,"0.1–1%"), (1e-4,1e-3,"0.01–0.1%"),
        (1e-5,1e-4,"$10^{-5}$–$10^{-4}$"), (0,1e-5,"$<10^{-5}$"),
        (-1,0,"never seen in\n$10^6$ samples")]
def binof(c):
    v = vol.get(c)
    if v is None: return len(BINS)-1
    for i,(lo,hi,_) in enumerate(BINS[:-1]):
        if lo < v <= hi: return i
    return 0                      # v >= 10%: commonest bin (never fires here)
A = collections.Counter(); U = collections.Counter(); nov = []
for s in SEEDS:
    a, u = cum[s]["agent"], cum[s]["uniform"]
    for c in a: A[binof(c)] += 1
    for c in u: U[binof(c)] += 1
    nov.append(len(a - u))
n = len(SEEDS)
av = [A[i]/n for i in range(len(BINS))]; uv = [U[i]/n for i in range(len(BINS))]
pop = [len([c for c in vol if binof(c) == i]) for i in range(len(BINS))]

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5), dpi=200,
                         gridspec_kw={"width_ratios": [1, 1.12]})
ax = axes[0]
ax.fill_between(bs, alo, ahi, color=BLUE, alpha=0.17, lw=0)
ax.fill_between(bs, ulo, uhi, color=MUTED, alpha=0.18, lw=0)
ax.plot(bs, am, "-o", color=BLUE, lw=2.3, ms=6, zorder=4, label="gradient exploration")
ax.plot(bs, um, "--s", color=MUTED, lw=1.9, ms=5, zorder=4, label="random sampling")
for i in (0, len(bs)-1):
    ax.annotate(f"+{am[i]-um[i]:.0f}", xy=(bs[i], am[i]), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=9.5,
                color=BLUE, fontweight="bold")
ax.set_xscale("log")
ax.set_xlabel("budget (ground states evaluated)", color=TEXT)
ax.set_ylabel("distinct Chern sectors found", color=TEXT)
ax.set_title("(a)  equal-budget coverage", fontsize=10.5, color=TEXT, loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper left")

ax = axes[1]
y = np.arange(len(BINS)); h = 0.36
ax.barh(y - h/2, av, h, color=BLUE, zorder=3)
ax.barh(y + h/2, uv, h, color=MUTED, zorder=3)
for i, (a, u) in enumerate(zip(av, uv)):
    ax.annotate(f"{a:.0f}", xy=(a, i-h/2), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=BLUE)
    ax.annotate(f"{u:.0f}", xy=(u, i+h/2), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=MUTED)
    if u > 0.3 and a/u > 1.3:
        ax.annotate(f"{a/u:.1f}$\\times$", xy=(max(a, u), i), xytext=(32, 0),
                    textcoords="offset points", va="center", fontsize=10.5,
                    color=ORANGE, fontweight="bold")
    if i < len(BINS)-1:
        ax.annotate(f"({pop[i]} exist)", xy=(0, i), xytext=(-6, 0),
                    textcoords="offset points", va="center", ha="right",
                    fontsize=7, color=MUTED)
ax.set_yticks(y); ax.set_yticklabels([b[2] for b in BINS], fontsize=8.5)
ax.invert_yaxis(); ax.set_xlim(0, 74)
ax.set_xlabel("sectors found at $3\\times10^5$ budget (mean of 5 seeds)", color=TEXT)
ax.set_ylabel("sector volume under the Gaussian measure", color=TEXT, fontsize=9)
ax.set_title("(b)  where the advantage lives", fontsize=10.5, color=TEXT, loc="left")
ax.annotate("both saturate every bin\nthe reference resolves", xy=(35, 1.2),
            fontsize=8.5, color=MUTED)
ax.annotate(f"{np.mean(nov):.0f} sectors per run that random\n"
            f"sampling never touches at equal budget",
            xy=(0.985, 0.42), xycoords="axes fraction", ha="right",
            fontsize=8.5, color=ORANGE)
for a in axes:
    a.grid(True, axis="x" if a is axes[1] else "y", color="#ececec", lw=0.6, zorder=0)
    a.spines[["top", "right"]].set_visible(False)
    a.spines[["left", "bottom"]].set_color(MUTED)
    a.tick_params(colors=MUTED, labelcolor=TEXT)
fig.suptitle("50-parameter generalized Chern insulator: converged labels ($M=768$)",
             color=TEXT, fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("fig_headline.png", bbox_inches="tight")
fig.savefig("fig_headline.pdf", bbox_inches="tight")
print(f"written fig_headline  (a: {bs[0]}->{bs[-1]}, +{am[0]-um[0]:.0f}->+{am[-1]-um[-1]:.0f};"
      f"  b: novel {np.mean(nov):.1f}/run, rarest ratio {av[-1]/uv[-1]:.1f}x)")
