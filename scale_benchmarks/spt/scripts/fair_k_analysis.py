"""Like-for-like comparison across d = 6...16 under ONE criterion:

    "how many ED-solved states until EVERY phase has >= k of them"

k = 1   : mere contact (oracle-graded: you only know you touched it if an
          oracle labels the point)
k >= 50 : classification grade — enough states for the phase to be a
          recognizable cluster rather than an outlier

Both sides pay the same currency: one solved ground state = one query. On
the agentic side EVERY solved state is counted, including the bootstrap
clouds (200/center) that are ~57% of its budget.

Bootstrap reconstruction is exact from the query counter and the code path:
  init            : 400 samples around t_init
  stock retrain j : all (j+2) accumulated centers x 200  (quadratic)
  efficient       : 2 new centers x 200 (stall point + relocation target)
Centers are the trajectory points bracketing each query-counter jump.
"""

import json
import multiprocessing as mp

import numpy as np

A = "autobench_data/"
FILES = {
    ("stock", 6):  [A + "spt_benchmark_d6_K8_L6_pbc_20260812_005625.npz"],
    ("stock", 8):  [A + "spt_benchmark_d8_K8_L8_pbc_20260812_011301.npz"],
    ("stock", 10): [A + "spt_benchmark_d10_K8_L10_pbc_20260812_032051.npz"],
    ("stock", 12): [A + "spt_benchmark_d12_K4_L12_pbc_20260812_034321.npz",
                    A + "spt_benchmark_d12_K4_L12_pbc_20260812_034528.npz"],
    ("stock", 14): [A + "spt_benchmark_d14_K2_L14_pbc_20260813_045806.npz",
                    A + "spt_benchmark_d14_K3_L14_pbc_20260814_095751.npz",
                    A + "spt_benchmark_d14_K3_L14_pbc_20260814_095924.npz"],
    ("efficient", 6):  [A + "EFF_K16_d6_20260817.npz"],
    ("efficient", 8):  [A + "EFF_K16_d8_20260817.npz"],
    ("efficient", 10): [A + "EFF_K16_d10_20260815.npz"],
    ("efficient", 12): [A + "EFF_K16_d12_20260816.npz"],
}
import glob as _g
FILES[("efficient", 14)] = sorted(_g.glob(A + "spt_benchmark_d14_K1_*.npz"))
FILES[("efficient", 16)] = sorted(_g.glob(A + "spt_benchmark_d16_K1_*.npz"))

DS = [6, 8, 10, 12, 14, 16]
KS = [1, 50, 200]
N_INIT, N_BOOT, SIGMA = 400, 200, 0.1


# ---------------------------------------------------------------- labels
def wind_slow(t):
    c = np.asarray(t, float).copy()
    c[0] = -c[0]
    c = c[::-1]
    nz = np.nonzero(np.abs(c) > 1e-13 * np.max(np.abs(c)))[0]
    c = c[nz[0]:]
    return 0 if len(c) <= 1 else int((np.abs(np.roots(c)) < 1.0).sum())


def wind_batch(T, chunk=8000):
    """Physical winding for a batch of coupling vectors, via batched
    companion-matrix eigenvalues."""
    T = np.atleast_2d(np.asarray(T, float))
    out = np.empty(len(T), dtype=np.int16)
    for s in range(0, len(T), chunk):
        B = T[s:s + chunk].copy()
        B[:, 0] *= -1.0                       # physical convention
        P = B[:, ::-1]                        # highest degree first
        lead, scale = P[:, 0], np.abs(P).max(axis=1)
        bad = np.abs(lead) < 1e-10 * scale
        Pn = P / np.where(bad, 1.0, lead)[:, None]
        n = P.shape[1] - 1
        C = np.zeros((len(P), n, n))
        C[:, 0, :] = -Pn[:, 1:]
        i = np.arange(n - 1)
        C[:, i + 1, i] = 1.0
        ev = np.linalg.eigvals(C)
        cnt = (np.abs(ev) < 1.0).sum(axis=1).astype(np.int16)
        for j in np.nonzero(bad)[0]:          # degenerate leading coeff
            cnt[j] = wind_slow(B[j] * np.array([-1.0] + [1.0] * (n)))
        out[s:s + chunk] = cnt
    return out


def cloud(center, d, n, rng):
    S = center[None, :] + SIGMA * rng.standard_normal((n, d))
    return S / np.linalg.norm(S, axis=1, keepdims=True)


# ---------------------------------------------------------------- events
def run_events(ts, qs, d, kind, rng):
    """(query_position, phase) for every solved state of one run."""
    pts, pos = [cloud(ts[0], d, N_INIT, rng)], [np.arange(1, N_INIT + 1)]
    centers = [ts[0]]
    for i in (np.nonzero(np.diff(qs) > 1)[0] + 1):
        q0 = int(qs[i - 1])
        if kind == "efficient":               # stall point + relocation target
            new = [ts[i - 1], ts[i]]
            centers += new
            for c in new:
                pts.append(cloud(c, d, N_BOOT, rng))
                pos.append(np.full(N_BOOT, q0 + 1))
        else:                                 # re-sample every known center
            centers.append(ts[i - 1])
            for c in centers:
                pts.append(cloud(c, d, N_BOOT, rng))
                pos.append(np.full(N_BOOT, q0 + 1))
    pts.append(np.asarray(ts))                # the walk itself
    pos.append(np.asarray(qs, int))
    P = np.concatenate(pts, axis=0)
    Q = np.concatenate(pos)
    W = wind_batch(P)
    o = np.argsort(Q, kind="stable")
    return Q[o], W[o], int(qs[-1])


def campaign(d, kind, n_perm=300):
    """Per-run event streams, then permute run order to get a distribution of
    'queries until every phase has >= k states' (matches the uniform side's
    median-over-realizations treatment)."""
    files = FILES.get((kind, d))
    if not files:
        return None
    rng = np.random.default_rng(1000 + d + (7 if kind == "efficient" else 0))
    runs = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        if "histories" not in z.files:
            continue
        for h in z["histories"]:
            ts = np.asarray(h["t_per_step"], float)
            qs = np.asarray(h["query_per_step"], int)
            if len(ts):
                runs.append(run_events(ts, qs, d, kind, rng))
    if not runs:
        return None
    total = sum(r[2] for r in runs)
    dist = {k: [] for k in KS}
    for _ in range(n_perm):
        order = rng.permutation(len(runs))
        counts = np.zeros(d, dtype=np.int64)
        hit = {k: None for k in KS}
        base = 0
        for i in order:
            Q, W, tot = runs[i]
            for q, w in zip(Q, W):
                counts[w] += 1
                mn = counts.min()
                for k in KS:
                    if hit[k] is None and mn >= k:
                        hit[k] = base + int(q)
            base += tot
            if all(hit[k] is not None for k in KS):
                break
        for k in KS:
            dist[k].append(hit[k] if hit[k] is not None else np.inf)
    out = {}
    for k in KS:
        a = np.array(dist[k], float)
        fin = np.isfinite(a)
        out[k] = dict(median=float(np.median(a[fin])) if fin.any() else None,
                      p10=float(np.percentile(a[fin], 10)) if fin.any() else None,
                      p90=float(np.percentile(a[fin], 90)) if fin.any() else None,
                      frac_reached=float(fin.mean()))
    counts = np.zeros(d, dtype=np.int64)
    for Q, W, _ in runs:
        for w in W: counts[w] += 1
    return {"q": out, "total_queries": total, "n_runs": len(runs),
            "counts": counts.tolist(),
            "n_states": int(sum(len(r[1]) for r in runs))}


def uniform_q(d, p, k, trials=2000, seed=7):
    """Poissonized: draws until k-th arrival of phase w ~ Gamma(k, 1/p_w);
    the criterion is the max over phases. Median over trials."""
    rng = np.random.default_rng(seed + d)
    g = rng.gamma(shape=k, scale=1.0 / np.asarray(p), size=(trials, len(p)))
    m = g.max(axis=1)
    return dict(median=float(np.median(m)), p10=float(np.percentile(m, 10)),
                p90=float(np.percentile(m, 90)))


def work(d):
    p = np.array(json.load(open("dscan.json"))[str(d)]["p"])
    r = {"d": d, "uniform": {k: uniform_q(d, p, k) for k in KS},
         "p_min": float(p.min())}
    for kind in ["stock", "efficient"]:
        r[kind] = campaign(d, kind)
    print(f"  d={d} done", flush=True)
    return r


if __name__ == "__main__":
    with mp.get_context("fork").Pool(6) as pool:
        res = pool.map(work, DS)
    json.dump({str(r["d"]): r for r in res}, open("fair_k_summary.json", "w"),
              indent=1, default=str)

    print("\n queries until EVERY phase has >= k solved states"
          "   (median over run orders / sampling realizations)\n")
    print("  d |   k |     uniform |       stock |   efficient")
    print(" ---+-----+-------------+-------------+------------")
    for r in res:
        for k in KS:
            def cell(m):
                if not m:
                    return "          -"
                v = m["q"][k]
                if v["median"] is None:
                    return f">{m['total_queries']:>10,}"
                tag = "" if v["frac_reached"] > 0.99 else "*"
                return f"{v['median']:>10,.0f}{tag or ' '}"
            print(f" {r['d']:2d} | {k:3d} | {r['uniform'][k]['median']:>11,.0f} |"
                  f" {cell(r['stock'])} | {cell(r['efficient'])}")
    print("\n * = criterion not met in some run orders (fraction shown in json)")
