"""Uniform-sampling baseline, measured properly: one (d, budget) cell per task,
5 independent realisations each."""
import os, json, sys
import numpy as np
sys.path.insert(0, "/scratch/yuxzhang/final")
from fair_k_analysis import wind_batch

D = int(os.environ["SW_D"]); N = int(os.environ["SW_BUDGET"])
v = []
for s in range(5):
    r = np.random.default_rng(1000 + s)
    seen, n = set(), 0
    while n < N:
        m = min(25000, N - n)
        T = r.standard_normal((m, D)); T /= np.linalg.norm(T, axis=1, keepdims=True)
        seen |= set(wind_batch(T).tolist()); n += m
    v.append(len(seen))
print(f"  d={D} N={N:,}: {sorted(v)}  median {int(np.median(v))}", flush=True)
json.dump(dict(d=D, budget=N, vals=sorted(v), median=float(np.median(v))),
          open(f"u_d{D}_b{N}.json", "w"))
