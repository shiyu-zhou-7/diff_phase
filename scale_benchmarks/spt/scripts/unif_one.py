"""One (d, budget, realisation) per task — the 5-in-one version blew the wall
clock at d = 200 (about 9 h)."""
import os, json, sys
import numpy as np
sys.path.insert(0, "/scratch/yuxzhang/final")
from fair_k_analysis import wind_batch
D = int(os.environ["SW_D"]); N = int(os.environ["SW_BUDGET"]); S = int(os.environ["SW_REP"])
r = np.random.default_rng(1000 + S)
seen, n = set(), 0
while n < N:
    m = min(25000, N - n)
    T = r.standard_normal((m, D)); T /= np.linalg.norm(T, axis=1, keepdims=True)
    seen |= set(wind_batch(T).tolist()); n += m
print(f"  d={D} N={N:,} rep={S}: {len(seen)}/{D}", flush=True)
json.dump(dict(d=D, budget=N, rep=S, cov=len(seen)),
          open(f"u1_d{D}_b{N}_r{S}.json", "w"))
