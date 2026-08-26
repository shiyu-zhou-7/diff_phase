"""Is the residual M=768 label error SYMMETRIC between agent and baseline theta?

The chunk-0 spot-check showed M=768 vs M=1536 agreeing only 97% on AGENT theta,
where random Gaussian theta converge at 768.  If the agent walks into
harder-to-resolve regions, the residual error could be asymmetric and bias the
comparison.  Measure both sides on the same footing.
"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, "/scratch/yuxzhang/chern")
from chern_fast import chern_fast

H, DIM, BUDGET = 16, 50, 300000
N = int(os.environ.get("CONV_N", "1000"))
SEED = int(os.environ.get("VAL_SEED", "200"))

def lab(th, M):
    c, g = chern_fast(th, H, M=M)
    return c if (c is not None and g > 1e-6) else None

if __name__ == "__main__":
    t0 = time.time()
    A = np.load(f"/scratch/yuxzhang/chern/theta_agent_s{SEED}.npy")
    U = np.random.default_rng(90000 + SEED).standard_normal((BUDGET, DIM))
    pick = np.random.default_rng(31337)
    out = {}
    for tag, TH in (("agent", A), ("uniform", U)):
        idx = pick.choice(TH.shape[0], size=N, replace=False)
        rows = []
        for i in idx:
            c7 = lab(TH[i], 768); c15 = lab(TH[i], 1536); c9 = lab(TH[i], 96)
            if c7 is not None and c15 is not None:
                rows.append((c7 == c15, c9 == c15 if c9 is not None else False,
                             abs(c7 - c15)))
        ok7 = sum(r[0] for r in rows); ok9 = sum(r[1] for r in rows)
        out[tag] = dict(n=len(rows), agree_768=ok7, agree_96=ok9,
                        mean_abs_dev_768=float(np.mean([r[2] for r in rows])),
                        max_abs_dev_768=int(max(r[2] for r in rows)))
        print(f"  {tag:8s}: n={len(rows)}  M768 vs M1536 = {ok7}/{len(rows)} "
              f"({100*ok7/len(rows):.1f}%)   M96 vs M1536 = {ok9}/{len(rows)} "
              f"({100*ok9/len(rows):.1f}%)   mean|dev@768|="
              f"{out[tag]['mean_abs_dev_768']:.3f}", flush=True)
    out["secs"] = round(time.time()-t0, 1)
    json.dump(out, open(f"convsym_s{SEED}.json", "w"), indent=1)
