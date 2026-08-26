"""Stage B: label agent theta (dumped) and baseline theta (regenerated from the
seed) on a CONVERGED Brillouin-zone grid.

The baseline needs no walk: chern_sets.py draws it from
np.random.default_rng(90000+SEED).standard_normal(DIM), so it is reproducible.
Also labels at M=96 to quantify the shift, and spot-checks a subsample at
M=1536 to confirm M=768 is converged on THESE theta (not just on random ones).
"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, "/scratch/yuxzhang/chern")
from chern_fast import chern_fast

H, DIM = 16, 50
M_CONV = int(os.environ.get("M_CONV", "768"))
M_OLD = 96
M_CHECK = int(os.environ.get("M_CHECK", "1536"))
NCHUNK = int(os.environ.get("NCHUNK", "10"))
BUDGET = int(os.environ.get("VAL_Q", "300000"))
TASK = int(os.environ["SLURM_ARRAY_TASK_ID"])
SEED = 200 + TASK // NCHUNK
CHUNK = TASK % NCHUNK

def lab(th, M):
    c, g = chern_fast(th, H, M=M)
    return c if (c is not None and g > 1e-6) else None

if __name__ == "__main__":
    t0 = time.time()
    A = np.load(f"/scratch/yuxzhang/chern/theta_agent_s{SEED}.npy")
    rb = np.random.default_rng(90000 + SEED)
    U = rb.standard_normal((BUDGET, DIM))        # identical stream to chern_sets.py
    lo_a, hi_a = [int(round(x)) for x in
                  np.linspace(0, A.shape[0], NCHUNK + 1)[[CHUNK, CHUNK + 1]]]
    lo_u, hi_u = [int(round(x)) for x in
                  np.linspace(0, U.shape[0], NCHUNK + 1)[[CHUNK, CHUNK + 1]]]
    res = {}
    for tag, TH, lo, hi in (("agent", A, lo_a, hi_a), ("uniform", U, lo_u, hi_u)):
        new, old = set(), set()
        agree = tot = 0
        for i in range(lo, hi):
            cn = lab(TH[i], M_CONV); co = lab(TH[i], M_OLD)
            if cn is not None: new.add(int(cn))
            if co is not None: old.add(int(co))
            if cn is not None and co is not None:
                tot += 1; agree += (cn == co)
        res[tag] = dict(n=hi-lo, set_conv=sorted(new), set_old=sorted(old),
                        pointwise_agree=agree, pointwise_tot=tot)
        print(f"  s{SEED} chunk{CHUNK} {tag}: n={hi-lo} "
              f"|new|={len(new)} |old|={len(old)} "
              f"agree={agree}/{tot} ({100*agree/max(tot,1):.1f}%)", flush=True)
    # convergence spot-check on this chunk's agent theta
    if CHUNK == 0:
        rs = np.random.default_rng(7).choice(np.arange(lo_a, hi_a),
                                             size=min(400, hi_a-lo_a), replace=False)
        ok = n = 0
        for i in rs:
            c1 = lab(A[i], M_CONV); c2 = lab(A[i], M_CHECK)
            if c1 is not None and c2 is not None:
                n += 1; ok += (c1 == c2)
        res["conv_check"] = dict(M=M_CONV, vs=M_CHECK, agree=ok, tot=n)
        print(f"  s{SEED} CONVERGENCE CHECK M={M_CONV} vs M={M_CHECK}: "
              f"{ok}/{n} ({100*ok/max(n,1):.1f}%)", flush=True)
    res["secs"] = round(time.time()-t0, 1)
    json.dump(res, open(f"relab_s{SEED}_c{CHUNK}.json", "w"), indent=1)
