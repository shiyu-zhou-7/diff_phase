"""Unbiased re-evaluation of the Hyperband winners on FRESH seeds.

The bracket reports max-of-27 evaluated on the same run, which is
selection-biased upward.  Re-running the winning config on unseen seeds gives
an honest estimate."""
import sys, os, time, json
import numpy as np
sys.path.insert(0, "/scratch/yuxzhang/diff_phase/spt/scripts")
sys.path.insert(0, "/scratch/yuxzhang/chern")
import jax
import chern_hyperband as HB          # reuse Trial / GRAD / feat

WINNERS = {
    "w0": dict(lr=0.432, trail_w=1.13, trail_s=0.046, trail_n=29584,
               lazy=2, radius=0.046, max_steps=623, clip=1.462),
    "w1": dict(lr=0.059, trail_w=8.47, trail_s=0.034, trail_n=273,
               lazy=1, radius=0.315, max_steps=312, clip=0.242),
    "w2": dict(lr=1.240, trail_w=0.73, trail_s=0.083, trail_n=114,
               lazy=5, radius=0.087, max_steps=319, clip=1.008),
}
WHICH = os.environ["VAL_CFG"]
SEED = int(os.environ["VAL_SEED"])
BUDGET = int(os.environ.get("VAL_Q", "67500"))

if __name__ == "__main__":
    cfg = WINNERS[WHICH]
    t0 = time.time()
    tr = HB.Trial(cfg, 90000 + SEED*13)          # fresh seed stream
    cov = tr.run_to(BUDGET)
    rb = np.random.default_rng(90000 + SEED); U = set()
    for _ in range(BUDGET):
        c, g = HB.chern_fast(rb.standard_normal(HB.DIM), HB.H, M=HB.M_LAB)
        if c is not None and g > 1e-6: U.add(c)
    print(f"  {WHICH} dim={HB.DIM} seed={SEED} budget={BUDGET:,}: "
          f"agent {cov} vs random {len(U)}  diff {cov-len(U):+d}"
          f"   [{time.time()-t0:.0f}s]", flush=True)
    json.dump(dict(cfg_name=WHICH, dim=HB.DIM, seed=SEED, budget=BUDGET,
                   agent=cov, uniform=len(U), cfg=cfg),
              open(f"prod_{WHICH}_H{HB.H}_Q{BUDGET}_s{SEED}.json", "w"), indent=1)
