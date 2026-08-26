"""Stage C: rebuild the sector-volume reference on the converged grid.

Same measure and same gap cut as the in-job baseline (raw Gaussian, g > 1e-6).
"""
import sys, os, time, json, collections
import numpy as np
sys.path.insert(0, "/scratch/yuxzhang/chern")
from chern_fast import chern_fast

H, DIM = 16, 50
M_CONV = int(os.environ.get("M_CONV", "768"))
NTOT = int(os.environ.get("VOL_N", "1000000"))
NTASK = int(os.environ.get("VOL_TASKS", "20"))
TASK = int(os.environ["SLURM_ARRAY_TASK_ID"])

if __name__ == "__main__":
    t0 = time.time()
    per = NTOT // NTASK
    rng = np.random.default_rng(555000 + TASK)     # documented, per-chunk stream
    cnt = collections.Counter(); ng = 0
    for _ in range(per):
        c, g = chern_fast(rng.standard_normal(DIM), H, M=M_CONV)
        if c is not None and g > 1e-6: cnt[int(c)] += 1
        else: ng += 1
    print(f"  volref task {TASK}: n={per} sectors={len(cnt)} gapless={ng} "
          f"[{time.time()-t0:.0f}s]", flush=True)
    json.dump(dict(task=TASK, n=per, M=M_CONV, gapless=ng,
                   counts={str(k): v for k, v in cnt.items()}),
              open(f"volref_M{M_CONV}_t{TASK}.json", "w"), indent=1)
