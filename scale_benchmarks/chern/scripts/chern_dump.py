"""Stage A: re-run the 5 production walks and dump every PAID theta.

The trajectory is independent of the label grid (the gradient comes from feat(),
and label() only fills Trial.seen), so this reproduces the stored runs exactly.
Self-check: the M=96 sector count must equal the value already in
set_w2_H16_Q300000_s{SEED}.json, and the number of recorded theta must equal
the budget exactly.
"""
import sys, os, time, json
import numpy as np
sys.path.insert(0, "/scratch/yuxzhang/diff_phase/spt/scripts")
sys.path.insert(0, "/scratch/yuxzhang/chern")
import chern_hyperband as HB

W2 = dict(lr=1.240, trail_w=0.73, trail_s=0.083, trail_n=114,
          lazy=5, radius=0.087, max_steps=319, clip=1.008)
SEED = int(os.environ["VAL_SEED"])
BUDGET = int(os.environ.get("VAL_Q", "300000"))

PAID = []
_orig_label = HB.label
def _recording_label(th):
    PAID.append(np.array(th, dtype=np.float64, copy=True))
    return _orig_label(th)
HB.label = _recording_label          # picked up via module globals at call time

if __name__ == "__main__":
    t0 = time.time()
    tr = HB.Trial(W2, 90000 + SEED*13)
    cov = tr.run_to(BUDGET)
    TH = np.asarray(PAID)
    np.save(f"theta_agent_s{SEED}.npy", TH)
    out = dict(seed=SEED, budget=BUDGET, n_paid=int(TH.shape[0]),
               overshoot=int(TH.shape[0] - BUDGET),
               agent_M96=int(cov), secs=round(time.time()-t0, 1))
    # self-check against the stored production result
    ref = f"/scratch/yuxzhang/chern/set_w2_H16_Q{BUDGET}_s{SEED}.json"
    if os.path.exists(ref):
        r = json.load(open(ref))
        out["stored_M96"] = int(r["agent"])
        out["reproduced"] = bool(int(r["agent"]) == int(cov))
    print(f"  seed={SEED}: paid={TH.shape[0]} (budget {BUDGET}, overshoot "
          f"{TH.shape[0]-BUDGET})  M96 count={cov}  "
          f"stored={out.get('stored_M96','?')}  match={out.get('reproduced','?')}"
          f"   [{out['secs']}s]", flush=True)
    json.dump(out, open(f"dump_s{SEED}.json", "w"), indent=1)
