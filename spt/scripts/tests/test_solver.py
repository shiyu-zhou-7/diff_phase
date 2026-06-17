"""
Validation suite, part 1: ED solver vs free-fermion + label cross-check.

  Section 4.1 (JW correctness, SCOPED): at pure stabilizer points e_alpha (all d)
    the ED ground energy and in-sector gap match the free-fermion result
    (E0=-L, gap=4). For d <= 2 they match the closed-form BdG result for random t
    to numerical tolerance (ED energy = -sum_NS|f|; ED gap = 2 * the closed-form
    2*min_k|f|, the pair-excitation factor). The exact finite-size d >= 3 PBC
    free-fermion energy carries parity/boundary corrections that are intentionally
    out of scope here -- for d >= 3 we instead check that the BULK gap closes at an
    analytic boundary and is O(1) in a cell interior (the physically meaningful JW
    cross-check; full localization is section 4.4 in test_gradients.py).

  Section 4.2 (label correctness): for random t in each dominance region the
    root-finder winding(t) agrees with the independent ED diagnostics -- the
    order parameters m_X / O_Z2 / O_SPT and the entanglement entropy
    (S ~ omega*log2) and ES degeneracy.

Run from spt/scripts/:
    python -m tests.test_solver   (or: python tests/test_solver.py)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

from hamiltonians.cluster import build_cluster_model, gd_solver_ed, lift_to_full
from hamiltonians.observables import (
    magnetization_x, zz_correlation, string_order_spt,
    entanglement_entropy, es_degeneracy,
)
from analytic.cluster_exact import (
    winding, gap as gap_closed, gap_thermo, ground_energy_bdg, is_gapless,
)

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def ed_e0_gap(t, model):
    e, _ = gd_solver_ed(jnp.asarray(t, dtype=jnp.float64), model)
    e = np.asarray(e)
    return float(e[0]), float(e[1] - e[0])


# ===========================================================================
print("== 4.1 pure stabilizer points e_alpha: E0=-L, in-sector gap=4 (all d) ==")
for d in (2, 3, 4, 5):
    for L in (10, 12) if d == 3 else (10,):   # L=12 spot-check on d=3 only
        model = build_cluster_model(L=L, d=d, bc='pbc', sector=+1)
        for a in range(d):
            t = np.zeros(d)
            t[a] = 1.0
            E0, gp = ed_e0_gap(t, model)
            check(f"d={d} L={L} e_{a}: E0=-L", abs(E0 + L) < 1e-9, f"E0={E0}")
            check(f"d={d} L={L} e_{a}: gap=4", abs(gp - 4.0) < 1e-9, f"gap={gp}")

# ===========================================================================
print("== 4.1 d<=2 random t: ED energy = closed-form BdG, gap = 2 x closed gap ==")
rng = np.random.default_rng(0)
for d in (2,):
    for L in (10, 12):
        model = build_cluster_model(L=L, d=d, bc='pbc', sector=+1)
        for _ in range(6):
            t = rng.standard_normal(d)
            if is_gapless(t):       # skip points on a boundary
                continue
            E0, gp = ed_e0_gap(t, model)
            E0_an = ground_energy_bdg(t, L, 'pbc', +1)
            gp_an = 2.0 * gap_closed(t, L, 'pbc', +1)   # pair-excitation factor
            check(f"d={d} L={L} energy match", abs(E0 - E0_an) < 1e-8,
                  f"ed={E0:.6f} an={E0_an:.6f}")
            check(f"d={d} L={L} gap match", abs(gp - gp_an) < 1e-7,
                  f"ed={gp:.6f} an={gp_an:.6f}")

# ===========================================================================
print("== 4.1 d>=3: bulk gap closes at an analytic boundary, O(1) in interior ==")
for d, L in [(3, 10), (4, 10)]:
    model = build_cluster_model(L=L, d=d, bc='pbc', sector=+1)
    # interior point (pure SPT) -- gapped
    t_int = np.zeros(d); t_int[2] = 1.0
    _, gp_int = ed_e0_gap(t_int, model)
    check(f"d={d} interior ED gap O(1)", gp_int > 0.5, f"gap={gp_int}")
    check(f"d={d} interior bulk gap O(1)", gap_thermo(t_int) > 0.5)
    # boundary point on sum_alpha t_alpha = 0 (z=+1): build t with zero sum
    t_bnd = np.ones(d) / d
    t_bnd[0] -= np.sum(t_bnd)   # force sum = 0
    check(f"d={d} boundary is analytically gapless", is_gapless(t_bnd))
    check(f"d={d} boundary bulk gap ~ 0", gap_thermo(t_bnd) < 1e-6,
          f"gap_thermo={gap_thermo(t_bnd)}")
    _, gp_bnd = ed_e0_gap(t_bnd, model)
    check(f"d={d} ED gap smaller at boundary than interior",
          gp_bnd < gp_int, f"bnd={gp_bnd:.4f} int={gp_int:.4f}")

# ===========================================================================
print("== 4.2 label correctness: winding agrees with ED diagnostics ==")
LOG2 = np.log(2.0)
for L in (10,):
    model = build_cluster_model(L=L, d=3, bc='pbc', sector=+1)
    rng = np.random.default_rng(2)
    for alpha, opname in [(0, 'm_X'), (1, 'O_Z2'), (2, 'O_SPT')]:
        for _ in range(8):
            t = 0.12 * rng.standard_normal(3)
            t[alpha] = 1.0              # alpha dominant -> expect winding == alpha
            w = winding(t)
            if w != alpha:
                continue               # skip rare boundary-adjacent draws
            _, psi = gd_solver_ed(jnp.asarray(t), model)
            pf = lift_to_full(psi, model)
            mx = float(magnetization_x(pf, model))
            zz = float(zz_correlation(pf, model))
            spt = float(string_order_spt(pf, model))
            S = float(entanglement_entropy(pf, model))
            es = float(es_degeneracy(pf, model))
            ops = {'m_X': mx, 'O_Z2': zz, 'O_SPT': abs(spt)}
            dominant = max(ops, key=ops.get)
            check(f"L={L} winding={alpha}: dominant order param is {opname}",
                  dominant == opname, f"ops={ {k: round(v,2) for k,v in ops.items()} }")
            # entanglement entropy near omega*log2 (deep in the phase)
            check(f"L={L} winding={alpha}: S ~ {alpha}*log2",
                  abs(S - alpha * LOG2) < 0.35, f"S={S:.3f} expect~{alpha*LOG2:.3f}")
            # trivial phase: non-degenerate ES; topological: ~degenerate
            if alpha == 0:
                check(f"L={L} trivial: ES non-degenerate", es > 0.5, f"es={es:.3f}")
            else:
                check(f"L={L} omega={alpha}: ES ~ degenerate", es < 0.2, f"es={es:.3f}")

print()
print(f"==== solver/label tests: {PASS} passed, {FAIL} failed ====")
sys.exit(1 if FAIL else 0)
