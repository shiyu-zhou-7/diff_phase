"""
Unit tests for the analytic ground-truth module (analytic/cluster_exact.py).

These are SELF-CONSISTENT checks of the symbol/root math -- they do not touch
the ED solver (that cross-check is validation section 4.1 in test_solver.py).
They cover: winding at pure stabilizer points and in dominance regions, the
gapless flag, the closed-form boundary hyperplanes, the BdG gap/energy at pure
points, and boundary-crossing classification (real z=+-1 and complex pair k*).

Run from spt/scripts/:
    python -m tests.test_analytic
or:
    python tests/test_analytic.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from analytic.cluster_exact import (
    f_symbol, g_roots, winding, is_gapless, classify,
    allowed_momenta, single_particle_energies, gap, gap_thermo,
    ground_energy_bdg, boundary_value_z_plus1, boundary_value_z_minus1,
    boundary_crossings, _closing_kstar,
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


# ---------------------------------------------------------------------------
print("== winding at pure stabilizer points e_alpha ==")
for d in (2, 3, 4, 5, 8):
    for alpha in range(d):
        t = np.zeros(d)
        t[alpha] = 1.0
        w = winding(t)
        check(f"d={d} e_{alpha} -> omega={alpha}", w == alpha, f"got {w}")

# ---------------------------------------------------------------------------
print("== winding in dominance regions (small off-axis perturbation) ==")
rng = np.random.default_rng(0)
for d, alpha in [(3, 0), (3, 1), (3, 2), (5, 0), (5, 2), (5, 4)]:
    for _ in range(20):
        t = 0.08 * rng.standard_normal(d)
        t[alpha] = 1.0  # alpha strongly dominant
        check(f"d={d} t_{alpha} dominant -> omega={alpha}",
              winding(t) == alpha, f"got {winding(t)} for t={np.round(t,3)}")

# ---------------------------------------------------------------------------
print("== gapless flag on the boundary hyperplanes ==")
# z=+1 boundary: -t_0 + t_1 + t_2 = 0  (alpha=0 plus convention: ttilde_0 = -t_0)
# t = np.array([1.0, -1.0, 0.0])      # old convention: g(1) = 0
t = np.array([1.0, 1.0, 0.0])         # g(1) = -1 + 1 = 0
check("sum t = 0 is gapless", is_gapless(t), f"roots={np.round(g_roots(t),4)}")
check("boundary_value_z_plus1 == 0", abs(boundary_value_z_plus1(t)) < 1e-12)
# z=-1 boundary: -t_0 - t_1 + t_2 = 0
# t = np.array([1.0, 2.0, 1.0])       # old convention: g(-1) = 1 - 2 + 1 = 0
t = np.array([-1.0, 2.0, 1.0])        # g(-1) = 1 - 2 + 1 = 0
check("alt-sum t = 0 is gapless", is_gapless(t))
check("boundary_value_z_minus1 == 0", abs(boundary_value_z_minus1(t)) < 1e-12)
# generic interior point is gapped
check("interior point not gapless", not is_gapless(np.array([1.0, 0.2, 0.05])))

# ---------------------------------------------------------------------------
print("== classify bundle agrees with winding/is_gapless ==")
for _ in range(50):
    d = rng.integers(2, 6)
    t = rng.standard_normal(d)
    c = classify(t)
    check("classify omega matches winding", c['omega'] == winding(t),
          f"{c['omega']} vs {winding(t)}")
    check("classify gapless matches is_gapless", c['gapless'] == is_gapless(t))

# ---------------------------------------------------------------------------
print("== allowed momenta sets (PBC parity, OBC) ==")
L = 10
ns = allowed_momenta(L, 'pbc', +1)
check("NS set size L", ns.shape[0] == L)
check("NS set is half-integer grid", np.allclose(ns, 2*np.pi*(np.arange(L)+0.5)/L))
R = allowed_momenta(L, 'pbc', -1)
check("R set includes k=0", np.any(np.isclose(R, 0.0)))
obc = allowed_momenta(L, 'obc')
check("OBC modes in (0,pi)", np.all((obc > 0) & (obc < np.pi)))

# ---------------------------------------------------------------------------
print("== BdG gap & energy at pure points (normalization) ==")
for d in (2, 3, 4):
    for alpha in range(d):
        for L in (10, 12, 14):
            t = np.zeros(d)
            t[alpha] = 1.0
            # |f(k)| = |t_alpha| = 1 for all k -> gap = 2, E0 = -L
            check(f"gap(e_{alpha}, L={L}) == 2", abs(gap(t, L) - 2.0) < 1e-9,
                  f"got {gap(t,L)}")
            check(f"E0(e_{alpha}, L={L}) == -L", abs(ground_energy_bdg(t, L) + L) < 1e-9,
                  f"got {ground_energy_bdg(t,L)}")
# random gapped point: finite-size gap >= thermodynamic gap (allowed k miss the
# continuous minimum), both positive.
t = np.array([1.0, 0.3, 0.1])
check("finite gap >= thermo gap (gapped)", gap(t, 12) >= gap_thermo(t) - 1e-9,
      f"{gap(t,12)} vs {gap_thermo(t)}")
check("thermo gap > 0 at interior point", gap_thermo(t) > 1e-6)

# ---------------------------------------------------------------------------
print("== boundary crossing: real z=+1 line ==")
# d=2: g(z) = -t_0 + t_1 z (alpha=0 plus convention), single root z = t_0/t_1.
# Fix t_1=1, sweep t_0 = 0.5 + s so the root = 0.5 + s crosses z=+1 at s=0.5
# (t_0=+1). The root stays real positive throughout, so ONLY the z=+1 boundary
# is hit (g(+1)=-t_0+1 vanishes at s=0.5; g(-1)=-t_0-1 stays negative).
s = np.linspace(0.0, 1.0, 401)
# t_path = np.stack([-0.5 - s, np.ones_like(s)], axis=1)   # old convention
t_path = np.stack([0.5 + s, np.ones_like(s)], axis=1)
cr = boundary_crossings(t_path, s)
check("one z=+1 crossing found", len(cr) == 1 and cr[0]['type'] == 'z=+1',
      f"got {cr}")
if cr:
    check("z=+1 crossing near analytic s*=0.5", abs(cr[0]['s'] - 0.5) < 0.02,
          f"got s={cr[0]['s']}")
    check("z=+1 kstar == 0", cr[0]['kstar'] == 0.0)
    check("z=+1 c == 0.5", cr[0]['c'] == 0.5)
    check("z=+1 |delta_omega| == 1", abs(cr[0]['delta_omega']) == 1)

# ---------------------------------------------------------------------------
print("== boundary crossing: real z=-1 line ==")
# Same d=2 family but root real negative: t_1=1, t_0 = -0.5 - s, root = -(0.5+s)
# crosses z=-1 at s=0.5. Only g(-1)=-t_0-1 vanishes (g(+1)=-t_0+1 stays positive).
# t_path = np.stack([0.5 + s, np.ones_like(s)], axis=1)    # old convention
t_path = np.stack([-0.5 - s, np.ones_like(s)], axis=1)
cr = boundary_crossings(t_path, s)
check("one z=-1 crossing found", len(cr) == 1 and cr[0]['type'] == 'z=-1',
      f"got {cr}")
if cr:
    check("z=-1 crossing near analytic s*=0.5", abs(cr[0]['s'] - 0.5) < 0.02,
          f"got s={cr[0]['s']}")
    check("z=-1 kstar == pi", abs(cr[0]['kstar'] - np.pi) < 1e-9)
    check("z=-1 c == 0.5", cr[0]['c'] == 0.5)
    check("z=-1 |delta_omega| == 1", abs(cr[0]['delta_omega']) == 1)

# ---------------------------------------------------------------------------
print("== boundary crossing: complex pair (c=1, |dw|=2) ==")
# d=3: sweep t_2 dominant -> t_0 dominant through a complex-pair crossing while
# keeping sum and alt-sum away from zero so the real boundaries are not hit.
# t(s) = (-s, 0, 1-s): g = (1-s) z^2 + s (ttilde_0 = -t_0 = s). roots
# z^2 = -s/(1-s) -> pure imaginary, |z| = sqrt(s/(1-s)) crosses 1 at s=0.5,
# k* = pi/2. sum and alt-sum both = 1 (never 0). Clean complex crossing.
# t_path = np.stack([s, np.zeros_like(s), 1.0 - s], axis=1)   # old convention
t_path = np.stack([-s, np.zeros_like(s), 1.0 - s], axis=1)
cr = boundary_crossings(t_path, s)
check("one complex crossing found", len(cr) == 1 and cr[0]['type'] == 'complex',
      f"got {cr}")
if cr:
    check("complex crossing near s*=0.5", abs(cr[0]['s'] - 0.5) < 0.01,
          f"got s={cr[0]['s']}")
    check("complex c == 1.0", cr[0]['c'] == 1.0)
    check("complex |delta_omega| == 2", abs(cr[0]['delta_omega']) == 2)
    check("complex kstar == pi/2", abs(cr[0]['kstar'] - np.pi/2) < 1e-2,
          f"got {cr[0]['kstar']}")
# verify the predicted closing momentum at the crossing point directly
# (t = (-0.5, 0, 0.5) -> ttilde = (0.5, 0, 0.5) -> g = 0.5 z^2 + 0.5)
check("k* of g=(0.5)z^2+0.5 is pi/2",
      abs(_closing_kstar(np.array([-0.5, 0.0, 0.5])) - np.pi/2) < 1e-9,
      f"got {_closing_kstar(np.array([-0.5,0.0,0.5]))}")

# ---------------------------------------------------------------------------
print()
print(f"==== analytic tests: {PASS} passed, {FAIL} failed ====")
sys.exit(1 if FAIL else 0)
