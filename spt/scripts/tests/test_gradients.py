"""
Validation suite, part 2: gradients and boundary localization.

  Section 4.3 (gradient correctness): away from boundaries,
    - autodiff dE0/dt_alpha == Hellmann-Feynman <psi0|T_alpha|psi0> and == central
      finite difference;
    - autodiff gradient of a rho0-based loss (half-cut entanglement entropy) ==
      central finite difference;
    - gradients are finite (no NaN) in cell interiors and GROW as a boundary is
      approached (the eigenvector-derivative denominator is the in-sector gap).

  Section 4.4 (boundary localization): scanning a straight line in t,
    - across the z=+1 hyperplane sum_alpha t_alpha = 0, the ED gap minimum and the
      loss-gradient peak localize at the analytic crossing within the O(1/L)
      finite-size shift;
    - across a complex-pair (c=1) crossing, same, and the analytic predicted
      closing momentum k* matches the continuous-k argmin of |f(k)|.

Run from spt/scripts/:
    python -m tests.test_gradients   (or: python tests/test_gradients.py)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

from hamiltonians.cluster import (
    build_cluster_model, gd_solver_ed, ground_state_vector, lift_to_full, H_sector,
)
from hamiltonians.observables import renyi2_entropy
from analytic.cluster_exact import (
    winding, gap_thermo, boundary_crossings, f_symbol, is_gapless,
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


L = 10
d = 3
model = build_cluster_model(L=L, d=d, bc='pbc', sector=+1)


def E0_fn(t):
    e, _ = gd_solver_ed(t, model)
    return e[0]


def ent_fn(t):
    psi = ground_state_vector(t, model, 0.0)   # custom-jvp eigenvector adjoint
    pf = lift_to_full(psi, model)
    return renyi2_entropy(pf, model)   # differentiable rho0-based entanglement loss


def central_fd(fn, t, i, h=1e-4):
    tp = t.at[i].add(h)
    tm = t.at[i].add(-h)
    return (fn(tp) - fn(tm)) / (2.0 * h)


# ===========================================================================
print("== 4.3 energy gradient: autodiff == Hellmann-Feynman == finite diff ==")
grad_E0 = jax.grad(E0_fn)
rng = np.random.default_rng(0)
for _ in range(5):
    t = 0.1 * rng.standard_normal(d)
    t = t.at[rng.integers(d)].set(1.0) if False else jnp.asarray(t)
    t = jnp.asarray(np.array(t))
    # ensure a gapped interior point: make one coupling dominant
    base = np.array(t)
    base[0] = 1.0
    t = jnp.asarray(base)
    if is_gapless(np.array(t)):
        continue
    g_ad = np.array(grad_E0(t))
    _, psi = gd_solver_ed(t, model)
    # Hellmann-Feynman: dE0/dt_alpha = <psi0|T_alpha|psi0>
    g_hf = np.array([float(psi @ (model.T_sector[a] @ psi)) for a in range(d)])
    g_fd = np.array([float(central_fd(E0_fn, t, i)) for i in range(d)])
    check("autodiff == Hellmann-Feynman", np.allclose(g_ad, g_hf, atol=1e-8),
          f"ad={np.round(g_ad,5)} hf={np.round(g_hf,5)}")
    check("autodiff == finite diff", np.allclose(g_ad, g_fd, atol=1e-4),
          f"ad={np.round(g_ad,5)} fd={np.round(g_fd,5)}")

# ===========================================================================
print("== 4.3 rho0-loss (entanglement entropy) gradient: autodiff == finite diff ==")
grad_ent = jax.grad(ent_fn)
for _ in range(4):
    base = 0.1 * rng.standard_normal(d)
    base[2] = 1.0   # interior of the SPT cell -> gapped, S ~ 2 log2
    t = jnp.asarray(base)
    if is_gapless(np.array(t)):
        continue
    g_ad = np.array(grad_ent(t))
    g_fd = np.array([float(central_fd(ent_fn, t, i, h=1e-4)) for i in range(d)])
    check("autodiff entropy-grad == finite diff",
          np.allclose(g_ad, g_fd, atol=1e-3),
          f"ad={np.round(g_ad,4)} fd={np.round(g_fd,4)}")
    check("entropy gradient finite (no NaN)", np.all(np.isfinite(g_ad)))

# ===========================================================================
print("== 4.3 STATE-loss gradient grows toward a boundary, finite in interior ==")
# The Hellmann-Feynman ENERGY gradient is bounded (<=L) and does NOT diverge at a
# boundary. The divergence is in a STATE-dependent (rho0) loss, whose eigenvector
# derivative has the in-sector gap in its denominator. Line from the SPT interior
# (e_2) toward the z=+1 boundary at s=1 (-t_0 + t_1 + t_2 -> 0, alpha=0 plus
# convention); sample up to s=0.9 to stay just shy of the exact closing.
ts = np.linspace(0.0, 0.9, 19)
t_int = np.array([0.0, 0.0, 1.0])
# t_bnd = np.array([0.5, 0.5, -1.0])   # old convention: sum = 0 at s=1
t_bnd = np.array([-0.5, 0.5, -1.0])    # -t_0 + t_1 + t_2 = 0 at s=1
gnorms = np.array([float(jnp.linalg.norm(grad_ent(jnp.asarray((1 - s) * t_int + s * t_bnd))))
                   for s in ts])
check("all entropy-grad norms finite away from boundary", np.all(np.isfinite(gnorms)))
check("entropy-grad norm near boundary > interior",
      gnorms[-3:].max() > 3.0 * gnorms[:4].mean() + 1e-9,
      f"interior~{gnorms[:4].mean():.4f} near-bnd~{gnorms[-3:].max():.4f}")

# ===========================================================================
print("== 4.4 boundary localization: z=+1 line (single real root through +1) ==")
# Clean single z=+1 crossing on the t_2=0 face: g(z) = -t_0 + t_1 z (alpha=0
# plus convention), single root z = t_0/t_1. With t_1=1, t_0 = 0.5 + s the root
# = 0.5 + s crosses z=+1 at s*=0.5 (t_0 = +1). The other root structure is
# absent (degree 1), so no spurious crossings.
s_grid = np.linspace(0.0, 1.0, 201)
# t_path = np.stack([-0.5 - s_grid, np.ones_like(s_grid), np.zeros_like(s_grid)], axis=1)  # old
t_path = np.stack([0.5 + s_grid, np.ones_like(s_grid), np.zeros_like(s_grid)], axis=1)
cr = boundary_crossings(t_path, s_grid)
check("one analytic crossing on the line", len(cr) == 1, f"got {len(cr)}")
s_star = 0.5
if cr:
    check("crossing is z=+1 type", cr[0]['type'] == 'z=+1', f"{cr[0]}")
    check("analytic crossing near s*=0.5", abs(cr[0]['s'] - s_star) < 0.01)
# ED gap along the line
gaps = []
for t in t_path[::4]:
    e, _ = gd_solver_ed(jnp.asarray(t), model)
    e = np.array(e)
    gaps.append(e[1] - e[0])
gaps = np.array(gaps)
s_sub = s_grid[::4]
s_min = s_sub[int(np.argmin(gaps))]
check("ED gap minimum localizes at analytic s* (within O(1/L))",
      abs(s_min - s_star) < 0.12, f"s_min={s_min:.3f} s*={s_star:.3f}")
# STATE-loss gradient peak localizes too (energy grad is bounded and would not).
gnorm_line = np.array([float(jnp.linalg.norm(grad_ent(jnp.asarray(t)))) for t in t_path[::4]])
s_gpeak = s_sub[int(np.argmax(gnorm_line))]
check("loss-gradient peak localizes at analytic s*",
      abs(s_gpeak - s_star) < 0.12, f"s_gpeak={s_gpeak:.3f} s*={s_star:.3f}")

# ===========================================================================
print("== 4.4 boundary localization: complex-pair crossing (c=1), check k* ==")
# t(s) = (-s, 0, 1-s): g=(1-s)z^2+s (ttilde_0 = -t_0 = s), complex pair crosses
# |z|=1 at s*=0.5, k*=pi/2.
# t_path2 = np.stack([s_grid, np.zeros_like(s_grid), 1.0 - s_grid], axis=1)   # old
t_path2 = np.stack([-s_grid, np.zeros_like(s_grid), 1.0 - s_grid], axis=1)
cr2 = boundary_crossings(t_path2, s_grid)
check("one complex crossing on the line", len(cr2) == 1 and cr2[0]['type'] == 'complex',
      f"got {cr2}")
if cr2:
    check("complex crossing near s*=0.5", abs(cr2[0]['s'] - 0.5) < 0.01)
    kstar = cr2[0]['kstar']
    check("analytic k* == pi/2", abs(kstar - np.pi / 2) < 1e-2, f"k*={kstar}")
    # predicted k* matches the continuous-k argmin of |f| at the crossing point
    # t_cross = np.array([0.5, 0.0, 0.5])   # old convention
    t_cross = np.array([-0.5, 0.0, 0.5])
    ks = np.linspace(0, 2 * np.pi, 4001, endpoint=False)
    k_argmin = ks[int(np.argmin(np.abs(f_symbol(t_cross, ks))))]
    k_argmin = min(k_argmin, 2 * np.pi - k_argmin)   # fold to (0,pi]
    check("k* matches argmin_k |f(k)| at crossing",
          abs(k_argmin - np.pi / 2) < 5e-3, f"argmin k={k_argmin:.4f}")
# ED gap min on this line near s*=0.5
gaps2 = []
for t in t_path2[::4]:
    e, _ = gd_solver_ed(jnp.asarray(t), model)
    e = np.array(e)
    gaps2.append(e[1] - e[0])
gaps2 = np.array(gaps2)
s_min2 = s_grid[::4][int(np.argmin(gaps2))]
check("ED gap minimum localizes at complex s*=0.5 (within O(1/L))",
      abs(s_min2 - 0.5) < 0.12, f"s_min={s_min2:.3f}")

print()
print(f"==== gradient/boundary tests: {PASS} passed, {FAIL} failed ====")
sys.exit(1 if FAIL else 0)
