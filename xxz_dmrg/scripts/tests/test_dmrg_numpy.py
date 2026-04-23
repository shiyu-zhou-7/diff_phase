"""
Correctness test: compare numpy DMRG against JAX DMRG.

For each parameter point we run both implementations with the same L, bond dim,
and sweep count, then check that all 3L site observables (sz, sx, sy) agree to
within ABS_TOL.  We use L=10 so each run is fast.

Run from xxz_dmrg/scripts/:
    python -m tests.test_dmrg_numpy
or:
    python tests/test_dmrg_numpy.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

from dmrg.hamiltonians import XXZhX
from dmrg.dmrg        import run_dmrg
from dmrg.dmrg_numpy  import run_dmrg_numpy
from configs.config   import DMRGConfig

# ── test settings ─────────────────────────────────────────────────────────────
L        = 10
ABS_TOL  = 1e-3   # max allowed absolute difference in any observable
SWEEPS   = 15     # more sweeps → better convergence → tighter agreement

# parameter points: (delta, h) spanning different phases of XXZ+hX
PARAMS = [
    (-2.0,  0.0),   # ferromagnetic XXZ, no field
    ( 0.5,  0.0),   # XXZ easy-plane, no field
    ( 1.0,  2.0),   # field-dominated
    (-1.0, -1.0),   # near the boundary where we got stuck
]
# ──────────────────────────────────────────────────────────────────────────────

def build_observables(L):
    sx = jnp.array([[0, 1],    [1, 0]],    dtype=jnp.complex128)
    sy = jnp.array([[0, -1j],  [1j, 0]],   dtype=jnp.complex128)
    sz = jnp.array([[1.0, 0.0],[0.0,-1.0]], dtype=jnp.complex128)
    obs = [(i, sz) for i in range(L)] + \
          [(i, sx) for i in range(L)] + \
          [(i, sy) for i in range(L)]
    return obs


def run_jax(delta, h, dmrg_cfg, obs):
    model = XXZhX(L, delta, h)
    psi   = run_dmrg(L, model, dmrg_cfg)
    vals  = psi.get_site_exp_val(obs)
    return np.array([np.real(np.asarray(v)) for v in vals])


def run_numpy(delta, h, dmrg_cfg, obs):
    psi  = run_dmrg_numpy(L, delta, h, dmrg_cfg)
    vals = psi.get_site_exp_val(obs)
    return np.array([np.real(np.asarray(v)) for v in vals])


def test_all():
    dmrg_cfg         = DMRGConfig()
    dmrg_cfg.sweeps  = SWEEPS
    obs              = build_observables(L)

    all_passed = True
    for delta, h in PARAMS:
        jax_obs = run_jax(delta, h, dmrg_cfg, obs)
        np_obs  = run_numpy(delta, h, dmrg_cfg, obs)

        max_diff = np.max(np.abs(jax_obs - np_obs))
        passed   = max_diff < ABS_TOL
        status   = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"[{status}] delta={delta:+.1f}, h={h:+.1f} | "
              f"max |jax - numpy| = {max_diff:.2e}  (tol={ABS_TOL:.0e})")

        if not passed:
            worst = np.argmax(np.abs(jax_obs - np_obs))
            op_names = (["sz"] * L + ["sx"] * L + ["sy"] * L)
            site     = worst % L
            print(f"       worst observable: {op_names[worst]}[site {site}] "
                  f"jax={jax_obs[worst]:.6f}  numpy={np_obs[worst]:.6f}")

    print()
    if all_passed:
        print("All tests passed.")
    else:
        print("Some tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    test_all()
