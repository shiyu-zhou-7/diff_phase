"""
Generate a *post-fix* z2gauge dataset where every sample's ITE initial
state v_0 is drawn INDEPENDENTLY from N(0, I_D), rather than sharing the
pinned-twin PRNGKey(0) used by the production pipeline.

Two narrow ranges:
  - deconfined: 500 samples uniformly drawn from h ∈ [-0.15, -0.10]
  - confined:   500 samples uniformly drawn from h ∈ [-1.50, -0.70]

Output:
  ../data/data_ite_deconfined_2x3_h-0.15_to_-0.1_n500_variedseed.pkl
  ../data/data_ite_confined_2x3_h-1.5_to_-0.7_n500_variedseed.pkl

The Z2 ground subspace is (quasi-)degenerate at small |h| (topological
4-fold degeneracy on the 2x3 torus), and ITE projects onto whichever
superposition of degenerate states the initial v_0 has nonzero overlap
with. By using a different v_0 per sample we explicitly *break* the
pinned-twin convention — samples within each phase will land on
different points in the ground manifold.

Sign convention: post-fix. `h` is passed directly to ITE; no negation.

Usage (cluster):
    cd z2gauge/scripts && python generate_variedseed_data.py
"""
import os, sys, pickle, time

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", False)   # match production ITE precision (float32)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hamiltonian.z2ham import sum_star_operators, transverse_field


# ── tunables ──────────────────────────────────────────────────────────────────
Lx, Ly = 2, 3
D = 2 ** (2 * Lx * Ly)
J_A = -1.0
N_STEPS = 150
DT = 1e-2

DECONFINED_H_RANGE = (-0.15, -0.10)
CONFINED_H_RANGE   = (-1.50, -0.70)
N_DEC, N_CONF = 500, 500

# Master keys for v_0 randomness (per-phase). Each sample's v_0 is drawn
# independently from the (B, D) array these produce.
DEC_INIT_KEY  = jax.random.PRNGKey(1001)
CONF_INIT_KEY = jax.random.PRNGKey(2002)

# Separate keys for the h-sampling.
H_DEC_KEY  = jax.random.PRNGKey(7777)
H_CONF_KEY = jax.random.PRNGKey(8888)


def ite_batched_varied_init(j_a, h_batch, S, T, *,
                            n_steps, dt, key, eps=1e-12):
    """Batched ITE where each sample's initial v_0 is independently sampled
    from N(0, I_D). Same scan body as `ite_ground_state_batched` -- only the
    initialization differs (no broadcast of a single v_0)."""
    B = h_batch.shape[0]
    dtype = S.dtype

    V = jax.random.normal(key, (B, D), dtype=dtype)
    V = V / (jnp.linalg.norm(V, axis=-1, keepdims=True) + eps)

    j_a_c = jnp.asarray(j_a, dtype=dtype)
    h_c = h_batch.astype(dtype)

    def body(V, _):
        SV = V @ S
        TV = V @ T
        HV = j_a_c * SV + h_c[:, None] * TV
        E = jnp.sum(V * HV, axis=-1)
        V_new = V - dt * (HV - E[:, None] * V)
        norms = jnp.linalg.norm(V_new, axis=-1, keepdims=True)
        return V_new / (norms + eps), E

    V, _ = jax.lax.scan(body, V, xs=None, length=n_steps)

    SV = V @ S
    TV = V @ T
    HV = j_a_c * SV + h_c[:, None] * TV
    E_final = jnp.real(jnp.sum(V * HV, axis=-1))
    V = jnp.real(V)
    norms = jnp.linalg.norm(V, axis=-1, keepdims=True)
    V = V / (norms + eps)
    return V, E_final


def _save_phase(phase, V, E, h_arr, out_path):
    data = []
    for i in range(len(h_arr)):
        data.append({
            'v': np.asarray(V[i]),
            'E': float(E[i]),
            'j_a': J_A,
            'h': float(h_arr[i]),
            'Lx': Lx,
            'Ly': Ly,
            'phase': phase,
        })
    with open(out_path, 'wb') as f:
        pickle.dump(data, f)
    energies = [d['E'] for d in data]
    print(f'  saved {len(data)} samples -> {out_path}')
    print(f'  size: {os.path.getsize(out_path) / 1e6:.2f} MB')
    print(f'  E range: [{min(energies):.4f}, {max(energies):.4f}]')


# ── main ─────────────────────────────────────────────────────────────────────
print(f'Lattice {Lx}x{Ly}, D={D}, n_steps={N_STEPS}, dt={DT}')
print('Building operators...')
t0 = time.time()
S_ops = sum_star_operators(Lx, Ly)
T_ops = transverse_field(Lx, Ly)
print(f'  done ({time.time() - t0:.1f}s)')

# Deconfined
print(f'\n=== deconfined: N={N_DEC}, h ∈ {DECONFINED_H_RANGE} ===')
t0 = time.time()
h_dec = jax.random.uniform(H_DEC_KEY, shape=(N_DEC,),
                           minval=DECONFINED_H_RANGE[0],
                           maxval=DECONFINED_H_RANGE[1])
print(f'  h range sampled: [{float(h_dec.min()):+.4f}, {float(h_dec.max()):+.4f}]')
V_dec, E_dec = ite_batched_varied_init(J_A, h_dec, S_ops, T_ops,
                                        n_steps=N_STEPS, dt=DT, key=DEC_INIT_KEY)
V_dec.block_until_ready()
print(f'  ITE done ({time.time() - t0:.1f}s)')
dec_path = f'../data/data_ite_deconfined_{Lx}x{Ly}_h-0.15_to_-0.1_n{N_DEC}_variedseed.pkl'
_save_phase('deconfined', V_dec, E_dec, h_dec, dec_path)

# Free JAX state between phases.
jax.clear_caches()

# Confined
print(f'\n=== confined: N={N_CONF}, h ∈ {CONFINED_H_RANGE} ===')
t0 = time.time()
h_conf = jax.random.uniform(H_CONF_KEY, shape=(N_CONF,),
                            minval=CONFINED_H_RANGE[0],
                            maxval=CONFINED_H_RANGE[1])
print(f'  h range sampled: [{float(h_conf.min()):+.4f}, {float(h_conf.max()):+.4f}]')
V_conf, E_conf = ite_batched_varied_init(J_A, h_conf, S_ops, T_ops,
                                          n_steps=N_STEPS, dt=DT, key=CONF_INIT_KEY)
V_conf.block_until_ready()
print(f'  ITE done ({time.time() - t0:.1f}s)')
conf_path = f'../data/data_ite_confined_{Lx}x{Ly}_h-1.5_to_-0.7_n{N_CONF}_variedseed.pkl'
_save_phase('confined', V_conf, E_conf, h_conf, conf_path)

print('\nDone.')
