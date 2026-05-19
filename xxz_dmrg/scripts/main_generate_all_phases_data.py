"""
Generate DMRG observable vectors for:
  (a) ~1250 (Δ, h) points spanning all 5 regions of the XXZ + transverse-field
      phase diagram (Néel-Z, FM-Z, Néel-Y, paramagnet, h=0 liquid), and
  (b) every (Δ, h) along the 437956 active-phase-discovery trajectory.

The output pickle is consumed downstream by:
  - main_train_all_phases_ae.py  (trains a fresh AE on the training set)
  - plot_trajectory_in_latent.py (encodes trajectory, PCA, plots)

Set DRY_RUN=1 to run a small local version for schema validation.
"""

import os
import numpy as np
from datetime import datetime

import jax
from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp

from configs.config import DMRGConfig
from workflows.active_phase_discovery import generate_data_xxzh
from utils.io import save_pickle, load_pickle


DRY_RUN = bool(int(os.environ.get('DRY_RUN', '0')))
MODE    = os.environ.get('MODE', 'all')   # 'all', 'train_only', 'traj_only'
assert MODE in ('all', 'train_only', 'traj_only'), f"Invalid MODE={MODE!r}"

L              = 20
SEED           = 42
PREV_PICKLE    = '../models/active_phase_discovery_checkpoint_latest.pkl'
RUN_PICKLE     = '../models/active_phase_resume_history_run_20260505_235151.pkl'
REAL_START     = np.array([[-1.5, -0.3]])  # (Δ, h)
PER_REGION     = 10 if DRY_RUN else 250
TRAJ_DECIMATE  = int(os.environ.get('TRAJ_DECIMATE', '50' if DRY_RUN else '1'))

# (label, delta_range, h_range_or_None)  -- None ⇒ h fixed at 0
PHASE_REGIONS = [
    ('Neel_Z', (1.2,   1.7),   (1.5,   2.0)),
    ('FM_Z',   (-2.0, -1.5),   (-0.5,  0.0)),
    ('Neel_Y', (0.0,   0.5),   (0.5,   1.0)),
    ('PM',     (-1.25, -0.75), (-1.25, -0.75)),
    ('Liquid', (-0.5,  0.5),   None),
]


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Timestamp:        {timestamp}")
print(f"MODE:             {MODE}")
print(f"DRY_RUN:          {DRY_RUN}")
print(f"Per-region count: {PER_REGION}")
print(f"Trajectory decim: {TRAJ_DECIMATE}")

sx = jnp.array([[0, 1], [1, 0]])
sz = jnp.array([[1, 0], [0, -1]])
sy = jnp.array([[0, -1j], [1j, 0]])
observables_list = (
    [(i, sz) for i in range(L)] +
    [(i, sx) for i in range(L)] +
    [(i, sy) for i in range(L)]
)
D = len(observables_list)
print(f"Observable dim D: {D}")

dmrg_cfg = DMRGConfig()

rng = np.random.default_rng(SEED)

train_params = train_obs = train_phase = None
if MODE != 'traj_only':
    # ── Sample + run DMRG over training set ────────────────────────────────
    train_params_chunks = []
    train_obs_chunks    = []
    train_phase_chunks  = []

    for label, d_range, h_range in PHASE_REGIONS:
        print(f"\n===== Region '{label}' "
              f"(Δ∈{d_range}, h={'0 (fixed)' if h_range is None else h_range}) =====")
        deltas = rng.uniform(d_range[0], d_range[1], size=PER_REGION)
        if h_range is None:
            hs = np.zeros(PER_REGION)
        else:
            hs = rng.uniform(h_range[0], h_range[1], size=PER_REGION)

        X = np.asarray(generate_data_xxzh(L, deltas, hs, observables_list, dmrg_cfg))

        train_params_chunks.append(np.stack([deltas, hs], axis=1))
        train_obs_chunks.append(X)
        train_phase_chunks.extend([label] * PER_REGION)

    train_params = np.concatenate(train_params_chunks, axis=0)
    train_obs    = np.concatenate(train_obs_chunks,    axis=0)
    train_phase  = np.array(train_phase_chunks)
    print(f"\nTraining set: params={train_params.shape}, obs={train_obs.shape}")

traj_params = traj_obs = None
if MODE != 'train_only':
    # ── Build trajectory (real start + prev + run) ────────────────────────
    prev = load_pickle(PREV_PICKLE)
    prev_params = np.array(prev['hist']['ham_params'])
    run  = load_pickle(RUN_PICKLE)
    run_params  = np.array(run['hist']['ham_params'])

    traj_full = np.vstack([REAL_START, prev_params, run_params])
    n_full = len(traj_full)
    idx = list(range(0, n_full, TRAJ_DECIMATE))
    if idx[-1] != n_full - 1:
        idx.append(n_full - 1)
    traj_params = traj_full[idx]
    print(f"\nTrajectory: {traj_params.shape}  "
          f"(prev={len(prev_params)}, run={len(run_params)}, "
          f"+1 real-start, decim={TRAJ_DECIMATE}, "
          f"first={traj_params[0].tolist()}, last={traj_params[-1].tolist()})")

    # ── Run DMRG over trajectory ──────────────────────────────────────────
    print(f"\n===== DMRG over trajectory ({len(traj_params)} pts) =====")
    traj_obs = np.asarray(generate_data_xxzh(
        L, traj_params[:, 0], traj_params[:, 1], observables_list, dmrg_cfg
    ))
    print(f"Trajectory obs: {traj_obs.shape}")

# ── Save ─────────────────────────────────────────────────────────────────
mode_suffix = '' if MODE == 'all' else f'_{MODE}'
out_path = f'../data/all_phases_latent_data{mode_suffix}_{timestamp}.pkl'
save_pickle({
    'train_params':   train_params,
    'train_obs':      train_obs,
    'train_phase':    train_phase,
    'traj_params':    traj_params,
    'traj_obs':       traj_obs,
    'L':              L,
    'seed':           SEED,
    'dry_run':        DRY_RUN,
    'mode':           MODE,
    'phase_regions':  PHASE_REGIONS,
    'prev_pickle':    PREV_PICKLE,
    'run_pickle':     RUN_PICKLE,
    'real_start':     REAL_START,
    'traj_decimate':  TRAJ_DECIMATE,
    'timestamp':      timestamp,
}, out_path)
print(f"\nSaved: {out_path}")
