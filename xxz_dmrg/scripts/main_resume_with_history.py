"""
Resume active phase discovery from a checkpoint with a history-aware centroid.

The AE is trained on data from two small-radius circles:
  - Circle 1 (history):  around (HISTORY_DELTA, HISTORY_H), the original
                         starting point of the previous run.
  - Circle 2 (current):  around the stuck position loaded from the checkpoint.

Both circles use SAMPLE_RADIUS=0.01 to stay within a single phase each.
The centroid passed to active_phase_discovery is the average of the two
per-circle centroids in the shared latent space, giving the optimizer a
directional bias that reflects the history of the run.

Usage:
    python main_resume_with_history.py

Checkpoint loaded : ../models/active_phase_discovery_checkpoint_latest.pkl
Checkpoint written: ../models/active_phase_resume_history_<timestamp>.pkl  (never overwrites latest)
"""

import numpy as np
from datetime import datetime
from dataclasses import replace

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Timestamp: {timestamp}")

import jax
from jax import config
config.update("jax_enable_x64", True)

from contextlib import nullcontext
from utils.gpu_utils import get_gpu_device

_gpu = get_gpu_device()
_devctx = jax.default_device(_gpu) if _gpu is not None else nullcontext()
if _gpu is not None:
    print(f"Using GPU: {_gpu}")
else:
    print("No GPU found, using CPU")

import jax.numpy as jnp

from configs.config import AEConfig, HamConfig, ActiveConfig, DMRGConfig
from models.autoencoder import init_params as init_ae, fetch_latent
from training.ae_train import train_autoencoder
from workflows.active_phase_discovery import active_phase_discovery, sample_params, generate_data_xxzh
from utils.io import save_pickle, load_pickle

# ── tunables ──────────────────────────────────────────────────────────────────
CHECKPOINT_IN  = '../models/active_phase_discovery_checkpoint_latest.pkl'
CHECKPOINT_OUT = f'../models/active_phase_resume_history_{timestamp}.pkl'
RESUME_CHECKPOINT_PATH = f'../models/active_phase_resume_history_run_{timestamp}.pkl'

L               = 20
BOOTSTRAP_N     = 500    # samples per circle (1000 total)
SAMPLE_RADIUS   = 0.01   # tight radius to stay within one phase

HISTORY_DELTA   = -1.5   # original starting point of the previous run
HISTORY_H       = -0.3

MAX_OUTER_ITERS = 6
# ──────────────────────────────────────────────────────────────────────────────

ae_cfg   = AEConfig()
ham_cfg  = HamConfig()
dmrg_cfg = DMRGConfig()
act_cfg  = replace(
    ActiveConfig(),
    sample_radius_delta=SAMPLE_RADIUS,
    sample_radius_h=SAMPLE_RADIUS,
    num_samples_when_stalled=BOOTSTRAP_N,
)

with _devctx:
    # ── 1. load checkpoint ────────────────────────────────────────────────────
    print(f"Loading checkpoint: {CHECKPOINT_IN}")
    ckpt = load_pickle(CHECKPOINT_IN)
    ham_param  = ckpt['ham_param']
    init_delta = float(ham_param[0])
    init_h     = float(ham_param[1])
    print(f"Resuming from     (delta={init_delta:.5f}, h={init_h:.5f})")
    print(f"History origin at (delta={HISTORY_DELTA:.5f}, h={HISTORY_H:.5f})")
    print("Discarding checkpoint AE params and centroid — rebootstrapping from scratch.")

    # ── 2. build observables list (must match training) ───────────────────────
    sx = jnp.array([[0, 1], [1, 0]])
    sz = jnp.array([[1, 0], [0, -1]])
    sy = jnp.array([[0, -1j], [1j, 0]])
    observables_list = (
        [(i, sz) for i in range(L)] +
        [(i, sx) for i in range(L)] +
        [(i, sy) for i in range(L)]
    )
    D = len(observables_list)
    print(f"Observable dimension D={D}")

    key = jax.random.PRNGKey(ae_cfg.seed)

    # ── 3a. generate data around history point ────────────────────────────────
    print(f"\nSampling {BOOTSTRAP_N} points with radius {SAMPLE_RADIUS} "
          f"around history point (delta={HISTORY_DELTA}, h={HISTORY_H})...")
    key, sub = jax.random.split(key)
    deltas_hist, hs_hist = sample_params(
        HISTORY_DELTA, HISTORY_H, BOOTSTRAP_N, SAMPLE_RADIUS, SAMPLE_RADIUS, sub
    )
    X_history = generate_data_xxzh(L, deltas_hist, hs_hist, observables_list, dmrg_cfg)
    print(f"History data shape: {X_history.shape}")

    # ── 3b. generate data around current stuck point ──────────────────────────
    print(f"\nSampling {BOOTSTRAP_N} points with radius {SAMPLE_RADIUS} "
          f"around resume point (delta={init_delta:.5f}, h={init_h:.5f})...")
    key, sub = jax.random.split(key)
    deltas_res, hs_res = sample_params(
        init_delta, init_h, BOOTSTRAP_N, SAMPLE_RADIUS, SAMPLE_RADIUS, sub
    )
    X_resume = generate_data_xxzh(L, deltas_res, hs_res, observables_list, dmrg_cfg)
    print(f"Resume data shape: {X_resume.shape}")

    # ── 3c. combine ───────────────────────────────────────────────────────────
    X_combined = jnp.concatenate([X_history, X_resume], axis=0)
    print(f"Combined data shape: {X_combined.shape}")

    # ── 4. train fresh AE on combined data ────────────────────────────────────
    print("\nInitializing fresh AE params...")
    layers = [D, 20, ae_cfg.latent_dim, 20, D]
    key, sub = jax.random.split(key)
    fresh_ae_params = init_ae(layers, sub)

    print(f"Training AE for {ae_cfg.epochs} epochs on {len(X_combined)} samples...")
    fresh_ae_params = train_autoencoder(
        fresh_ae_params, X_combined,
        epochs=ae_cfg.epochs,
        lr=ae_cfg.lr,
        weight_decay=ae_cfg.weight_decay,
        drop_p=ae_cfg.dropout_p,
        center_coeff=act_cfg.center_coeff,
        seed=ae_cfg.seed,
    )

    # ── 5. compute history-aware centroid ─────────────────────────────────────
    eval_key = jax.random.PRNGKey(0)
    Z_history = fetch_latent(fresh_ae_params, X_history, eval_key)
    Z_resume  = fetch_latent(fresh_ae_params, X_resume,  eval_key)

    C_history = jnp.mean(Z_history, axis=0)
    C_resume  = jnp.mean(Z_resume,  axis=0)
    new_centroid = (C_history + C_resume) / 2.0

    print(f"Centroid shape: {new_centroid.shape}")
    print(f"  C_history (delta={HISTORY_DELTA}, h={HISTORY_H}): {np.array(C_history[:4])} ...")
    print(f"  C_resume  (delta={init_delta:.5f}, h={init_h:.5f}):  {np.array(C_resume[:4])} ...")
    print(f"  Averaged centroid: {np.array(new_centroid[:4])} ...")

    # ── 6. save bootstrap checkpoint (does NOT touch latest.pkl) ─────────────
    save_pickle({
        'params':         fresh_ae_params,
        'centroid':       np.array(new_centroid),
        'C_history':      np.array(C_history),
        'C_resume':       np.array(C_resume),
        'ham_param':      np.array([init_delta, init_h]),
        'history_origin': np.array([HISTORY_DELTA, HISTORY_H]),
        'source_checkpoint': CHECKPOINT_IN,
        'timestamp':      timestamp,
    }, CHECKPOINT_OUT)
    print(f"\nBootstrap checkpoint saved: {CHECKPOINT_OUT}")

    # ── 7. resume active phase discovery ─────────────────────────────────────
    print(f"\nResuming active_phase_discovery from (delta={init_delta:.5f}, h={init_h:.5f})")
    print(f"  sample_radius_delta      = {act_cfg.sample_radius_delta}")
    print(f"  sample_radius_h          = {act_cfg.sample_radius_h}")
    print(f"  num_samples_when_stalled = {act_cfg.num_samples_when_stalled}")
    print(f"  max_outer_iters          = {MAX_OUTER_ITERS}")
    print(f"  checkpoint_path          = {RESUME_CHECKPOINT_PATH}\n")

    params, centroid, hist = active_phase_discovery(
        L=L,
        init_delta=init_delta,
        init_h=init_h,
        ae_cfg=ae_cfg,
        ham_cfg=ham_cfg,
        act_cfg=act_cfg,
        dmrg_cfg=dmrg_cfg,
        init_ae_params=fresh_ae_params,
        ferro_centroid=new_centroid,
        max_outer_iters=MAX_OUTER_ITERS,
        checkpoint_path=RESUME_CHECKPOINT_PATH,
        checkpoint_every_steps=50,
        checkpoint_every_outer=1,
    )

    save_pickle(
        {'params': params, 'centroid': np.array(centroid), 'hist': hist},
        f'../models/active_phase_resume_history_final_{timestamp}.pkl',
    )
    print(f"Saved final output: ../models/active_phase_resume_history_final_{timestamp}.pkl")
