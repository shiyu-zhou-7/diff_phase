"""
Resume active phase discovery from a checkpoint, rebootstrapping the AE
with a tight sample radius to stay within a single phase near a boundary.

Usage:
    python main_resume_small_radius.py

Checkpoint loaded : ../models/active_phase_discovery_checkpoint_latest.pkl
Checkpoint written: ../models/active_phase_resume_<timestamp>.pkl  (never overwrites latest)
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
from dmrg.hamiltonians import XXZhX
from dmrg.dmrg import run_dmrg
from models.autoencoder import init_params as init_ae, fetch_latent
from training.ae_train import train_autoencoder
from workflows.active_phase_discovery import active_phase_discovery, sample_params, generate_data_xxzh
from utils.io import save_pickle, load_pickle

# ── tunables ──────────────────────────────────────────────────────────────────
CHECKPOINT_IN  = '../models/active_phase_discovery_checkpoint_latest.pkl'
CHECKPOINT_OUT = f'../models/active_phase_resume_{timestamp}.pkl'
RESUME_CHECKPOINT_PATH = f'../models/active_phase_resume_run_{timestamp}.pkl'

L               = 20
BOOTSTRAP_N     = 500    # samples for focused AE retraining
SAMPLE_RADIUS   = 0.01   # tight radius to stay within one phase

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
    ham_param = ckpt['ham_param']
    init_delta = float(ham_param[0])
    init_h     = float(ham_param[1])
    print(f"Resuming from (delta={init_delta:.5f}, h={init_h:.5f})")
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

    # ── 3. generate focused bootstrap data ───────────────────────────────────
    print(f"Sampling {BOOTSTRAP_N} points with radius {SAMPLE_RADIUS} around stuck position...")
    key = jax.random.PRNGKey(ae_cfg.seed)
    key, sub = jax.random.split(key)
    deltas, hs = sample_params(init_delta, init_h, BOOTSTRAP_N, SAMPLE_RADIUS, SAMPLE_RADIUS, sub)
    X_bootstrap = generate_data_xxzh(L, deltas, hs, observables_list, dmrg_cfg)
    print(f"Bootstrap data shape: {X_bootstrap.shape}")

    # ── 4. train fresh AE on focused data ────────────────────────────────────
    print("Initializing fresh AE params...")
    layers = [D, 20, ae_cfg.latent_dim, 20, D]
    key, sub = jax.random.split(key)
    fresh_ae_params = init_ae(layers, sub)

    print(f"Training AE for {ae_cfg.epochs} epochs...")
    fresh_ae_params = train_autoencoder(
        fresh_ae_params, X_bootstrap,
        epochs=ae_cfg.epochs,
        lr=ae_cfg.lr,
        weight_decay=ae_cfg.weight_decay,
        drop_p=ae_cfg.dropout_p,
        center_coeff=act_cfg.center_coeff,
        seed=ae_cfg.seed,
    )

    # ── 5. compute new centroid ───────────────────────────────────────────────
    Z = fetch_latent(fresh_ae_params, X_bootstrap, jax.random.PRNGKey(0))
    new_centroid = jnp.mean(Z, axis=0)
    print(f"New centroid shape: {new_centroid.shape}")

    # ── 6. save bootstrap checkpoint (does NOT touch latest.pkl) ─────────────
    save_pickle({
        'params':    fresh_ae_params,
        'centroid':  np.array(new_centroid),
        'ham_param': np.array([init_delta, init_h]),
        'source_checkpoint': CHECKPOINT_IN,
        'timestamp': timestamp,
    }, CHECKPOINT_OUT)
    print(f"Bootstrap checkpoint saved: {CHECKPOINT_OUT}")

    # ── 7. resume active phase discovery ─────────────────────────────────────
    print(f"\nResuming active_phase_discovery from (delta={init_delta:.5f}, h={init_h:.5f})")
    print(f"  sample_radius_delta = {act_cfg.sample_radius_delta}")
    print(f"  sample_radius_h     = {act_cfg.sample_radius_h}")
    print(f"  num_samples_when_stalled = {act_cfg.num_samples_when_stalled}")
    print(f"  max_outer_iters = {MAX_OUTER_ITERS}")
    print(f"  checkpoint_path = {RESUME_CHECKPOINT_PATH}\n")

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
        f'../models/active_phase_resume_final_{timestamp}.pkl',
    )
    print(f"Saved final output: ../models/active_phase_resume_final_{timestamp}.pkl")
