"""
Train a combined AE on the **post-fix** (new sign convention) z2gauge datasets.

The previous combined AE (`z2gauge_ite_autoencoder_combined_Lx2Ly3.pkl`, dated
2026-05-05) was trained from data files generated BEFORE commit 8a7585a, which
silently negated `h` before handing it to ITE. As a result, the old combined
AE knows about negative-h *deconfined* states and positive-h *confined*
states — opposite gauge sectors for the two phases. Trajectories from the
post-fix active-phase runs (which all use the new no-negation convention)
land in the negative-h gauge sector for both phases, so the AE only correctly
encodes their deconfined parts and treats the confined parts as OOD,
extrapolating them toward whatever cluster it does know about. Hence the
"trajectories stay in the deconfined cluster even after crossing the boundary"
artifact in the previous overlays.

This script retrains the combined AE on the post-fix data files:
    data_ite_confined_2x3_h-1.0_to_-0.4_n1000.pkl       (negative-h confined)
    data_ite_deconfined_2x3_h-0.2_to_-0.001_n1000.pkl   (negative-h deconfined)
and saves the result as
    z2gauge_ite_autoencoder_combined_postfix_Lx2Ly3.pkl

Architecture / hyperparameters come from the current AEConfig
(layer_widths derived from D_in = 2^(2*Lx*Ly), latent_dim = whatever the
default tuple's middle entry says).

Usage:
    cd z2gauge/scripts && python main_train_combined_ae.py
    LX=2 LY=3 python main_train_combined_ae.py        # same as default
"""
import os, sys, pickle
from dataclasses import replace

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import AEConfig
from models.autoencoder import init_params, fetch_latent
from training.ae_train import train_autoencoder
from utils.io import save_pickle, timestamp


# ── tunables (env-var overridable) ────────────────────────────────────────────
Lx = int(os.environ.get('LX', 2))
Ly = int(os.environ.get('LY', 3))
SEED = int(os.environ.get('SEED', AEConfig().seed))
N = 2 * Lx * Ly
D = 2 ** N

CONF_PATH = os.environ.get('CONF_DATA',
                           f'../data/data_ite_confined_{Lx}x{Ly}_h-1.0_to_-0.4_n1000.pkl')
DEC_PATH  = os.environ.get('DEC_DATA',
                           f'../data/data_ite_deconfined_{Lx}x{Ly}_h-0.2_to_-0.001_n1000.pkl')

print(f'Lattice {Lx}x{Ly}  (N={N} qubits, D_in={D})')
print(f'Confined data:   {CONF_PATH}')
print(f'Deconfined data: {DEC_PATH}')

# ── load and stack ───────────────────────────────────────────────────────────
with open(CONF_PATH, 'rb') as f:
    data_conf = pickle.load(f)
with open(DEC_PATH, 'rb') as f:
    data_dec = pickle.load(f)

X_conf = np.array([d['v'].real for d in data_conf], dtype=np.float32)
X_dec  = np.array([d['v'].real for d in data_dec], dtype=np.float32)

h_conf = np.array([float(d['h']) for d in data_conf])
h_dec  = np.array([float(d['h']) for d in data_dec])

print(f'  confined  : {X_conf.shape}, h ∈ [{h_conf.min():+.4f}, {h_conf.max():+.4f}]')
print(f'  deconfined: {X_dec.shape}, h ∈ [{h_dec.min():+.4f}, {h_dec.max():+.4f}]')

X_all = np.concatenate([X_conf, X_dec], axis=0)
phase_all = np.array(['confined']*len(X_conf) + ['deconfined']*len(X_dec))

# Shuffle so SGD doesn't see the two phases sequentially.
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(X_all))
X_all_s = X_all[perm]
phase_all_s = phase_all[perm]

print(f'Combined training set: {X_all_s.shape}')

# ── AE config + train ────────────────────────────────────────────────────────
ae_cfg = AEConfig()
hidden_dim = ae_cfg.layer_widths[1]
latent_dim = ae_cfg.layer_widths[2]
ae_cfg = replace(ae_cfg, layer_widths=(D, hidden_dim, latent_dim, hidden_dim, D))
print(f'AE layer widths: {ae_cfg.layer_widths}')
print(f'AE epochs={ae_cfg.epochs}, lr={ae_cfg.lr}, dropout={ae_cfg.dropout_p}, '
      f'center_coeff={ae_cfg.center_coeff}, seed={SEED}')

key = jax.random.PRNGKey(SEED)
key, sub = jax.random.split(key)
params = init_params(list(ae_cfg.layer_widths), sub, scale=ae_cfg.init_scale)

hist = train_autoencoder(
    params,
    X_all_s,
    epochs=ae_cfg.epochs,
    lr=ae_cfg.lr,
    drop_p=ae_cfg.dropout_p,
    center_coeff=ae_cfg.center_coeff,
    seed=SEED,
    log_every=ae_cfg.log_every,
    eval_every=ae_cfg.eval_every,
)
trained = hist['params']

# Save: AE params + centroid (mean of soft-normalized latents over training set)
Z = fetch_latent(trained, X_all_s, jax.random.PRNGKey(0))
centroid = jnp.mean(Z, axis=0)

out_path = f'../models/z2gauge_ite_autoencoder_combined_postfix_Lx{Lx}Ly{Ly}.pkl'
save_pickle({'params': trained, 'centroid': np.array(centroid)}, out_path)
print(f'Saved combined-postfix AE to {out_path}')
