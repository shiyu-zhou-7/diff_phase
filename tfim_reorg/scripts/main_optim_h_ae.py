"""
Drive the transverse-field strength h by minimizing latent distance away from
a target centroid in the AE latent space:

    L(h) = -||z(h) - centroid|| / D

AE auto-selection by |h_init|:
  - |h_init| < 1  →  use the SSB AE  (`ae_params_*.pkl`,        `ae_latent_centroids_*.pkl`)
  - |h_init| ≥ 1  →  use the para AE (`para_ae_params_*.pkl`,   `para_ae_centroid_*.pkl`)

  In each branch the lexicographically-latest matching artifact is loaded.
  Override either with env vars AE_PARAMS / AE_CENTROIDS.
  Override the starting field via H_INIT (e.g. `H_INIT=-1.5 python main_optim_h_ae.py`).
  Override total epochs via EPOCHS.

Writes:
    ../data/optim_h_ae_<ts>.pkl
    ../figures/optim_h_ae_<ts>.pdf

Usage:
    cd tfim_reorg/scripts && python main_optim_h_ae.py
    H_INIT=-1.5 python main_optim_h_ae.py
    H_INIT=-1.5 EPOCHS=50 python main_optim_h_ae.py
"""
import sys, os
import glob
import numpy as np
from dataclasses import replace, asdict

import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import HamConfig, AEConfig, OptimConfig
from hamiltonians.tfim import build_tfim_chain
from training.optim_h import optimize_h_ae
from utils.io import save_pickle, load_pickle, timestamp
from utils.plotting import plot_h_loss_combined


def _latest(pattern):
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f'No file matched pattern: {pattern}')
    return matches[-1]


# ── tunables ──────────────────────────────────────────────────────────────────
H_INIT_DEFAULT = -0.4
h_init = float(os.environ.get('H_INIT', H_INIT_DEFAULT))

ham_cfg = HamConfig(N=10, J=-1.0, h_init=h_init)
opt_cfg = OptimConfig(lr=0.1, epochs=1000, log_every=25)
if 'EPOCHS' in os.environ:
    opt_cfg = replace(opt_cfg, epochs=int(os.environ['EPOCHS']),
                      log_every=max(1, int(os.environ['EPOCHS']) // 10))
DROP_P  = 0.1

# AE selection: SSB if |h_init|<1, para if |h_init|>=1. Env vars still override.
if abs(ham_cfg.h_init) < 1.0:
    ae_kind             = 'ssb'
    default_params_glob = '../models/ae_params_*.pkl'
    default_centroid_glob = '../models/ae_latent_centroids_*.pkl'
else:
    ae_kind             = 'para'
    default_params_glob = '../models/para_ae_params_*.pkl'
    default_centroid_glob = '../models/para_ae_centroid_*.pkl'

ae_params_path    = os.environ.get('AE_PARAMS')    or _latest(default_params_glob)
ae_centroids_path = os.environ.get('AE_CENTROIDS') or _latest(default_centroid_glob)

ts = timestamp()
print(f'Timestamp: {ts}')
print(f'|h_init|={abs(ham_cfg.h_init):.4f}  →  using {ae_kind!r} AE')
print(f'AE params path:    {ae_params_path}')
print(f'AE centroids path: {ae_centroids_path}')
print(f'Ham config:   {asdict(ham_cfg)}')
print(f'Optim config: {asdict(opt_cfg)}')

# ── load AE + centroid ────────────────────────────────────────────────────────
ae_blob       = load_pickle(ae_params_path)
ae_params     = ae_blob['params']
trained_ae_cfg = ae_blob.get('config', {})

centroids_blob = load_pickle(ae_centroids_path)
# Both ssb and para centroid files store the centroid under 'centroid_ssb'
# (the para file aliases 'centroid_para' to the same array for back-compat).
centroid_target = jnp.asarray(centroids_blob['centroid_ssb'])
print(f'centroid shape: {centroid_target.shape}')

# ── build TFIM and run optimization ──────────────────────────────────────────
ham_X, ham_ZZ = build_tfim_chain(ham_cfg.N, ham_cfg.J)

h_list, loss_list = optimize_h_ae(
    ham_cfg.h_init,
    ham_X,
    ham_ZZ,
    latent_target=centroid_target,
    params=ae_params,
    epochs=opt_cfg.epochs,
    lr=opt_cfg.lr,
    drop_p=DROP_P,
    log_every=opt_cfg.log_every,
    seed=0,
)

# ── save ──────────────────────────────────────────────────────────────────────
data_path = f'../data/optim_h_ae_{ts}.pkl'
fig_path  = f'../figures/optim_h_ae_{ts}.pdf'

save_pickle({
    'h_list':            h_list,
    'loss_list':         loss_list,
    'ham_cfg':           asdict(ham_cfg),
    'opt_cfg':           asdict(opt_cfg),
    'ae_kind':           ae_kind,            # 'ssb' or 'para' (auto-selected by |h_init|)
    'ae_params_path':    ae_params_path,
    'ae_centroids_path': ae_centroids_path,
    'trained_ae_cfg':    trained_ae_cfg,
    'centroid':          np.asarray(centroid_target),
}, data_path)

plot_h_loss_combined(h_list, loss_list, fig_path)

print(f'\nSaved:')
print(f'  {data_path}')
print(f'  {fig_path}')
print(f'Final h = {h_list[-1]:+.6f}, final loss = {loss_list[-1]:+.6e}')
