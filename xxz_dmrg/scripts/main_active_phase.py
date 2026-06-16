import os
import numpy as np
from datetime import datetime
from dataclasses import replace

from jax import config
config.update("jax_enable_x64", True)

# Get current time string
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f'Timestamp: {timestamp}')

# GPU configuration via a single default_device context
import jax
from contextlib import nullcontext
from utils.gpu_utils import get_gpu_device
_gpu = get_gpu_device()
_devctx = jax.default_device(_gpu) if _gpu is not None else nullcontext()
if _gpu is not None:
    print(f"Using GPU: {_gpu}")
else:
    print("No GPU found, using CPU")

from workflows.active_phase_discovery import active_phase_discovery
from utils.io import save_pickle, load_pickle
from configs.config import AEConfig, HamConfig, ActiveConfig, DMRGConfig

# ── run parameterization via env vars (fresh seed-varied runs) ──────────────
AE_SEED    = int(os.environ.get('AE_SEED', AEConfig().seed))
INIT_DELTA = float(os.environ.get('INIT_DELTA', 2.0))
INIT_H     = float(os.environ.get('INIT_H', 0.0))
FRESH      = bool(int(os.environ.get('FRESH', '0')))   # 1 -> ignore pretrained AE, bootstrap from scratch
ae_cfg     = replace(AEConfig(), seed=AE_SEED)
run_tag    = f'd{INIT_DELTA:g}_h{INIT_H:g}_seed{AE_SEED}'
CKPT_LATEST = f'../models/active_phase_fresh_{run_tag}_latest.pkl'
print(f'AE_SEED={AE_SEED}  INIT=(delta={INIT_DELTA}, h={INIT_H})  FRESH={FRESH}  run_tag={run_tag}')

with _devctx:
    # Optionally: load a pretrained AE and centroid (skipped when FRESH=1).
    if FRESH:
        print('FRESH=1: ignoring any pretrained AE; bootstrapping a brand-new AE from scratch.')
        init_params = None
        ferro_centroid = None
    else:
        try:
            file = '../models/xxzhdmrg_autoencoder_params.pkl'
            # file = '../models/xxzhdmrg1site_autoencoder_params_latent5_delta-2.5TO-1.5_h0.0TO0.5_L20.pkl'
            checkpoint = load_pickle(file)
            # checkpoint = load_pickle('../models/xxzhdmrg1site_autoencoder_params_latent5_delta-2.5TO-1.5_h0.0TO0.5_L20.pkl')
            init_params = checkpoint['params']
            ferro_centroid = checkpoint['centroid']
            print(f'LOADED CHECKPOINT file: {file}.')
            print(f"Centriod shape: {ferro_centroid.shape}")
        except Exception:
            print('NO CHECKPOINT FOUND. INITIALIZING FROM SCRATCH.')
            init_params = None
            ferro_centroid = None

    params, centroid, hist = active_phase_discovery(
        L=20,
        # init_delta=-1.5,   # old hardcoded fresh start
        # init_h=-0.3,
        init_delta=INIT_DELTA,
        init_h=INIT_H,
        ae_cfg=ae_cfg,
        ham_cfg=HamConfig(),
        act_cfg=ActiveConfig(),
        dmrg_cfg=DMRGConfig(),
        init_ae_params=init_params,
        ferro_centroid=ferro_centroid,
        max_outer_iters=6,
        # checkpoint_path='../models/active_phase_discovery_checkpoint_latest.pkl',  # old shared name (would clobber across runs)
        checkpoint_path=CKPT_LATEST,
        checkpoint_every_steps=50,
        checkpoint_every_outer=1,
        run_tag=run_tag,
    )

    out = f'../models/active_phase_fresh_final_{run_tag}_lr{HamConfig().lr}_{timestamp}.pkl'
    save_pickle({'params': params, 'centroid': np.array(centroid), 'hist': hist}, out)
    print(f'Saved {out}')
