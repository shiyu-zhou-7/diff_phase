"""
Cluster test for active phase discovery with L=10.
Uses full training parameters with L=10 to verify before submitting L=20 job.
For the full L=20 cluster run, use main_active_phase.py.
"""
import numpy as np
from datetime import datetime

from jax import config
config.update("jax_enable_x64", True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f'Timestamp: {timestamp}')

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

with _devctx:
    try:
        file = '../models/xxzhdmrg_autoencoder_params.pkl'
        checkpoint = load_pickle(file)
        init_params = checkpoint['params']
        ferro_centroid = checkpoint['centroid']
        print(f'LOADED CHECKPOINT file: {file}.')
        print(f"Centroid shape: {ferro_centroid.shape}")
    except Exception:
        print('NO CHECKPOINT FOUND. INITIALIZING FROM SCRATCH.')
        init_params = None
        ferro_centroid = None

    params, centroid, hist = active_phase_discovery(
        L=10,
        init_delta=-1.5,
        init_h=-0.3,
        ae_cfg=AEConfig(),
        ham_cfg=HamConfig(),
        act_cfg=ActiveConfig(),
        dmrg_cfg=DMRGConfig(sweeps=7),
        init_ae_params=init_params,
        ferro_centroid=ferro_centroid,
        max_outer_iters=6,
        checkpoint_path='../models/active_phase_discovery_test_checkpoint.pkl',
        checkpoint_every_steps=50,
        checkpoint_every_outer=1,
    )

    save_pickle(
        {'params': params, 'centroid': np.array(centroid), 'hist': hist},
        f'../models/active_phase_discovery_test_{timestamp}.pkl'
    )
    print(f'Saved ../models/active_phase_discovery_test_{timestamp}.pkl')
