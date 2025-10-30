import numpy as np
from datetime import datetime

from jax import config
config.update("jax_enable_x64", True)

# GPU configuration via a single default_device context
import jax
from contextlib import nullcontext
_gpu = next((d for d in jax.devices() if d.platform == 'gpu'), None)
_devctx = jax.default_device(_gpu) if _gpu is not None else nullcontext()
if _gpu is not None:
    print(f"Using GPU: {_gpu}")
else:
    print("No GPU found, using CPU")

from workflows.active_phase_discovery import active_phase_discovery
from utils.io import save_pickle, load_pickle
from configs.config import AEConfig, HamConfig, ActiveConfig, DMRGConfig

# Get current time string
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

with _devctx:
    # Optionally: load a pretrained AE and centroid
    try:
        checkpoint = load_pickle('../models/xxzhdmrg_autoencoder_params.pkl')
        init_params = checkpoint['params']
        ferro_centroid = checkpoint['centroid']
        print('LOADED CHECKPOINT.')
    except Exception:
        print('NO CHECKPOINT FOUND. INITIALIZING FROM SCRATCH.')
        init_params = None
        ferro_centroid = None

    params, centroid, hist = active_phase_discovery(
        L=20,
        init_delta=-1.5,
        init_h=-0.3,
        ae_cfg=AEConfig(),
        ham_cfg=HamConfig(),
        act_cfg=ActiveConfig(),
        dmrg_cfg=DMRGConfig(),
        init_ae_params=init_params,
        ferro_centroid=ferro_centroid,
        max_outer_iters=6,
    )

    save_pickle({'params': params, 'centroid': np.array(centroid), 'hist': hist}, f'../models/active_phase_discovery_checkpoint_lr{HamConfig().lr}_{timestamp}.pkl')
    print(f'Saved ../models/active_phase_discovery_checkpoint_lr{HamConfig().lr}_{timestamp}.pkl')
