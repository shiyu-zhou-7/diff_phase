import numpy as np
from datetime import datetime

from jax import config
config.update("jax_enable_x64", True)

# Get current time string
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f'Timestamp: {timestamp}')

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

with _devctx:
    # Optionally: load a pretrained AE and centroid
    try:
        # file = '../models/xxzhdmrg_autoencoder_params.pkl'
        file = '../models/xxzhdmrg1site_autoencoder_params_latent5_delta-2.5TO-1.5_h0.0TO0.5_L20.pkl'
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
