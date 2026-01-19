import numpy as np
from datetime import datetime
import os

from jax import config
config.update("jax_enable_x64", True)

from workflows.active_phase_discovery import active_phase_discovery
from utils.io import save_pickle, load_pickle
from configs.config import AEConfig, HamConfig, ActiveConfig

# Get current time string
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Optionally: load a pretrained AE and centroid
Lx = 2
Ly = 3
N = 2 * Lx * Ly
checkpoint_path = f'../models/z2gauge_ite_autoencoder_params_Lx{Lx}Ly{Ly}.pkl'
init_params = None
ferro_centroid = None
print(f'z2gauge ite active phase discovery with Lx={Lx} and Ly={Ly}')

if os.path.exists(checkpoint_path):
    try:
        checkpoint = load_pickle(checkpoint_path)
        init_params = checkpoint['params']
        ferro_centroid = checkpoint['centroid']
        print('LOADED CHECKPOINT.')
    except FileNotFoundError:
        print('NO CHECKPOINT FOUND. INITIALIZING FROM SCRATCH.')
    except KeyError as e:
        print(f'CHECKPOINT FILE EXISTS BUT MISSING KEY: {e}. INITIALIZING FROM SCRATCH.')
    except Exception as e:
        print(f'ERROR LOADING CHECKPOINT: {type(e).__name__}: {e}. INITIALIZING FROM SCRATCH.')
else:
    print('NO CHECKPOINT FOUND. INITIALIZING FROM SCRATCH.')

params, centroid, hist = active_phase_discovery(
    Lx=Lx,
    Ly=Ly,
    h_init=-1.0,
    ae_cfg=AEConfig(),
    ham_cfg=HamConfig(),
    act_cfg=ActiveConfig(),
    init_ae_params=init_params,
    ferro_centroid=ferro_centroid,
    max_outer_iters=10,
)

save_pickle({'params': params, 'centroid': np.array(centroid), 'hist': hist}, f'../models/active_phase_discovery_checkpoint_lr{HamConfig().lr}_{timestamp}.pkl')
print(f'Saved ../models/active_phase_discovery_checkpoint_lr{HamConfig().lr}_{timestamp}.pkl')
