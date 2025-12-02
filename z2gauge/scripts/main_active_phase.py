import numpy as np
from datetime import datetime

from jax import config
config.update("jax_enable_x64", True)

from workflows.active_phase_discovery import active_phase_discovery
from utils.io import save_pickle, load_pickle
from configs.config import AEConfig, HamConfig, ActiveConfig

# Get current time string
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Optionally: load a pretrained AE and centroid
Lx = 2
Ly = 1
N = 2 * Lx * Ly
try:
    checkpoint = load_pickle(f'../models/z2gauge_autoencoder_params_Lx{Lx}Ly{Ly}.pkl')
    init_params = checkpoint['params']
    ferro_centroid = checkpoint['centroid']
    print('LOADED CHECKPOINT.')
except Exception:
    print('NO CHECKPOINT FOUND. INITIALIZING FROM SCRATCH.')
    init_params = None
    ferro_centroid = None

params, centroid, hist = active_phase_discovery(
    Lx=Lx,
    Ly=Ly,
    h_init=-1.5,
    ae_cfg=AEConfig(),
    ham_cfg=HamConfig(),
    act_cfg=ActiveConfig(),
    init_ae_params=init_params,
    ferro_centroid=ferro_centroid,
    max_outer_iters=6,
)

save_pickle({'params': params, 'centroid': np.array(centroid), 'hist': hist}, f'../models/active_phase_discovery_checkpoint_lr{HamConfig().lr}_{timestamp}.pkl')
print(f'Saved ../models/active_phase_discovery_checkpoint_lr{HamConfig().lr}_{timestamp}.pkl')
