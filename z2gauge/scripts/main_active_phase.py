import numpy as np
from dataclasses import replace
from datetime import datetime
import os

from jax import config
config.update("jax_enable_x64", True)

from workflows.active_phase_discovery import active_phase_discovery
from utils.io import save_pickle, load_pickle
from configs.config import AEConfig, HamConfig, ActiveConfig

# Get current time string
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ── env-var overrides so the same script can be launched with different settings
# from a SLURM job array without editing source.  Examples:
#   sbatch --export=ALL,H_INIT=-1.5,LX=2,LY=3,HAM_LR=0.1,MAX_STEPS_BLOCK=100,MAX_ITERS=1 \
#          gpu_run_active_phase.sbatch
#   H_INIT=-0.5 python main_active_phase.py
H_INIT           = float(os.environ.get('H_INIT', -1.0))
Lx               = int(os.environ.get('LX', 3))
Ly               = int(os.environ.get('LY', 3))
MAX_ITERS        = int(os.environ.get('MAX_ITERS', 10))
HAM_LR           = os.environ.get('HAM_LR')            # h-Adam step (None -> HamConfig default)
MAX_STEPS_BLOCK  = os.environ.get('MAX_STEPS_BLOCK')   # max inner h-steps per outer iter
PARAM_TOL_CHANGE = os.environ.get('PARAM_TOL_CHANGE')  # set to 0 to disable early-stop break in inner loop

N = 2 * Lx * Ly

# Auto-select the AE that matches the phase of h_init.
# Convention: deconfined when |h|<0.3, confined when |h|>0.3.
phase = 'deconfined' if abs(H_INIT) < 0.3 else 'confined'
checkpoint_path = f'../models/z2gauge_ite_autoencoder_{phase}_Lx{Lx}Ly{Ly}.pkl'
init_params = None
ferro_centroid = None

# Build configs, applying env overrides where provided.
ham_cfg = HamConfig()
if HAM_LR is not None:
    ham_cfg = replace(ham_cfg, lr=float(HAM_LR))
if MAX_STEPS_BLOCK is not None:
    ham_cfg = replace(ham_cfg, max_steps_block=int(MAX_STEPS_BLOCK))
if PARAM_TOL_CHANGE is not None:
    ham_cfg = replace(ham_cfg, param_tol_change=float(PARAM_TOL_CHANGE))

print(f'z2gauge ite active phase discovery with Lx={Lx} Ly={Ly} h_init={H_INIT} '
      f'(|h_init|={abs(H_INIT):.4f} -> phase={phase!r}) '
      f'max_outer_iters={MAX_ITERS} ham_lr={ham_cfg.lr} '
      f'max_steps_block={ham_cfg.max_steps_block} param_tol_change={ham_cfg.param_tol_change}')
print(f'AE checkpoint path: {checkpoint_path}')

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
    h_init=H_INIT,
    ae_cfg=AEConfig(),
    ham_cfg=ham_cfg,
    act_cfg=ActiveConfig(),
    init_ae_params=init_params,
    ferro_centroid=ferro_centroid,
    max_outer_iters=MAX_ITERS,
)

save_pickle({'params': params, 'centroid': np.array(centroid), 'hist': hist}, f'../models/active_phase_discovery_checkpoint_lr{ham_cfg.lr}_{timestamp}.pkl')
print(f'Saved ../models/active_phase_discovery_checkpoint_lr{ham_cfg.lr}_{timestamp}.pkl')
