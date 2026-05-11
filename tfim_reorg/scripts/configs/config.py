"""
Central place for TFIM hyperparameters and small toggles.

Same pattern as xxz_dmrg/scripts/configs/config.py: each main_*.py builds
these at the top with `replace(...)`, no hyperparameters baked deeper in.
"""
from dataclasses import dataclass


@dataclass
class AEConfig:
    seed: int = 83948
    layer_widths: tuple = (1024, 500, 20, 500, 1024)
    dropout_p: float = 0.1
    lr: float = 1e-4
    epochs: int = 10000
    eval_every: int = 20      # snapshot latent space every N epochs
    log_every: int = 100      # print loss every N epochs
    init_scale: float = 0.01  # weight init scale


@dataclass
class HamConfig:
    N: int = 10                       # chain length (Hilbert dim = 2**N)
    J: float = -1.0                   # ferro coupling
    h_init: float = -0.4              # initial transverse field for h-optimization runs
    seed: int = 83948                 # RNG for kick noise + bootstrap sampling
    lr: float = 0.01                  # h-optim Adam lr (active-phase workflow)
    max_steps_block: int = 200        # max inner Adam steps per outer block
    stall_window: int = 25            # consecutive low-progress steps to declare stall
    stall_tol_loss: float = 1e-4
    stall_tol_grad: float = 1e-4
    param_tol_change: float = 1e-4
    nan_lr_mult: float = 10.0         # temporary lr multiplier on NaN recovery
    nan_lr_steps: int = 5             # how many steps to keep boosted lr


@dataclass
class ActiveConfig:
    bootstrap_radius_init: float = 0.3        # initial circle radius around h_init
    bootstrap_radius_retrain: float = 0.1     # tight retrain radius around current h
    num_samples_init: int = 1000              # ED samples for the FIRST (initial) bootstrap circle
    num_samples_bootstrap: int = 500          # ED samples per circle on every retrain
    max_outer_iters: int = 5                  # max bootstrap retrains per run
    center_coeff: float = 1e-3                # stubbed — no-op until ae_loss adds the var-penalty term
    nan_jump_scale: float = 2.0               # multiplier on last_dir for momentum kick
    nan_jump_noise: float = 0.05              # random kick magnitude (first NaN, no last_dir)


@dataclass
class OptimConfig:
    lr: float = 0.1
    epochs: int = 1000
    log_every: int = 25
    target_phase: str = 'para'   # 'para' or 'ssb' — controls loss sign in ED-magnetization variant
