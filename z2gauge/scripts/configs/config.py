"""
Central place for hyperparameters and small toggles.
"""
from dataclasses import dataclass

@dataclass
class AEConfig:
    seed: int = 45283497
    latent_dim: int = 32
    dropout_p: float = 0.05
    lr: float = 1e-4
    weight_decay: float = 1e-4  # AdamW style
    epochs: int = 10000
    mini_epochs: int = 10000      # for quick refresh

@dataclass
class HamConfig:
    seed: int = 83948
    lr: float = 1e-4
    max_steps_block: int = 1000
    stall_window: int = 25
    stall_tol_loss: float = 1e-4
    stall_tol_grad: float = 1e-6
    param_tol_change: float = 1e-5
    nan_lr_mult: float = 1e2
    nan_lr_steps: int = 10       # how many steps to keep boosted LR

@dataclass
class ActiveConfig:
    sample_radius_delta: float = 0.2
    sample_radius_h: float = 0.2
    num_samples_when_stalled: int = 1000
    center_coeff: float = 1e-3  # latent variance penalty in AE loss
    nan_jump_scale: float = 2.0   # jump length multiplier along last_dir
    nan_jump_noise: float = 0.05  # fallback random jump (if no last_dir yet)