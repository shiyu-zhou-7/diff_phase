"""
Central place for hyperparameters and small toggles.
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class AEConfig:
    seed: int = 45283497
    latent_dim: int = 15
    dropout_p: float = 0.05
    lr: float = 1e-5
    weight_decay: float = 0 # AdamW style
    epochs: int = 50000
    mini_epochs: int = 5000      # for quick refresh
    use_gpu: bool = True         # Enable GPU usage
    batch_size: int = 32         # Batch size for GPU processing

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
    num_samples_when_stalled: int = 100
    center_coeff: float = 1e-3  # latent variance penalty in AE loss
    nan_jump_scale: float = 2.0   # jump length multiplier along last_dir
    nan_jump_noise: float = 0.05  # fallback random jump (if no last_dir yet)

@dataclass
class DMRGConfig:
    max_bond: int = 10          # χ (bond dimension)
    sweeps: int = 1            # number of finite sweeps
    tol: float = 1e-8           # energy convergence tol
    normalize: bool = True
    warm_start: bool = True     # reuse MPS between nearby (Δ,h)
    # optimization over (Δ,h)
    lr: float = 3e-3            # step size for param updates
    # grad_method: str = "fd"     # "fd" = finite difference, "none" = gradient-free step
    fd_eps: float = 1e-3        # finite-diff epsilon
    clip_step: float = 0.2      # cap on |Δ|,|h| update per step
    lanczos_bool: bool = False       # use Lanczos for eigensolver (else dense)
    use_gpu: bool = True        # Enable GPU usage for DMRG
    memory_efficient: bool = True    # Use memory-efficient operations on GPU
