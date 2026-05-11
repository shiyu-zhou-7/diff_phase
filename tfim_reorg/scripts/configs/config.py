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
    lr: float = 0.1
    epochs: int = 1000
    eval_every: int = 20      # snapshot latent space every N epochs
    log_every: int = 100      # print loss every N epochs
    init_scale: float = 0.01  # weight init scale


@dataclass
class HamConfig:
    N: int = 10               # chain length (Hilbert dim = 2**N)
    J: float = -1.0           # ferro coupling
    h_init: float = -0.4      # initial transverse field for h-optimization runs


@dataclass
class OptimConfig:
    lr: float = 0.1
    epochs: int = 1000
    log_every: int = 25
    target_phase: str = 'para'   # 'para' or 'ssb' — controls loss sign in ED-magnetization variant
