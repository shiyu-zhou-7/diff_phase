"""
Central place for z2gauge hyperparameters and small toggles.

Same pattern as tfim_reorg/scripts/configs/config.py: each main_*.py builds
these at the top with `replace(...)`, no hyperparameters baked deeper in.

z2gauge specifics vs. TFIM defaults:
  - Phase boundary lives at |h| = 0.3 (not |h| = 1)
  - Ground state solved by ITE (n_steps=150, dt=1e-2) rather than ED
  - Hilbert dim = 2^(2*Lx*Ly); layer_widths is built per-run from (Lx, Ly)
  - Smaller sample budgets per circle (ITE is slower than eigh)
"""
from dataclasses import dataclass, field


@dataclass
class AEConfig:
    seed: int = 83948
    # layer_widths is built per-run from (Lx, Ly) — main_active_phase.py
    # injects (D_in, 500, 10, 500, D_in) where D_in = 2^(2*Lx*Ly).
    layer_widths: tuple = (4096, 500, 30, 500, 4096)
    dropout_p: float = 0.1
    lr: float = 1e-4
    epochs: int = 50000
    eval_every: int = 20      # snapshot latent space every N epochs (used only when x_para_eval supplied)
    log_every: int = 100      # print loss every N epochs
    init_scale: float = 0.01  # weight init scale
    center_coeff: float = 1e-3  # latent-variance penalty weight in ae_loss


@dataclass
class HamConfig:
    Lx: int = 2                       # lattice x extent
    Ly: int = 3                       # lattice y extent
    J: float = -1.0                   # star-operator coupling (j_a)
    h_init: float = -0.4              # initial transverse field
    seed: int = 83948                 # RNG for kick noise + bootstrap sampling
    lr: float = 5e-4                  # h-optim Adam lr
    max_steps_block: int = 1000       # max inner Adam steps per outer block
    stall_window: int = 25            # consecutive low-progress steps to declare stall
    stall_tol_loss: float = 1e-4
    stall_tol_grad: float = 1e-4      # threshold on |EMA(grad)| magnitude for stall detection
    param_tol_change: float = 1e-4    # vestigial (EMA-grad stall is used instead)
    nan_lr_mult: float = 10.0         # temporary lr multiplier on NaN recovery
    nan_lr_steps: int = 5             # how many steps to keep boosted lr


@dataclass
class ActiveConfig:
    bootstrap_radius_init: float = 0.05       # initial circle radius around h_init (z2gauge boundary at |h|=0.3 → keep circles well inside one phase)
    bootstrap_radius_retrain: float = 0.05    # retrain radius around each historical center
    num_samples_init: int = 1000              # ITE samples for the FIRST (initial) bootstrap circle
    num_samples_bootstrap: int = 500          # ITE samples per circle on every retrain
    max_outer_iters: int = 5                  # max bootstrap retrains per run
    nan_jump_scale: float = 2.0               # multiplier on last_dir for momentum kick
    nan_jump_noise: float = 0.05              # random kick magnitude (first NaN, no last_dir)


@dataclass
class ITEConfig:
    dt: float = 1e-2
    n_steps: int = 150
    eps: float = 1e-12
    # Fixed init-state key so the same H(h) always converges to the same Z2 twin.
    # Both the data-gen path (generate_confined_data.py) and the optim path
    # (training/optim_h.py) pin this to PRNGKey(0) so training latents and
    # optim-time latents share a single twin-selection convention.
    seed: int = 0
