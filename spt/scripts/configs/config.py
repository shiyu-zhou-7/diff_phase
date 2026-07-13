"""
Hyperparameters for the SPT cluster-chain active-phase pipeline.

Same pattern as xxz_dmrg/tfim_reorg: each main_*.py builds these at the top with
`replace(...)`, no hyperparameters baked deeper in.
"""
from dataclasses import dataclass, field


@dataclass
class AEConfig:
    seed: int = 83948
    latent_dim: int = 8           # latent sphere dimension
    hidden: int = 32              # encoder/decoder hidden width
    dropout_p: float = 0.0        # feature vectors are clean -> no dropout by default
    lr: float = 1e-3
    weight_decay: float = 0.0     # AdamW style
    epochs: int = 4000
    mini_epochs: int = 4000       # quicker refresh on retrain
    center_coeff: float = 1e-3    # latent-variance penalty in ae_loss
    init_scale: float = 1e-2


@dataclass
class ClusterConfig:
    L: int = 10                   # chain length (sector dim = 2**(L-1))
    d: int = 3                    # number of open couplings t_0..t_{d-1}
    bc: str = 'pbc'               # 'pbc' (default) or 'obc'
    sector: int = +1              # fixed P = +1 sector
    kappa: float = 0.0            # interacting control term (off by default)
    eta: float = 0.0              # resolvent shift in the state-derivative adjoint
    seed: int = 83948
    # inner optimization over the d-vector t
    lr: float = 5e-2
    max_steps_block: int = 500    # max inner Adam steps per outer block
    param_tol_change: float = 1e-5  # |dt|_inf below this for stall
    stall_window: int = 20
    nan_lr_mult: float = 10.0     # temporary lr boost after a NaN jump
    nan_lr_steps: int = 5
    # anti-chatter stabilizers (do NOT read the analytic label -> no cheating):
    #   cosine-decay the inner lr over the block so the optimizer settles into a
    #   cell instead of overshooting across boundaries; clip the step so the
    #   gradient DIVERGENCE at a boundary cannot fling t far past it.
    lr_decay: bool = True
    lr_final_frac: float = 0.02   # cosine decays lr -> lr_final_frac * lr over the block
    grad_clip: float = 0.5        # max global-norm of the (pre-lr) update; <=0 disables


@dataclass
class ActiveConfig:
    t_radius_init: float = 0.1        # sampling radius (Gaussian sigma) around init t
    t_radius_retrain: float = 0.1     # radius per historical center on retrain
    num_samples_init: int = 400       # ED samples for the FIRST bootstrap
    num_samples_bootstrap: int = 200  # ED samples per center on each retrain
    max_outer_iters: int = 6          # max bootstrap retrains per run
    nan_jump_scale: float = 2.0       # momentum overshoot = scale * last_dir
    nan_jump_noise: float = 0.1       # random kick magnitude (first NaN, no last_dir)
    normalize_sphere: bool = True     # project t onto the unit sphere S^{d-1}
    converge_tol: float = 1e-3        # end the run once a block moves t less than
                                      # this on the sphere (settled fixed point;
                                      # no analytic label read)
