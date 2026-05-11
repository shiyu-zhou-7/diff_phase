"""
Drive the transverse-field strength h via gradient descent on a magnetization
loss computed on the ED ground state. No autoencoder involved.

Reads:  (none — ED solver builds H from scratch)
Writes:
    ../data/optim_h_ed_<ts>.pkl
    ../figures/optim_h_ed_<ts>.pdf

Usage:
    cd tfim_reorg/scripts && python main_optim_h_ed.py
"""
import sys, os
from dataclasses import replace, asdict

from jax import config
config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import HamConfig, OptimConfig
from hamiltonians.tfim import build_tfim_chain
from training.optim_h import optimize_h_ed
from utils.io import save_pickle, timestamp
from utils.plotting import plot_h_trajectory


# ── tunables ──────────────────────────────────────────────────────────────────
ham_cfg = HamConfig(N=10, J=-1.0, h_init=-0.3)
opt_cfg = OptimConfig(lr=0.3, epochs=500, log_every=25, target_phase='para')

ts = timestamp()
print(f'Timestamp: {ts}')
print(f'Ham config:   {asdict(ham_cfg)}')
print(f'Optim config: {asdict(opt_cfg)}')

# ── build the TFIM chain ──────────────────────────────────────────────────────
ham_X, ham_ZZ = build_tfim_chain(ham_cfg.N, ham_cfg.J)

# ── optimize h ────────────────────────────────────────────────────────────────
h_list, loss_list = optimize_h_ed(
    ham_cfg.h_init,
    ham_X,
    ham_ZZ,
    epochs=opt_cfg.epochs,
    lr=opt_cfg.lr,
    target_phase=opt_cfg.target_phase,
    log_every=opt_cfg.log_every,
)

# ── save ──────────────────────────────────────────────────────────────────────
data_path = f'../data/optim_h_ed_{ts}.pkl'
fig_path  = f'../figures/optim_h_ed_{ts}.pdf'

save_pickle({
    'h_list':    h_list,
    'loss_list': loss_list,
    'ham_cfg':   asdict(ham_cfg),
    'opt_cfg':   asdict(opt_cfg),
}, data_path)

plot_h_trajectory(h_list, fig_path)

print(f'\nSaved:')
print(f'  {data_path}')
print(f'  {fig_path}')
print(f'Final h = {h_list[-1]:+.6f}, final loss = {loss_list[-1]:.6e}')
