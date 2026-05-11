"""
Train a fresh autoencoder on a small radius-r interval around `h_init`, then
run gradient descent on h using that AE's local centroid as the latent target.

For samples in the SSB phase (|h| < 1) the training-time wavefunction is a
random real superposition of the two near-degenerate ground states (single
uniform angle on [0, 2π)). Outside the SSB phase the deterministic ED ground
state is used directly.

The h-optim step deliberately feeds `v[:, 0]` only (no superposition), to keep
the loss differentiable in h; this matches the original pattern.

Reads:    (none — generates data via ED on the fly)

Writes (with timestamp suffix <ts>):
    ../models/local_ae_params_<h_tag>_<ts>.pkl
    ../models/local_ae_centroid_<h_tag>_<ts>.pkl
    ../data/optim_h_ae_local_<h_tag>_<ts>.pkl
    ../figures/local_ae_train_loss_<h_tag>_<ts>.pdf
    ../figures/optim_h_ae_local_<h_tag>_<ts>.pdf

Usage:
    cd tfim_reorg/scripts && python main_train_and_optim_ae.py
    H_INIT=-1.5 python main_train_and_optim_ae.py
    H_INIT=-0.3 EPOCHS_AE=50 EPOCHS_OPTIM=50 python main_train_and_optim_ae.py
"""
import sys, os
import numpy as np
from dataclasses import replace, asdict

import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import AEConfig, HamConfig, OptimConfig
from hamiltonians.tfim import build_tfim_chain, gd_solver_ed
from models.autoencoder import init_params, encoder
from training.ae_train import train_autoencoder
from training.optim_h import optimize_h_ae
from utils.io import save_pickle, timestamp
from utils.plotting import plot_loss_curve, plot_h_loss_combined


# ── tunables ──────────────────────────────────────────────────────────────────
H_INIT_DEFAULT = -0.4
RADIUS         = 0.05    # half-width of the sampling interval around h_init
N_SAMPLES      = 500
SSB_BOUND      = 1.0     # |h| < SSB_BOUND => degenerate ground-state manifold

ham_cfg = HamConfig(N=10, J=-1.0)
ae_cfg  = AEConfig()
opt_cfg = OptimConfig(lr=0.1, epochs=1000, log_every=25)

# Env-var overrides (handy for smoke tests / cross-h_init scans without editing)
h_init = float(os.environ.get('H_INIT', H_INIT_DEFAULT))
if 'EPOCHS_AE' in os.environ:
    ae_cfg = replace(ae_cfg, epochs=int(os.environ['EPOCHS_AE']),
                     log_every=max(1, int(os.environ['EPOCHS_AE']) // 10))
if 'EPOCHS_OPTIM' in os.environ:
    opt_cfg = replace(opt_cfg, epochs=int(os.environ['EPOCHS_OPTIM']),
                      log_every=max(1, int(os.environ['EPOCHS_OPTIM']) // 10))
ham_cfg = replace(ham_cfg, h_init=h_init)

ts = timestamp()
print(f'Timestamp: {ts}')
print(f'h_init={h_init:+.4f}  radius={RADIUS}  n_samples={N_SAMPLES}  ssb_bound={SSB_BOUND}')
print(f'AE config:    {asdict(ae_cfg)}')
print(f'Optim config: {asdict(opt_cfg)}')
print(f'Ham config:   {asdict(ham_cfg)}')

# ── build chain ──────────────────────────────────────────────────────────────
ham_X, ham_ZZ = build_tfim_chain(ham_cfg.N, ham_cfg.J)

# ── sample h values ──────────────────────────────────────────────────────────
key = jax.random.PRNGKey(ae_cfg.seed)
key, sub = jax.random.split(key)
h_values = h_init + RADIUS * (2.0 * jax.random.uniform(sub, (N_SAMPLES,)) - 1.0)
h_values_np = np.asarray(h_values)

# For each sample: ED, then random superposition of the two lowest if |h|<SSB_BOUND.
# (Python loop because we need branching on |h| and per-sample RNG; eigh on a
# 2**N x 2**N matrix is one fast LAPACK call so the loop is ~25s for N=10.)
print(f'\nGenerating {N_SAMPLES} ground states...')
n_ssb = 0
x_list = []
for i, h_i_f in enumerate(h_values_np):
    e, v = gd_solver_ed(float(h_i_f), ham_X, ham_ZZ)
    if abs(h_i_f) < SSB_BOUND:
        key, sub = jax.random.split(key)
        theta = float(jax.random.uniform(sub, (), minval=0.0, maxval=2.0 * jnp.pi))
        psi = jnp.cos(theta) * v[:, 0] + jnp.sin(theta) * v[:, 1]
        n_ssb += 1
    else:
        psi = v[:, 0]
    x_list.append(np.asarray(psi))
    if (i + 1) % max(1, N_SAMPLES // 10) == 0:
        print(f'  generated {i+1}/{N_SAMPLES}')
x_train = np.stack(x_list, axis=0).astype(np.float64)
print(f'x_train shape: {x_train.shape}  ({n_ssb} samples used SSB-superposition)')

# Sanity: layer widths must match input dim
input_dim = x_train.shape[-1]
if ae_cfg.layer_widths[0] != input_dim or ae_cfg.layer_widths[-1] != input_dim:
    lw = list(ae_cfg.layer_widths)
    lw[0] = input_dim
    lw[-1] = input_dim
    ae_cfg = replace(ae_cfg, layer_widths=tuple(lw))
    print(f'[warn] rebuilt layer_widths to {ae_cfg.layer_widths}')

# ── init AE ─────────────────────────────────────────────────────────────────
key, sub = jax.random.split(key)
params = init_params(list(ae_cfg.layer_widths), sub, scale=ae_cfg.init_scale)
print(f'\nlayers: {ae_cfg.layer_widths}')

# ── train AE on local neighborhood (no test/eval split — only one circle) ──
hist = train_autoencoder(
    params,
    x_train,
    epochs=ae_cfg.epochs,
    lr=ae_cfg.lr,
    drop_p=ae_cfg.dropout_p,
    log_every=ae_cfg.log_every,
    eval_every=ae_cfg.eval_every,
    seed=ae_cfg.seed,
)
trained_params = hist['params']

# ── compute local centroid (mean of normalized latents) ────────────────────
mid = len(trained_params) // 2
key, sub = jax.random.split(key)
z_local = encoder(trained_params[0:mid], jnp.asarray(x_train), 0.0, sub)
centroid_local = np.array(np.mean(z_local, axis=0))
print(f'\ncentroid shape: {centroid_local.shape}')

# ── filename tag ───────────────────────────────────────────────────────────
h_tag = f'h{h_init:+.2f}'.replace('+', 'p').replace('-', 'm')

params_path     = f'../models/local_ae_params_{h_tag}_{ts}.pkl'
centroid_path   = f'../models/local_ae_centroid_{h_tag}_{ts}.pkl'
ae_loss_pdf     = f'../figures/local_ae_train_loss_{h_tag}_{ts}.pdf'
optim_data_path = f'../data/optim_h_ae_local_{h_tag}_{ts}.pkl'
optim_fig_path  = f'../figures/optim_h_ae_local_{h_tag}_{ts}.pdf'

save_pickle({
    'params':     trained_params,
    'h_init':     h_init,
    'h_values':   h_values_np,
    'radius':     RADIUS,
    'n_ssb':      n_ssb,
    'config':     asdict(ae_cfg),
    'loss_list':  hist['loss_list'],
}, params_path)

save_pickle({
    'centroid':    centroid_local,
    'h_init':      h_init,
    'radius':      RADIUS,
    'params_path': params_path,
}, centroid_path)

plot_loss_curve(hist['loss_list'], ae_loss_pdf, ylabel='AE loss', logy=True,
                title=f'local AE  h_init={h_init:+.2f}')

# ── run h-optim using the freshly-trained local AE ─────────────────────────
print(f'\n=== optimize_h_ae from h_init={h_init:+.4f} (local AE) ===')
h_list, loss_list = optimize_h_ae(
    h_init,
    ham_X,
    ham_ZZ,
    latent_target=jnp.asarray(centroid_local),
    params=trained_params,
    epochs=opt_cfg.epochs,
    lr=opt_cfg.lr,
    drop_p=ae_cfg.dropout_p,
    log_every=opt_cfg.log_every,
    seed=0,
)

save_pickle({
    'h_init':           h_init,
    'h_list':           h_list,
    'loss_list':        loss_list,
    'ham_cfg':          asdict(ham_cfg),
    'opt_cfg':          asdict(opt_cfg),
    'ae_cfg':           asdict(ae_cfg),
    'ae_params_path':   params_path,
    'ae_centroid_path': centroid_path,
}, optim_data_path)

plot_h_loss_combined(h_list, loss_list, optim_fig_path,
                     title=f'h_init={h_init:+.2f}  (local AE)')

print(f'\nSaved:')
for p in (params_path, centroid_path, ae_loss_pdf, optim_data_path, optim_fig_path):
    print(f'  {p}')
print(f'Final h = {h_list[-1]:+.6f}, final loss = {loss_list[-1]:+.6e}')
