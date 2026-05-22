"""
One-off overlay: project a single active-phase trajectory onto the latent-space
PCA defined by the *combined* AE + the narrow-range (deconfined h ∈ [0.1, 0.15],
confined h ∈ [-1.2, -1.0]) latents from `data/latent_eval.ipynb`.

Reproduces the latent space behind
    figures/latent_eval_pca_narrow_dec0.1_to_0.15_conf-1.2_to_-1.0_Lx2Ly3.pdf
and overlays the trajectory of the bundle path given as the only positional arg
(defaults to active_phase_hm0.09_20260518_115424.pkl, job 497507).

The trajectory's ψ(h_t) values are recomputed via ITE *on this machine* (to be
run on the cluster so BLAS matches the AE's training data), then encoded with
the combined AE and projected onto the same 2-component PCA basis.

Usage (on cluster):
    cd z2gauge/scripts && python overlay_trajectory_on_narrow_latent.py [bundle.pkl]
"""
import os, sys, pickle, glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hamiltonian.z2ham import sum_star_operators, transverse_field
from hamiltonian.ite import ite_ground_state_batched
from models.autoencoder import fetch_latent


# ── locate bundle ─────────────────────────────────────────────────────────────
if len(sys.argv) >= 2:
    bundle_path = sys.argv[1]
else:
    bundle_path = '../data/active_phase_hm0.09_20260518_115424.pkl'

print(f'Bundle: {bundle_path}')
bundle = pickle.load(open(bundle_path, 'rb'))
h_per_step    = bundle['history']['h_per_step']
event_per_step = bundle['history']['event_per_step']
ham_cfg_snap   = bundle['cfg_snapshot']['ham_cfg']
Lx = ham_cfg_snap['Lx']
Ly = ham_cfg_snap['Ly']
J  = ham_cfg_snap['J']
h_init = bundle['cfg_snapshot']['h_init']
print(f'  Lx={Lx} Ly={Ly} J={J} h_init={h_init:+.4f}')
print(f'  total steps = {len(h_per_step)} ({event_per_step.count("normal")} normal, {event_per_step.count("retrain")} retrain, {event_per_step.count("nan_kick")} nan_kick)')

# ── combined AE + narrow data ─────────────────────────────────────────────────
ae_path    = f'../models/z2gauge_ite_autoencoder_combined_Lx{Lx}Ly{Ly}.pkl'
narrow_dec_pkl  = f'../data/data_ite_deconfined_{Lx}x{Ly}_h0.1_to_0.15_n500.pkl'
narrow_conf_pkl = f'../data/data_ite_confined_{Lx}x{Ly}_h-1.2_to_-1.0_n500.pkl'

print(f'AE:           {ae_path}')
print(f'Narrow dec:   {narrow_dec_pkl}')
print(f'Narrow conf:  {narrow_conf_pkl}')

ae = pickle.load(open(ae_path, 'rb'))
ae_params = ae['params']

data_conf = pickle.load(open(narrow_conf_pkl, 'rb'))
data_dec  = pickle.load(open(narrow_dec_pkl, 'rb'))

# Same stacking order as the notebook: confined first, then deconfined.
X_narrow = np.stack([d['v'].real for d in (list(data_conf) + list(data_dec))]).astype(np.float32)
phase_narrow = np.array(['confined']*len(data_conf) + ['deconfined']*len(data_dec))
h_narrow     = np.array([float(d['h']) for d in (list(data_conf) + list(data_dec))])
print(f'Narrow X:     {X_narrow.shape}')
print(f'  confined  h ∈ [{h_narrow[phase_narrow=="confined"].min():+.4f}, {h_narrow[phase_narrow=="confined"].max():+.4f}]')
print(f'  deconfined h ∈ [{h_narrow[phase_narrow=="deconfined"].min():+.4f}, {h_narrow[phase_narrow=="deconfined"].max():+.4f}]')

# Encode narrow data through combined AE
Z_narrow = np.asarray(fetch_latent(ae_params, X_narrow, jax.random.PRNGKey(0)))
print(f'Z_narrow:     {Z_narrow.shape}  (mean ‖z‖ = {np.linalg.norm(Z_narrow, axis=1).mean():.3f})')

# 2-component PCA via numpy SVD (sklearn unavailable in venv)
mu = Z_narrow.mean(axis=0, keepdims=True)
Zc = Z_narrow - mu
U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
P = Vt[:2]
var_ratio = (S[:2] ** 2) / (S ** 2).sum()
print(f'PCA var ratios: PC1={var_ratio[0]:.2%}, PC2={var_ratio[1]:.2%}')


def to_pca(Z):
    return (Z - mu) @ P.T


Z2_narrow = to_pca(Z_narrow)

# ── recompute trajectory ψ(h_t) and project ──────────────────────────────────
# Keep 'normal' + 'nan_kick' entries only (drop synthetic 'retrain' markers).
real_idx = [i for i, e in enumerate(event_per_step) if e in ('normal', 'nan_kick')]
n_real = len(real_idx)
MAX_POINTS = 80
stride = max(1, n_real // MAX_POINTS)
keep_within_real = set(range(0, n_real, stride))
keep_within_real.update(k for k, i in enumerate(real_idx) if event_per_step[i] == 'nan_kick')
keep_within_real.add(n_real - 1)
keep_within_real = sorted(keep_within_real)
traj_step_idx = [real_idx[k] for k in keep_within_real]
traj_h = np.array([float(h_per_step[i]) for i in traj_step_idx], dtype=np.float32)
traj_event_tags = [event_per_step[i] for i in traj_step_idx]
print(f'Trajectory: {n_real} real entries -> {len(traj_h)} kept (stride={stride})')
print(f'  h range visited: [{traj_h.min():+.4f}, {traj_h.max():+.4f}]')

print('Building z2gauge operators...')
S_ops = sum_star_operators(Lx, Ly)
T_ops = transverse_field(Lx, Ly)

print(f'Solving ITE for {len(traj_h)} trajectory points (batched)...')
V_traj, _ = ite_ground_state_batched(
    J, jnp.asarray(traj_h),
    S_ops, T_ops,
    n_steps=150, dt=1e-2, key=jax.random.PRNGKey(0),
)
V_traj = np.asarray(V_traj)
print(f'  V_traj shape: {V_traj.shape}')

Z_traj = np.asarray(fetch_latent(ae_params, V_traj, jax.random.PRNGKey(0)))
Z2_traj = to_pca(Z_traj)

# ── plot ──────────────────────────────────────────────────────────────────────
out_path = bundle_path.replace('../data/', '../figures/').replace('active_phase_', 'active_phase_overlay_narrow_').replace('.pkl', '.pdf')
print(f'Output: {out_path}')

fig, ax = plt.subplots(figsize=(7.5, 6), dpi=150)

# Narrow scatter (notebook ordering: confined red, deconfined blue)
for p, color in [('confined', 'tab:red'), ('deconfined', 'tab:blue')]:
    mask = phase_narrow == p
    ax.scatter(Z2_narrow[mask, 0], Z2_narrow[mask, 1],
               c=color, s=22, alpha=0.5, edgecolors='none',
               label=f'{p} samples', zorder=2)

# Trajectory: line + alpha-graded dots
n_traj = len(Z2_traj)
alphas = np.linspace(0.25, 1.0, n_traj) if n_traj > 1 else np.array([1.0])
ax.plot(Z2_traj[:, 0], Z2_traj[:, 1], '-', color='dimgray', lw=1.2, zorder=3)
gray_rgba = np.zeros((n_traj, 4))
gray_rgba[:, :3] = 0.4
gray_rgba[:, 3]  = alphas
ax.scatter(Z2_traj[:, 0], Z2_traj[:, 1], c=gray_rgba, s=22, zorder=4)

# Start and end markers
ax.scatter([Z2_traj[0, 0]], [Z2_traj[0, 1]],
           c='limegreen', marker='s', s=130, edgecolor='black', lw=1, zorder=7,
           label=f'start (h={traj_h[0]:+.3f})')
ax.scatter([Z2_traj[-1, 0]], [Z2_traj[-1, 1]],
           c='black', marker='*', s=220, zorder=8,
           label=f'end (h={traj_h[-1]:+.3f})')

# NaN-kick markers (if any)
nan_idx_in_traj = [k for k, tag in enumerate(traj_event_tags) if tag == 'nan_kick']
if nan_idx_in_traj:
    nx = [Z2_traj[k, 0] for k in nan_idx_in_traj]
    ny = [Z2_traj[k, 1] for k in nan_idx_in_traj]
    ax.scatter(nx, ny, c='red', marker='X', s=100, zorder=6, label='NaN kick')

ax.set_xlabel(f'PC1 ({var_ratio[0]:.0%})', fontsize=14)
ax.set_ylabel(f'PC2 ({var_ratio[1]:.0%})', fontsize=14)
ax.set_title(f'h_init={h_init:+.2f} trajectory in combined-AE narrow PCA',
             fontsize=13)
ax.tick_params(axis='both', labelsize=11)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10, framealpha=0.92)

fig.savefig(out_path, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out_path}')
