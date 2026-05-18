"""
Plot the latent-space PC1-PC2 trajectory for a single z2gauge active-phase-discovery run.

Uses the final AE from the bundle. Samples background data from both phases
(deconfined: |h| < 0.3 and confined: |h| > 0.3) on a fixed h-grid, solves each
via ITE, encodes with the final AE, fits PCA(2) on the combined background
latents, and projects:
  - background deconfined and confined latents
  - the final centroid
  - a subsampled version of the run's trajectory

Usage:
    python plot_active_phase_pca_trajectory.py [bundle_path]

No argument:
    Glob the newest ../data/active_phase_*.pkl and plot it.

Output:
    ../figures/active_phase_pca_h{tag}_<ts>.pdf
"""
import sys, os, glob, re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.io import load_pickle
from hamiltonian.z2ham import sum_star_operators, transverse_field, hamiltonian
from hamiltonian.ite import ite_ground_state
from models.autoencoder import fetch_latent


_TWIN_KEY = jax.random.PRNGKey(0)


# ── locate bundle ─────────────────────────────────────────────────────────────
if len(sys.argv) >= 2:
    bundle_path = sys.argv[1]
else:
    candidates = sorted(glob.glob('../data/active_phase_*.pkl'))
    candidates = [p for p in candidates if not p.endswith('_ckpt.pkl')]
    if not candidates:
        raise SystemExit('No active_phase_*.pkl bundle found in ../data/')
    bundle_path = candidates[-1]
    print(f'No path given; using latest bundle: {bundle_path}')

bundle = load_pickle(bundle_path)
final_ae_params = bundle['final_ae_params']
final_centroid  = np.asarray(bundle['final_centroid'])
h_per_step      = bundle['history']['h_per_step']
event_per_step  = bundle['history']['event_per_step']
bootstrap_history = bundle['bootstrap_history']
ham_cfg_snap = bundle['cfg_snapshot']['ham_cfg']
Lx = ham_cfg_snap['Lx']
Ly = ham_cfg_snap['Ly']
J  = ham_cfg_snap['J']

# Output filename mirrors the bundle's tag and timestamp.
basename = os.path.basename(bundle_path)
m = re.match(r'active_phase_h([mp]\d+\.\d+)_(\d{8}_\d{6})\.pkl$', basename)
if m:
    h_tag, ts = m.group(1), m.group(2)
else:
    h_tag = f'{bundle["cfg_snapshot"]["h_init"]:+.2f}'.replace('+', 'p').replace('-', 'm')
    ts = basename.rstrip('.pkl').split('_')[-1]
out_path = f'../figures/active_phase_pca_h{h_tag}_{ts}.pdf'

# ── build operators and encoder ───────────────────────────────────────────────
print(f'Building star + transverse-field operators for Lx={Lx} Ly={Ly}...')
star_ops  = sum_star_operators(Lx, Ly)
trans_ops = transverse_field(Lx, Ly)


def encode_h(h):
    """ITE-solve at h, return the soft-normalized latent of the ground state."""
    h32 = jnp.float32(h)
    H = hamiltonian(jnp.float32(J), h32, star_ops, trans_ops).astype(jnp.float32)
    v, _ = ite_ground_state(H, n_steps=150, dt=1e-2, key=_TWIN_KEY)
    z = fetch_latent(
        final_ae_params, v,
        rng_key=jax.random.PRNGKey(0),
        drop_p=0.0,
        normalize=True, eps=1e-8,
    )
    return np.asarray(z)


# ── background sampling ───────────────────────────────────────────────────────
# deconfined: |h| < 0.3 (sample on the negative side close to 0)
# confined:   |h| > 0.3 (sample on the negative side, going down to ~h=-1)
DECONFINED_H = np.linspace(-0.29, -0.01, 100)
CONFINED_H   = np.linspace(-1.0,  -0.31, 100)

print('Encoding background (100 deconfined + 100 confined)...')
bg_dec  = np.stack([encode_h(float(h)) for h in DECONFINED_H])
bg_conf = np.stack([encode_h(float(h)) for h in CONFINED_H])
bg_all  = np.concatenate([bg_dec, bg_conf], axis=0)

# ── fit PCA on background only (numpy SVD; sklearn not available in venv) ────
bg_mean = bg_all.mean(axis=0, keepdims=True)
_, S, Vt = np.linalg.svd(bg_all - bg_mean, full_matrices=False)
pca_components = Vt[:2]                          # (2, latent_dim)
var_ratio = (S[:2] ** 2) / (S ** 2).sum()
print(f'PCA variance ratios: PC1={var_ratio[0]:.2%}, PC2={var_ratio[1]:.2%}')


def pca_transform(Y):
    return (Y - bg_mean) @ pca_components.T


bg_dec_pca  = pca_transform(bg_dec)
bg_conf_pca = pca_transform(bg_conf)
centroid_pca = pca_transform(final_centroid.reshape(1, -1))[0]

# ── trajectory subsampling: keep 'normal' + 'nan_kick' entries ────────────────
real_idx = [i for i, e in enumerate(event_per_step) if e in ('normal', 'nan_kick')]
nan_idx_within_real = [k for k, i in enumerate(real_idx) if event_per_step[i] == 'nan_kick']

MAX_POINTS = 40
stride = max(1, len(real_idx) // MAX_POINTS)
keep_within_real = set(range(0, len(real_idx), stride))
keep_within_real.update(nan_idx_within_real)
keep_within_real.add(len(real_idx) - 1)
keep_within_real = sorted(keep_within_real)

traj_step_indices = [real_idx[k] for k in keep_within_real]
traj_h_values     = [float(h_per_step[i]) for i in traj_step_indices]
traj_event_tags   = [event_per_step[i] for i in traj_step_indices]

print(f'Trajectory: {len(real_idx)} real entries -> {len(traj_h_values)} kept (stride={stride})')

print('Encoding trajectory points...')
traj_latents = np.stack([encode_h(h) for h in traj_h_values])
traj_pca = pca_transform(traj_latents)

n_traj = len(traj_pca)
alphas = np.linspace(0.2, 1.0, n_traj) if n_traj > 1 else np.array([1.0])

# ── retrain marker positions: closest trajectory point to each retrain h ─────
retrain_traj_indices = []
for h_center in bootstrap_history[1:]:
    target_h = float(h_center)
    diffs = [abs(traj_h_values[k] - target_h) for k in range(n_traj)]
    retrain_traj_indices.append(int(np.argmin(diffs)))

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)

ax.scatter(bg_dec_pca[:, 0], bg_dec_pca[:, 1],
           c='tab:blue', alpha=0.45, s=22, label='deconfined samples', zorder=1)
ax.scatter(bg_conf_pca[:, 0], bg_conf_pca[:, 1],
           c='tab:red', alpha=0.45, s=22, label='confined samples', zorder=1)

if n_traj > 1:
    ax.plot(traj_pca[:, 0], traj_pca[:, 1], '-', color='dimgray', lw=1.2, zorder=3)
gray_rgba = np.zeros((n_traj, 4))
gray_rgba[:, :3] = 0.4
gray_rgba[:, 3]  = alphas
ax.scatter(traj_pca[:, 0], traj_pca[:, 1],
           c=gray_rgba, s=18, zorder=4)

ax.scatter([traj_pca[0, 0]], [traj_pca[0, 1]],
           c='limegreen', marker='s', s=120, edgecolor='black', lw=1,
           label='start', zorder=7)
ax.scatter([traj_pca[-1, 0]], [traj_pca[-1, 1]],
           c='black', marker='*', s=180, label='end', zorder=8)

if retrain_traj_indices:
    rx = [traj_pca[i, 0] for i in retrain_traj_indices]
    ry = [traj_pca[i, 1] for i in retrain_traj_indices]
    ax.scatter(rx, ry, c='tab:orange', marker='D', s=80,
               edgecolor='black', lw=0.5, label='retrain', zorder=6)

nan_traj_indices = [k for k, tag in enumerate(traj_event_tags) if tag == 'nan_kick']
if nan_traj_indices:
    nx = [traj_pca[i, 0] for i in nan_traj_indices]
    ny = [traj_pca[i, 1] for i in nan_traj_indices]
    ax.scatter(nx, ny, c='red', marker='X', s=100, label='NaN kick', zorder=6)

ax.scatter([centroid_pca[0]], [centroid_pca[1]],
           c='black', marker='+', s=200, lw=2, label='centroid', zorder=9)

ax.set_xlabel(f'PC1 ({var_ratio[0]:.0%})', fontsize=14)
ax.set_ylabel(f'PC2 ({var_ratio[1]:.0%})', fontsize=14)
ax.tick_params(axis='both', labelsize=12)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10, frameon=True)

fig.savefig(out_path, bbox_inches='tight')
print(f'Saved: {out_path}')
