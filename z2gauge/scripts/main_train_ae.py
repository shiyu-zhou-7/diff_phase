import os
import jax
import jax.numpy as jnp
import numpy as np
import pickle

from jax import config
config.update("jax_enable_x64", True)

from models.autoencoder import init_params, autoencoder, fetch_latent
from training.ae_train import train_autoencoder
from hamiltonian.z2ham import *
from utils.io import save_pickle
from configs.config import AEConfig, ActiveConfig
import matplotlib.pyplot as plt

# ── env-var overrides so the same script trains either phase's AE.
#   sbatch --export=ALL,LX=2,LY=3,DATA_PATH=../data/data_ite_confined_2x3_h-1.0_to_-0.4_n1000.pkl,PHASE_TAG=confined run_ae.sbatch
# Defaults preserve the prior hardcoded behavior (Lx=Ly=3, default data file, no phase suffix).
Lx         = int(os.environ.get('LX', 3))
Ly         = int(os.environ.get('LY', 3))
DATA_PATH  = os.environ.get('DATA_PATH', f'../data/data_ite_confined_{Lx}x{Ly}_n1000.pkl')
PHASE_TAG  = os.environ.get('PHASE_TAG', 'params')   # output file suffix; 'params' preserves the legacy name

# ----- load ferro dataset -----
N = 2 * Lx * Ly
print(f"Lattice size: {Lx}x{Ly}, Qubits: {N}")
print(f"Data path:    {DATA_PATH}")
print(f"Phase tag:    {PHASE_TAG}")

with open(DATA_PATH, 'rb') as f:
    data = np.array(pickle.load(f))
np.random.shuffle(data)

# simple train/test split 
ratio = 0.3
test_size = int(len(data) * ratio)
idx = np.random.choice(len(data), test_size, replace=False)
train = np.delete(data, idx)
X_train = np.array([d['v'].real for d in train])
X_test  = np.array([d['v'].real for d in data[idx]])

N = int(np.log2(X_train.shape[1]))
D = 2 ** N

# ----- build & train -----
ae_cfg = AEConfig()
act_cfg = ActiveConfig()
key = jax.random.PRNGKey(ae_cfg.seed)
layers = [D, 512, ae_cfg.latent_dim, 512, D]
key, sub = jax.random.split(key)
params = init_params(layers, sub)

params = train_autoencoder(params, X_train,
                           epochs=ae_cfg.epochs,
                           lr=ae_cfg.lr,
                           weight_decay=ae_cfg.weight_decay,
                           drop_p=ae_cfg.dropout_p,
                           center_coeff=act_cfg.center_coeff,
                           seed=ae_cfg.seed)

Z = fetch_latent(params, X_train, jax.random.PRNGKey(0))
ferro_centroid = jnp.mean(Z, axis=0)

X_rc = autoencoder(params, X_test, 0, key)   ## testing mode, set drop_p = 0

# ----- Calculate Wilson loops and entanglement entropy -----
print("\n" + "=" * 60)
print("WILSON LOOP AND ENTANGLEMENT ENTROPY ANALYSIS")
print("=" * 60)

# Analyze all test samples (target loop size only)
n_samples = len(X_test)
target_loop_size = min(2, Lx, Ly)
loop_label = f"{target_loop_size}x{target_loop_size}"
wilson_sample_orig = []
wilson_sample_rc = []
entropy_sample_orig = []
entropy_sample_rc = []

for idx in range(n_samples):
    print(f"\n--- Sample {idx+1} ---")
    
    # Original and reconstructed states
    state_orig = X_test[idx]
    state_rc = X_rc[idx]
    
    # Normalize states
    state_orig = state_orig / jnp.linalg.norm(state_orig)
    state_rc = state_rc / jnp.linalg.norm(state_rc)
    
    # Target Wilson loop only
    if target_loop_size > 0:
        W_target = wilson_loop(Lx, Ly, 0, 0, target_loop_size, target_loop_size)
        w_orig_target = float(expectation_value(W_target, state_orig))
        w_rc_target = float(expectation_value(W_target, state_rc))
        print(f"Wilson loop ({loop_label}): Original = {w_orig_target:.4f}, Reconstructed = {w_rc_target:.4f}, Δ = {abs(w_orig_target - w_rc_target):.4f}")
        wilson_sample_orig.append(w_orig_target)
        wilson_sample_rc.append(w_rc_target)
    else:
        print("Wilson loop undefined (target size zero).")
    
    # Entanglement entropy
    subsystem = list(range(N // 2))
    S_orig = float(entanglement_entropy(state_orig, subsystem, N))
    S_rc = float(entanglement_entropy(state_rc, subsystem, N))
    print(f"Entanglement entropy: Original = {S_orig:.4f}, Reconstructed = {S_rc:.4f}, Δ = {abs(S_orig - S_rc):.4f}")
    entropy_sample_orig.append(S_orig)
    entropy_sample_rc.append(S_rc)

# Calculate average values across all test samples
print(f"\n{'='*60}")
print("AVERAGE OVER ALL TEST SAMPLES")
print("=" * 60)

if wilson_sample_orig:
    avg_w_orig = np.mean(wilson_sample_orig)
    std_w_orig = np.std(wilson_sample_orig)
    avg_w_rc = np.mean(wilson_sample_rc)
    std_w_rc = np.std(wilson_sample_rc)
    print(f"Average Wilson loop ({loop_label}): Original = {avg_w_orig:.4f} ± {std_w_orig:.4f}, Reconstructed = {avg_w_rc:.4f} ± {std_w_rc:.4f}")
else:
    print("Average Wilson loop: undefined (target size zero).")

avg_entropy_orig = np.mean(entropy_sample_orig)
std_entropy_orig = np.std(entropy_sample_orig)
avg_entropy_rc = np.mean(entropy_sample_rc)
std_entropy_rc = np.std(entropy_sample_rc)
print(f"Average entanglement entropy: Original = {avg_entropy_orig:.4f} ± {std_entropy_orig:.4f}, Reconstructed = {avg_entropy_rc:.4f} ± {std_entropy_rc:.4f}")

# Plot Wilson loop and entanglement entropy per sample
plt.figure(figsize=(12, 5))
fs = 15
sample_indices = np.arange(len(entropy_sample_orig))

plt.subplot(1, 2, 1)
if wilson_sample_orig:
    plt.scatter(np.arange(len(wilson_sample_orig)), wilson_sample_orig, color='orange', label='Exact', s=35)
    plt.scatter(np.arange(len(wilson_sample_rc)), wilson_sample_rc, color='blue', label='Reconstructed', s=35)
else:
    plt.text(0.5, 0.5, "Not available", ha='center', va='center', fontsize=fs)
plt.xlabel('Sample index', fontsize=fs)
plt.ylabel(f'<W> ({loop_label})', fontsize=fs)
plt.title(f'Wilson loop ({loop_label}) per sample', fontsize=fs)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=fs-2)

plt.subplot(1, 2, 2)
plt.scatter(sample_indices, entropy_sample_orig, color='orange', label='Exact', s=35)
plt.scatter(sample_indices, entropy_sample_rc, color='blue', label='Reconstructed', s=35)
plt.xlabel('Sample index', fontsize=fs)
plt.ylabel('Entanglement entropy S', fontsize=fs)
plt.title('Entanglement entropy per sample', fontsize=fs)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=fs-2)

plt.tight_layout()
plt.savefig(f'../figures/z2gauge_ite_wilson_entropy_reconstruction_{Lx}x{Ly}_epoch{ae_cfg.epochs}.pdf', bbox_inches='tight')
print(f"\nPlot saved to ../figures/z2gauge_wilson_entropy_reconstruction_{Lx}x{Ly}_epoch{ae_cfg.epochs}.pdf")

out_path = f'../models/z2gauge_ite_autoencoder_{PHASE_TAG}_Lx{Lx}Ly{Ly}.pkl'
save_pickle({'params': params, 'centroid': np.array(ferro_centroid)}, out_path)
print(f'Saved AE params to {out_path}')
