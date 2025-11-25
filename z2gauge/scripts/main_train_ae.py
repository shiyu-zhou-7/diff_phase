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

# ----- load ferro dataset -----
with open('../data/data_para.pkl', 'rb') as f:
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
layers = [D, 256, ae_cfg.latent_dim, 256, D]
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

# Infer lattice size from the Hilbert space dimension
# N is number of qubits, for 2D lattice: N = 2 * Lx * Ly
Lx = Ly = int(np.sqrt(N / 2))
print(f"Lattice size: {Lx}x{Ly}, Qubits: {N}")

# Analyze a few test samples
n_samples = min(5, len(X_test))
for idx in range(n_samples):
    print(f"\n--- Sample {idx+1} ---")
    
    # Original and reconstructed states
    state_orig = X_test[idx]
    state_rc = X_rc[idx]
    
    # Normalize states
    state_orig = state_orig / jnp.linalg.norm(state_orig)
    state_rc = state_rc / jnp.linalg.norm(state_rc)
    
    # Calculate Wilson loops for various sizes
    print("Wilson loops:")
    for size in range(1, min(Lx, Ly) + 1):
        W = wilson_loop(Lx, Ly, 0, 0, size, size)
        w_orig = float(expectation_value(W, state_orig))
        w_rc = float(expectation_value(W, state_rc))
        print(f"  Size {size}×{size}: Original = {w_orig:.4f}, Reconstructed = {w_rc:.4f}, Δ = {abs(w_orig - w_rc):.4f}")
    
    # Calculate entanglement entropy
    subsystem = list(range(N // 2))
    S_orig = float(entanglement_entropy(state_orig, subsystem, N))
    S_rc = float(entanglement_entropy(state_rc, subsystem, N))
    print(f"Entanglement entropy: Original = {S_orig:.4f}, Reconstructed = {S_rc:.4f}, Δ = {abs(S_orig - S_rc):.4f}")

# Calculate average values across all test samples
print(f"\n{'='*60}")
print("AVERAGE OVER ALL TEST SAMPLES")
print("=" * 60)

wilson_orig = {size: [] for size in range(1, min(Lx, Ly) + 1)}
wilson_rc = {size: [] for size in range(1, min(Lx, Ly) + 1)}
entropy_orig = []
entropy_rc = []

for idx in range(len(X_test)):
    state_orig = X_test[idx] / jnp.linalg.norm(X_test[idx])
    state_rc = X_rc[idx] / jnp.linalg.norm(X_rc[idx])
    
    # Wilson loops
    for size in range(1, min(Lx, Ly) + 1):
        W = wilson_loop(Lx, Ly, 0, 0, size, size)
        w_orig = float(expectation_value(W, state_orig))
        w_rc = float(expectation_value(W, state_rc))
        wilson_orig[size].append(w_orig)
        wilson_rc[size].append(w_rc)
    
    # Entanglement entropy
    subsystem = list(range(N // 2))
    S_orig = float(entanglement_entropy(state_orig, subsystem, N))
    S_rc = float(entanglement_entropy(state_rc, subsystem, N))
    entropy_orig.append(S_orig)
    entropy_rc.append(S_rc)

print("Average Wilson loop values:")
for size in range(1, min(Lx, Ly) + 1):
    avg_orig = np.mean(wilson_orig[size])
    std_orig = np.std(wilson_orig[size])
    avg_rc = np.mean(wilson_rc[size])
    std_rc = np.std(wilson_rc[size])
    print(f"  Size {size}×{size}: Original = {avg_orig:.4f} ± {std_orig:.4f}, Reconstructed = {avg_rc:.4f} ± {std_rc:.4f}")

avg_entropy_orig = np.mean(entropy_orig)
std_entropy_orig = np.std(entropy_orig)
avg_entropy_rc = np.mean(entropy_rc)
std_entropy_rc = np.std(entropy_rc)
print(f"Average entanglement entropy: Original = {avg_entropy_orig:.4f} ± {std_entropy_orig:.4f}, Reconstructed = {avg_entropy_rc:.4f} ± {std_entropy_rc:.4f}")

# Plot Wilson loop and entanglement entropy: original vs reconstructed
plt.figure(figsize=(12, 5))
fs = 15

plt.subplot(1, 2, 1)
sizes = list(range(1, min(Lx, Ly) + 1))
avg_orig = [np.mean(wilson_orig[size]) for size in sizes]
std_orig = [np.std(wilson_orig[size]) for size in sizes]
avg_rc = [np.mean(wilson_rc[size]) for size in sizes]
std_rc = [np.std(wilson_rc[size]) for size in sizes]

plt.errorbar(sizes, avg_orig, yerr=std_orig, fmt='o-', capsize=5, label='Original', color='blue', linewidth=2)
plt.errorbar(sizes, avg_rc, yerr=std_rc, fmt='s--', capsize=5, label='Reconstructed', color='orange', linewidth=2)
plt.xlabel('Wilson loop size', fontsize=fs)
plt.ylabel('<W> expectation value', fontsize=fs)
plt.title('Wilson Loop Comparison', fontsize=fs)
plt.legend(fontsize=fs-2)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
bins = np.linspace(min(min(entropy_orig), min(entropy_rc)), max(max(entropy_orig), max(entropy_rc)), 20)
plt.hist(entropy_orig, bins=bins, alpha=0.6, label='Original', color='blue', edgecolor='black')
plt.hist(entropy_rc, bins=bins, alpha=0.6, label='Reconstructed', color='orange', edgecolor='black')
plt.xlabel('Entanglement entropy S', fontsize=fs)
plt.ylabel('Frequency', fontsize=fs)
plt.title('Entanglement Entropy Distribution', fontsize=fs)
plt.legend(fontsize=fs-2)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'../figures/z2gauge_wilson_entropy_reconstruction_epoch{ae_cfg.epochs}.pdf', bbox_inches='tight')
print(f"\nPlot saved to ../figures/z2gauge_wilson_entropy_reconstruction_epoch{ae_cfg.epochs}.pdf")

save_pickle({'params': params, 'centroid': np.array(ferro_centroid)}, '../models/xxzh_autoencoder_params.pkl')
print('Saved AE params to ../models/xxzh_autoencoder_params.pkl')