# diffphase

## What This Project Does

Automated quantum phase discovery via differentiable DMRG + autoencoders. The core idea: run DMRG to get ground-state observables, encode them with an autoencoder into a latent space, then gradient-optimize Hamiltonian parameters to explore phase boundaries (active learning loop).

Three quantum models are implemented:
- **XXZ** (`xxz_dmrg/`) — primary, DMRG-based
- **Z2 gauge** (`z2gauge/`) — ITE-based solver
- **TFIM** (`tfim_reorg/`) — simpler, used for dev/testing

## Structure

```
xxz_dmrg/scripts/
  configs/config.py          # AEConfig, HamConfig, ActiveConfig, DMRGConfig
  dmrg/                      # DMRG algorithm, MPO Hamiltonians, MPS utilities
  models/autoencoder.py      # JAX feedforward AE with latent variance regularization
  training/                  # ae_train.py, dmrg_optimize.py
  workflows/active_phase_discovery.py  # Main active learning loop
  main_active_phase.py       # Entry point: full active phase discovery
  main_train_ae.py           # Entry point: train AE only

z2gauge/scripts/             # Mirrors xxz_dmrg structure; ITE solver instead of DMRG
tfim_reorg/                  # Mirrors xxz_dmrg structure; simpler Hamiltonian
manuscript/                  # Figure-making notebooks
slurm/                       # Cluster job scripts
```

## Active Learning Workflow

1. **Bootstrap:** sample random Hamiltonian params → DMRG ground states → train AE → compute latent centroid
2. **Outer loop:** gradient-based inner optimization (move params toward centroid), active sampling, AE retraining
3. **Checkpointing** after each outer iteration (pickle files in `data/`)

## Key Dependencies

| Package | Version | Role |
|---------|---------|------|
| `jax` / `jaxlib` | 0.6.2 | AD, numerical compute, GPU/Metal |
| `optax` | 0.1.8 | Adam/AdamW optimizers |
| `numpy` | 1.26.4 | Arrays |
| `scipy` | 1.15.3 | Scientific utilities |
| `matplotlib` | 3.8.0 | Plotting |

Python 3.12 via `rye`. Virtual env at `.venv/`.

## Known Issues

- **Z2 gauge NaN gradients**: `jnp.linalg.eigh()` with degenerate eigenvalues produces NaN gradients. Fixed by switching the Z2 solver from exact diagonalization to imaginary time evolution (ITE), which avoids the problematic eigendecomposition entirely.
