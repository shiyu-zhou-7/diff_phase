"""
Generate per-phase ITE training datasets for the z2gauge AE workflow.

Produces two pickles for Lx=2, Ly=3 (negative-h convention, matching
main_active_phase.py's `h` parameter):

  - data_ite_confined_2x3_h-1.0_to_-0.4_n1000.pkl     (h ∈ [−1.0, −0.4],   1000 samples)
  - data_ite_deconfined_2x3_h-0.2_to_-0.001_n1000.pkl (h ∈ [−0.2, −0.001], 1000 samples)

Both are intended for separately training a confined AE and a deconfined AE,
which `main_active_phase.py` will then auto-select between based on |h_init|.

Usage:
    cd z2gauge/scripts && python generate_train_data_negh.py
"""
import os
import sys
import pickle
import time

import jax
from jax import config
config.update("jax_enable_x64", True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from generate_confined_data import (
    generate_confined_phase_data,
    generate_deconfined_phase_data,
)


def save(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    energies = [float(d['E']) for d in data]
    print(f"  saved {len(data)} samples → {filename}")
    print(f"  size: {os.path.getsize(filename) / 1e6:.2f} MB")
    print(f"  E range: [{min(energies):.4f}, {max(energies):.4f}]")


if __name__ == "__main__":
    Lx, Ly = 2, 3
    n_samples = 1000

    # Confined: |h| > 0.3, negative side
    t0 = time.time()
    print("\n=== generating confined data h ∈ [-1.0, -0.4] ===")
    data_c = generate_confined_phase_data(
        Lx, Ly,
        j_a_fixed=-1.0,
        h_range=(-1.0, -0.4),
        n_samples=n_samples,
        ite_steps=150,        # 300 default caused OOM at sample ~770 on 64 GB; 150 is plenty (matches earlier successful runs)
    )
    save(data_c, f'../data/data_ite_confined_{Lx}x{Ly}_h-1.0_to_-0.4_n{n_samples}.pkl')
    print(f"  elapsed: {time.time() - t0:.1f} s")

    # Release XLA JIT cache before starting the second phase to avoid LLVM
    # "Cannot allocate memory" when the cumulative cached graphs exceed
    # LLVM's per-process limit (manifested as SIGSEGV at sample ~770 of
    # deconfined when both phases ran in one process — jobs 491499, 491527).
    print("\nClearing JAX caches between phases...")
    jax.clear_caches()

    # Deconfined: |h| < 0.3, negative side
    t0 = time.time()
    print("\n=== generating deconfined data h ∈ [-0.2, -0.001] ===")
    data_d = generate_deconfined_phase_data(
        Lx, Ly,
        j_a_fixed=-1.0,
        h_range=(-0.2, -0.001),
        n_samples=n_samples,
        ite_steps=150,        # same — keep parity with confined path; 150 sufficient for ITE projection here
    )
    save(data_d, f'../data/data_ite_deconfined_{Lx}x{Ly}_h-0.2_to_-0.001_n{n_samples}.pkl')
    print(f"  elapsed: {time.time() - t0:.1f} s")

    print("\nDone.")
