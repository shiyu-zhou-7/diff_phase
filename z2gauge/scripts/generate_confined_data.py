import jax
import jax.numpy as jnp
import numpy as np
import pickle
from tqdm import tqdm

from jax import config
config.update("jax_enable_x64", True)

# Import Hamiltonian functions
import sys
sys.path.append('.')
from hamiltonian.z2ham import sum_star_operators, transverse_field, hamiltonian
from hamiltonian.ite import ite_ground_state_from_params, ite_ground_state_batched


def generate_confined_phase_data(
    Lx,
    Ly,
    j_a_fixed=-1.0,
    h_range=(0.5, 3.0),
    n_samples=1000,
    ite_steps=300,
    ite_dt=1e-2,
    seed=3458923,
):
    """
    Generate ground state wavefunctions in the confined phase.
    
    Confined phase: h >> j_a (large transverse field dominates)
    
    Args:
        Lx, Ly: Lattice dimensions
        j_a_fixed: Fixed coupling for star operators (negative for attractive)
        h_range: Range of transverse field strengths (h_min, h_max)
        n_samples: Number of different h values to sample
    
    Returns:
        List of dictionaries containing ground states and parameters
    """
    n_qubits = 2 * Lx * Ly
    dim = 2**n_qubits
    
    print(f"Generating confined phase data for {Lx}×{Ly} lattice")
    print(f"Number of qubits: {n_qubits}, Hilbert space dim: {dim}")
    print(f"j_a = {j_a_fixed}, h ∈ [{h_range[0]}, {h_range[1]}]")
    print(f"Samples: {n_samples}\n")
    
    # Pre-compute operators (they don't change with parameters)
    print("Computing star operators...")
    star_ops = sum_star_operators(Lx, Ly)
    print("Computing transverse field...")
    trans_ops = transverse_field(Lx, Ly)
    
    data = []
    h_values = np.linspace(h_range[0], h_range[1], n_samples)

    # FIXED ITE init-state key so the same H(h) always converges to the same Z2
    # twin — matches optim_h.py:_TWIN_KEY = PRNGKey(0), so training and
    # optim-time latents share a single twin-selection convention.
    fixed_key = jax.random.PRNGKey(0)

    # Batched ITE path: one JAX dispatch for all n_samples (~28× faster than the
    # per-sample loop at D=4096, n_samples=1000). The batched function broadcasts
    # one v_0 (drawn from `fixed_key`) to every sample, preserving the pinned-twin
    # convention exactly.
    print(f"\nGenerating ground states (batched ITE: 1 call for {n_samples} samples)...")
    # Cast operators to float32 to match the per-sample function's internal precision.
    S32 = star_ops.astype(jnp.float32)
    T32 = trans_ops.astype(jnp.float32)
    h_batch = jnp.asarray(h_values, dtype=jnp.float32)
    V, E_final = ite_ground_state_batched(
        j_a_fixed, h_batch,
        S32, T32,
        n_steps=ite_steps, dt=ite_dt,
        key=fixed_key,
    )
    V.block_until_ready()

    for i, h in enumerate(h_values):
        data.append({
            'v': np.asarray(V[i]),     # Ground state wavefunction (numpy for pickle portability)
            'E': float(E_final[i]),    # Ground state energy
            'j_a': j_a_fixed,          # Star operator coupling
            'h': h,                    # Transverse field strength
            'Lx': Lx,
            'Ly': Ly,
            'phase': 'confined'
        })

    # --- Legacy single-sample loop (kept commented for easy revert) ---------
    # for i, h in enumerate(tqdm(h_values)):
    #     ground_state, ground_energy = ite_ground_state_from_params(
    #         j_a_fixed,
    #         float(h),                # NO sign flip — H(j_a, h) matches optim's hamiltonian(j_a, h)
    #         star_ops,
    #         trans_ops,
    #         n_steps=ite_steps,
    #         dt=ite_dt,
    #         key=fixed_key,
    #     )
    #     data.append({
    #         'v': ground_state,
    #         'E': ground_energy,
    #         'j_a': j_a_fixed,
    #         'h': h,
    #         'Lx': Lx,
    #         'Ly': Ly,
    #         'phase': 'confined',
    #     })
    # ------------------------------------------------------------------------

    return data


def generate_deconfined_phase_data(
    Lx,
    Ly,
    j_a_fixed=-1.0,
    h_range=(0.01, 0.3),
    n_samples=100,
    ite_steps=300,
    ite_dt=1e-2,
    seed=3458923,
):
    """
    Generate ground state wavefunctions in the deconfined phase via ITE.

    Deconfined phase: |j_a| >> h (star operators dominate). Uses imaginary-time
    evolution (matches generate_confined_phase_data); avoids the eigh path which
    can pick arbitrary basis-aligned eigenvectors in the degenerate ground
    subspace at small |h|.

    Args:
        Lx, Ly: Lattice dimensions
        j_a_fixed: Fixed star operator coupling (negative for attractive)
        h_range: Range of transverse field strengths (small)
        n_samples: Number of different h values to sample

    Returns:
        List of dictionaries containing ground states and parameters
    """
    n_qubits = 2 * Lx * Ly
    dim = 2**n_qubits

    print(f"Generating deconfined phase data (ITE) for {Lx}×{Ly} lattice")
    print(f"Number of qubits: {n_qubits}, Hilbert space dim: {dim}")
    print(f"j_a = {j_a_fixed}, h ∈ [{h_range[0]}, {h_range[1]}]")
    print(f"Samples: {n_samples}, ite_steps={ite_steps}, dt={ite_dt}\n")

    print("Computing star operators...")
    star_ops = sum_star_operators(Lx, Ly)
    print("Computing transverse field...")
    trans_ops = transverse_field(Lx, Ly)

    data = []
    h_values = np.linspace(h_range[0], h_range[1], n_samples)

    # Same fixed-key + no-sign-flip convention as generate_confined_phase_data.
    fixed_key = jax.random.PRNGKey(0)

    print(f"\nGenerating ground states (batched ITE: 1 call for {n_samples} samples)...")
    S32 = star_ops.astype(jnp.float32)
    T32 = trans_ops.astype(jnp.float32)
    h_batch = jnp.asarray(h_values, dtype=jnp.float32)
    V, E_final = ite_ground_state_batched(
        j_a_fixed, h_batch,
        S32, T32,
        n_steps=ite_steps, dt=ite_dt,
        key=fixed_key,
    )
    V.block_until_ready()

    for i, h in enumerate(h_values):
        data.append({
            'v': np.asarray(V[i]),
            'E': float(E_final[i]),
            'j_a': j_a_fixed,
            'h': h,
            'Lx': Lx,
            'Ly': Ly,
            'phase': 'deconfined'
        })

    # --- Legacy single-sample loop (kept commented for easy revert) ---------
    # for i, h in enumerate(tqdm(h_values)):
    #     ground_state, ground_energy = ite_ground_state_from_params(
    #         j_a_fixed,
    #         float(h),
    #         star_ops,
    #         trans_ops,
    #         n_steps=ite_steps,
    #         dt=ite_dt,
    #         key=fixed_key,
    #     )
    #     data.append({
    #         'v': ground_state,
    #         'E': ground_energy,
    #         'j_a': j_a_fixed,
    #         'h': h,
    #         'Lx': Lx,
    #         'Ly': Ly,
    #         'phase': 'deconfined',
    #     })
    # ------------------------------------------------------------------------

    return data


def generate_phase_transition_data(Lx, Ly, n_samples=200):
    """
    Generate ground states across the phase transition.
    
    Varies the ratio h/|j_a| from small (deconfined) to large (confined).
    
    Args:
        Lx, Ly: Lattice dimensions
        n_samples: Number of samples across transition
    
    Returns:
        List of dictionaries containing ground states and parameters
    """
    n_qubits = 2 * Lx * Ly
    dim = 2**n_qubits
    
    print(f"Generating phase transition data for {Lx}×{Ly} lattice")
    print(f"Number of qubits: {n_qubits}, Hilbert space dim: {dim}")
    print(f"Samples: {n_samples}\n")
    
    # Pre-compute operators
    print("Computing star operators...")
    star_ops = sum_star_operators(Lx, Ly)
    print("Computing transverse field...")
    trans_ops = transverse_field(Lx, Ly)
    
    data = []
    
    # Vary h while keeping j_a fixed
    j_a_fixed = -1.0
    h_values = np.linspace(0.01, 3.0, n_samples)
    
    print("\nGenerating ground states across phase transition...")
    for i, h in enumerate(tqdm(h_values)):
        # Construct Hamiltonian
        H = hamiltonian(j_a=j_a_fixed, h=-h, star_ops=star_ops, trans_ops=trans_ops)
        
        # Get ground state
        eigenvalues, eigenvectors = jnp.linalg.eigh(H)
        ground_state = eigenvectors[:, 0]
        ground_energy = eigenvalues[0]
        
        # Determine phase based on h/|j_a| ratio
        ratio = h / abs(j_a_fixed)
        phase = 'confined' if ratio > 0.5 else 'deconfined'
        
        # Store data
        data.append({
            'v': ground_state,
            'E': ground_energy,
            'j_a': j_a_fixed,
            'h': h,
            'ratio': ratio,
            'Lx': Lx,
            'Ly': Ly,
            'phase': phase
        })
    
    return data

"""
if __name__ == "__main__":
    # Configuration
    Lx, Ly = 2, 3  # Lattice size
    
    # Choose which dataset to generate
    print("=" * 60)
    print("Z2 GAUGE THEORY GROUND STATE GENERATION")
    print("=" * 60)
    print("\nOptions:")
    print("1. Confined phase only")
    print("2. Deconfined phase only")
    print("3. Phase transition (both phases)")
    print()
    
    # choice = input("Select option (1/2/3) [default: 3]: ").strip() or "3"
    choice = None
    
    if choice == "1":
        # Generate confined phase data
        data = generate_confined_phase_data(Lx, Ly, j_a_fixed=-1.0, h_range=(-1.5, -0.7), n_samples=1000)
        filename = f'../data/data_ite_confined_{Lx}x{Ly}_n1000.pkl'
        
    elif choice == "2":
        # Generate deconfined phase data
        data = generate_deconfined_phase_data(Lx, Ly, j_a_fixed=-1.0, h_range=(0.0001, 0.1), n_samples=1000)
        filename = f'../data/data_ite_deconfined_{Lx}x{Ly}_n1000.pkl'
        
    else:  # choice == "3"
        # Generate phase transition data
        data = generate_phase_transition_data(Lx, Ly, n_samples=200)
        filename = '../data/data_ite_phase_transition_{Lx}x{Ly}_n200.pkl'
    
    # Save data
    print(f"\nSaving data to {filename}...")
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✓ Saved {len(data)} ground states")
    print(f"✓ File size: {len(pickle.dumps(data)) / 1e6:.2f} MB")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    energies = [d['E'] for d in data]
    print(f"Energy range: [{min(energies):.4f}, {max(energies):.4f}]")
    
    # if 'phase' in data[0]:
    #     phases = [d['phase'] for d in data]
    #     n_confined = phases.count('confined')
    #     n_deconfined = phases.count('deconfined')
    #     print(f"Confined samples: {n_confined}")
    #     print(f"Deconfined samples: {n_deconfined}")
    
    print("\nDone!")

"""


if __name__ == "__main__":
    # Configuration
    Lx, Ly = 3, 3  # Lattice size
    
    # Choose which dataset to generate
    print("=" * 60)
    print("Z2 GAUGE THEORY GROUND STATE GENERATION")
    print("=" * 60)

    # Generate confined phase data
    data = generate_confined_phase_data(Lx, Ly, j_a_fixed=-1.0, h_range=(-1.5, -0.7), n_samples=1000)
    filename = f'../data/data_ite_confined_{Lx}x{Ly}_n1000.pkl'
    # Save data
    print(f"\nSaving data to {filename}...")
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✓ Saved {len(data)} ground states")
    print(f"✓ File size: {len(pickle.dumps(data)) / 1e6:.2f} MB")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    energies = [d['E'] for d in data]
    print(f"Energy range: [{min(energies):.4f}, {max(energies):.4f}]")
    
    
    # Generate deconfined phase data
    data = generate_deconfined_phase_data(Lx, Ly, j_a_fixed=-1.0, h_range=(0.0001, 0.1), n_samples=1000)
    filename = f'../data/data_ite_deconfined_{Lx}x{Ly}_n1000.pkl'
    
    # Save data
    print(f"\nSaving data to {filename}...")
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✓ Saved {len(data)} ground states")
    print(f"✓ File size: {len(pickle.dumps(data)) / 1e6:.2f} MB")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    energies = [d['E'] for d in data]
    print(f"Energy range: [{min(energies):.4f}, {max(energies):.4f}]")
    
    # if 'phase' in data[0]:
    #     phases = [d['phase'] for d in data]
    #     n_confined = phases.count('confined')
    #     n_deconfined = phases.count('deconfined')
    #     print(f"Confined samples: {n_confined}")
    #     print(f"Deconfined samples: {n_deconfined}")
    
    print("\nDone!")

