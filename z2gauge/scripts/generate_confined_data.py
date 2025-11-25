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


def generate_confined_phase_data(Lx, Ly, j_a_fixed=-1.0, h_range=(0.5, 3.0), n_samples=100):
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
    
    print("\nGenerating ground states...")
    for i, h in enumerate(tqdm(h_values)):
        # Construct Hamiltonian
        H = hamiltonian(j_a=j_a_fixed, h=-h, star_ops=star_ops, trans_ops=trans_ops)
        
        # Get ground state (lowest eigenvalue and eigenvector)
        eigenvalues, eigenvectors = jnp.linalg.eigh(H)
        ground_state = eigenvectors[:, 0]
        ground_energy = eigenvalues[0]
        
        # Store data
        data.append({
            'v': ground_state,  # Ground state wavefunction
            'E': ground_energy,  # Ground state energy
            'j_a': j_a_fixed,    # Star operator coupling
            'h': h,              # Transverse field strength
            'Lx': Lx,
            'Ly': Ly,
            'phase': 'confined'
        })
    
    return data


def generate_deconfined_phase_data(Lx, Ly, j_a_range=(-3.0, -0.5), h_fixed=0.1, n_samples=100):
    """
    Generate ground state wavefunctions in the deconfined phase.
    
    Deconfined phase: |j_a| >> h (star operators dominate)
    
    Args:
        Lx, Ly: Lattice dimensions
        j_a_range: Range of star operator couplings (j_a_min, j_a_max)
        h_fixed: Fixed transverse field strength (small)
        n_samples: Number of different j_a values to sample
    
    Returns:
        List of dictionaries containing ground states and parameters
    """
    n_qubits = 2 * Lx * Ly
    dim = 2**n_qubits
    
    print(f"Generating deconfined phase data for {Lx}×{Ly} lattice")
    print(f"Number of qubits: {n_qubits}, Hilbert space dim: {dim}")
    print(f"j_a ∈ [{j_a_range[0]}, {j_a_range[1]}], h = {h_fixed}")
    print(f"Samples: {n_samples}\n")
    
    # Pre-compute operators
    print("Computing star operators...")
    star_ops = sum_star_operators(Lx, Ly)
    print("Computing transverse field...")
    trans_ops = transverse_field(Lx, Ly)
    
    data = []
    j_a_values = np.linspace(j_a_range[0], j_a_range[1], n_samples)
    
    print("\nGenerating ground states...")
    for i, j_a in enumerate(tqdm(j_a_values)):
        # Construct Hamiltonian
        H = hamiltonian(j_a=j_a, h=-h_fixed, star_ops=star_ops, trans_ops=trans_ops)
        
        # Get ground state
        eigenvalues, eigenvectors = jnp.linalg.eigh(H)
        ground_state = eigenvectors[:, 0]
        ground_energy = eigenvalues[0]
        
        # Store data
        data.append({
            'v': ground_state,
            'E': ground_energy,
            'j_a': j_a,
            'h': h_fixed,
            'Lx': Lx,
            'Ly': Ly,
            'phase': 'deconfined'
        })
    
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


if __name__ == "__main__":
    # Configuration
    Lx, Ly = 3, 3  # Lattice size
    
    # Choose which dataset to generate
    print("=" * 60)
    print("Z2 GAUGE THEORY GROUND STATE GENERATION")
    print("=" * 60)
    print("\nOptions:")
    print("1. Confined phase only")
    print("2. Deconfined phase only")
    print("3. Phase transition (both phases)")
    print()
    
    choice = input("Select option (1/2/3) [default: 3]: ").strip() or "3"
    
    if choice == "1":
        # Generate confined phase data
        data = generate_confined_phase_data(Lx, Ly, j_a_fixed=-1.0, h_range=(0.5, 3.0), n_samples=100)
        filename = '../data/data_confined.pkl'
        
    elif choice == "2":
        # Generate deconfined phase data
        data = generate_deconfined_phase_data(Lx, Ly, j_a_range=(-3.0, -0.5), h_fixed=0.1, n_samples=100)
        filename = '../data/data_deconfined.pkl'
        
    else:  # choice == "3"
        # Generate phase transition data
        data = generate_phase_transition_data(Lx, Ly, n_samples=200)
        filename = '../data/data_para.pkl'
    
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
    
    if 'phase' in data[0]:
        phases = [d['phase'] for d in data]
        n_confined = phases.count('confined')
        n_deconfined = phases.count('deconfined')
        print(f"Confined samples: {n_confined}")
        print(f"Deconfined samples: {n_deconfined}")
    
    print("\nDone!")

