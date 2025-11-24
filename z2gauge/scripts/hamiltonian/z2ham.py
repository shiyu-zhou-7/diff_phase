import jax.numpy as jnp
from functools import reduce


def pauli_z():
    """Pauli Z operator."""
    return jnp.array([[1., 0.], [0., -1.]])


def pauli_i():
    """Identity operator."""
    return jnp.eye(2)


def pauli_x():
    """Pauli X operator."""
    return jnp.array([[0., 1.], [1., 0.]])


def transverse_field(Lx, Ly):
    """
    Constructs the transverse field sum_i sigma^x_i.
    
    Args:
        Lx, Ly: Lattice dimensions
    
    Returns:
        Sum of Pauli X on all qubits (edges)
    """
    n_qubits = 2 * Lx * Ly
    total = jnp.zeros((2**n_qubits, 2**n_qubits))
    
    for i in range(n_qubits):
        # X on qubit i, I on all others
        ops = [pauli_x() if j == i else pauli_i() for j in range(n_qubits)]
        total += reduce(jnp.kron, ops)
    
    return total


def star_operator(Lx, Ly, star_x, star_y):
    """
    Constructs a single star operator A_s at vertex (star_x, star_y).
    
    A_s = sigma_z on 4 edges touching the vertex.
    
    Args:
        Lx, Ly: Lattice dimensions
        star_x, star_y: Vertex coordinates
    
    Returns:
        Star operator as a dense matrix
    """
    n_qubits = 2 * Lx * Ly  # qubits live on edges
    
    # Get the 4 edge indices around this vertex
    # Horizontal edges: indexed 0 to Lx*Ly-1
    # Vertical edges: indexed Lx*Ly to 2*Lx*Ly-1
    edge_left = star_y * Lx + (star_x - 1) % Lx  # horizontal edge to the left
    edge_right = star_y * Lx + star_x  # horizontal edge to the right
    edge_bottom = Lx * Ly + ((star_y - 1) % Ly) * Lx + star_x  # vertical edge below
    edge_top = Lx * Ly + star_y * Lx + star_x  # vertical edge above
    
    edges = [edge_left, edge_right, edge_bottom, edge_top]
    
    # Build tensor product: Z on edges in star, I elsewhere
    ops = [pauli_z() if i in edges else pauli_i() for i in range(n_qubits)]
    
    return reduce(jnp.kron, ops)


def sum_star_operators(Lx, Ly):
    """
    Constructs sum_s A_s where A_s = prod_{i in s} sigma^z_i.
    
    Args:
        Lx, Ly: Lattice dimensions
    
    Returns:
        Sum of all star operators
    """
    total = jnp.zeros((2**(2*Lx*Ly), 2**(2*Lx*Ly)))
    
    for x in range(Lx):
        for y in range(Ly):
            total += star_operator(Lx, Ly, x, y)
    
    return total


def hamiltonian(j_a, h, star_ops, trans_ops):
    """
    Constructs the Hamiltonian H = j_a * sum_s A_s + h * sum_i sigma^x_i.
    
    Args:
        Lx, Ly: Lattice dimensions
        j_a: Coupling strength for star operators
        h: Coupling strength for transverse field
    
    Returns:
        Full Hamiltonian matrix
    """
    return j_a * star_ops + h * trans_ops


def wilson_loop(Lx, Ly, loop_x_start, loop_y_start, loop_width, loop_height):
    """
    Constructs a Wilson loop operator W_C = prod_{i in C} sigma^z_i.
    
    A Wilson loop is a product of sigma_z along a closed loop of edges.
    For a rectangular loop starting at (loop_x_start, loop_y_start) with given width and height.
    
    Args:
        Lx, Ly: Lattice dimensions
        loop_x_start, loop_y_start: Starting position of the loop
        loop_width, loop_height: Width and height of the rectangular loop
    
    Returns:
        Wilson loop operator
    """
    n_qubits = 2 * Lx * Ly
    edges_in_loop = []
    
    # Bottom horizontal edge
    for i in range(loop_width):
        x = (loop_x_start + i) % Lx
        y = loop_y_start % Ly
        edges_in_loop.append(y * Lx + x)
    
    # Right vertical edge
    for j in range(loop_height):
        x = (loop_x_start + loop_width) % Lx
        y = (loop_y_start + j) % Ly
        edges_in_loop.append(Lx * Ly + y * Lx + x)
    
    # Top horizontal edge (going backwards)
    for i in range(loop_width):
        x = (loop_x_start + loop_width - 1 - i) % Lx
        y = (loop_y_start + loop_height) % Ly
        edges_in_loop.append(y * Lx + x)
    
    # Left vertical edge (going backwards)
    for j in range(loop_height):
        x = loop_x_start % Lx
        y = (loop_y_start + loop_height - 1 - j) % Ly
        edges_in_loop.append(Lx * Ly + y * Lx + x)
    
    # Build operator: Z on edges in loop, I elsewhere
    ops = [pauli_z() if i in edges_in_loop else pauli_i() for i in range(n_qubits)]
    
    return reduce(jnp.kron, ops)


def expectation_value(operator, state):
    """
    Compute expectation value <state|operator|state>.
    
    Args:
        operator: Operator matrix
        state: State vector (normalized)
    
    Returns:
        Expectation value (real number)
    """
    return jnp.real(jnp.dot(jnp.conj(state), operator @ state))


def reduced_density_matrix(state, qubits_to_keep, n_qubits):
    """
    Compute reduced density matrix by tracing out qubits not in qubits_to_keep.
    
    Args:
        state: Full state vector
        qubits_to_keep: List of qubit indices to keep
        n_qubits: Total number of qubits
    
    Returns:
        Reduced density matrix
    """
    # Create full density matrix
    rho = jnp.outer(state, jnp.conj(state))
    
    # Reshape into tensor: (2,2,2,...) for bra and ket indices
    rho_shape = [2] * (2 * n_qubits)
    rho_tensor = rho.reshape(rho_shape)
    
    # Trace out qubits not in qubits_to_keep
    qubits_to_trace = sorted([i for i in range(n_qubits) if i not in qubits_to_keep], reverse=True)
    
    # Trace one qubit at a time in reverse order (so indices don't shift)
    remaining_qubits = n_qubits
    for qubit in qubits_to_trace:
        # Axis for bra and ket of this qubit (ket indices come after bra indices)
        axis_bra = qubit
        axis_ket = remaining_qubits + qubit
        
        rho_tensor = jnp.trace(rho_tensor, axis1=axis_bra, axis2=axis_ket)
        remaining_qubits -= 1
    
    # Reshape back to matrix form
    n_keep = len(qubits_to_keep)
    return rho_tensor.reshape(2**n_keep, 2**n_keep)


def von_neumann_entropy(rho):
    """
    Compute von Neumann entropy S = -Tr(rho log rho).
    
    Args:
        rho: Density matrix
    
    Returns:
        Von Neumann entropy
    """
    # Get eigenvalues
    eigenvalues = jnp.linalg.eigvalsh(rho)
    
    # Filter out zero/negative eigenvalues (numerical errors)
    eigenvalues = jnp.where(eigenvalues > 1e-12, eigenvalues, 1e-12)
    
    # S = -sum(lambda * log(lambda))
    return -jnp.sum(eigenvalues * jnp.log(eigenvalues))


def entanglement_entropy(state, subsystem_qubits, n_qubits):
    """
    Compute entanglement entropy of a subsystem.
    
    Args:
        state: Full state vector
        subsystem_qubits: List of qubit indices in subsystem
        n_qubits: Total number of qubits
    
    Returns:
        Entanglement entropy
    """
    rho_A = reduced_density_matrix(state, subsystem_qubits, n_qubits)
    return von_neumann_entropy(rho_A)


def wilson_loop_expectation(state, Lx, Ly, loop_width, loop_height):
    """
    Compute Wilson loop expectation value for a given state.
    
    Deconfined phase: <W> != 0
    Confined phase: <W> -> 0 (exponentially decaying with loop size)
    
    Args:
        state: Ground state vector
        Lx, Ly: Lattice dimensions
        loop_width, loop_height: Loop dimensions
    
    Returns:
        Wilson loop expectation value
    """
    W = wilson_loop(Lx, Ly, 0, 0, loop_width, loop_height)
    return expectation_value(W, state)


def verify_phase(state, Lx, Ly):
    """
    Helper function to verify deconfined vs confined phase.
    
    Args:
        state: Ground state vector
        Lx, Ly: Lattice dimensions
    
    Returns:
        Dictionary with phase indicators
    """
    n_qubits = 2 * Lx * Ly
    
    # Compute Wilson loop for various sizes
    wilson_values = {}
    for size in range(1, min(Lx, Ly) + 1):
        wilson_values[size] = float(wilson_loop_expectation(state, Lx, Ly, size, size))
    
    # Compute entanglement entropy for a subsystem
    # Use half of the qubits
    subsystem = list(range(n_qubits // 2))
    S_ent = float(entanglement_entropy(state, subsystem, n_qubits))
    
    return {
        'wilson_loops': wilson_values,
        'entanglement_entropy': S_ent,
        'phase': 'deconfined' if abs(wilson_values[1]) > 0.1 else 'confined'
    }


if __name__ == "__main__":
    # Example usage
    Lx, Ly = 2, 2
    
    print(f"Toric code: {Lx}x{Ly} lattice")
    print(f"Qubits: {2*Lx*Ly}, Hilbert space dim: {2**(2*Lx*Ly)}\n")
    
    # Construct Hamiltonian in deconfined phase (large j_a, small h)
    print("=" * 50)
    print("DECONFINED PHASE (j_a >> h)")
    print("=" * 50)
    star_ops = sum_star_operators(Lx, Ly)
    trans_ops = transverse_field(Lx, Ly)
    H_deconfined = hamiltonian(j_a=-1.0, h=-0.1, star_ops=star_ops, trans_ops=trans_ops)
    
    # Get ground state
    eigenvalues, eigenvectors = jnp.linalg.eigh(H_deconfined)
    ground_state = eigenvectors[:, 0]
    print(f"Ground state energy: {eigenvalues[0]:.4f}")
    
    # Verify phase
    phase_info = verify_phase(ground_state, Lx, Ly)
    print(f"Phase: {phase_info['phase']}")
    print(f"Wilson loops: {phase_info['wilson_loops']}")
    print(f"Entanglement entropy: {phase_info['entanglement_entropy']:.4f}")
    
    # Construct Hamiltonian in confined phase (small j_a, large h)
    print("\n" + "=" * 50)
    print("CONFINED PHASE (h >> j_a)")
    print("=" * 50)
    H_confined = hamiltonian(j_a=-0.1, h=-1.0, star_ops=star_ops, trans_ops=trans_ops)
    
    # Get ground state
    eigenvalues, eigenvectors = jnp.linalg.eigh(H_confined)
    ground_state = eigenvectors[:, 0]
    print(f"Ground state energy: {eigenvalues[0]:.4f}")
    
    # Verify phase
    phase_info = verify_phase(ground_state, Lx, Ly)
    print(f"Phase: {phase_info['phase']}")
    print(f"Wilson loops: {phase_info['wilson_loops']}")
    print(f"Entanglement entropy: {phase_info['entanglement_entropy']:.4f}")
