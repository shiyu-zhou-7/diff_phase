import jax.numpy as jnp

class TFI:
    """ Transverse Field Ising (TFI) model implemented in JAX. """
    def __init__(self, L, g, J, bc="finite"):
        assert bc in ["finite", "infinite"]
        self.bc = bc
        self.g = g
        self.J = J
        self.d = 2
        self.L = L

        # Pauli matrices
        sx = jnp.array([[0, 1], [1, 0]])
        sy = jnp.array([[0, -1.0j], [1.0j, 0]])
        sz = jnp.array([[1.0, 0.0], [0.0, -1.0]])
        idn = jnp.eye(2)

        # Construct MPO tensors
        self.Ws = []
        for i in range(L):
            w = jnp.zeros((3, 3, self.d, self.d), dtype=float)
            w = w.at[0, 0].set(idn).at[2, 2].set(idn)
            w = w.at[0, 1].set(sx)
            w = w.at[0, 2].set(-g * sz)
            w = w.at[1, 2].set(-J * sx)
            self.Ws.append(w)

    def get_H_bonds(self):
        """ Returns a list of local operators corresponding to the TFI model. """
        num_bonds = self.L if self.bc == "infinite" else self.L - 1

        # Pauli matrices
        sx = jnp.array([[0, 1], [1, 0]])
        sz = jnp.array([[1.0, 0.0], [0.0, -1.0]])
        idn = jnp.eye(2)

        ops = []

        for site in range(num_bonds):
            gL = gR = 0.5 * self.g
            if self.bc == "finite":
                if site == 0:
                    gL = self.g
                if site == self.L - 2:
                    gR = self.g
            H_local = (-self.J * jnp.kron(sx, sx) -
                       gL * jnp.kron(sz, idn) -
                       gR * jnp.kron(idn, sz))
            ops.append(H_local.reshape([self.d] * 4))

        self.H_bonds = ops
        return ops

class XXZ:
    """
    XXZ model implemented in JAX.
    """
    def __init__(self, L, Jx, Jy, Jz, h = 0, bc="finite"):
        """
        Initialize the XXZ model.

        Parameters:
        L : int
            Number of sites in the chain.
        Jx, Jy, Jz : float
            Coupling constants for the x, y, and z directions.
        bc : str
            Boundary conditions, either "finite" or "infinite".
        """
        assert bc in ["finite", "infinite"]
        self.bc = bc
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.d = 2
        self.L = L
        self.h = h

        # Pauli matrices
        sx = jnp.array([[0, 1], [1, 0]])
        sy = jnp.array([[0, -1j], [1j, 0]])
        sz = jnp.array([[1.0, 0.0], [0.0, -1.0]])
        idn = jnp.eye(2)

        # Construct MPO tensors
        self.Ws = []
        for i in range(L):
            w = jnp.zeros((5, 5, self.d, self.d), dtype=complex)
            w = w.at[0, 0].set(idn)
            w = w.at[0, 1].set(sx)
            w = w.at[0, 2].set(sy)
            w = w.at[0, 3].set(sz)
            w = w.at[0, 4].set(h * sz)
            w = w.at[1, 4].set(Jx * sx)
            w = w.at[2, 4].set(Jy * sy)
            w = w.at[3, 4].set(Jz * sz)
            w = w.at[4, 4].set(idn)
            self.Ws.append(w)

    def get_H_bonds(self):
        """
        Returns a list of local operators corresponding to the XXZ model.
        """
        num_bonds = self.L if self.bc == "infinite" else self.L - 1

        # Pauli matrices
        sx = jnp.array([[0, 1], [1, 0]])
        sy = jnp.array([[0, -1j], [1j, 0]])
        sz = jnp.array([[1.0, 0.0], [0.0, -1.0]])
        idn = jnp.eye(2)

        ops = []

        for site in range(num_bonds):
            hL = hR = self.h/2
            if self.bc == "finite":
                if site == 0:
                    hL = self.h
                if site == self.L - 2:
                    hR = self.h
            H_local = (self.Jx * jnp.kron(sx, sx) +
                       self.Jy * jnp.kron(sy, sy) +
                       self.Jz * jnp.kron(sz, sz) + hL * jnp.kron(sz, idn) +
                       hR * jnp.kron(idn, sz))
            ops.append(H_local.reshape([self.d] * 4))

        self.H_bonds = ops
        return ops
        
class AXY3:
    """
    Model implementing Equation (1) with alternating couplings, a z-field, 
    and three-site interactions.
    """
    def __init__(self, L, J0, J1, gamma, h, Omega, bc="finite"):
        """
        Initialize the model.

        Args:
            L: int - Number of sites in the chain.
            J0: float - Uniform nearest-neighbor coupling constant.
            J1: float - Alternating nearest-neighbor coupling constant.
            gamma: float - Anisotropy parameter for the XY interaction.
            h: float - Strength of the Z field.
            Omega: float - Coupling constant for the three-site interaction.
            bc: str - Boundary conditions, either "finite" or "infinite".
        """
        assert bc in ["finite", "infinite"]
        self.bc = bc
        self.J0 = J0
        self.J1 = J1
        self.gamma = gamma
        self.h = h
        self.Omega = Omega
        self.d = 2
        self.L = L

        # Pauli matrices
        sx = jnp.array([[0, 1], [1, 0]])
        sy = jnp.array([[0, -1j], [1j, 0]])
        sz = jnp.array([[1.0, 0.0], [0.0, -1.0]])
        idn = jnp.eye(2)

        # Construct MPO tensors
        self.Ws = []
        for i in range(L):
            Jn = (J0 + (-1)**i * J1) / 2  # Alternating coupling constant

            w = jnp.zeros((6, 6, self.d, self.d), dtype=complex)
            w = w.at[0, 0].set(idn).at[5, 5].set(idn)  # Identity terms
            #w = w.at[0, 1].set(sx)  # σ^x
            #w = w.at[0, 2].set(sy)  # σ^y
            #w = w.at[2, 3].set(sz)  # σ^z
            #w = w.at[1, 4].set(sz)  # σ^z
            #w = w.at[0, 5].set(h * sz)  # Z-field term (-h σ^z)
            #w = w.at[1, 5].set(1 + gamma)*Jn * sx  # Jn σ^x ⊗ σ^x
            #w = w.at[2, 5].set(1 - gamma)*Jn * sy  # Jn σ^y ⊗ σ^y
            #w = w.at[3, 5].set(Omega * sy)  # Three-site σ^x ⊗ σ^z ⊗ σ^x
            #w = w.at[4, 5].set(Omega * sx)  # Three-site σ^y ⊗ σ^z ⊗ σ^y
            w = w.at[0, 1].set(sx)  # σ^x
            w = w.at[0, 2].set(sy)  # σ^y
            #w = w.at[3, 2].set(idn)  # σ^z
            #w = w.at[4, 1].set(idn)  # σ^z
            w = w.at[0, 5].set(h * sz)  # Z-field term (-h σ^z)
            w = w.at[1, 5].set(1 + gamma)*Jn * sx  # Jn σ^x ⊗ σ^x
            w = w.at[2, 5].set(1 - gamma)*Jn * sy  # Jn σ^y ⊗ σ^y
            w = w.at[3, 5].set(Omega * sy)  # Three-site σ^x ⊗ σ^z ⊗ σ^x
            w = w.at[4, 5].set(Omega * sx)  # Three-site σ^y ⊗ σ^z ⊗ σ^y
            self.Ws.append(w)

    def get_H_bonds(self):
        """
        Returns a list of local operators corresponding to the model.

        Includes two-site and three-site interaction terms.
        """
        num_bonds = self.L if self.bc == "infinite" else self.L - 1

        # Pauli matrices
        sx = jnp.array([[0, 1], [1, 0]])
        sy = jnp.array([[0, -1j], [1j, 0]])
        sz = jnp.array([[1.0, 0.0], [0.0, -1.0]])
        idn = jnp.eye(2)

        ops = []

        for site in range(num_bonds):
            # Alternating coupling constant
            Jn = (self.J0 + (-1)**site * self.J1) / 2

            # Two-site nearest-neighbor interaction
            H_local = (
                Jn * (1 + self.gamma) * jnp.kron(sx, sx) +
                Jn * (1 - self.gamma) * jnp.kron(sy, sy) +
                Jn * jnp.kron(sz, sz)
            )
            ops.append(H_local.reshape([self.d] * 4))

        # Three-site interaction terms
        for site in range(1, self.L - 1):
            H_three = (
                self.Omega * jnp.kron(jnp.kron(sx, sz), sx) +
                self.Omega * jnp.kron(jnp.kron(sy, sz), sy)
            )
            ops.append(H_three.reshape([self.d] * 6))

        self.H_bonds = ops
        return ops