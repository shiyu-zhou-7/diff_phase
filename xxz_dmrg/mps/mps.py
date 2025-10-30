import jax
import jax.numpy as jnp
from jax.numpy.linalg import svd


class MPS:
    """ 1D MPS class
    Attributes
    ----------
    Ss: Schmidt values at each site (determines the entanglement entropy across
    a bond).

    Bs: B matrices making up the MPS. Once the MPS is converged, it is easy to
    move between the A and B canonical forms via A_n = S_n^{-1} B_n S_{n+1}. 
    The index convention is vL p vR.

    num_bonds: The number of bonds. For the infinite MPS, this is the length
    of the MPS (there is one additional bond to allow the environments to grow
    at each stage). 

    L: Length of the MPS.
    """
    def __init__(self, Ss, Bs, bc="finite"):
        self.Ss = Ss
        self.Bs = Bs
        assert bc in ["finite", "infinite"]
        self.bc = bc
        self.num_bonds = len(Bs) - 1 if bc == "finite" else len(Bs)
        self.L = len(Bs)

    def __copy__(self):
        return MPS(self.Ss, self.Bs, self.bc)
        
    def get_theta(self, ind, k=2):
        """ Returns the k-site wavefunction for the MPS in mixed canonical 
        form at i, i+1 """
        assert ind <= self.num_bonds
        assert k in [1, 2]
        if k == 2:
            theta = jnp.tensordot(jnp.diag(self.Ss[ind]), self.Bs[ind], axes=(1, 0))
            theta = jnp.tensordot(theta, self.Bs[(ind + 1) % self.L], axes=(2, 0))
            return theta
        elif k == 1:
            theta = jnp.tensordot(self.Bs[ind], jnp.diag(self.Ss[ind]), axes=(0, 1))
            return theta
    
    def get_bond_exp_val(self, ops):
        """ ops should be a list of local two-site operators. """
        exp_vals = [None] * len(ops)
        for l_site in range(self.num_bonds):
            print("Computing exp val at bond ", l_site)
            theta = self.get_theta(l_site)
            theta_ = jnp.conj(theta)
            op = ops[l_site].reshape(2, 2, 2, 2)
            exp_vals[l_site] = jnp.einsum("ijkl,mnjk,imnl", theta, op, theta_)
        return exp_vals
    
    def get_site_exp_val(self, ops):
        """
        ops: iterable of (site, op) where
            - site: int in [0, L-1]
            - op:   (2,2) array acting on the physical index at `site`
        Returns: list[scalar] expectation values <op_site>
        """
        # assert len(ops) == self.L
        exp_vals = []
        # for site in range(self.L):
        for site, op in ops:    # <-- site first, operator second
            theta = self.get_theta(site, k=1)
            theta_ = jnp.conj(theta)
            # exp_val = jnp.tensordot(theta, ops[site], axes=(0, 1))
            exp_val = jnp.tensordot(theta, op, axes=(0, 1))
            exp_val = jnp.tensordot(exp_val, theta_, axes=([0, 2, 1], [1, 0, 2]))
            exp_vals.append(exp_val)
        return exp_vals

    def get_local_expectations(self, ops):
        """
        Compute local expectation values for a mixed list of 1-site and 2-site ops.

        Args
        ----
        ops : iterable of items where each item is either
            - (site, op)                  for a 1-site operator, or
            - ([site1, site2], op)        for a 2-site operator on consecutive sites
        'site' and 'site1/site2' are ints; 'op' shapes:
            * 1-site: (2, 2)
            * 2-site: (4, 4) or (2, 2, 2, 2)

        Returns
        -------
        list of scalar expectation values in the same order as 'ops'.
        """
        exp_vals = []
        d = 2

        for item in ops:
            # Unpack item
            idx, op = item

            # ------------- One-site operator -------------
            if isinstance(idx, (int, jnp.ndarray)):  # (site, op)
                site = int(idx)
                theta = self.get_theta(site, k=1)       # (χ_L, d, χ_R)
                theta_ = jnp.conj(theta)

                if op.ndim != 2 or op.shape != (d, d):
                    raise ValueError(f"1-site op must have shape (2,2); got {op.shape}")

                # <θ| O |θ> over physical index
                x = jnp.tensordot(theta, op, axes=(0, 1))                
                val = jnp.tensordot(x, theta_, axes=([0, 2, 1], [1, 0, 2]))
                exp_vals.append(val)

            # ------------- Two-site operator -------------
            else:  # ([site1, site2], op)
                site1, site2 = idx
                # enforce adjacency modulo L
                assert site2 == (site1 + 1) % self.L, \
                    f"Two-site operator must act on consecutive sites, got {site1}, {site2}"

                theta = self.get_theta(site1, k=2)      # (χ_L, d1, d2, χ_R)
                theta_ = jnp.conj(theta)

                # Accept (4,4) or (2,2,2,2)
                if op.ndim == 2 and op.shape == (d*d, d*d):
                    op4 = op.reshape(d, d, d, d)
                elif op.ndim == 4 and op.shape == (d, d, d, d):
                    op4 = op
                else:
                    raise ValueError(f"2-site op must be (4,4) or (2,2,2,2); got {op.shape}")

                # Contract: <θ| O |θ>, where O acts on the two physical legs
                # x = jnp.tensordot(theta, op4, axes=([1, 2], [0, 1]))    # (χ_L, d1, d2, χ_R) × (d1,d2,d1,d2)
                # val = jnp.tensordot(x, theta_, axes=([0, 2, 1], [0, 1, 2]))
                # x = jnp.tensordot(theta, op4, axes=([1, 2], [0, 1]))
                # val = jnp.tensordot(x, theta_, axes=([0, 1, 2, 3], [0, 3, 1, 2]))
                val = jnp.einsum("ijkl,mnjk,imnl", theta, op4, theta_)
                exp_vals.append(val)

        return exp_vals

    
    def entanglement_entropy(self, k=1, bond=None):
        """ Returns the kth Renyi entropy across a bond. """
        print(f"Returning {k}th Renyi entropy")
        bonds = range(self.num_bonds)
        result = []
        if bond is not None:
            S = self.Ss[bond].copy()
            Sr = (S * S) ** k
            return -jnp.sum(Sr * jnp.log(Sr))
        for i in bonds:
            S = self.Ss[i].copy()
            Sr = (S * S) ** k
            result.append(-jnp.sum(Sr * jnp.log(Sr)))
        return jnp.array(result)

    def update_theta(self, i, theta, shape, chi_max):
        j = (i + 1) % self.L
        A, Sj, B = split_and_truncate(theta, shape, chi_max)
        Si = self.Ss[i]
        Bprev = jnp.tensordot(jnp.diag(1.0 / Si), A, axes=(1, 0))
        Bprev = jnp.tensordot(Bprev, jnp.diag(Sj), axes=(2, 0))
        oldSV = self.Ss[j]
        self.Ss[j] = Sj
        self.Bs[i] = Bprev
        self.Bs[j] = B
        return oldSV, Sj


def split_and_truncate(theta, shape, chi_max, eps=1.e-14):
    """ Splits theta, performs an SVD, and trims the matrices to chi_max. """
    chiL, dL, dR, chiR = shape
    theta_matrix = theta.reshape((chiL * dL, chiR * dR))
    U, Sfull, V = svd(theta_matrix, full_matrices=False)
    
    # chi_keep = jnp.sum(Sfull > eps)  # Number of singular values > eps
    # chi_keep = min(chi_keep, chi_max)
    chi_keep = chi_max
    
    A = U[:, :chi_keep]
    B = V[:chi_keep, :]
    S = Sfull[:chi_keep]
    S = S / jnp.linalg.norm(S)  # Normalize Schmidt values
    
    A = A.reshape([chiL, dL, chi_keep])
    B = B.reshape([chi_keep, dR, chiR])
    return A, S, B

def get_random_MPS(L, d, bond_dim = 2, bc="finite", seed=None):
    """
    Generate a random MPS (Matrix Product State).

    Args:
        L: int - Number of sites.
        d: int - Local Hilbert space dimension (e.g., 2 for spin-1/2 systems).
        bond_dim: int - Maximum bond dimension.
        bc: str - Boundary conditions ("finite" or "infinite").
        seed: int or None - Random seed for reproducibility.

    Returns:
        MPS object with random tensors.
    """
    key = jax.random.PRNGKey(seed) if seed is not None else jax.random.PRNGKey(0)

    Bs = []
    Ss = []

    # Create random tensors for the MPS
    for i in range(L):
        bond_left = 1 if i == 0 else bond_dim
        bond_right = 1 if i == L - 1 else bond_dim

        # Random tensor for site i
        B_shape = (bond_left, d, bond_right)
        key, subkey = jax.random.split(key)
        B = jax.random.normal(subkey, shape=B_shape)

        # Singular values for site i
        S_shape = (bond_right,)
        key, subkey = jax.random.split(key)
        S = jnp.abs(jax.random.normal(subkey, shape=S_shape)) + 1e-10  # Ensure non-zero

        Bs.append(B)
        Ss.append(S)

    return MPS(Ss, Bs, bc)
    
def get_FM_MPS(L, d, bc="finite"):
    B = jnp.zeros([1, d, 1], float)
    B = B.at[0, 0, 0].set(1.0)
    S = jnp.ones([1], float)
    Bs = [B.copy() for _ in range(L)]
    Ss = [S.copy() for _ in range(L)]
    return MPS(Ss, Bs, bc)