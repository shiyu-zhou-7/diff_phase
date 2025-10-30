# --- place these four lines at the very TOP of the script -------------
import os
os.environ["JAX_ENABLE_X64"] = "True"        # high-precision path
os.environ["JAX_DEBUG_NANS"] = "True"        # raise on inf / nan
# os.environ["JAX_DISABLE_JIT"] = "1"        # <- UNcomment only for 1-off debugging
# ----------------------------------------------------------------------

import jax, jax.numpy as jnp
import numpy as np

#############################################################################

Id = np.eye(2)

Sz = np.zeros([2,2], dtype=complex)
Sz[0,0] = 1.
Sz[1,1] = -1.

Sx = np.zeros([2,2], dtype=complex)
Sx[0,1] = 1.
Sx[1,0] = 1.

Sy = np.zeros([2,2], dtype=complex)
Sy[0,1] = -1j
Sy[1,0] = 1j

##############################################################################

def prod(N, sites, ops):
    """
    input N (int): size of the system, 
    input sites (list): integer labels of the lattice sites that the operators are acted on
    input ops (list): operators act on sites
    output Out: an operator of dimension (2**N, 2**N) of operators acting on sites
    """
    arg = np.argsort(np.asarray(sites))
    sites = np.array(sites)[arg]
    ops = np.array(ops)[arg]
    cnt = 0
    Out = None
    if sites[cnt] == 0:
        Out = ops[cnt]
        cnt+=1

    else:
        Out = Id

    for i in range(1,N):
        if cnt == len(sites):
            Out = np.kron(Out,Id)
            continue

        if i==sites[cnt]:
            Out = np.kron(Out,ops[cnt])
            cnt+=1
        else:
            Out = np.kron(Out,Id)

    return Out


def H_xxz(ham_xx, ham_yy, ham_zz, delta, J=1):
    H_mat = (ham_xx + ham_yy + ham_zz * delta) * J
    return H_mat

##############################################################################

N = 10

ham_xx = prod(N,[0,1],[Sx,Sx])
ham_yy = prod(N,[0,1],[Sy,Sy])
ham_zz = prod(N,[0,1],[Sz,Sz])

for i in range(1,N-1):
    ham_xx += prod(N, [i,i+1], [Sx, Sx])
    ham_yy += prod(N, [i,i+1], [Sy, Sy])
    ham_zz += prod(N, [i,i+1], [Sz, Sz])

##############################################################################

def gd_solver_ed(delta, ham_xx, ham_yy, ham_zz):
    H_np = H_xxz(ham_xx, ham_yy, ham_zz, delta)

    if not np.isfinite(H_np).all():
        raise FloatingPointError("Hamiltonian contains NaN/Inf")
    
    H = jnp.asarray(H_np, dtype=jnp.complex128)

    evals, evecs = jnp.linalg.eigh(H)        # <-- gradients supported
    return evals, evecs

###############################################################################

e, v = gd_solver_ed(-2, ham_xx, ham_yy, ham_zz)
print("Eigenvalues:", v.dtype)