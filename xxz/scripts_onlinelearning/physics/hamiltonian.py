import jax
import jax.numpy as jnp
import numpy as np

Id = jnp.eye(2)

Sz = jnp.array([[1., 0.], [0., -1.]], dtype=jnp.complex128)
Sx = jnp.array([[0., 1.], [1.,  0.]], dtype=jnp.complex128)
Sy = jnp.array([[0., -1j], [1j, 0.]], dtype=jnp.complex128)


def prod(N, sites, ops):
    arg = jnp.argsort(jnp.asarray(sites))
    sites = jnp.array(sites)[arg]
    ops = jnp.array(ops)[arg]
    cnt = 0
    Out = Id if sites[0] != 0 else ops[0]
    cnt = cnt + (1 if sites[0] == 0 else 0)
    for i in range(1, N):
        if cnt == len(sites):
            Out = jnp.kron(Out, Id)
            continue
        if i == sites[cnt]:
            Out = jnp.kron(Out, ops[cnt])
            cnt += 1
        else:
            Out = jnp.kron(Out, Id)
    return Out


def build_chain_operators(N):
    ham_xx = prod(N, [0,1], [Sx, Sx])
    ham_yy = prod(N, [0,1], [Sy, Sy])
    ham_zz = prod(N, [0,1], [Sz, Sz])
    ham_x  = prod(N, [0],   [Sx])
    for i in range(1, N-1):
        ham_xx += prod(N, [i, i+1], [Sx, Sx])
        ham_yy += prod(N, [i, i+1], [Sy, Sy])
        ham_zz += prod(N, [i, i+1], [Sz, Sz])
        ham_x  += prod(N, [i],      [Sx])
    ham_x += prod(N, [N-1], [Sx])
    return ham_xx, ham_yy, ham_zz, ham_x


def H_xxzh(delta, h, ham_xx, ham_yy, ham_zz, ham_x, J=1.0):
    return (ham_xx + ham_yy + ham_zz * delta) * J + h * ham_x


def eigh_ground_state(H):
    e, v = jnp.linalg.eigh(jnp.real(H).astype(jnp.float64))
    return e[0], v[:, 0]


def hslabeltoocc(hslabel, N):
    """
    Converts integer hilbert space label (hslabel) to anyon labels on bonds.
    N: number of bonds in the chain.
    Returns an integer array with 1 representing tau and 0 representing the identity
    """

    return np.array(list(np.binary_repr(hslabel, N)), dtype=int)


def occtohslabel(occ, N):
    """
    Converts array of anyon labels on bonds to integer label.
    N: number of bonds in the chain.
    """

    return int(''.join(map(str, occ)),2)


## average magnetization of a wavfunction
def cal_m(v):
    m_tot = 0
    N = int(np.log2(len(v)))
    for i in range(len(v)):
        state = hslabeltoocc(i, N)
        num_up = jnp.count_nonzero(state)
        m = abs(2*num_up - N)
        m_tot += m* (jnp.conj(v[i])*v[i])
    return m_tot/N


def cal_m_reconstructed(x):
    m_list = []
    for v in x:
        m_list.append(cal_m(v))
    return m_list