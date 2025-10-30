import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap, block_until_ready, pure_callback

import pickle
import numpy as np
import matplotlib.pyplot as plt

from functools import partial

########################################################################################

#  part 1: 
#  auto-differentiate H respect to h to drive to the appropriate phase 

########################################################################################

from jax import config
config.update("jax_enable_x64", True)

########################################################################################

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


def prod(N, sites, ops):
    """
    input N (int): size of the system, 
    input sites (list): integer labels of the lattice sites that the operators are acted on
    input ops (list): operators act on sites
    output Out: an operator of dimension (2**N, 2**N) of operators acting on sites
    """
    arg = jnp.argsort(jnp.asarray(sites))
    sites = jnp.array(sites)[arg]
    ops = jnp.array(ops)[arg]
    cnt = 0
    Out = None
    if sites[cnt] == 0:
        Out = ops[cnt]
        cnt+=1

    else:
        Out = Id

    for i in range(1,N):
        if cnt == len(sites):
            Out = jnp.kron(Out,Id)
            continue

        if i==sites[cnt]:
            Out = jnp.kron(Out,ops[cnt])
            cnt+=1
        else:
            Out = jnp.kron(Out,Id)

    return Out

########################################################################################

Id = jnp.eye(2)
Sz = jnp.zeros([2,2])
Sz = Sz.at[0,0].set(1.)
Sz = Sz.at[1,1].set(-1.)
Sx = jnp.zeros([2,2])
Sx = Sx.at[0,1].set(1.)
Sx = Sx.at[1,0].set(1.)

########################################################################################

## setup the 1d transverse field ising model
# partial(jit, static_argnums=(0,1,2))
@jit
def H_tfim(h, ham_X, ham_ZZ):
    H_mat = ham_X * h + ham_ZZ
    return H_mat


## ED solver
# partial(jit, static_argnums=(0,))
@jit
def gd_solver_ed(h, ham_X, ham_ZZ):
    H = H_tfim(h, ham_X, ham_ZZ)
    e, v = jnp.linalg.eigh(H)
    # print(e)
    return e, v


## average magnetization
def cal_m(v):
    m_tot = 0
    N = int(np.log2(len(v)))
    # print(N)
    for i in range(len(v)):
        state = hslabeltoocc(i, N)
        num_up = jnp.count_nonzero(state)
        m = abs(2*num_up - N)
        m_tot += m* (jnp.conj(v[i])*v[i])
    return m_tot/N


## loss function
@jit
def loss(h, ham_X, ham_ZZ):
    e, v = gd_solver_ed(h, ham_X, ham_ZZ)
    m = cal_m(v[:,0])
    # l = (1-m)**2  ## drive to the ssb phase
    l = m**2   ## drive to the para phase
    return l

########################################################################################

## size of the ising chain
N = 10
J = jnp.asarray(-1.)
h = jnp.asarray(-1.5)   ## initial strenght of the transverse field

ham_ZZ = prod(N,[0,1],[Sz,Sz])
ham_X = prod(N,[0],[Sx])
for i in range(1,N-1):
  ham_ZZ += prod(N, [i,i+1], [Sz, Sz])
  ham_X += prod(N, [i], [Sx])
ham_X += prod(N, [N-1], [Sx])
ham_ZZ *= J

########################################################################################

learning_rate = 0.3
epochs = 200

loss_list = []
h_list = []

## gradient descent to find h that minimizes the loss function
for i in range(epochs):
    value, gradient = value_and_grad(loss,argnums=(0,))(h, ham_X, ham_ZZ)
    h -= learning_rate * gradient[0]
    loss_list.append(value)
    h_list.append(h)
    print('epoch=',i, ', loss=',value, ', gradient=',gradient, ', h_new=',h)

########################################################################################

# plt.plot(range(epochs), h_list, 'o-')
# # plt.axhline(y = 0, color = 'grey', linestyle = '--')
# plt.ylabel('h',fontsize=14)
# plt.xlabel('epoch',fontsize=14)

# plt.savefig('audiff_tfimH_hSSB.pdf', bbox_inches='tight')

