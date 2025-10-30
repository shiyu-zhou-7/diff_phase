import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap, block_until_ready, pure_callback

import pickle
import numpy as np
import matplotlib.pyplot as plt

from functools import partial

from jax import config
config.update("jax_enable_x64", True)



"""""""""""""""""""""
 part 3: 
 Auto-differentiate through a trained neural network and ED solver 
 to find phase's fixed point
"""""""""""""""""""""


""" defind functions """

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


## define the single spin Pauli operators: Identity, Sz, Sx
## defined in terms of jax.numpy
Id = jnp.eye(2)
Sz = jnp.zeros([2,2])
Sz = Sz.at[0,0].set(1.)
Sz = Sz.at[1,1].set(-1.)
Sx = jnp.zeros([2,2])
Sx = Sx.at[0,1].set(1.)
Sx = Sx.at[1,0].set(1.)


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
def predict(params, x):
  activation = x
  for w, b in params[:-1]:
    # print(activation.shape, w.shape, b.shape)
    activation = jax.nn.relu(jnp.dot(activation, w) + b)

  ## the last layer does not have the activation function
  fx = (jnp.dot(activation, params[-1][0]) + params[-1][1])
  # print(np.reshape(fx,()))
  return jnp.reshape(fx,())


def cal_m(v):
    """
    Calculate the average magnetization of a given state
    input v: a state in hilbert space
    output m_tot: the total magnetization
    """
    m_tot = 0
    for i in range(len(v)):
        state = hslabeltoocc(i, N)
        num_up = jnp.count_nonzero(state)
        m = abs(2*num_up - N)
        m_tot += m* (jnp.conj(v[i])*v[i])
    return m_tot


## loss function
# partial(jit, static_argnums=(2,))
@jit
def loss_es(h, ham_X, ham_ZZ, params):
    e, v = gd_solver_ed(h, ham_X, ham_ZZ)
    label = predict(params, v[:,0])
    l = (1 - label)**2  # learning the ssb phase
    # l = label**2 # learning the para phase
    return l


""" setup Hamiltonian """

## size of the ising chain
N = 10

## strength of the coupling
J = jnp.asarray(-1.)
## initial strength of the transverse field
h = jnp.asarray(-0.5)

## pre-compute the terms (J*sum_i Z_i Z_{i+1}) and (sum_i X_i) 
## the Hamiltonian is H = H_ZZ + h*H_X
ham_ZZ = prod(N,[0,1],[Sz,Sz])
ham_X = prod(N,[0],[Sx])
for i in range(1,N-1):
  ham_ZZ += prod(N, [i,i+1], [Sz, Sz])
  ham_X += prod(N, [i], [Sx])
ham_X += prod(N, [N-1], [Sx])
ham_ZZ *= J
# print(ham_X.shape)
# print(ham_ZZ.shape)


""" auto differentiate """

epochs = 700
lr = 0.1

h_list = []
loss_list = []

## load trained dnn parameters
# model_params = pickle.load(open('./TFIM_TEST/tfimClassifier_nnParams_500_500_500.pickle', 'rb'))
model_params = pickle.load(open('./tfimClassifier_nnParams_500_500.pickle', 'rb'))

## gradient descent to find h that minimizes the loss function
for i in range(epochs):
    value, gradient = jit(value_and_grad(loss_es, argnums=0))(h, ham_X, ham_ZZ, model_params)
    h -= lr * gradient
    loss_list.append(value)
    h_list.append(h)
    if i%10 == 0:
        print('epoch=',i, ', loss=',value, ', gradient=',gradient, ', h_new=',h)