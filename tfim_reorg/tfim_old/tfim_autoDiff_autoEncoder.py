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
 part 5:
 Using the learnt auto-encoder to learn the phases of the transverse Ising model (tfim).
 Specifically, we start with one phase of tfim and ask the learnt auto-encoder to go to the other phase.
 The structure of auto-encoder is the following,
 o              o
    o       o
 o      o       o
    o       o
 o              o
 In order to drive the wavefunction of tfim to a different phase, we design the following loss function,
 L = - sqrt(||x_in - x_out||_2)
 Notice that minimize the L with an extra minus sign is to maximize the distance between x_in and x_out
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
# partial(jit, static_argnums=(0,1,2))g
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

def dropout(x, drop_p):
    # print(x.shape)
    bernoulli_list = np.random.binomial(size=x.shape, n=1, p=(1-drop_p))
    x = x * bernoulli_list
    return x

def reconstruct(params, x):
  ## input params: a list of parameters for the network; its length is the depth of the network
  ## input x: the training data
  ## output fx: the output after passing the network
  ## this function performs the network on the input data x given the network parameters x

  ## input
  activation = x

  ## dropout probability
  drop_p = 0.1

  ## encoder
  for w, b in params[0:1]:
    activation = jax.nn.relu(jnp.dot(activation, w) + b)
    ## dropout
    activation = dropout(activation, drop_p)

  ## latent space
  w, b = params[1]
  activation = jnp.dot(activation, w) + b
  latent_prm = activation

  ## decoder 
  for w, b in params[2:-1]:
    activation = jax.nn.relu(jnp.dot(activation, w) + b)
    activation = dropout(activation, drop_p)

  ## neural network 
#   latent_prm = 0
#   for w, b in params[:-1]:
#     activation = jax.nn.relu(jnp.dot(activation, w) + b)  

  ## output 
  x_out = jnp.dot(activation, params[-1][0]) + params[-1][1]

  return x_out, jnp.mean(latent_prm)


@jit
def loss_es(h, ham_X, ham_ZZ, params):
    e, v = gd_solver_ed(h, ham_X, ham_ZZ)
    x = v[:,0]
    x_out, latent_prm = reconstruct(params, x)
    l = (x_out - x)**2
    l = 1 * jnp.mean(jnp.sqrt(l))
    return l


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


""" setup Hamiltonian """

## size of the ising chain
N = 10

## strength of the coupling
J = jnp.asarray(-1.)
## initial strength of the transverse field
h = jnp.asarray(-0.4)

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


""" load the model """
size_train = 1000
num_epochs = 10000
architecture = [2**10, 1000, 1, 1000, 2**10] 
hyper_param = '_'.join(str(e) for e in architecture)
# model_params = pickle.load(open('./TFIM_TEST/tfimAutoEncoder_nnParams_500_256_1_256_500.pickle', 'rb'))
data = pickle.load(open('./tfim_test/data/tfimAutoEncoder_ssb_nnParams_trainSize%d_%s_epoch%d.pickle'%(size_train, hyper_param, num_epochs), 'rb'))
model_params = data['model']


""" train """
epochs = 700
lr = 0.1

h_list = []
loss_list = []

## gradient descent to find h that minimizes the loss function
for i in range(epochs):
    value, gradient = jit(value_and_grad(loss_es, argnums=0))(h, ham_X, ham_ZZ, model_params)
    h -= lr * gradient
    loss_list.append(value)
    h_list.append(h)
    if i%10 == 0:
        print('epoch=',i, ', loss=',value, ', gradient=',gradient, ', h_new=',h)

