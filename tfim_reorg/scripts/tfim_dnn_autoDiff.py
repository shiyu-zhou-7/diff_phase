import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap, block_until_ready, pure_callback

import pickle
import numpy as np
import matplotlib.pyplot as plt

from functools import partial

from jax import config
config.update("jax_enable_x64", True)


########################################################################################

#  part 3: 
#  Auto-differentiate through a trained neural network and ED solver 
#  to find phase's fixed point

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

@jit
def H_tfim(h, ham_X, ham_ZZ):
    H_mat = ham_X * h + ham_ZZ
    return H_mat


@jit
def gd_solver_ed(h, ham_X, ham_ZZ):
    # exact diagonalization
    H = H_tfim(h, ham_X, ham_ZZ)
    e, v = jnp.linalg.eigh(H)
    return e, v


def predict(params, x):
  activation = x
  for w, b in params[:-1]:
    activation = jax.nn.relu(jnp.dot(activation, w) + b)

  ## the last layer does not have the activation function
  fx = (jnp.dot(activation, params[-1][0]) + params[-1][1])
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
        m_tot += m * (jnp.conj(v[i])*v[i])
    return m_tot / N


@jit
def loss_es(h, ham_X, ham_ZZ, params):
    _, v = gd_solver_ed(h, ham_X, ham_ZZ)
    label = predict(params, v[:,0])
    # label = cal_m(v[:,0])
    # loss = (1 - label)**2  # learning the ssb phase
    loss = label**2 # learning the para phase
    return loss

########################################################################################

Id = jnp.eye(2)
Sz = jnp.zeros([2,2])
Sz = Sz.at[0,0].set(1.)
Sz = Sz.at[1,1].set(-1.)
Sx = jnp.zeros([2,2])
Sx = Sx.at[0,1].set(1.)
Sx = Sx.at[1,0].set(1.)

########################################################################################

N = 10
J = jnp.asarray(-1.)
h = jnp.asarray(-1.5)

ham_X = prod(N,[0],[Sx])
ham_ZZ = prod(N,[0,1],[Sz,Sz])
for i in range(1,N-1):
  ham_ZZ += prod(N, [i,i+1], [Sz, Sz])
  ham_X += prod(N, [i], [Sx])
ham_X += prod(N, [N-1], [Sx])
ham_ZZ *= J

########################################################################################    

epochs = 200
lr = 5

h_list = []
loss_list = []

# load trained dnn parameters
model_params = pickle.load(open('../models/tfimClassifier_nnParams_500_500.pkl', 'rb'))

# gradient descent to find h that minimizes the loss function
for i in range(epochs):
    value, gradient = jit(value_and_grad(loss_es, argnums=0))(h, ham_X, ham_ZZ, model_params)
    h -= lr * gradient
    loss_list.append(value)
    h_list.append(h)
    if i%10 == 0:
        print('epoch=',i, ', loss=',value, ', gradient=',gradient, ', h_new=',h)

########################################################################################

plt.figure()
plt.plot(range(epochs), h_list, 'o', color='blue', label='h')
plt.plot(range(epochs), loss_list, 'o', color='orange', label='loss')
plt.legend(fontsize=13)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('h', fontsize=14)
# plt.savefig(f'../figures/tfim_dnn_autoDiff_para_h0={-1.5}.pdf', bbox_inches='tight')

########################################################################################