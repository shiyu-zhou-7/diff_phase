import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap, block_until_ready, pure_callback

import pickle
import numpy as np
import matplotlib.pyplot as plt

from functools import partial

import optax

from jax import config
config.update("jax_enable_x64", True)

########################################################################################

#  part 5:
#  Using the learnt (ssb) auto-encoder to learn a different phases of the transverse Ising model (tfim).
#  Specifically, we start with one phase of tfim (ssb) and ask the learnt auto-encoder to go to the other phase.
#  The structure of auto-encoder is the following, 
#  o              o
#     o       o
#  o      o       o
#     o       o
#  o              o
#  Note that the autoencoder is symmetric with respect to the latent space.
#  In order to drive the wavefunction of tfim to a different phase, we design the following loss function,
#  L = - sqrt(||x_in - x_out||_2) / (# of samples)
#  Notice that minimize the L with an extra minus sign is to maximize the distance between x_in and x_out

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

@jit
def H_tfim(h, ham_X, ham_ZZ):
    H_mat = ham_X * h + ham_ZZ
    return H_mat

@jit
def gd_solver_ed(h, ham_X, ham_ZZ):
    H = H_tfim(h, ham_X, ham_ZZ)
    e, v = jnp.linalg.eigh(H)
    return e, v


def cal_m(v):
    m_tot = 0
    N = int(np.log2(len(v)))
    for i in range(len(v)):
        state = hslabeltoocc(i, N)
        num_up = jnp.count_nonzero(state)
        m = abs(2*num_up - N)
        m_tot += m* (jnp.conj(v[i])*v[i])
    return m_tot / N

########################################################################################

#@jit
def dropout(x, drop_p, rng_key):
    keep_prob = 1.0 - drop_p
    bernoulli_list = jax.random.bernoulli(rng_key, keep_prob, shape=x.shape)
    x = x * bernoulli_list
    return x

#@jit
def encoder(params, x, drop_p, rng_key):
    activation = x
    for w, b in params[0:-1]:
        activation = jax.nn.relu(jnp.dot(activation, w) + b)
        rng_key, subkey = jax.random.split(rng_key)
        activation = dropout(activation, drop_p, subkey)
    w, b = params[-1]
    latent_representation = jnp.dot(activation, w) + b
    return latent_representation

#@jit
def decoder(params, latent_rep, drop_p, rng_key):
    activation = latent_rep
    for w, b in params[0:-1]:
        activation = jax.nn.relu(jnp.dot(activation, w) + b)
        rng_key, subkey = jax.random.split(rng_key)
        activation = dropout(activation, drop_p, subkey)
    x_out = jnp.dot(activation, params[-1][0]) + params[-1][1]
    return x_out

#@jit
def autoencoder(params, x, drop_p, rng_key):
    # assuming symmetric autoencoder
    mid_index = len(params) // 2

    rng_key, subkey = jax.random.split(rng_key)
    latent_rep = encoder(params[0:mid_index], x, drop_p, subkey)

    rng_key, subkey = jax.random.split(rng_key)
    x_out = decoder(params[mid_index:], latent_rep, drop_p, subkey)

    def acti_loss(x):
        # note x is a one-dimensional vector here
        dn = jnp.sum(x**2)  # Sum the squares of all elements in the vector
        return x / jnp.sqrt(dn)  # Normalize the vector

    x_out = acti_loss(x_out)   ## normalize the output
    return x_out


def fetch_latent(params, x, drop_p, rng_key):
    # assuming symmetric autoencoder
    mid_index = len(params) // 2

    rng_key, subkey = jax.random.split(rng_key)
    latent_rep = encoder(params[0:mid_index], x, drop_p, subkey)
    return latent_rep


#@jit
# def loss_es(h, ham_X, ham_ZZ, params, drop_p, rng_key):
#     _, v = gd_solver_ed(h, ham_X, ham_ZZ)
#     x_out = autoencoder(params, v[:,0], drop_p, rng_key)
#     l = (x_out - v[:,0])**2
#     l_sum = jnp.sum(l)  # Directly sum the squared differences
#     return -1 * jnp.sqrt(l_sum) / len(l)  # Compute the mean sqrt of l and then apply the minus sign


@jit
def loss_es(h, ham_X, ham_ZZ, latent_ssb, params, drop_p, rng_key):
    _, v = gd_solver_ed(h, ham_X, ham_ZZ)
    latent = fetch_latent(params, v[:,0], drop_p, rng_key)
    l = (latent - latent_ssb)**2
    l_sum = jnp.sum(l)  # Directly sum the squared differences
    return -1 * jnp.sqrt(l_sum) / len(l)  # Compute the mean sqrt of l and then apply the minus sign


#@jit
def update(h, ham_X, ham_ZZ, latent_ssb, params, opt_state, opt, drop_p, rng_key):
    value, grads = value_and_grad(loss_es, argnums=0)(h, ham_X, ham_ZZ, latent_ssb, params, drop_p, rng_key)
    updates, opt_state = opt.update(grads, opt_state)
    h_new = optax.apply_updates(h, updates)
    return h_new, value


## average magnetization of reconstructed wavefunction
def cal_m_reconstructed(x):
    m_list = []
    for v in x:
        m_list.append(cal_m(v))
    return sum(m_list) / len(m_list)

########################################################################################

N = 10
J = jnp.asarray(-1.)
h = jnp.asarray(-0.4)

ham_ZZ = prod(N,[0,1],[Sz,Sz])
ham_X = prod(N,[0],[Sx])
for i in range(1,N-1):
  ham_ZZ += prod(N, [i,i+1], [Sz, Sz])
  ham_X += prod(N, [i], [Sx])
ham_X += prod(N, [N-1], [Sx])
ham_ZZ *= J

########################################################################################

seed = 83948
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key)

########################################################################################

num_epochs = 2000
architecture = [2**10, 500, 20, 500, 2**10] 
hyper_param = '_'.join(str(e) for e in architecture)
# file_path = f'../models/tfimAutoEncoder_ssb_nnParams_adam_epoch{num_epochs}_layersP{hyper_param}.pickle'
file_path = f'../models/tfimAutoEncoder_latentnorm_ssb_nnParams_adam_epoch{num_epochs}_layersP{hyper_param}.pickle'
with open(file_path, 'rb') as file:
    model_params = pickle.load(file)

########################################################################################

file_path = f'../models/tfimdmrg_AutoEncoder_latentnorm_ssb_componentMeans_adam_epoch{num_epochs}_layersP{hyper_param}_latentmeans.pickle'
with open(file_path, 'rb') as file:
    latent_ssb = pickle.load(file)
print(latent_ssb.shape)

########################################################################################

epochs = 1000
lr = 0.1
drop_p = 0.1

params = model_params

# define optimizer 
opt = optax.adam(learning_rate=lr)
opt_state = opt.init(h)  # always the same pattern - handling state externally

h_list = []
loss_list = []

## gradient descent to find h that minimizes the loss function
for i in range(epochs):
    h, loss_new = update(h, ham_X, ham_ZZ, latent_ssb, params, opt_state, opt, drop_p, key)
    loss_list.append(loss_new)
    h_list.append(h)
    if i%25 == 0:
        print('epoch=',i, 'loss=',loss_new, 'h_new=',h)

########################################################################################

plt.figure()
plt.plot(loss_list, 'o', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(f'../figures/tfim_autoDoff_autoEncoder_latentnorm_es_ssb2para_loss_sgd_epoch{num_epochs}_layersP{hyper_param}.pdf', bbox_inches='tight')

plt.figure()
plt.plot(h_list, 'o', color='blue')
plt.xlabel('Epochs')
plt.ylabel('h')
plt.savefig(f'../figures/tfim_autoDoff_autoEncoder_latentnorm_es_ssb2para_h_sgd_epoch{num_epochs}_layersP{hyper_param}.pdf', bbox_inches='tight')

########################################################################################
