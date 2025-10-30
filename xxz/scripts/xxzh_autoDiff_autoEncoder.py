import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap, block_until_ready, pure_callback
from jax import debug

import pickle
import numpy as np
import matplotlib.pyplot as plt

from functools import partial

import optax

from jax import config
config.update("jax_enable_x64", True)

# jax.config.update("jax_debug_nans", True) 

# import os, importlib
# os.environ["JAX_DISABLE_JIT"] = "1"
# import importlib, sys
# importlib.reload(sys.modules['jax'])  

########################################################################################

# using the learnt (ferro) auto-encoder to learn a different phases of the xxz model.
# H_xxz = J * sum(sigma_x[i] * sigma_x[i+1] + sigma_y[i] * sigma_y[i+1] + delta * sigma_z[i] * sigma_z[i+1])
# specifically, we start with one phase of tfim (ssb) and ask the learnt auto-encoder to go to the other phase.
# auto-differentiate the parameter delta to go through different phases

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

Sz = jnp.zeros([2,2], dtype=complex)
Sz = Sz.at[0,0].set(1.)
Sz = Sz.at[1,1].set(-1.)

Sx = jnp.zeros([2,2], dtype=complex)
Sx = Sx.at[0,1].set(1.)
Sx = Sx.at[1,0].set(1.)

Sy = jnp.zeros([2,2], dtype=complex)
Sy = Sy.at[0,1].set(-1j)
Sy = Sy.at[1,0].set(1j)

########################################################################################

# @jit
def H_xxzh(delta, h, ham_xx, ham_yy, ham_zz, ham_x, J=1):
    H_mat = (ham_xx + ham_yy + ham_zz * delta) * J + h * ham_x
    return H_mat

# @jit
def gd_solver_ed(delta, h, ham_xx, ham_yy, ham_zz, ham_x):
    H = jnp.real(H_xxzh(delta, h, ham_xx, ham_yy, ham_zz, ham_x)).astype(jnp.float64)

    # sigma = 1e-8
    # eps = sigma * jax.random.normal(key, H.shape[-1:], dtype=H.dtype)
    # eps = jax.lax.stop_gradient(eps)
    # # print(eps.shape)

    # H += jnp.diag(eps)

    e, v = jnp.linalg.eigh(H)
    return e, v

# def cal_m(obs_val):
#     # every third observable data is the z_i expectation value
#     if obs_val.ndim == 1:
#         return np.mean(obs_val[2::3])
#     else:
#         return np.mean(obs_val[:, 2::3], axis=1)

def cal_m(v):
    m_tot = 0
    N = int(np.log2(len(v)))
    for i in range(len(v)):
        state = hslabeltoocc(i, N)
        num_up = jnp.count_nonzero(state)
        m = abs(2*num_up - N)
        m_tot += m * (jnp.conj(v[i])*v[i])
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

    norm = jnp.linalg.norm(latent_representation)
    return latent_representation / norm

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

    # x_out = acti_loss(x_out)   ## normalize the output
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


# def cal_obs(v, obs_list):
#     obs_stack = jnp.stack(obs_list) 
#     expvals = jax.vmap(lambda O: jnp.vdot(v, O @ v))(obs_stack)
#     return jnp.real(expvals) 


# def loss_obs(delta, ham_xx, ham_yy, ham_zz, latent_ferro, obs_list, params, drop_p, rng_key):
#     _, v = gd_solver_ed(delta, ham_xx, ham_yy, ham_zz)
#     obs_value_arr = cal_obs(v[:,0], obs_list)

#     latent = fetch_latent(params, obs_value_arr, drop_p, rng_key)
#     l = (latent - latent_ferro)**2
#     l_sum = jnp.sum(l)  

#     return -1 * jnp.sqrt(l_sum) / len(l)


# def update_obs(delta, ham_xx, ham_yy, ham_zz, latent_ferro, obs_list, params, opt_state, opt, drop_p, rng_key):
#     value, grads = value_and_grad(loss_obs, argnums=0)(delta, ham_xx, ham_yy, ham_zz, latent_ferro, obs_list, params, drop_p, rng_key)  
#     # print('grads:', grads)
#     updates, opt_state = opt.update(grads, opt_state)
#     delta_new = optax.apply_updates(delta, updates)
#     return delta_new, value


# @jit
def loss_es(ham_param, ham_xx, ham_yy, ham_zz, ham_x, latent_ferro, params, drop_p, rng_key):
    delta, h = ham_param[0], ham_param[1]
    _, v = gd_solver_ed(delta, h, ham_xx, ham_yy, ham_zz, ham_x)
    latent = fetch_latent(params, v[:,0].real, drop_p, rng_key)
    l = (latent - latent_ferro)**2
    l_sum = jnp.sum(l)  # Directly sum the squared differences
    return -1 * jnp.sqrt(l_sum) / len(l)  # Compute the mean sqrt of l and then apply the minus sign


#@jit
def update(ham_param, ham_xx, ham_yy, ham_zz, ham_x, latent_ferro, params, opt_state, opt, drop_p, rng_key):
    value, grads = value_and_grad(loss_es, argnums=0)(ham_param, ham_xx, ham_yy, ham_zz, ham_x, latent_ferro, params, drop_p, rng_key)  
    # print(grads)
    updates, opt_state = opt.update(grads, opt_state)
    ham_param = optax.apply_updates(ham_param, updates)
    return ham_param, value

########################################################################################

N = 10

ham_xx = prod(N,[0,1],[Sx,Sx])
ham_yy = prod(N,[0,1],[Sy,Sy])
ham_zz = prod(N,[0,1],[Sz,Sz])
ham_x = prod(N, [0], [Sx])

for i in range(1,N-1):
    ham_xx += prod(N, [i,i+1], [Sx, Sx])
    ham_yy += prod(N, [i,i+1], [Sy, Sy])
    ham_zz += prod(N, [i,i+1], [Sz, Sz])
    ham_x += prod(N, [i], [Sx])
ham_x += prod(N, [N-1], [Sx]) 

########################################################################################

obs_list = []

for i in range(10):
    obs_list.append(prod(N, [i], [Sx]))
    obs_list.append(prod(N, [i], [Sy]))
    obs_list.append(prod(N, [i], [Sz]))

########################################################################################

seed = 83948
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key)

########################################################################################

# load the learnt auto-encoder parameters

num_epochs = 10000
architecture = [2**10, 256, 10, 256, 2**10] 

hyper_param = '_'.join(str(e) for e in architecture)
# file_path = f'../models/xxzAutoEncoder_ferroNonDebias_nnParams_epoch{num_epochs}_layersP{hyper_param}.pickle'
file_path = f'../models/xxzhAutoEncoder_ferrodb_nnParams_epoch{num_epochs}_layersP{hyper_param}.pickle'
# file_path = f'../models/xxzAutoEncoder_ferro_obs_nnParams_epoch{num_epochs}_layersP{hyper_param}.pickle'
with open(file_path, 'rb') as file:

    model_params = pickle.load(file)
# print('model_params datatype:', model_params[0][0].dtype) 

########################################################################################

# load the learnt latent representation of the ferro phase
file_path = f'../models/xxzhAutoEncoder_ferrodb_latent_componentMeans_epoch{num_epochs}_layersP{hyper_param}.pickle'
# file_path = f'../models/xxzAutoEncoder_ferroNonDebias_latent_componentMeans_epoch{num_epochs}_layersP{hyper_param}.pickle'
# file_path = f'../models/xxzAutoEncoder_ferro_obs_latent_componentMeans_epoch{num_epochs}_layersP{hyper_param}.pickle'
with open(file_path, 'rb') as file:
    latent_ferro = pickle.load(file)
# print('latent ferro datatype', latent_ferro.dtype)

########################################################################################

epochs = 10000
lr = 1e-4
drop_p = 0.1

params = model_params

J = jnp.asarray(1.) 
delta0 = -1.5
h0 = 0.3
ham_param = jnp.array([delta0, h0])
print(f"initial parameters: delta={delta0}, h={h0}")

# define optimizer 
opt = optax.adam(learning_rate=lr)    ## AdamW, LBFGS, SGD (momentum), etc.
opt_state = opt.init(ham_param)  # always the same pattern - handling state externally

delta_list = []
h_list = []
loss_list = []

# loss0 = loss_es(delta, ham_xx, ham_yy, ham_zz, latent_ferro,
#                 params, 0.0, key)
# loss1 = loss_es(delta+1e-5, ham_xx, ham_yy, ham_zz, latent_ferro,
#                 params, 0.0, key)
# print("finite diff  :", (loss1 - loss0)/ 1e-5)
# print("jax grad     :", jax.grad(loss_es, argnums=0)(delta, ham_xx, ham_yy, ham_zz, latent_ferro,
#                 params, 0.0, key))


## gradient descent to find h that minimizes the loss function
for i in range(epochs):
    ham_param, loss_new = update(ham_param, ham_xx, ham_yy, ham_zz, ham_x, latent_ferro, params, opt_state, opt, drop_p, key)
    # delta, loss_new = update_obs(delta, ham_xx, ham_yy, ham_zz, latent_ferro, obs_list, params, opt_state, opt, drop_p, key)
    loss_list.append(loss_new)
    delta_list.append(ham_param[0])
    h_list.append(ham_param[1])
    if i%25 == 0:
        print('epoch=',i, 'loss=',loss_new, 'delta_new=',ham_param[0], 'h_new=', ham_param[1])


########################################################################################

file_name = f'../data/xxzhAutoDiff_autoEncoder_es_hamParam_epoch{num_epochs}_layersP{hyper_param}.pickle'
with open(file_name, 'wb') as file:
    pickle.dump({'delta': delta_list, 'h': h_list}, file)

########################################################################################

plt.figure()
plt.plot(loss_list, 'o-', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(f'../figures/xxzhAutoDiff_autoEncoder_es_loss_delta0{delta0}_h0{h0}_epoch{num_epochs}_layersP{hyper_param}.pdf', bbox_inches='tight')

plt.figure()
plt.plot(delta_list, 'o-', color='blue')
plt.xlabel('Epochs')
plt.ylabel(r'$\Delta$')
plt.savefig(f'../figures/xxzhAutoDiff_autoEncoder_es_delta_delta0{delta0}_h0{h0}_epoch{num_epochs}_layersP{hyper_param}.pdf', bbox_inches='tight')

plt.figure()
plt.plot(h_list, 'o-', color='blue')
plt.xlabel('Epochs')
plt.ylabel(r'$h$')
plt.savefig(f'../figures/xxzhAutoDiff_autoEncoder_es_h_delta0{delta0}_h0{h0}_epoch{num_epochs}_layersP{hyper_param}.pdf', bbox_inches='tight')

########################################################################################
