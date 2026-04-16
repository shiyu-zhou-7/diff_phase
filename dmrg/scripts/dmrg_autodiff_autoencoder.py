import jax
import jax.numpy as np   ## to be consistent with the dmrg code
from jax import grad, value_and_grad, jit, vmap, block_until_ready, pure_callback

import pickle
import numpy as nnp
import matplotlib.pyplot as plt

from functools import partial

import optax

from dmrg_tfim_mps_jax import *

from jax import config
config.update("jax_enable_x64", True)

########################################################################################

def dmrg_E(L,J,g,conf=1e-4,test=False,chi_max=10, max_sweep = 5):
    # Initialize random MPS and MPO
    # chi_max: max bond dimension - b/c we are using jax, we are limited to relatively small chi; 
    # setting to 10 for now
    init_mps = random_MPS(L=L,d=2,chi_max=chi_max)
    mpo = Ising_MPO(L=L,d=2,h=g,J=J)
    dmrg = DMRG(init_mps,mpo,eps=-1.,chi_max=chi_max,test=False)
    
    diff = 5.*conf
    counter = 0
    while diff > conf:
        if test:
            print(counter)
        counter += 1
        ### note: we could also simply use the current dmrg.e value to speed things up. This is just to give justification to the expect_mpo(mps,mpo) function
        e_in = expect_mpo(dmrg.MPS,mpo)
        dmrg.left_to_right()
        dmrg.right_to_left()
        e_end = expect_mpo(dmrg.MPS,mpo)
        diff = abs(e_in - e_end)
        # print(f'sweep: {counter}, E: {dmrg.e[-1]}')
        if counter > max_sweep:
            break
    return dmrg.MPS, dmrg.e[-1]

def cal_observables(mps, operator_list):
    expect_list = []
    for mpo in operator_list:
        expect_list.append(np.real(expect_mpo(mps, mpo)))
    return expect_list

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
        activation = jax.nn.relu(np.dot(activation, w) + b)
        rng_key, subkey = jax.random.split(rng_key)
        activation = dropout(activation, drop_p, subkey)
    w, b = params[-1]
    latent_representation = np.dot(activation, w) + b
    return latent_representation

#@jit
def decoder(params, latent_rep, drop_p, rng_key):
    activation = latent_rep
    for w, b in params[0:-1]:
        activation = jax.nn.relu(np.dot(activation, w) + b)
        rng_key, subkey = jax.random.split(rng_key)
        activation = dropout(activation, drop_p, subkey)
    x_out = np.dot(activation, params[-1][0]) + params[-1][1]
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
        dn = np.sum(x**2)  # Sum the squares of all elements in the vector
        return x / np.sqrt(dn)  # Normalize the vector

    x_out = acti_loss(x_out)   ## normalize the output
    return x_out

def fetch_latent(params, x, drop_p, rng_key):
    # assuming symmetric autoencoder
    mid_index = len(params) // 2

    rng_key, subkey = jax.random.split(rng_key)
    latent_rep = encoder(params[0:mid_index], x, drop_p, subkey)
    return latent_rep

########################################################################################

# @jit
def loss(g, L, operator_list, latent_ssb, params, drop_p, rng_key, chi_max=10):
    mps, _ = dmrg_E(L, np.asarray(1), g, conf=1e-4, test=False, chi_max=chi_max, max_sweep=5)
    exp_list = np.asarray(cal_observables(mps, operator_list))
    latent = fetch_latent(params, exp_list, drop_p, rng_key)
    l = (latent - latent_ssb)**2
    l_sum = np.sum(l)  # directly sum the squared differences
    return -1 * np.sqrt(l_sum) / len(l)  # compute the mean sqrt of l and then apply the minus sign

def update(g, L, operator_list, latent_ssb, params, opt_state, opt, drop_p, rng_key):
    value, grads = value_and_grad(loss, argnums=(0))(g, L, operator_list, latent_ssb, params, drop_p, rng_key)
    updates, opt_state = opt.update(grads, opt_state)
    g_new = optax.apply_updates(g, updates)
    return g_new, value

########################################################################################

L = 50

########################################################################################

sxx = np.array([[0., 1.], [1., 0.]])
syy = np.array([[0., -1j], [1j, 0.]])
szz = np.array([[1., 0.], [0., -1.]])

# pauli_string: 0,1,2,3 for I, X, Y, Z
# for example: 3000.., means ZIII...

operator_list= []
for i in range (L):
    mpo = Pauli_MPO(L=L, d=2,pauli_string = '0'*(i)+'1'+'0'*(L-i-1))
    operator_list.append(mpo)
for i in range (L):
    mpo = Pauli_MPO(L=L,d=2,pauli_string = '0'*(i)+'3'+'0'*(L-i-1))
    operator_list.append(mpo)

########################################################################################

seed = 89435234
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key)

########################################################################################

num_epochs = 10000
lr = 5e-5
architecture = [100, 25, 5, 25, 100] 
hyper_param = '_'.join(str(e) for e in architecture)
file_path = f'../models/tfimdmrg_AutoEncoder_latentnorm_ssb_nnParams_adam_epoch{num_epochs}_lr{lr}_layersP{hyper_param}.pickle'
with open(file_path, 'rb') as file:
    model_params = pickle.load(file)

########################################################################################

file_path = f'../models/tfimdmrg_AutoEncoder_latentnorm_ssb_componentMeans_adam_epoch{num_epochs}_layersP{hyper_param}_latentmeans.pickle'
with open(file_path, 'rb') as file:
    latent_ssb = pickle.load(file)
# print(latent_ssb.shape)

########################################################################################

epochs = 100
lr = 0.05
drop_p = 0.

g = np.asarray(0.2)
print(type(g))

params = model_params

# define optimizer 
opt = optax.adam(learning_rate=lr)
opt_state = opt.init(g)  

g_list = []
loss_list = []

## gradient descent to find h that minimizes the loss function
for i in range(epochs):
    # print(f'-----training setp {i+1}-----')
    g, loss_new = update(g, L, operator_list, latent_ssb, params, opt_state, opt, drop_p, key)
    loss_list.append(loss_new)
    g_list.append(g)
    if i%25 == 0:
        print('epoch=',i, 'loss=',loss_new, 'g_new=',g)

########################################################################################

# save the data
data = {
    'g_list': g_list,
    'loss_list': loss_list,
}
with open(f'tfimdmrg_autoDoff_autoEncoder_latentnorm_ssb2para   trajectory_epoch{epochs}_layersP{hyper_param}.pickle', 'wb') as f:
    pickle.dump(data, f)

########################################################################################

# plt.figure()
# plt.plot(loss_list, 'o', color='blue')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.savefig(f'../figures/tfimdmrg_autoDoff_autoEncoder_latentnorm_ssb2para_loss_epoch{num_epochs}_layersP{hyper_param}.pdf', bbox_inches='tight')

# plt.figure()
# plt.plot(g_list, 'o', color='blue')
# plt.xlabel('Epochs')
# plt.ylabel('h')
# plt.savefig(f'../figures/tfimdmrg_autoDoff_autoEncoder_latentnorm_ssb2para_h_epoch{num_epochs}_layersP{hyper_param}.pdf', bbox_inches='tight')

########################################################################################

gs = g_list

fig, ax = plt.subplots()
ax.plot(range(epochs), gs, 'o-', color='blue', markersize=2, lw=1)

ax.set_xlabel("Epoch step", fontsize=14)
ax.set_ylabel(r"$g$", fontsize=14)
if len(gs) > 0:
    g_min, g_max = min(gs), max(gs)
    g_range = g_max - g_min
    padding = 0.1 * g_range if g_range > 0 else 0.1
    ax.set_ylim(g_min - padding, g_max + padding)
ax.set_title("Hamiltonian Parameter Trajectory", fontsize=15)
ax.grid(True)
plt.tight_layout()
plt.savefig(f'../figures/dmrg_tfim_L50_trajectory_g.3.pdf', bbox_inches='tight')

