import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap, block_until_ready, pure_callback

import pickle
import numpy as np
import random
import matplotlib.pyplot as plt

from scipy.stats import bernoulli
from functools import partial
import optax

from jax import config
config.update("jax_enable_x64", True)


""" load data """
with open('./tfim_test/ssb_unbias.pickle', 'rb') as f:
  data_ssb = pickle.load(f)
# print('size of the data:', len(data_ssb))

""" organize the data """
random.seed(10)

x_dim = len(data_ssb[0]['wf'])
size_data = len(data_ssb)
size_train = 1000
size_test = 200

num_train = random.sample(range(0,size_data), size_train)
load_order = random.sample(range(0,size_data), size_test)

x_test = []
h_test = []

for i in load_order:
    if i in num_train: continue
    x_test.append(data_ssb[i]['wf'])
    h_test.append(data_ssb[i]['h'])
x_test = np.array(x_test)
print(x_test.shape)


""" load the model """
size_train = 1000
num_epochs = 10000
architecture = [2**10, 1000, 1, 1000, 2**10] 
hyper_param = '_'.join(str(e) for e in architecture)
with open('./tfim_test/data/tfimAutoEncoder_ssb_nnParams_trainSize%d_%s_epoch%d.pickle'%(size_train, hyper_param, num_epochs), 'rb') as f:
    data = pickle.load(f)
params = data['model']


""" helper functions """
def reconstruct(params, x):
    # input
    activation = x

    """ encoder """
    for w, b in params[0:1]:
        activation = jax.nn.relu(jnp.dot(activation, w) + b)

    """ latent space """
    w, b = params[1]
    activation = jnp.dot(activation, w) + b
    latent_prm = activation

    """ decoder """
    for w, b in params[2:-1]:
        activation = jax.nn.relu(jnp.dot(activation, w) + b)

    ## neural network 
    #   latent_prm = 0
    #   for w, b in params[:-1]:
    #     activation = jax.nn.relu(jnp.dot(activation, w) + b)  

    """ output """
    x_out = jnp.dot(activation, params[-1][0]) + params[-1][1]

    return x_out, latent_prm

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

## average magnetization of reconstructed wavefunction
def cal_m_reconstructed(x):
    m_list = []
    for v in x:
        m_list.append(cal_m(v))
    return m_list

def loss(params, x):
  x_out, latent_prm = reconstruct(params, x)
  l = (x_out - x)**2
  l = jnp.einsum('ij->i', l)
  return jnp.sqrt(l)


""" evaluate the model """
recon_x, latent_prm = reconstruct(params, x_test)
m_list_ori = cal_m_reconstructed(x_test)
m_list = cal_m_reconstructed(recon_x)
l_list = loss(params, x_test)


""" plot """
fs = 14
name_dir = './tfim_test/figures/tfimAutoEncoder_trainSize%d_ssb_%s_epoch%d'%(size_train, hyper_param, num_epochs)

## latent parameter vs h
plt.figure()
plt.plot(h_test, latent_prm, 'x', markersize=3)
plt.ylabel('latent parameter',fontsize=fs)
plt.xlabel('h',fontsize=fs)
plt.savefig(name_dir+'/tfim_ae_test_latent_vs_h.pdf', bbox_inches='tight')
plt.close()

## reconstruction loss vs h
plt.figure()
plt.plot(h_test, l_list, 'x', markersize=3)
plt.ylabel('loss',fontsize=fs)
plt.xlabel('h',fontsize=fs)
plt.savefig(name_dir+'/tfim_ae_test_loss_vs_h.pdf', bbox_inches='tight')
plt.close()

## magnetization vs h
plt.figure()
plt.plot(h_test, m_list, 'o', label='rec')
plt.plot(h_test, m_list_ori, 'o', label='ori')
plt.legend(fontsize=fs)
plt.ylabel('<m>',fontsize=fs)
plt.xlabel('h',fontsize=fs)
plt.savefig(name_dir+'/tfim_ae_test_m.pdf', bbox_inches='tight')
plt.close()






