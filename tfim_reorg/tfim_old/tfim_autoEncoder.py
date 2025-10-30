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

## font size of the plot
fs = 15

"""""""""""""""""""""
 part 4: 
 Compress the wavefunction from a ssb phase of the transverse Ising model with auto-encoder 
 and then re-represent the wavefunction. It has the following architecture,
 o              o
    o       o
 o      o       o
    o       o
 o              o
 Loss function: L = sqrt(||x_in - x_out||_2) / (# of samples)
"""""""""""""""""""""


""" load data """
## label: param: 1; ssb: 0
## training size: 1191
## testing size: 800
## data structure: {'data':[], 'label': int}

with open('./tfim_test/ssb_unbias.pickle', 'rb') as f:
  data_ssb = pickle.load(f)

# print('data size:',len(data_ssb))
# print('sample:', data_ssb[0])

# print(data_ssb[-1]['h'])
# print(data_ssb[-1]['gd'][0])

""" organize the data """
## separate the data and the label
## make sure all of them are arrays so that it is easy to perform calculation

random.seed(10)

x_dim = len(data_ssb[0]['wf'])

size_data = len(data_ssb)
size_train = 1000
# size_test = size_data - size_train

num_train = random.sample(range(0,size_data), size_train)

load_order = np.arange(0,size_data,1)
random.shuffle(load_order)

# x_train = np.zeros((size_train, x_dim))
# x_test = np.zeros((size_test, x_dim))

# h_train = np.zeros(size_train)
# h_test = np.zeros(size_test)

x_train = []
h_train = []

for i in num_train:
    x_train.append(data_ssb[i]['wf'])
    h_train.append(data_ssb[i]['h'])
x_train = np.array(x_train)
print(x_train.shape)

# load a subset of ssb data 
# x_train = []
# h_train = []
# for i in load_order:
#     # if cal_m(data_ssb[i]['gd'][0]) < 0.8: continue
#     if data_ssb[i]['h'] > -0.1: continue
#     sign_data = np.sign(data_ssb[i]['gd'][0][0])
#     x_train.append(data_ssb[i]['gd'][0]*sign_data)
#     h_train.append(data_ssb[i]['h'])
# x_train = np.array(x_train)
# print(x_train.shape)


# print('shape of the trainning data', x_train.shape)
# print('shape of the testing data', x_test.shape)
# print()
# print(x_train[0])
# print(cal_m(x_train[0]))

# m_list = []
# for v in x_train:
#   m_list.append(cal_m(v))
# plt.plot(m_list, 'o')
# plt.xlabel('loading order',fontsize=fs)
# plt.ylabel('<m>', fontsize=fs)
# plt.savefig('train_m.pdf', bbox_inches='tight')
# plt.close()


""" initialize the auto-encoder """
## define the auto-encoder network depths and width and initialize a network

## define seed 
seed = 0
key = jax.random.PRNGKey(seed)

def init_params(layer_widths, parent_key, scale=0.01):
  params = []
  keys = jax.random.split(parent_key, num=(len(layer_widths)-1))

  for in_width, out_width, key in zip(layer_widths[:-1], layer_widths[1:], keys):
    weight_key, bias_key = jax.random.split(key)
    params.append([scale * jax.random.normal(weight_key, shape=(in_width, out_width)),
                   scale * jax.random.normal(bias_key, shape=(out_width,))
                  ])
  return params

## auto-encoder network architecutre
layer_widths = [x_dim, 1000, 1, 1000, x_dim] 
initial_MLP_params = init_params(layer_widths, key)
print(len(initial_MLP_params))
print([(x[0].shape, x[1].shape) for x in initial_MLP_params])


""" define funcions """

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
  # use a modified softmax to normalize the x_out
  def acti_loss(x):
    dn = x**2
    dn = jnp.einsum('ij->i', dn)
    return x / jnp.sqrt(dn)[:, np.newaxis]
  x_out = acti_loss(x_out)

  return x_out, jnp.mean(latent_prm)


def loss(params, x):
  x_out, latent_prm = reconstruct(params, x)
  l = (x_out - x)**2
  l = jnp.einsum('ij->i', l)
  return jnp.mean(jnp.sqrt(l))
  

# @jit
def update(params, x, lr, opt_state, opt_sgd):
  value, grads = jit(value_and_grad(loss, argnums=0))(params, x)
#   print('loss=', value)
  updates, opt_state = opt_sgd.update(grads, opt_state)
  params_new = optax.apply_updates(params, updates)
#   params_new = jax.tree_map(lambda p, g: p-lr*g, params, grads)
  return params_new, value


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



""" train """

lr = 0.1
num_epochs = 10000

params = initial_MLP_params

""" define optimizer """
opt_sgd = optax.sgd(learning_rate=lr)
opt_state = opt_sgd.init(params)  # always the same pattern - handling state externally

## record the value of loss function during training 
loss_list = []

for i in range(num_epochs):
#   if i == 1500: lr = 0.05
#   if i == 3000: lr = 0.01
#   if i == 4000: lr = 0.005
  params, loss_new = update(params, x_train, lr, opt_state, opt_sgd)
  loss_list.append(loss_new)
  print('epoch=',i, 'loss=',loss_new)

  ## check if need to update the learning rate
#   if i > 600:
#     if (sum(loss_list[-10:])/10 - loss_new) < 0.01:
#         lr = lr/2.
#         print('new learning rate:', lr)


""" write down the trained auto encoder parameters """
""" write down the loss_epoch, h_train, latent_prm_h, l_h, m_rec_h, m_ori_h """

## latent parameter vs h
x_out, latent_prm = reconstruct(params, x_train)

## reconstruction loss vs h
l= (x_out - x_train)**2
l = jnp.einsum('ij->i', l)
# print(l.shape)
l = jnp.sqrt(l)

# plt.plot(loss_list, 'o')
# plt.show()

## magnetization vs h
m_list_train = cal_m_reconstructed(x_out)
m_list = cal_m_reconstructed(x_train)

hyper_param = '_'.join(str(e) for e in layer_widths)
with open('./tfim_test/data/tfimAutoEncoder_ssb_nnParams_trainSize%d_%s_epoch%d.pickle'%(size_train, hyper_param, num_epochs),'wb') as f:
    pickle.dump({'model': list(params), 
    'loss_epoch': loss_list, 
    'h_list': h_train,
    'latent_prm_h': latent_prm,
    'loss_h': l,
    'm_rec_h': m_list_train,
    'm_ori_h': m_list
    }, 
    f)





