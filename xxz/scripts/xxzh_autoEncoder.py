import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap, block_until_ready, pure_callback

import pickle
import numpy as np
import matplotlib.pyplot as plt

import optax

from functools import partial

########################################################################################

# use auto-encoder structure to learn the ferromagnetic phase of the xxz+h in x direction model

########################################################################################

## font size of plots
fs = 15

########################################################################################

from jax import config
config.update("jax_enable_x64", True)

########################################################################################

np.random.seed(45297)

with open('../data/data_ferro_xxzh_debiased.pkl', 'rb') as f:
  data = np.array(pickle.load(f))
np.random.shuffle(data)

test_size = int(len(data)*0.3)
test_indices = np.random.choice(len(data), test_size, replace=False)

data_test = data[test_indices]
data_train = np.delete(data, test_indices)

x_train = np.array([d['v'].real for d in data_train])
param_train = np.array([d['param'] for d in data_train])

x_test = np.array([d['v'].real for d in data_test])
param_test = np.array([d['param'] for d in data_test])

print('all data shapes: ', x_train.shape, param_train.shape, x_test.shape, param_test.shape)
print(f'training size is {len(data_train)} and testing size is {len(data_test)}')

########################################################################################

# initialize the auto-encoder
# define the auto-encoder network depths and width and initialize a network

def init_params(layer_widths, parent_key, scale=0.01):
  params = []
  keys = jax.random.split(parent_key, num=(len(layer_widths)-1))

  for in_width, out_width, key in zip(layer_widths[:-1], layer_widths[1:], keys):
    weight_key, bias_key = jax.random.split(key)
    params.append([scale * jax.random.normal(weight_key, shape=(in_width, out_width)),
                   scale * jax.random.normal(bias_key, shape=(out_width,))
                  ])
  return params

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


## average magnetization of a wavfunction
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

########################################################################################

def dropout(x, drop_p, rng_key):
    keep_prob = 1.0 - drop_p
    bernoulli_list = jax.random.bernoulli(rng_key, keep_prob, shape=x.shape)
    x = x * bernoulli_list
    return x


def encoder(params, x, drop_p, rng_key):
    activation = x
    for w, b in params[0:-1]:
        activation = jax.nn.relu(jnp.dot(activation, w) + b)
        rng_key, subkey = jax.random.split(rng_key)
        activation = dropout(activation, drop_p, subkey)
    w, b = params[-1]
    latent_representation = jnp.dot(activation, w) + b

    # normalize the latent representation to unit norm
    norm = jnp.linalg.norm(latent_representation, axis=1, keepdims=True)
    return latent_representation / norm


def decoder(params, latent_rep, drop_p, rng_key):
    activation = latent_rep
    for w, b in params[0:-1]:
        activation = jax.nn.relu(jnp.dot(activation, w) + b)
        rng_key, subkey = jax.random.split(rng_key)
        activation = dropout(activation, drop_p, subkey)
    x_out = jnp.dot(activation, params[-1][0]) + params[-1][1]
    return x_out


def autoencoder(params, x, drop_p, rng_key):
    # assuming symmetric autoencoder
    mid_index = len(params) // 2

    rng_key, subkey = jax.random.split(rng_key)
    latent_rep = encoder(params[0:mid_index], x, drop_p, subkey)

    rng_key, subkey = jax.random.split(rng_key)
    x_out = decoder(params[mid_index:], latent_rep, drop_p, subkey)

    def acti_loss(x):
      dn = x**2
      dn = jnp.einsum('ij->i', dn)
      return x / jnp.sqrt(dn)[:, np.newaxis]
    
    # normalize the output wavefunction to unit norm
    x_out = acti_loss(x_out)   
    return x_out


def loss(params, x, drop_p, rng_key):
    x_out = autoencoder(params, x, drop_p, rng_key)
    l = (x_out - x)**2
    l = jnp.einsum('ij->i', l)
    return jnp.abs(jnp.mean(jnp.sqrt(l)))**2


def update(params, x, opt_state, opt_sgd, drop_p, rng_key):
    value, grads = jit(value_and_grad(loss, argnums=0))(params, x, drop_p, rng_key)
    updates, opt_state = opt_sgd.update(grads, opt_state)
    params_new = optax.apply_updates(params, updates)
    # params_new = jax.tree_map(lambda p, g: p - 0.1*g, params, grads)
    return params_new, value


## average magnetization of reconstructed wavefunction
def cal_m_reconstructed(x):
    m_list = []
    for v in x:
        m_list.append(cal_m(v))
    return m_list

########################################################################################

seed = 83948
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key)

layer_widths = [x_train.shape[-1], 256, 10, 256, x_train.shape[-1]] 
print('layers:', layer_widths)
initial_MLP_params = init_params(layer_widths, subkey)

########################################################################################

lr = 1e-4
num_epochs = 10000

drop_p = 0.1

params = initial_MLP_params

## define optimizer 
opt = optax.adam(learning_rate=lr)
opt_state = opt.init(params)  # always the same pattern - handling state externally

## record the value of loss function during training 
loss_list = []
m_list = []

for i in range(num_epochs):
  params, loss_new = update(params, x_train, opt_state, opt, drop_p, key)
  loss_list.append(loss_new)
  if i % 100 == 0:
    m_list = cal_m_reconstructed(autoencoder(params, x_test, 0, key))
    print('epoch=',i, 'loss=',loss_new,
          'val_loss=', loss(params, x_test, 0, key), 
          'magnetization=', sum(m_list) / len(m_list)
            )
    
########################################################################################

# save the trained parameters

hyper_param = '_'.join(str(e) for e in layer_widths)
with open(f'../models/xxzhAutoEncoder_ferrodb_nnParams_epoch{num_epochs}_layersP{hyper_param}.pickle','wb') as f:
    pickle.dump(params, f)

########################################################################################

# save the training loss curve

plt.figure()
plt.plot(loss_list, 'o', color='blue')
plt.xlabel('epoch', fontsize=fs)
plt.ylabel('loss', fontsize=fs)
plt.yscale('log')
plt.savefig(f'../figures/xxzhAutoEncoder_ferrodb_trainingloss_epoch{num_epochs}_layersP{hyper_param}.pdf', bbox_inches='tight')



########################################################################################################################################################################
########################################################################################################################################################################

# TESTING   


# load the trained parameters for testing

hyper_param = '_'.join(str(e) for e in layer_widths)
params = pickle.load(open(f'../models/xxzhAutoEncoder_ferrodb_nnParams_epoch{num_epochs}_layersP{hyper_param}.pickle', 'rb'))

########################################################################################

print(f'length of the testing set, {len(x_test)}')
x_rc = autoencoder(params, x_test, 0, key)   ## testing mode, set drop_p = 0
m_rc = cal_m_reconstructed(x_rc)
m_test = cal_m_reconstructed(x_test)

plt.figure()
plt.plot(range(len(m_test)), m_test, 'o', color='blue', label=f'<m>')
plt.plot(range(len(m_test)), m_rc, 'o', color='orange', label=f'<m>_rec')
plt.legend(fontsize=fs-2)
plt.xlabel('delta', fontsize=fs)
plt.ylabel('<m>', fontsize=fs)
plt.savefig(f'../figures/xxzhAutoEncoder_ferrodb_reconstructed_magnetization_epoch{num_epochs}_layersP{hyper_param}.pdf', bbox_inches='tight')

########################################################################################

## latent space study
def fetch_latent(params, x, drop_p, rng_key):
    # assuming symmetric autoencoder
    mid_index = len(params) // 2

    rng_key, subkey = jax.random.split(rng_key)
    latent_rep = encoder(params[0:mid_index], x, drop_p, subkey)
    return latent_rep

def pairwise_distances(vectors):
    norm_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    # Calculate the difference matrix, and the resulting shape will be (n, n, d)
    diff = norm_vectors[:, np.newaxis, :] - norm_vectors[np.newaxis, :, :]
    sq_diff = diff ** 2
    sum_sq_diff = np.sum(sq_diff, axis=2)
    distances = np.sqrt(sum_sq_diff)

    # Get the indices for the upper triangle, excluding the diagonal
    rows, cols = np.triu_indices(n=distances.shape[0], k=1)
    upper_triangle_elements = distances[rows, cols]
    return upper_triangle_elements

########################################################################################

## component-wise mean of latent space
latent = fetch_latent(params, x_test, 0, key)
component_means = np.mean(latent, axis=0)

# save the component means of the latent representation of ferro states for later training
with open(f'../models/xxzhAutoEncoder_ferrodb_latent_componentMeans_epoch{num_epochs}_layersP{hyper_param}.pickle','wb') as f:
    pickle.dump(component_means, f)

########################################################################################

latent = fetch_latent(params, x_test, 0, key)
print('latent ferro shape:', latent.shape)

pair_diff = pairwise_distances(latent)
print('pairwise distance of ferro states', np.average(pair_diff))

plt.figure()
plt.hist(pair_diff, bins=50, color='blue', alpha=0.7)
plt.xlabel('pairwise distances', fontsize=fs)
plt.ylabel('frequency', fontsize=fs)
plt.savefig(f'../figures/xxzhAutoEncoder_ferrodb_trainlatent_distance_hist_epoch{num_epochs}_layersP{hyper_param}.pdf', bbox_inches='tight')

########################################################################################

# def stats_latent(latent):
#     norm_latent = latent / np.linalg.norm(latent, axis=1, keepdims=True)
#     return np.var(norm_latent, axis=0), np.std(norm_latent, axis=0)

# latent = fetch_latent(params, x_test, 0, key)
# latent = fetch_latent(params, x_train, 0, key)
# var_latent, std_latent = stats_latent(latent)
# print(np.mean(var_latent), np.mean(std_latent), var_latent.shape)

# plt.figure()
# plt.plot(var_latent, 'o', color='blue', label='variance')
# plt.xlabel('latent dimension', fontsize=fs)
# plt.ylabel('variance', fontsize=fs)
# plt.savefig(f'../figures/xxzAutoEncoder_ferro_trainlatent_variance_epoch{num_epochs}_layersP{hyper_param}.pdf', bbox_inches='tight')

#########################################################################################

## use the latent representation trained on ferro phase to see 
## if states from other phase would have the latent similar representation

## load other states data
with open('../data/data_neelY_xxzh.pkl', 'rb') as f:
# with open('../data/data_nlroX4_xxzh.pkl', 'rb') as f:
    data_neelY = np.array(pickle.load(f))
x_neelY = np.array([d['v'].real for d in data_neelY])
param_neelY = np.array([d['param'] for d in data_neelY])

## fetch latent representation of gapless states
latent_neelY = fetch_latent(params, x_neelY, 0, key)
print('latent neelY shape:', latent_neelY.shape)

## calculate pairwise distances in the latent space
pair_diff_neelY = pairwise_distances(latent_neelY)
print('pairwise distance of neelY states:', np.average(pair_diff_neelY))

plt.figure()
plt.hist(pair_diff_neelY, bins=50, color='blue', alpha=0.7)
plt.xlabel('pairwise distances', fontsize=fs)
plt.ylabel('frequency', fontsize=fs)
plt.savefig(f'../figures/xxzhAutoEncoder_neelY_latent_distance_hist_epoch{num_epochs}_layersP{hyper_param}.pdf', bbox_inches='tight')

#########################################################################################

## calculate the difference between the component-wise means of the latent representations
## between ferro and states from other phases

latent_ferro = fetch_latent(params, x_test, 0, key)
latent_neelY = fetch_latent(params, x_neelY, 0, key)

component_means_ferro = np.mean(latent_ferro, axis=0)
component_means_neelY = np.mean(latent_neelY, axis=0)

diff_component_means = abs(component_means_ferro - component_means_neelY)

print('component means ferro:', component_means_ferro, np.std(latent_ferro, axis=0))
print('component means neelY:', component_means_neelY, np.std(latent_neelY, axis=0))
print('difference in component means:', diff_component_means)
print('mse between component means:', np.mean(diff_component_means**2))

#########################################################################################

## calculate the mse between the latent representations of every ferro state and the mean

latent_ferro = fetch_latent(params, x_test, 0, key)
latent_neelY = fetch_latent(params, x_neelY, 0, key)
component_means = np.mean(latent_ferro, axis=0)
print(f'shape of latent ferro: {latent_ferro.shape}, shape of component means: {component_means.shape}, shape of latent neelY: {latent_neelY.shape}')

mse_list_ferro = [float(np.mean((item - component_means)**2)) for item in latent_ferro]
mse_list_neelY = [float(np.mean((item - component_means)**2)) for item in latent_neelY]
mse_list = mse_list_ferro + mse_list_neelY
# delta_list = np.concatenate((delta_test, delta_gapless), axis=0)

plt.figure()
plt.plot(range(len(mse_list)), mse_list, 'o', color='blue')
plt.axvline(x=300, color='black', linestyle='--', linewidth=1)
plt.xlabel('delta', fontsize=fs)
plt.ylabel('MSE to component means', fontsize=fs)
plt.savefig(f'../figures/xxzhAutoEncoder_latent_mse_neelY_component_means_epoch{num_epochs}_layersP{hyper_param}.pdf', bbox_inches='tight')

#########################################################################################

## calculate the magnetization of the reconstructed gapless states

# x_rc_neelY = autoencoder(params, x_neelY, 0, key)   ## testing mode, set drop_p = 0
# m_rc_neelY = cal_m_reconstructed(x_rc_neelY)
# m_neelY = cal_m_reconstructed(x_neelY)

# plt.figure()
# plt.plot(range(len(m_neelY)), m_neelY, 'o', color='blue', label=f'<m>')
# plt.plot(range(len(m_rc_neelY)), m_rc_neelY, 'o', color='orange', label=f'<m>_rec')
# plt.legend(fontsize=fs-2)
# plt.xlabel('delta', fontsize=fs)
# plt.ylabel('<m>', fontsize=fs)
# plt.savefig(f'../figures/xxzhAutoEncoder_neelY_reconstructed_magnetization_epoch{num_epochs}_layersP{hyper_param}.pdf', bbox_inches='tight')

#########################################################################################

# def random_unit_vectors(n, d):
#     """Generate n random unit vectors in d dimensions."""
#     vecs = np.random.normal(0, 1, (n, d))
#     norms = np.linalg.norm(vecs, axis=1, keepdims=True)
#     return vecs / norms

# # Parameters
# num_vectors = 1000  # Reduced for computational feasibility in the example
# dimensions = 20

# # Generate random unit vectors
# vectors = random_unit_vectors(num_vectors, dimensions)

# # Calculate distances
# distances = np.sqrt(2 * (1 - np.dot(vectors, vectors.T)))
# upper_triangle_distances = distances[np.triu_indices(num_vectors, k=1)]

# # Analyze distances
# print("Mean Distance:", np.mean(upper_triangle_distances))
# print("Median Distance:", np.median(upper_triangle_distances))
# print("90th Percentile Distance:", np.percentile(upper_triangle_distances, 90))
# print("Small Distance Threshold (e.g., 10th Percentile):", np.percentile(upper_triangle_distances, 10))

# # Plot the histogram of the distances
# plt.figure(figsize=(10, 6))
# plt.hist(upper_triangle_distances, bins=50, color='blue', alpha=0.7)
# plt.title('Histogram of Distances Between Random Unit Vectors on a Hypersphere')
# plt.xlabel('Distance')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.savefig('../figures/histogram_distances_random_unit_vectors_d=20.png')

## Mean Distance: 1.4049771483137858
## Median Distance: 1.4145493938115403
## Average Variance: 0.049949897540107216
## 90th Percentile Distance: 1.6070694239348111
## Small Distance Threshold (e.g., 10th Percentile): 1.1906459658262973