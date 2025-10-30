import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap, block_until_ready, pure_callback

import pickle
import numpy as np
import matplotlib.pyplot as plt

import optax

from functools import partial

########################################################################################

# use auto-encoder structure to learn the ferromagnetic phase of the xxz model

########################################################################################

## font size of plots
fs = 15

########################################################################################

from jax import config
config.update("jax_enable_x64", True)

########################################################################################

np.random.seed(45297)

with open('../data/data_ferro_xxz_observables.pkl', 'rb') as f:
  data = np.array(pickle.load(f))
np.random.shuffle(data)

test_size = int(len(data)*0.3)
test_indices = np.random.choice(len(data), test_size, replace=False)

data_test = data[test_indices]
data_train = np.delete(data, test_indices)

x_train = np.array([d['obs_value'] for d in data_train])
delta_train = np.array([d['delta'] for d in data_train]).reshape(-1, 1)

x_test = np.array([d['obs_value'] for d in data_test])
delta_test = np.array([d['delta'] for d in data_test]).reshape(-1, 1)

print(x_train.shape, delta_train.shape, x_test.shape, delta_test.shape, x_train.dtype, x_test.dtype)
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


# def cal_m(v):
#     ## calculate the average magnetization given a wavefunction
#     m_tot = 0
#     N = int(np.log2(len(v)))
#     # print(N)
#     for i in range(len(v)):
#         state = hslabeltoocc(i, N)
#         num_up = jnp.count_nonzero(state)
#         m = abs(2*num_up - N)
#         m_tot += m* (jnp.conj(v[i])*v[i])
#     return m_tot/N

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
    
    # x_out = acti_loss(x_out)   ## normalize the output
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


def cal_m(obs_val):
    # every third observable data is the z_i expectation value
    if obs_val.ndim == 1:
        return np.mean(obs_val[2::3])
    else:
        return np.mean(obs_val[:, 2::3], axis=1)


########################################################################################

seed = 7582938
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key)

layer_widths = [x_train.shape[-1], 15, 5, 15, x_train.shape[-1]] 
print('layers:', layer_widths)
initial_MLP_params = init_params(layer_widths, subkey)

########################################################################################

N = 10

lr = 0.001
num_epochs = 1000

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
    # m_list = cal_m_reconstructed(to_complex(autoencoder(params, x_test, 0, key)))
    print('epoch=',i, 'loss=',loss_new,
          'val_loss=', loss(params, x_test, 0, key), 
          'magnetization_test=', np.mean(cal_m(autoencoder(params, x_test, 0, key))), 
          'magnetization_train=', np.mean(cal_m(autoencoder(params, x_train, 0, key))),
            )
    
########################################################################################

hyper_param = '_'.join(str(e) for e in layer_widths)
with open(f'../models/xxzAutoEncoder_ferro_obs_nnParams_epoch{num_epochs}_layersP{hyper_param}.pickle','wb') as f:
    pickle.dump(params, f)

########################################################################################

plt.figure()
plt.plot(loss_list, 'o', color='blue')
plt.xlabel('epoch', fontsize=fs)
plt.ylabel('loss', fontsize=fs)
plt.yscale('log')
plt.savefig(f'../figures/xxzAutoEncoder_ferro_obs_trainingloss_epoch{num_epochs}_layersP{hyper_param}.pdf', bbox_inches='tight')

########################################################################################

hyper_param = '_'.join(str(e) for e in layer_widths)
params = pickle.load(open(f'../models/xxzAutoEncoder_ferro_obs_nnParams_epoch{num_epochs}_layersP{hyper_param}.pickle', 'rb'))

########################################################################################

x_rc = autoencoder(params, x_test, 0, key)   ## testing mode, set drop_p = 0
m_rc = cal_m(x_rc)
m_test = cal_m(x_test)

plt.figure()
plt.plot(delta_test.reshape(-1), m_test, 'o', color='blue', label=f'<m>')
plt.plot(delta_test.reshape(-1), m_rc, 'o', color='orange', label=f'<m>_rec')
plt.legend(fontsize=fs-2)
plt.xlabel('delta', fontsize=fs)
plt.ylabel('<m>', fontsize=fs)
plt.savefig(f'../figures/xxzAutoEncoder_ferro_obs_reconstructed_magnetization_epoch{num_epochs}_layersP{hyper_param}.pdf', bbox_inches='tight')

########################################################################################

# latent space study
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

with open(f'../models/xxzAutoEncoder_ferro_obs_latent_componentMeans_epoch{num_epochs}_layersP{hyper_param}.pickle','wb') as f:
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
plt.savefig(f'../figures/xxzAutoEncoder_ferro_obs_trainlatent_distance_hist_epoch{num_epochs}_layersP{hyper_param}.pdf', bbox_inches='tight')

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
# plt.savefig(f'../figures/xxzAutoEncoder_ferro_obs_trainlatent_variance_epoch{num_epochs}_layersP{hyper_param}.pdf', bbox_inches='tight')

#########################################################################################

## use the latent representation trained on ferro phase to see 
## if the gapless states have the similar representation

## load gapless states data
with open('../data/data_gapless_xxz_observables.pkl', 'rb') as f:
    data_gapless = np.array(pickle.load(f))
x_gapless = np.array([d['obs_value'] for d in data_gapless])
delta_gapless = np.array([d['delta'] for d in data_gapless]).reshape(-1, 1)

## fetch latent representation of gapless states
latent_gapless = fetch_latent(params, x_gapless, 0, key)
print('latent gapless shape:', latent_gapless.shape)

# # calculate pairwise distances in the latent space
pair_diff_gapless = pairwise_distances(latent_gapless)
print('pairwise distance of gapless states:', np.average(pair_diff_gapless))

plt.figure()
plt.hist(pair_diff_gapless, bins=50, color='blue', alpha=0.7)
plt.xlabel('pairwise distances', fontsize=fs)
plt.ylabel('frequency', fontsize=fs)
plt.savefig(f'../figures/xxzAutoEncoder_gapless_obs_latent_distance_hist_epoch{num_epochs}_layersP{hyper_param}.pdf', bbox_inches='tight')

#########################################################################################

## calculate the difference between the component-wise means of the latent representations
## between ferro and gapless states

latent_ferro = fetch_latent(params, x_test, 0, key)
latent_gapless = fetch_latent(params, x_gapless, 0, key)

component_means_ferro = np.mean(latent_ferro, axis=0)
component_means_gapless = np.mean(latent_gapless, axis=0)

diff_component_means = abs(component_means_ferro - component_means_gapless)

print('component means ferro:', component_means_ferro, np.std(latent_ferro, axis=0))
print('component means gapless:', component_means_gapless, np.std(latent_gapless, axis=0))
print('difference in component means:', diff_component_means)
print('mse between component means:', np.mean(diff_component_means**2))

#########################################################################################

## calculate the mse between the latent representations of every ferro state and the mean

latent_ferro = fetch_latent(params, x_test, 0, key)
latent_gapless = fetch_latent(params, x_gapless, 0, key)
component_means = np.mean(latent_ferro, axis=0)

mse_list_ferro = [float(np.mean((item - component_means)**2)) for item in latent_ferro]
mse_list_gapless = [float(np.mean((item - component_means)**2)) for item in latent_gapless]
mse_list = mse_list_ferro + mse_list_gapless
delta_list = np.concatenate((delta_test, delta_gapless), axis=0)

plt.figure()
plt.plot(delta_list, mse_list, 'o', color='blue')
plt.xlabel('delta', fontsize=fs)
plt.ylabel('MSE to component means', fontsize=fs)
plt.savefig(f'../figures/xxzAutoEncoder_obs_latent_mse_component_means_epoch{num_epochs}_layersP{hyper_param}.pdf', bbox_inches='tight')

#########################################################################################

## calculate the magnetization of the reconstructed gapless states

x_rc_gapless = autoencoder(params, x_gapless, 0, key)   ## testing mode, set drop_p = 0
m_rc_gapless = cal_m(x_rc_gapless)
m_gapless = cal_m(x_gapless)

plt.figure()
plt.plot(delta_gapless.reshape(-1), m_gapless, 'o', color='blue', label=f'<m>')
plt.plot(delta_gapless.reshape(-1), m_rc_gapless, 'o', color='orange', label=f'<m>_rec')
plt.legend(fontsize=fs-2)
plt.xlabel('delta', fontsize=fs)
plt.ylabel('<m>', fontsize=fs)
plt.savefig(f'../figures/xxzAutoEncoder_gapless_obs_reconstructed_magnetization_epoch{num_epochs}_layersP{hyper_param}.pdf', bbox_inches='tight')

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