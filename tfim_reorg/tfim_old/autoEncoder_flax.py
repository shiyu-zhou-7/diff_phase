from flax import linen as nn
from flax.training import train_state

import jax
from jax import lax, numpy as jnp
from jax import grad, value_and_grad, jit, vmap, block_until_ready
from flax.training import train_state

import optax

import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

from functools import partial

from torch.utils.data import TensorDataset, dataloader
import torch.utils.data as data

from tqdm.auto import tqdm

from jax import config
config.update("jax_enable_x64", True)


""" 
learning resources:
https://colab.research.google.com/github/gordicaleksa/get-started-with-JAX/blob/main/Tutorial_4_Flax_Zero2Hero_Colab.ipynb#scrollTo=5hhcFZ7UlCov 
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html

"""

## initialize the random seed
rng = jax.random.PRNGKey(9203952)


class AutoEncoder(nn.Module):
    
    dim_x: int
    dim_latent: int
    layers_list: list[int]

    @nn.compact
    def __call__(self, x):
        """ encoder """
        for num_neurons in (self.layers_list):
            x = nn.Dense(num_neurons)(x)
            x = nn.relu(x)

        """ latent """
        if dim_latent != None:
            x = nn.Dense(self.dim_latent)(x)

        """ decoder """
        for num_neurons in reversed(self.layers_list):
            x = nn.Dense(num_neurons)(x)
            x = nn.relu(x)

        """ output """
        x = nn.Dense(self.dim_x)(x)

        return x


def load_data(size_train):

    """ load data file from ssb_unbias.pickle """

    with open('./ssb_unbias.pickle', 'rb') as f:
        data_ssb = pickle.load(f)

    size_data = len(data_ssb)
    num_train = random.sample(range(0,size_data), size_train)

    # x_train = []
    # h_train = []
    # for i in num_train:
        # x_train.append(data_ssb[i]['wf'])
        # h_train.append(data_ssb[i]['h'])
    # x_train = np.array(x_train)
    # h_train = np.array(h_train)
    # print(x_train.shape, h_train.shape)

    data = []
    for i in num_train:
        data.append((data_ssb[i]['wf'], data_ssb[i]['h']))
    
    print('Out of', size_data, 'samples, the', size_train, 'samples are selected for training.')
    
    return data


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


## loss function
def calculate_loss(state, params, batch):
    input, h = batch
    output = state.apply_fn(params, input)
    dif_sq = (input - output)**2
    loss = jnp.sqrt(jnp.einsum('ij->i', dif_sq))

    def squared_error(x):
        recon_x = model.apply(params, x)
        print('recon_x shape', recon_x.shape)
        loss = (recon_x - x)**2
        return jnp.sqrt(jnp.einsum('ij->i', loss))
    
    # return jnp.mean(squared_error(x), axis=0)
    return jnp.mean(loss, axis=0)


# def calculate_loss_acc(state, params, batch):
#     data_input, labels = batch
#     # Obtain the logits and predictions of the model for the input data
#     logits = state.apply_fn(params, data_input).squeeze(axis=-1)
#     pred_labels = (logits > 0).astype(jnp.float32)
#     # Calculate the loss and accuracy
#     loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
#     acc = (pred_labels == labels).mean()
#     return loss, acc


## define a training step
@jit
def train_step(state, batch):
    grad_fn = jax.value_and_grad(calculate_loss, argnums=1)
    loss, grads = grad_fn(state, state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state, loss


def train_model(state, data_loader, num_epochs=1):
    for epoch in tqdm(range(num_epochs)):
        for batch in data_loader:
            state, loss = train_step(state, batch)
            # We could use the loss and accuracy for logging here, e.g. in TensorBoard
            # For simplicity, we skip this part here
    return state



## load the data
size_train = 700
batch_size = 100
dataset = load_data(size_train)
data_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate)
# x, y = next(iter(data_loader))  
# print(x.shape)
# print(y)


## create an instance of the model 
dim_x = 1024
dim_latent = None
layers_list = [500,500]
model = AutoEncoder(dim_x=dim_x, dim_latent=dim_latent, layers_list=layers_list)
# print(model)


## initialize the model
x = jax.random.normal(rng, (batch_size, dim_x))
params = model.init(rng, x)   ## note that the biases are set to 0 during the initialization 
# print(jax.tree_util.tree_map(jnp.shape, params))
# print(params)
# print(model.apply(params, x))   


## define the optimizer
optimizer = optax.adam(learning_rate=1e-3)


## setup a train state (something specific to jax)
model_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)


## train the model
num_epochs = 100
trained_model_state = train_model(model_state, data_loader, num_epochs=100)


## need to write up the evaluation part
#print(model_state.apply_fn(trained_model_state.params, x))