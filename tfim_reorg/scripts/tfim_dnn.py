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

# import time

# start_time = time.time()

########################################################################################

# part 2: 
# train a neural network to classify the two phases of tfim

########################################################################################

## label: param: 0; ssb: 1
## data structure: a list of {'h': float, 'e': float, 'v': array, 'label': int}

np.random.seed(5902495)

with open('../data/tfim_data.pkl', 'rb') as f:
  data = np.array(pickle.load(f))
np.random.shuffle(data)

test_size = int(len(data)*0.3)
test_indices = np.random.choice(len(data), test_size, replace=False)

data_test = data[test_indices]
data_train = np.delete(data, test_indices)

x_train = np.array([d['v'] for d in data_train])
y_train = np.array([d['label'] for d in data_train]).reshape(-1, 1)

x_test = np.array([d['v'] for d in data_test])
y_test = np.array([d['label'] for d in data_test]).reshape(-1, 1)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(f'training size is {len(data_train)} and testing size is {len(data_test)}')

########################################################################################

## define the neural network depths and width and initialize a network

seed = 83948
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

## neural network architecutre
layer_widths = [x_train.shape[-1], 500, 500, 1] 
initial_MLP_params = init_params(layer_widths, key)

########################################################################################

def predict(params, x):
  ## input params: a list of parameters for the network; its length is the depth of the network
  ## input x: the training data
  ## output fx: the output after passing the network
  ## this function performs the network on the input data x given the network parameters x

  activation = x
  for w, b in params[:-1]:
    # print(activation.shape, w.shape, b.shape)
    activation = jax.nn.relu(jnp.dot(activation, w) + b)
    # print(activation)

  ## the last layer does not have the activation function
  fx = jnp.dot(activation, params[-1][0]) + params[-1][1]

  ## add the sigmoid function to the last layer
  # output = jax.nn.sigmoid(fx)
  # return output

  return fx


def loss(params, x, y):
  fx = predict(params, x)
  # fx = jnp.clip(fx, 1e-8, 1-1e-8)
  # loss = -1*jnp.mean(y*jnp.log(fx) + (1-y)*jnp.log(1-fx))   # binary cross entropy
  loss = jnp.mean((fx - y)**2)   # mean squared error
  return loss


@jit
def update(params, x, y, lr):
  grads = grad(loss)(params, x, y)
  params_new = jax.tree_map(lambda p, g: p-lr*g, params, grads)
  return params_new

########################################################################################

# lr = 0.1
# num_epochs = 2000

# params = initial_MLP_params

# print('starting training')
# for _ in range(num_epochs):
#   params = update(params, x_train, y_train, lr)

########################################################################################

def calculate_accuracy(params, x, y):
    # Obtain predicted probabilities from the neural network
    predictions = predict(params, x)
    
    # Convert probabilities to binary class labels
    predicted_labels = predictions > 0.5
    
    # Calculate the accuracy
    accuracy = jnp.mean(predicted_labels == y)
    return accuracy

def calculate_difference(params, x, y):
    predictions = predict(params, x)
    difference = jnp.mean(abs(predictions - y))
    return difference

# accuracy = calculate_accuracy(params, x_test, y_test)
# difference = calculate_difference(params, x_test, y_test)
# print('Accuracy:', accuracy)
# print('Difference:', difference)

########################################################################################

# with open('../models/tfimClassifier_nnParams_500_500.pkl','wb') as f:
#   pickle.dump(params, f)

########################################################################################

# end_time = time.time()
# print(f'time elapsed is {end_time-start_time} seconds')

########################################################################################

params = pickle.load(open('../models/tfimClassifier_nnParams_500_500.pkl', 'rb'))

h_test = np.array([d['h'] for d in data_test])
predictions = predict(params, x_test)
plt.plot(h_test, predictions, 'o')
plt.xlabel('h')
plt.ylabel('predicted label')
plt.savefig('../figures/tfim_dnn_classifier.pdf', bbox_inches='tight')
