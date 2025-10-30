import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap, block_until_ready, pure_callback

import pickle
import numpy as np
import matplotlib.pyplot as plt

from functools import partial

from jax import config
config.update("jax_enable_x64", True)



"""""""""""""""""""""
 part 2: 
 train a neural network to classify the phases of tfim
"""""""""""""""""""""

""" load data """
## label: param: 1; ssb: 0
## training size: 1191
## testing size: 800
## data structure: {'data':[], 'label': int}

with open('./tfim_test.pickle', 'rb') as f:
  data_train = pickle.load(f)

with open('./tfim_train.pickle', 'rb') as f:
  data_test = pickle.load(f)

print('training data size:',len(data_train))
print('testing data size:', len(data_test))
print('sample:', data_test[0])


""" organize the data """
## separate the data and the label
## make sure all of them are arrays so that it is easy to perform calculation

x_dim = len(data_train[0]['data'])

x_train = np.zeros((len(data_train), x_dim))
y_train = np.zeros((len(data_train), 1))
print(x_train.shape, y_train.shape)

for i in range(len(data_train)):
  x_train[i] = data_train[i]['data']
  y_train[i] = data_train[i]['label']

x_test = np.zeros((len(data_test), x_dim))
y_test = np.zeros((len(data_test), 1))
print(x_test.shape, y_test.shape)

for i in range(len(data_test)):
  x_test[i] = data_test[i]['data']
  y_test[i] = data_test[i]['label']


""" initialize a neural network """
## define the neural network depths and width and initialize a network

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

## neural network architecutre
layer_widths = [x_dim, 500, 500, 1] 
initial_MLP_params = init_params(layer_widths, key)
# print(len(initial_MLP_params))
# print([(x[0].shape, x[1].shape) for x in initial_MLP_params])


""" define funcions """

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


  return fx

def loss(params, x, y):
  fx = predict(params, x)
  l = jnp.mean((fx - y)**2)
  return l

@jit
def update(params, x, y, lr):
  grads = grad(loss)(params, x, y)
  # print(type(grads), type(params))
  # print(grads)
  # print(params)
  params_new = jax.tree_map(lambda p, g: p-lr*g, params, grads)
  return params_new


""" train """

lr = 0.1
num_epochs = 2000

params = initial_MLP_params

for _ in range(num_epochs):
  params = update(params, x_train, y_train, lr)


""" test """
def calculate_accuracy(params, x, y):
    # Obtain predicted probabilities from the neural network
    predictions = predict(params, x)
    
    # Convert probabilities to binary class labels
    predicted_labels = predictions > 0.5
    
    # Calculate the accuracy
    accuracy = jnp.mean(predicted_labels == y)
    return accuracy

accuracy = calculate_accuracy(params, x_test, y_test)
print('Accuracy:', accuracy)


""" plot """
# plt.plot(test_results, 'o')
# # plt.plot(y_test, 'o')
# plt.ylabel('label',fontsize=14)
# plt.xlabel('instances',fontsize=14)
# plt.show()

# plt.savefig('tfim_nn_test.pdf', bbox_inches='tight')


""" write down the trained neural network parameters """
# with open('tfimClassifier_nnParams_500_500.pickle','wb') as f:
#   pickle.dump(params, f)
