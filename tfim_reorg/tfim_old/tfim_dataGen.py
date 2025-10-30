import numpy as np
import random 
import pickle

"""""""""""""""""""""
 Generate data for training neural network
"""""""""""""""""""""


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


## setup the 1d transverse field ising model
def H_tfim(N, J, h):
    H_mat = np.zeros((2**N, 2**N))
    for i in range(2**N):
        state = hslabeltoocc(i,N)
        for site in range(N):
            if (site+1) < N:
                H_mat[i,i] += J*(2*state[site]-1)*(2*state[site+1]-1)
            stateflipped = state.copy()
            stateflipped[site] = abs(state[site]-1)
            H_mat[i,occtohslabel(stateflipped, N)] += h
    
    return H_mat


## ED solver
def gd_solver_ed(N, J, h):
    H = H_tfim(N, J, h)
    e, v = np.linalg.eigh(H)
    return e, v


## average magnetization
def cal_m(N, v):
    m_tot = 0
    for i in range(len(v)):
        state = hslabeltoocc(i, N)
        num_up = np.count_nonzero(state)
        m = abs(2*num_up - N)
        m_tot += m* (np.conj(v[i])*v[i])
    return m_tot/N


""" Generate data  """
## parameters
N = 10
J = -1.
#h_list = np.arange(-0.99,0.001,0.001)
h_list = np.arange(-2, -1.001, 0.001)

data_list = []

for h in h_list:
	e, v = gd_solver_ed(N, J, h)
	data_list.append({'h':h, 'e':e, 'gd':v[:,0], 'label':1})


## write data with all h to a file
f = open('tfim_para.pickle','wb')
pickle.dump(data_list, f)


""" Separate data to training and test """
## label: param: 1; ssb: 0
## data structure: {'data':[], 'label': int}

## define a set of training data 
select_ssb = random.sample(range(0,len(data_ssb)),400)
select_para = random.sample(range(0,len(data_para)),400)

train_data = []
for i in select_ssb:
    train_data.append({'data': data_ssb[i]['gd'][random.randint(0,1)], 'label': data_ssb[i]['label']})
for i in select_para:
    train_data.append({'data': data_para[i]['gd'], 'label': data_para[i]['label']})

## reshuffle the training data
random.shuffle(train_data)


## define a set of testing data
test_data = []
for i in range(len(data_ssb)):
    if i not in select_ssb:
        test_data.append({'data': data_ssb[i]['gd'][random.randint(0,1)], 'label': data_ssb[i]['label']})
for i in range(len(data_para)):
    if i not in select_para:
        test_data.append({'data': data_para[i]['gd'], 'label': data_para[i]['label']})


""" save the test and training data """
with open('tfim_train.pickle', 'wb') as f:
    pickle.dump(train_data, f)
    
with open('tfim_test.pickle', 'wb') as f:
    pickle.dump(test_data, f)

