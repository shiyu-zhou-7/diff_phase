import numpy as np
import random
import pickle

"""""""""""""""""""""
The ground states obtained from the exact diagonalization of TFIM Hamiltonian in the ordered states
are two-fold degenerate. In order to reduce the bias in the data, we create linear superpositions 
of the two degenerate ground states for each h value. 
"""""""""""""""""""""

# with open('./TFIM_TEST/tfim_ssb.pickle','rb') as f:
with open('./TFIM_TEST/tfim_ssb.pickle','rb') as f:
    data = pickle.load(f)

## the data contians information about
## h, ground states 'gd', energy 'e', phase label 'label'

np.random.seed(10)
size = 10

data_ub = []

for sample in data:
    for _ in range(size):
        g1 = sample['gd'][0]
        g2 = sample['gd'][1]
        ## create linear combination of g1 and g2
        c1, c2 = np.random.normal(size=2)
        norm = np.sqrt(c1**2+c2**2)
        c1 = c1/norm
        c2 = c2/norm
        wf = c1*g1 + c2*g2

        data_ub.append({'h': sample['h'], 'wf': wf})

print(len(data_ub))

with open('./TFIM_TEST/ssb_unbias.pickle','wb') as f:
    pickle.dump(data_ub, f)




