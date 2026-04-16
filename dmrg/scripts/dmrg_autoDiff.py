
import jax.numpy as np
from jax import value_and_grad
import pickle
import matplotlib.pyplot as plt

import optax

from dmrg_tfim_mps_jax import *

jax.config.update('jax_enable_x64', True)

########################################################################################

# auto-differentiate DMRG directly

########################################################################################

sxx = np.array([[0., 1.], [1., 0.]])
syy = np.array([[0., -1j], [1j, 0.]])
szz = np.array([[1., 0.], [0., -1.]])

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
        print(f'sweep: {counter}, E: {dmrg.e[-1]}')
        if counter > max_sweep:
            break
    return dmrg.MPS, dmrg.e[-1]


def ave_m(mps, mpo_list):
    ave = 0
    for mpo in mpo_list:
        ave += expect_mpo(mps, mpo)
    return ave.real.astype(float) / len(mpo_list)


def loss(g, L, chi_max):
    mps, E = dmrg_E(L, 1, g, conf=1e-4, test=False, chi_max=chi_max, max_sweep = 5)
    m = ave_m(mps, Sx_list)
    l = (1-m)**2  ## drive to the ssb phase
    # l = m**2   ## drive to the para phase
    return l


def update(g, L, chi_max, opt, opt_state):
    value, gradient = value_and_grad(loss,argnums=(0,))(g, L, chi_max)
    updates, opt_state = opt.update(gradient[0], opt_state)
    g_new = optax.apply_updates(g, updates)
    return g_new, value, gradient[0]

########################################################################################

L = 50
J = 1.

########################################################################################

Sx_list= []
for i in range (L):
    # mpo = Pauli_MPO(L=L,d = 2,pauli_string = '0'*(i)+'3'+'0'*(L-i-1))
    mpo = Pauli_MPO(L=L,d = 2,pauli_string = '0'*(i)+'1'+'0'*(L-i-1))
    Sx_list.append(mpo)

########################################################################################

g_0 = 0.3

########################################################################################

lr = 1e-2
epochs = 50

opt = optax.adam(lr)
opt_state = opt.init(g_0)

########################################################################################

# auto-differentiate through dmrg

loss_list = []
g_list = []

chi_max = 5
g = g_0

## gradient descent to find h that minimizes the loss function
for i in range(epochs):
    g, loss_new, grad = update(g, L, chi_max, opt, opt_state)
    loss_list.append(loss_new)
    g_list.append(g)
    print('epoch=',i, ', loss=',loss_new, ', g_new=',g, 'gradient=', grad)

########################################################################################

# save the data for plotting
data = {
    'g_list': g_list,
    'loss_list': loss_list,
}
with open('dmrg_autoDiff_trajectory.pkl', 'wb') as f:
    pickle.dump(data, f)

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