from .dmrg_xxz import *
import jax.numpy as jnp
from types import SimpleNamespace

def dmrg_E(L,delta,h,conf=1e-4,test=False,chi_max=10, max_sweep = 5):
    # Initialize random MPS and MPO
    # chi_max: max bond dimension - b/c we are using jax, we are limited to relatively small chi; 
    # setting to 10 for now
    init_mps = random_MPS(L=L,d=2,chi_max=chi_max)
    mpo = XXZhX_MPO(L=L,d=2,delta=delta,h=h)
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
        # print(f'sweep: {counter}, E: {dmrg.e[-1]}')
        if counter > max_sweep:
            break
    return dmrg.MPS, dmrg.e[-1]

def cal_observables(mps, operator_list):
    expect_list = []
    for mpo in operator_list:
        expect_list.append(jnp.real(expect_mpo(mps, mpo)))
    return expect_list