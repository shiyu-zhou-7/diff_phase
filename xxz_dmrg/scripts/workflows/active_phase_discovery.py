from venv import logger
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import optax
import matplotlib.pyplot as plt
from datetime import datetime

from configs.config import AEConfig, DMRGConfig, HamConfig, ActiveConfig
from dmrg.hamiltonians import XXZhX
from dmrg.mps import get_random_MPS
from dmrg.dmrg import DMRG
from models.autoencoder import init_params as init_ae, fetch_latent
from training.ae_train import train_autoencoder
from training.dmrg_optimize import make_opt_step
from utils.io import save_pickle
from utils.plotting import plot_trajectory

from dmrg1.dmrg_xxz import Pauli_MPO
from dmrg1.run_xxz import * 


def sample_params(center_delta, center_h, n, rad_d, rad_h, key):
    kd, kh = jax.random.split(key)
    deltas = center_delta + rad_d * jax.random.normal(kd, (n,))
    hs     = center_h     + rad_h * jax.random.normal(kh, (n,))
    return deltas, hs


def generate_data_xxzh(L, deltas, hs, observables_list, dmrg_cfg):
    """ observables_list: [(site, op), ...] """
    print(f'Generating new data {len(deltas)} for AE learning around delta={deltas} and h= {hs}...')
    rows = []
    for delta, h in zip(np.array(deltas), np.array(hs)):

        # initialize random product state and Hamiltonian
        # psi = get_random_MPS(L, d=2, bond_dim = 1)
        # H = XXZhX(L, delta, h)
        # dmrg = DMRG(psi, H, dmrg_cfg.max_bond, lanczos=dmrg_cfg.lanczos_bool)

        # perform DMRG sweeps
        # for _ in range(dmrg_cfg.sweeps):
            # dmrg.sweep()
            # psi = dmrg.psi

        # Compute observables
        # obs_list = psi.get_site_exp_val(observables_list)
        # obs_vec = jnp.stack(
            # [jnp.real(jnp.asarray(x)).reshape(()) for x in obs_list],
            # axis=0
        # ) # shape (len(observables_list),)


        # use one-site update rule to compute observables
        mps, _ = dmrg_E(L, delta, h, conf=1e-4, test=False, chi_max=dmrg_cfg.max_bond, max_sweep=dmrg_cfg.sweeps)
        obs = jnp.asarray(cal_observables(mps, observables_list))
        rows.append(obs)

    data = jnp.stack(rows, axis=0)
    return data


def active_phase_discovery(L=20, init_delta=-1.5, init_h=0.3,
                           ae_cfg: AEConfig = AEConfig(),
                           ham_cfg: HamConfig = HamConfig(),
                           act_cfg: ActiveConfig = ActiveConfig(),
                           dmrg_cfg: DMRGConfig = DMRGConfig(),
                           init_ae_params=None, ferro_centroid=None, max_outer_iters=6):
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    key = jax.random.PRNGKey(ae_cfg.seed)

    sx = jnp.array([[0, 1], [1, 0]])
    sz = jnp.array([[1, 0], [0, -1]])
    sy = jnp.array([[0, -1j], [1j, 0]])

    # observables_list = [(i, sz) for i in range(L)] + [(i, sx) for i in range(L)] + [(i, sy) for i in range(L)]

    # pauli_string: 0,1,2,3 for I, X, Y, Z
    # for example: 3000.., means ZIII...
    observables_list= []
    for i in range (L):
        mpo = Pauli_MPO(L=L, d=2,pauli_string = '0'*(i)+'3'+'0'*(L-i-1))
        observables_list.append(mpo)
    for i in range (L):
        mpo = Pauli_MPO(L=L,d=2,pauli_string = '0'*(i)+'1'+'0'*(L-i-1))
        observables_list.append(mpo)
    for i in range (L):
        mpo = Pauli_MPO(L=L,d=2,pauli_string = '0'*(i)+'2'+'0'*(L-i-1))
        observables_list.append(mpo)

    D = len(observables_list)  # data dimension
    print(f'Observable vector dimension D = {D}')

    # init AE if needed
    if init_ae_params is None:
        print('Initializing initial autoencoder parameters...')
        layers = [D, 20, ae_cfg.latent_dim, 20, D]
        key, sub = jax.random.split(key)
        init_ae_params = init_ae(layers, sub)

    # bootstrap centroid if not provided
    if ferro_centroid is None:
        key, sub = jax.random.split(key)
        deltas, hs = sample_params(init_delta, init_h, 1, 0.5, 0.5, sub)
        X = generate_data_xxzh(L, deltas, hs, observables_list, dmrg_cfg)
        init_ae_params = train_autoencoder(init_ae_params, X, epochs=ae_cfg.epochs, lr=ae_cfg.lr, weight_decay=ae_cfg.weight_decay, drop_p=ae_cfg.dropout_p, center_coeff=act_cfg.center_coeff, seed=ae_cfg.seed)
        Z = fetch_latent(init_ae_params, X, jax.random.PRNGKey(0))
        ferro_centroid = jnp.mean(Z, axis=0)

    # dmrg optimizer
    ham_param = jnp.array([init_delta, init_h], dtype=jnp.float64)
    opt = optax.adam(learning_rate=ham_cfg.lr)
    opt_state = opt.init(ham_param)
    opt_step = make_opt_step(opt, L, dmrg_cfg, observables_list)

    last_dir = jnp.zeros_like(ham_param)
    nan_boost_steps_left = 0 # counter for nan_lr_mult steps

    history = { 'ham_params': [], 'ham_loss': [] }

    for outer in range(max_outer_iters):

        print(f'===== Outer Iteration {outer+1} / {max_outer_iters} =====')

        recent_losses, recent_grads = [], []
        for step in range(ham_cfg.max_steps_block):
            ham_param_prev = jnp.array(ham_param)
            ham_param, opt_state, loss_val, gnorm = opt_step(ham_param, init_ae_params, ferro_centroid, opt_state)

            # Compute absolute change in each parameter
            delta_change = float(abs(ham_param[0] - ham_param_prev[0]))
            h_change     = float(abs(ham_param[1] - ham_param_prev[1]))
            max_change   = max(delta_change, h_change)

            # Logging
            if (step + 1) % 10 == 0 or step == 0:
                print(f"[HAM] step {step+1:4d} | loss={float(loss_val):.3e} | grad={float(gnorm):.2e} | "
                      f"Δ={float(ham_param[0]):.5f} h={float(ham_param[1]):.5f} | Δ_change={delta_change:.2e}, h_change={h_change:.2e}")
                
            # NaN/Inf guard -> break
            # bad = (~jnp.isfinite(loss)) | (~jnp.isfinite(gnorm))
            # bad = bad.item()
            # if bad:
                # print("[WARNING] Encountered NaN/Inf in loss or gradient. Stopping inner optimization.")
                # ham_param = ham_param_prev  # revert to previous safe parameters
                # break

            all_finite = (
            jnp.all(jnp.isfinite(ham_param))
            & jnp.isfinite(loss_val)
            & jnp.isfinite(gnorm)
            ).item()

            if not all_finite:
                print("[WARNING] NaN/Inf in params/loss/grad. Reverting and jumping past the region.")

                # 1) revert
                ham_param = ham_param_prev

                # 2) compute jump
                if jnp.all(last_dir == 0).item():
                    # first step or no history: small random nudge
                    key, sub = jax.random.split(key)
                    jump = act_cfg.nan_jump_noise * jax.random.normal(sub, shape=ham_param.shape)
                else:
                    jump = act_cfg.nan_jump_scale * last_dir

                ham_param = ham_param + jump

                # 3) temporary LR boost
                nan_boost_steps_left = ham_cfg.nan_lr_steps
                opt_boost = optax.adam(learning_rate=ham_cfg.lr * ham_cfg.nan_lr_mult)
                opt_state = opt_boost.init(ham_param)
                opt_step  = make_opt_step(opt_boost, L, dmrg_cfg, observables_list)

                # skip recording this failed step; continue to next iteration
                continue

                
            # Stop condition if parameter changes are tiny
            if max_change < ham_cfg.param_tol_change:
                print(f"[CONVERGED] Δ={float(ham_param[0]):.5f}, h={float(ham_param[1]):.5f} changed by < {ham_cfg.param_tol_change:.1e}, stopping inner optimization.")
                break

            # record history
            history['ham_params'].append(np.array(ham_param))
            history['ham_loss'].append(float(loss_val))
            last_dir = ham_param - ham_param_prev

            # recent_losses.append(float(loss_val))
            # recent_grads.append(float(gnorm))
            
            # if len(recent_losses) >= ham_cfg.stall_window:
            #     dL = abs(recent_losses[-1] - recent_losses[-ham_cfg.stall_window])
            #     gbar = np.mean(recent_grads[-ham_cfg.stall_window:])
            #     if dL < ham_cfg.stall_tol_loss or gbar < ham_cfg.stall_tol_grad:
            #         break

            # ---- manage LR boost window ----
            if nan_boost_steps_left > 0:
                nan_boost_steps_left -= 1
                if nan_boost_steps_left == 0:
                    # restore normal optimizer
                    opt  = optax.adam(learning_rate=ham_cfg.lr)
                    opt_state = opt.init(ham_param)
                    opt_step  = make_opt_step(opt)

        if step < ham_cfg.max_steps_block - 1:
            # retrain AE locally
            delta = float(ham_param[0])
            h = float(ham_param[1])
            key, sub = jax.random.split(key)
            deltas, hs = sample_params(delta, h, act_cfg.num_samples_when_stalled, act_cfg.sample_radius_delta, act_cfg.sample_radius_h, sub)
            X_local = generate_data_xxzh(L, deltas, hs, observables_list, dmrg_cfg)
            init_ae_params = train_autoencoder(init_ae_params, X_local, epochs=ae_cfg.mini_epochs, lr=ae_cfg.lr, weight_decay=ae_cfg.weight_decay, drop_p=ae_cfg.dropout_p, center_coeff=act_cfg.center_coeff, seed=ae_cfg.seed)
            Z_local = fetch_latent(init_ae_params, X_local, jax.random.PRNGKey(0))
            ferro_centroid = jnp.mean(Z_local, axis=0)

    # save hamiltonian parameters history
    print('Saving Hamiltonian parameters history...')
    save_pickle(history, f'../models/active_phase_ham_params_history_{timestamp}.pkl')

    # plotting
    deltas = [p[0] for p in history['ham_params']]
    hs = [p[1] for p in history['ham_params']]
    print('Plotting Hamiltonian parameters history...')
    plot_trajectory(deltas, hs, f'../figures/active_phase_ham_params_history_{timestamp}.png')

    return init_ae_params, ferro_centroid, history