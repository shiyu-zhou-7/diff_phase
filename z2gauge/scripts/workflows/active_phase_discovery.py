import jax
import jax.numpy as jnp
import numpy as np
import pickle
import optax
import matplotlib.pyplot as plt
from datetime import datetime

from configs.config import AEConfig, HamConfig, ActiveConfig
from hamiltonian.z2ham import *
from models.autoencoder import init_params as init_ae, fetch_latent
from training.ae_train import train_autoencoder
from training.ham_optimize import ham_update, make_ham_step
from utils.io import save_pickle


def sample_params(h_init, n, rad_h, key):
    _, kh = jax.random.split(key)
    hs     = h_init + rad_h * jax.random.normal(kh, (n,))
    return hs


def generate_states(hs, star_ops, trans_ops):
    states = []
    for h in np.array(hs):
        H = hamiltonian(-1.0, h, star_ops, trans_ops)
        _, eigenvectors = jnp.linalg.eigh(H)
        ground_state = eigenvectors[:, 0]
        ground_state = jnp.real(ground_state)
        ground_state = ground_state / (jnp.linalg.norm(ground_state) + 1e-12)
        states.append(jnp.asarray(ground_state))
    X = jnp.stack(states, axis=0)
    return X


def active_phase_discovery(Lx, Ly, h_init,
                           ae_cfg: AEConfig = AEConfig(),
                           ham_cfg: HamConfig = HamConfig(),
                           act_cfg: ActiveConfig = ActiveConfig(),
                           init_ae_params=None, ferro_centroid=None, max_outer_iters=6):
    key = jax.random.PRNGKey(ae_cfg.seed)

    N = 2 * Lx * Ly
    D = 2**N

    star_ops = sum_star_operators(Lx, Ly)
    trans_ops = transverse_field(Lx, Ly)

    # Get current time string
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # init AE if needed
    if init_ae_params is None:
        layers = [D, 512, ae_cfg.latent_dim, 512, D]
        key, sub = jax.random.split(key)
        init_ae_params = init_ae(layers, sub)

    # bootstrap centroid if not provided
    if ferro_centroid is None:
        key, sub = jax.random.split(key)
        hs = sample_params(h_init, 1000, 0.5, sub)
        X = generate_states(hs, star_ops, trans_ops)
        init_ae_params = train_autoencoder(init_ae_params, X, epochs=ae_cfg.epochs, lr=ae_cfg.lr, weight_decay=ae_cfg.weight_decay, drop_p=ae_cfg.dropout_p, center_coeff=act_cfg.center_coeff, seed=ae_cfg.seed)
        Z = fetch_latent(init_ae_params, X, jax.random.PRNGKey(0))
        ferro_centroid = jnp.mean(Z, axis=0)

    # ham optimizer
    ham_param = jnp.array([h_init], dtype=jnp.float64)
    ham_opt = optax.adam(learning_rate=ham_cfg.lr)
    ham_state = ham_opt.init(ham_param)
    ham_step = make_ham_step(ham_opt)

    last_dir = jnp.zeros_like(ham_param)
    nan_boost_steps_left = 0 # counter for nan_lr_mult steps

    history = { 'ham_params': [], 'ham_loss': [] }

    for outer in range(max_outer_iters):

        print(f'===== Outer Iteration {outer+1} / {max_outer_iters} =====')

        recent_losses, recent_grads = [], []
        for step in range(ham_cfg.max_steps_block):
            ham_param_prev = jnp.array(ham_param)
            ham_param, ham_state, L, gnorm = ham_step(ham_param, star_ops, trans_ops, init_ae_params, ferro_centroid, ham_state)

            # Compute absolute change in each parameter
            h_change     = float(abs(ham_param[0] - ham_param_prev[0]))
            max_change   = h_change

            # Logging
            if (step + 1) % 100 == 0 or step == 0:
                print(f"[HAM] step {step+1:4d} | loss={float(L):.3e} | grad={float(gnorm):.2e} | "
                      f"h={float(ham_param[0]):.5f} | h_change={h_change:.2e}")
                
            # NaN/Inf guard -> break
            # bad = (~jnp.isfinite(L)) | (~jnp.isfinite(gnorm))
            # bad = bad.item()
            # if bad:
                # print("[WARNING] Encountered NaN/Inf in loss or gradient. Stopping inner optimization.")
                # ham_param = ham_param_prev  # revert to previous safe parameters
                # break

            all_finite = (
            jnp.all(jnp.isfinite(ham_param))
            & jnp.isfinite(L)
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
                ham_opt_boost = optax.adam(learning_rate=ham_cfg.lr * ham_cfg.nan_lr_mult)
                ham_state = ham_opt_boost.init(ham_param)
                ham_step  = make_ham_step(ham_opt_boost)

                # skip recording this failed step; continue to next iteration
                continue

                
            # Stop condition if parameter changes are tiny
            if max_change < ham_cfg.param_tol_change:
                print(f"[CONVERGED] h={float(ham_param[0]):.5f} changed by < {ham_cfg.param_tol_change:.1e}, stopping inner optimization.")
                break

            # record history
            history['ham_params'].append(np.array(ham_param))
            history['ham_loss'].append(float(L))
            last_dir = ham_param - ham_param_prev

            # recent_losses.append(float(L))
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
                    ham_opt  = optax.adam(learning_rate=ham_cfg.lr)
                    ham_state = ham_opt.init(ham_param)
                    ham_step  = make_ham_step(ham_opt)

        if step < ham_cfg.max_steps_block - 1:
            # retrain AE locally
            h = float(ham_param[0])
            key, sub = jax.random.split(key)
            hs = sample_params(h, act_cfg.num_samples_when_stalled, act_cfg.sample_radius_h, sub)
            X_local = generate_states(hs, star_ops, trans_ops)
            init_ae_params = train_autoencoder(init_ae_params, X_local, epochs=ae_cfg.mini_epochs, lr=ae_cfg.lr, weight_decay=ae_cfg.weight_decay, drop_p=ae_cfg.dropout_p, center_coeff=act_cfg.center_coeff, seed=ae_cfg.seed)
            Z_local = fetch_latent(init_ae_params, X_local, jax.random.PRNGKey(0))
            ferro_centroid = jnp.mean(Z_local, axis=0)

    # save hamiltonian parameters history
    print('Saving Hamiltonian parameters history...')
    save_pickle(history, f'../models/active_phase_ham_params_history_h{h_init}_{Lx}x{Ly}_{timestamp}.pkl')

    # drawing
    print('Drawing Hamiltonian parameter trajectory...')
    hs = [float(p[0]) for p in history['ham_params']]  # Extract h values (single parameter)
    steps = np.arange(len(hs))  # Step numbers for x-axis
    
    fig, ax = plt.subplots()
    ax.plot(steps, hs, 'o-', color='blue', markersize=2, lw=1)

    ax.set_xlabel("Epoch step", fontsize=14)
    ax.set_ylabel(r"$h$", fontsize=14)
    if len(hs) > 0:
        h_min, h_max = min(hs), max(hs)
        h_range = h_max - h_min
        padding = 0.1 * h_range if h_range > 0 else 0.1
        ax.set_ylim(h_min - padding, h_max + padding)
    ax.set_title("Hamiltonian Parameter Trajectory", fontsize=15)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'../figures/active_phase_trajectory_h{h_init}_{Lx}x{Ly}_{timestamp}.pdf', bbox_inches='tight')

    return init_ae_params, ferro_centroid, history