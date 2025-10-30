import jax
import jax.numpy as jnp
import numpy as np
import pickle
import optax
import matplotlib.pyplot as plt
from datetime import datetime

from configs.config import AEConfig, HamConfig, ActiveConfig
from physics.hamiltonian import build_chain_operators, H_xxzh, eigh_ground_state
from models.autoencoder import init_params as init_ae, fetch_latent
from training.ae_train import train_autoencoder
from training.ham_optimize import ham_update, make_ham_step
from utils.io import save_pickle


def sample_params(center_delta, center_h, n, rad_d, rad_h, key):
    kd, kh = jax.random.split(key)
    deltas = center_delta + rad_d * jax.random.normal(kd, (n,))
    hs     = center_h     + rad_h * jax.random.normal(kh, (n,))
    return deltas, hs


def generate_states(deltas, hs, ham_ops):
    ham_xx, ham_yy, ham_zz, ham_x = ham_ops
    states = []
    for d, h in zip(np.array(deltas), np.array(hs)):
        H = H_xxzh(d, h, ham_xx, ham_yy, ham_zz, ham_x)
        _, v0 = eigh_ground_state(H)
        v0 = jnp.real(v0)
        v0 = v0 / (jnp.linalg.norm(v0) + 1e-12)
        states.append(np.asarray(v0))
    X = np.stack(states, axis=0)
    return X


def active_phase_discovery(N=10, init_delta=-1.5, init_h=0.3,
                           ae_cfg: AEConfig = AEConfig(),
                           ham_cfg: HamConfig = HamConfig(),
                           act_cfg: ActiveConfig = ActiveConfig(),
                           init_ae_params=None, ferro_centroid=None, max_outer_iters=6):
    key = jax.random.PRNGKey(ae_cfg.seed)
    ham_ops = build_chain_operators(N)
    D = 2**N

    # Get current time string
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # init AE if needed
    if init_ae_params is None:
        layers = [D, 256, ae_cfg.latent_dim, 256, D]
        key, sub = jax.random.split(key)
        init_ae_params = init_ae(layers, sub)

    # bootstrap centroid if not provided
    if ferro_centroid is None:
        key, sub = jax.random.split(key)
        deltas, hs = sample_params(init_delta, init_h, 1000, 0.5, 0.5, sub)
        X = generate_states(deltas, hs, ham_ops)
        init_ae_params = train_autoencoder(init_ae_params, X, epochs=ae_cfg.epochs, lr=ae_cfg.lr, weight_decay=ae_cfg.weight_decay, drop_p=ae_cfg.dropout_p, center_coeff=act_cfg.center_coeff, seed=ae_cfg.seed)
        Z = fetch_latent(init_ae_params, X, jax.random.PRNGKey(0))
        ferro_centroid = jnp.mean(Z, axis=0)

    # ham optimizer
    ham_param = jnp.array([init_delta, init_h], dtype=jnp.float64)
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
            ham_param, ham_state, L, gnorm = ham_step(ham_param, ham_ops, init_ae_params, ferro_centroid, ham_state)

            # Compute absolute change in each parameter
            delta_change = float(abs(ham_param[0] - ham_param_prev[0]))
            h_change     = float(abs(ham_param[1] - ham_param_prev[1]))
            max_change   = max(delta_change, h_change)

            # Logging
            if (step + 1) % 100 == 0 or step == 0:
                print(f"[HAM] step {step+1:4d} | loss={float(L):.3e} | grad={float(gnorm):.2e} | "
                      f"Δ={float(ham_param[0]):.5f} h={float(ham_param[1]):.5f} | Δ_change={delta_change:.2e}, h_change={h_change:.2e}")
                
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
                print(f"[CONVERGED] Δ={float(ham_param[0]):.5f}, h={float(ham_param[1]):.5f} changed by < {ham_cfg.param_tol_change:.1e}, stopping inner optimization.")
                break

            # record history
            history['ham_params'].append(np.array(ham_param))
            history['ham_loss'].append(float(L))
            # last_dir = ham_param - ham_param_prev

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
            delta = float(ham_param[0])
            h = float(ham_param[1])
            key, sub = jax.random.split(key)
            deltas, hs = sample_params(delta, h, act_cfg.num_samples_when_stalled, act_cfg.sample_radius_delta, act_cfg.sample_radius_h, sub)
            X_local = generate_states(deltas, hs, ham_ops)
            init_ae_params = train_autoencoder(init_ae_params, X_local, epochs=ae_cfg.mini_epochs, lr=ae_cfg.lr, weight_decay=ae_cfg.weight_decay, drop_p=ae_cfg.dropout_p, center_coeff=act_cfg.center_coeff, seed=ae_cfg.seed)
            Z_local = fetch_latent(init_ae_params, X_local, jax.random.PRNGKey(0))
            ferro_centroid = jnp.mean(Z_local, axis=0)

    # save hamiltonian parameters history
    print('Saving Hamiltonian parameters history...')
    save_pickle(history, f'../models/active_phase_ham_params_history_{timestamp}.pkl')

    # drawing
    print('Drawing Hamiltonian parameter trajectory...')
    deltas = [p[0] for p in history['ham_params']]
    hs     = [p[1] for p in history['ham_params']]
    fig, ax = plt.subplots()
    ax.plot(hs, deltas, 'o-', color='blue', markersize=2, lw=1)

    for i in range(0, len(deltas) - 1, 100):
        ax.annotate(
            '', 
            xy=(deltas[i + 1], hs[i + 1]), 
            xytext=(deltas[i], hs[i]),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.2, alpha=0.8)
        )

    ax.set_xlabel(r"$h$", fontsize=14)
    ax.set_ylabel(r"$\Delta$", fontsize=14)
    ax.set_xlim(min(hs)-0.5, max(hs)+0.5)
    ax.set_ylim(min(deltas)-0.5, max(deltas)+0.5)
    ax.set_title("Hamiltonian Parameter Trajectory", fontsize=15)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'../figures/active_phase_trajectory_{timestamp}.pdf', bbox_inches='tight')

    return init_ae_params, ferro_centroid, history