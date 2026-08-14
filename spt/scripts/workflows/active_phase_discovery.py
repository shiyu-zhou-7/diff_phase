"""
Active phase discovery for the generalized cluster chain.

Mirrors xxz_dmrg's active_phase_discovery (vector ham-param, NaN revert+jump, stall
on parameter change, retrain-with-centroid-averaging) with two additions needed for
the benchmark (section 5):
  * a SOLVER-QUERY counter (every ED solve, in data generation and in each inner
    step's loss, is one query),
  * a per-step WINDING label (analytic ground truth) so distinct phases visited can
    be plotted against cumulative queries.

Outer loop:
  1. Bootstrap: sample t around t_init -> ED features -> train AE -> centroid.
  2. Repeat up to max_outer_iters:
       a. Inner Adam block MAXIMIZING latent distance from the centroid. The
          gradient diverges at a phase boundary (gap closes) -> NaN -> revert and
          jump across; that is how a new phase is entered.
       b. Stall (|dt|_inf < tol over the window) ends the block.
       c. Retrain the AE on all accumulated bootstrap centers; average the centroid
          (retain memory of explored phases).

Public: run_active_phase_discovery(...).
"""
from dataclasses import asdict

import numpy as np
import jax
import jax.numpy as jnp
import optax

from configs.config import AEConfig, ClusterConfig, ActiveConfig
from hamiltonians.cluster import build_cluster_model
from models.autoencoder import init_params, fetch_latent
from training.ae_train import train_autoencoder
from training.dataset import (sample_params, generate_features, generate_wavefunctions,
                              feature_dim, normalize_rows)
from training.optim_t import make_opt_step
from analytic.cluster_exact import winding
from utils.io import save_pickle


class QueryCounter:
    def __init__(self):
        self.n = 0

    def add(self, k):
        self.n += int(k)


def _bootstrap(centers, sigma, n_per_center, model, ae_cfg, act_cfg, key, qc,
               prev_centroid=None, epochs=None):
    """Sample around each center, ED-solve features, (re)train a fresh AE, and
    return (ae_params, centroid, X_all). centroid is averaged with prev_centroid
    when given (memory of explored phases)."""
    keys = jax.random.split(key, len(centers) + 2)
    X_list = []
    for c, k in zip(centers, keys[:len(centers)]):
        ts = sample_params(c, sigma, n_per_center, k, normalize=act_cfg.normalize_sphere)
        # X, nq = generate_features(ts, model)          # old rho0-routed observable input
        X, nq = generate_wavefunctions(ts, model)       # sign-fixed full 2^L wavefunction
        qc.add(nq)
        X_list.append(np.asarray(X))
    X_all = jnp.asarray(np.concatenate(X_list, axis=0))

    D = X_all.shape[1]
    layers = [D, ae_cfg.hidden, ae_cfg.latent_dim, ae_cfg.hidden, D]
    params = init_params(layers, keys[-2], scale=ae_cfg.init_scale)
    params = train_autoencoder(
        params, X_all,
        epochs=(epochs if epochs is not None else ae_cfg.epochs),
        lr=ae_cfg.lr, weight_decay=ae_cfg.weight_decay,
        drop_p=ae_cfg.dropout_p, center_coeff=ae_cfg.center_coeff,
        seed=ae_cfg.seed, loss='fidelity',   # AE input is a wavefunction
    )
    Z = fetch_latent(params, X_all, jax.random.PRNGKey(0))
    centroid = jnp.mean(Z, axis=0)
    if prev_centroid is not None:
        centroid = 0.5 * (centroid + prev_centroid)
    return params, centroid, X_all


def run_active_phase_discovery(t_init, model, ae_cfg, cluster_cfg, act_cfg,
                               rng_key, checkpoint_path=None):
    """Top-level driver. Returns a result bundle (history, queries, winding trail,
    final AE/centroid, cfg snapshot)."""
    key = rng_key
    qc = QueryCounter()
    d = model.d
    t_init = np.asarray(t_init, dtype=float)
    if act_cfg.normalize_sphere:
        t_init = normalize_rows(t_init[None, :])[0]

    # 1. INITIAL BOOTSTRAP
    key, bkey = jax.random.split(key)
    print(f'[spt] initial bootstrap at t={np.round(t_init,3)} '
          f'radius={act_cfg.t_radius_init} n={act_cfg.num_samples_init}')
    ae_params, centroid, _ = _bootstrap(
        [t_init], act_cfg.t_radius_init, act_cfg.num_samples_init,
        model, ae_cfg, act_cfg, bkey, qc,
    )
    bootstrap_history = [t_init.copy()]

    # run-level history
    t_per_step, loss_per_step, grad_per_step = [], [], []
    omega_per_step, event_per_step, query_per_step = [], [], []

    # full-coverage early stop (benchmark budget): windings seen so far,
    # including the start point. Only consulted when act_cfg.stop_on_full_coverage.
    target_omegas = set(range(d))
    seen_omegas = {int(winding(np.asarray(t_init)))}

    cfg_snapshot = {
        't_init': t_init.tolist(),
        'ae_cfg': asdict(ae_cfg),
        'cluster_cfg': asdict(cluster_cfg),
        'act_cfg': asdict(act_cfg),
    }

    t = jnp.asarray(t_init, dtype=jnp.float64)

    def build_opt():
        """Per-block optimizer: cosine-decay the lr over the block so t settles
        into a cell instead of overshooting boundaries, and clip the global step
        norm so the gradient divergence at a boundary cannot fling t far past it.
        Neither stabilizer reads the analytic label."""
        if cluster_cfg.lr_decay:
            sched = optax.cosine_decay_schedule(
                cluster_cfg.lr, cluster_cfg.max_steps_block,
                alpha=cluster_cfg.lr_final_frac,
            )
        else:
            sched = cluster_cfg.lr
        chain = []
        if cluster_cfg.grad_clip and cluster_cfg.grad_clip > 0:
            chain.append(optax.clip_by_global_norm(cluster_cfg.grad_clip))
        chain.append(optax.adam(sched))
        return optax.chain(*chain)

    opt = build_opt()
    opt_state = opt.init(t)
    opt_step = make_opt_step(opt, model, cluster_cfg.eta)

    last_dir = jnp.zeros_like(t)
    boost_left = 0

    def _pack():
        return {
            'history': {
                't_per_step': [np.asarray(x).tolist() for x in t_per_step],
                'loss_per_step': loss_per_step,
                'grad_per_step': grad_per_step,
                'omega_per_step': omega_per_step,
                'event_per_step': event_per_step,
                'query_per_step': query_per_step,
            },
            'bootstrap_history': [np.asarray(c).tolist() for c in bootstrap_history],
            'final_t': np.asarray(t).tolist(),
            'final_ae_params': ae_params,
            'final_centroid': np.asarray(centroid),
            'total_queries': qc.n,
            'distinct_phases': sorted(set(int(w) for w in omega_per_step)),
            'cfg_snapshot': cfg_snapshot,
        }

    # 2. OUTER LOOP
    for outer in range(act_cfg.max_outer_iters + 1):
        key, ikey = jax.random.split(key)
        print(f'[spt] outer={outer} inner block at t={np.round(np.asarray(t),3)} '
              f'omega={winding(np.asarray(t))} queries={qc.n}')

        # fresh optimizer each block -> cosine lr decay restarts and the block
        # settles cleanly into a stall.
        opt = build_opt()
        opt_state = opt.init(t)
        opt_step = make_opt_step(opt, model, cluster_cfg.eta)
        boost_left = 0
        t_block_start = jnp.asarray(t)

        recent_change = []
        exit_reason = 'max_steps'
        consec_failed = 0

        for step in range(cluster_cfg.max_steps_block):
            t_prev = jnp.asarray(t)
            lr_mult = cluster_cfg.nan_lr_mult if boost_left > 0 else 1.0
            # apply temporary boost by scaling the update post-hoc
            t_new, opt_state_new, loss, gnorm = opt_step(t, ae_params, centroid, opt_state)
            if lr_mult != 1.0:
                t_new = t_prev + lr_mult * (t_new - t_prev)
            qc.add(1)   # one ED solve inside the loss

            finite = bool(
                np.all(np.isfinite(np.asarray(t_new)))
                and np.isfinite(float(loss)) and np.isfinite(float(gnorm))
            )
            if not finite:
                consec_failed += 1
                if consec_failed >= 3:
                    print(f'[spt]   unrecoverable NaN at step {step}; ending block')
                    exit_reason = 'nan_stuck'
                    break
                # revert + jump across the boundary
                if bool(np.all(np.asarray(last_dir) == 0)):
                    key, sub = jax.random.split(key)
                    jump = act_cfg.nan_jump_noise * jax.random.normal(sub, shape=t.shape)
                else:
                    jump = act_cfg.nan_jump_scale * last_dir
                t = t_prev + jump
                if act_cfg.normalize_sphere:
                    t = jnp.asarray(normalize_rows(np.asarray(t)[None, :])[0])
                opt_state = opt.init(t)
                boost_left = cluster_cfg.nan_lr_steps
                t_per_step.append(np.asarray(t)); loss_per_step.append(float('nan'))
                grad_per_step.append(float('nan')); omega_per_step.append(int(winding(np.asarray(t))))
                event_per_step.append('nan_jump'); query_per_step.append(qc.n)
                seen_omegas.add(omega_per_step[-1])
                if act_cfg.stop_on_full_coverage and target_omegas <= seen_omegas:
                    exit_reason = 'full_coverage'
                    break
                continue

            consec_failed = 0
            t, opt_state = t_new, opt_state_new
            if act_cfg.normalize_sphere:
                # Radially project t back onto the sphere. Keep the Adam state
                # (resetting it every step would kill momentum and stall progress);
                # the projection is a small O(step^2) correction near the sphere.
                t = jnp.asarray(normalize_rows(np.asarray(t)[None, :])[0])
            step_dir = t - t_prev
            last_dir = step_dir
            boost_left = max(0, boost_left - 1)

            t_per_step.append(np.asarray(t)); loss_per_step.append(float(loss))
            grad_per_step.append(float(gnorm)); omega_per_step.append(int(winding(np.asarray(t))))
            event_per_step.append('normal'); query_per_step.append(qc.n)
            seen_omegas.add(omega_per_step[-1])
            if act_cfg.stop_on_full_coverage and target_omegas <= seen_omegas:
                exit_reason = 'full_coverage'
                break

            change = float(np.max(np.abs(np.asarray(step_dir))))
            recent_change.append(change)
            if len(recent_change) > cluster_cfg.stall_window:
                recent_change.pop(0)
            if (len(recent_change) == cluster_cfg.stall_window
                    and max(recent_change) < cluster_cfg.param_tol_change):
                exit_reason = 'stall'
                break

        print(f'[spt]   inner exit={exit_reason} steps={step+1} '
              f't={np.round(np.asarray(t),3)} omega={winding(np.asarray(t))} queries={qc.n}')

        if exit_reason == 'full_coverage':
            print(f'[spt]   all {d} windings visited ({sorted(seen_omegas)}) -> stop run')
            break

        if outer == act_cfg.max_outer_iters:
            break

        # convergence stop: if the whole block barely moved t, the run has settled
        # into a fixed point and further blocks just waste queries. (Distance on
        # the sphere; no analytic label is read.)
        block_move = float(np.linalg.norm(np.asarray(t) - np.asarray(t_block_start)))
        if block_move < act_cfg.converge_tol:
            print(f'[spt]   block moved {block_move:.4f} < converge_tol -> stop run')
            break

        # retrain only on a genuine stall (optimizer settled into a cell)
        if exit_reason not in ('stall', 'nan_stuck'):
            continue

        bootstrap_history.append(np.asarray(t).copy())
        key, rkey = jax.random.split(key)
        print(f'[spt]   retrain on {len(bootstrap_history)} centers')
        ae_params, centroid, _ = _bootstrap(
            bootstrap_history, act_cfg.t_radius_retrain, act_cfg.num_samples_bootstrap,
            model, ae_cfg, act_cfg, rkey, qc,
            prev_centroid=centroid, epochs=ae_cfg.mini_epochs,
        )
        # next block rebuilds the optimizer; just clear the momentum direction
        last_dir = jnp.zeros_like(t)

        if checkpoint_path is not None:
            save_pickle(_pack(), checkpoint_path)
            print(f'[spt]   checkpoint -> {checkpoint_path}')

    return _pack()
