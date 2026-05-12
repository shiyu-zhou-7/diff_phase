"""
Active phase discovery workflow for TFIM.

Outer loop:
  1. Bootstrap an AE around h_init (training/bootstrap.py:bootstrap_ae).
  2. Repeat for active_cfg.max_outer_iters retrains:
       a. Inner block of Adam steps on h with the soft-normalized latent-distance
          loss (_ae_latent_loss from training/optim_h.py).
       b. NaN events trigger a kick + temporary lr boost. 3 consecutive failed
          kicks raise RuntimeError.
       c. Stall (param change < tol for stall_window steps) ends the inner block.
       d. Append the current h to bootstrap_history and retrain the AE on the
          full accumulated history (training/bootstrap.py:retrain_with_history).
  3. Return a results bundle for downstream plotting and pickling.

Public:
  run_active_phase_discovery(h_init, ham_X, ham_ZZ, ham_cfg, ae_cfg, active_cfg, rng_key)

Private helpers:
  _run_inner_block, _apply_kick
"""
from dataclasses import asdict

import jax
import jax.numpy as jnp
from jax import value_and_grad, jit
import optax

from training.bootstrap import bootstrap_ae, retrain_with_history
from training.optim_h import _ae_latent_loss
from utils.io import save_pickle


def _apply_kick(h_prev_safe, last_dir, active_cfg, rng_key):
    """Compute h after a NaN-recovery kick.

    First NaN (last_dir is None) -> random Gaussian kick scaled by nan_jump_noise.
    Subsequent NaN -> momentum overshoot kick = nan_jump_scale * last_dir.
    Returns (h_new, new_rng_key).
    """
    if last_dir is None:
        new_key, sub = jax.random.split(rng_key)
        kick = active_cfg.nan_jump_noise * jax.random.normal(sub)
        return h_prev_safe + kick, new_key
    return h_prev_safe + active_cfg.nan_jump_scale * last_dir, rng_key


def _run_inner_block(
    h_init, ae_params, centroid, ham_X, ham_ZZ,
    ham_cfg, ae_cfg, active_cfg, rng_key,
):
    """Inner Adam loop on h.

    Returns dict with:
      h, h_history, loss_history, grad_history, event_history (per step),
      exit_reason in {'stall', 'max_steps'}, last_dir (or None).
    """
    drop_p = ae_cfg.dropout_p
    opt = optax.adam(ham_cfg.lr)

    @jit
    def step_fn(h, opt_state, key, lr_mult):
        loss, grad = value_and_grad(_ae_latent_loss, argnums=0)(
            h, ham_X, ham_ZZ, centroid, ae_params, drop_p, key,
        )
        updates, new_opt_state = opt.update(grad, opt_state)
        h_new = h + updates * lr_mult
        return h_new, new_opt_state, loss, grad

    h = jnp.asarray(h_init, dtype=jnp.float64)
    opt_state = opt.init(h)

    h_history, loss_history, grad_history, event_history = [], [], [], []
    h_prev_safe = h
    last_dir = None
    boost_remaining = 0
    consecutive_stall = 0
    consecutive_failed_kicks = 0
    key = rng_key

    # EMA of gradient (smooth out dropout noise in the loss).
    # alpha = 1/stall_window so the smoothing horizon matches the stall window.
    ema_alpha = 1.0 / max(1, ham_cfg.stall_window)
    ema_grad = 0.0

    for step in range(ham_cfg.max_steps_block):
        key, sub = jax.random.split(key)
        lr_mult = ham_cfg.nan_lr_mult if boost_remaining > 0 else 1.0

        h_new, opt_state_new, loss, grad = step_fn(h, opt_state, sub, lr_mult)

        bad = (
            not bool(jnp.isfinite(h_new))
            or not bool(jnp.isfinite(loss))
            or not bool(jnp.isfinite(grad))
        )

        if bad:
            consecutive_failed_kicks += 1
            if consecutive_failed_kicks >= 3:
                raise RuntimeError(
                    f"Unrecoverable NaN at inner step {step}, "
                    f"h_prev_safe={float(h_prev_safe):.6f}; "
                    f"last 10 h-values = {h_history[-10:]}"
                )
            h_kicked, key = _apply_kick(h_prev_safe, last_dir, active_cfg, key)
            h = h_kicked
            opt_state = opt.init(h)
            boost_remaining = ham_cfg.nan_lr_steps

            h_history.append(float(h))
            loss_history.append(float('nan'))
            grad_history.append(float('nan'))
            event_history.append('nan_kick')
            continue

        consecutive_failed_kicks = 0
        step_dir = h_new - h

        h = h_new
        opt_state = opt_state_new
        h_prev_safe = h
        last_dir = float(step_dir)
        boost_remaining = max(0, boost_remaining - 1)

        # Stall detection on the EMA-smoothed gradient. The instantaneous grad
        # is inflated by dropout noise inside fetch_latent (~1e-2 even when h
        # has converged); EMA over the stall window removes the noise and
        # approaches the true mean gradient direction (~0 at a local minimum).
        ema_grad = (1.0 - ema_alpha) * ema_grad + ema_alpha * float(grad)
        # Require at least `stall_window` warmup steps before EMA can fire.
        if step >= ham_cfg.stall_window and abs(ema_grad) < ham_cfg.stall_tol_grad:
            consecutive_stall += 1
        else:
            consecutive_stall = 0

        h_history.append(float(h))
        loss_history.append(float(loss))
        grad_history.append(float(grad))
        event_history.append('normal')

        if consecutive_stall >= ham_cfg.stall_window:
            return {
                'h': float(h),
                'h_history': h_history,
                'loss_history': loss_history,
                'grad_history': grad_history,
                'event_history': event_history,
                'exit_reason': 'stall',
                'last_dir': last_dir,
            }

    return {
        'h': float(h),
        'h_history': h_history,
        'loss_history': loss_history,
        'grad_history': grad_history,
        'event_history': event_history,
        'exit_reason': 'max_steps',
        'last_dir': last_dir,
    }


def run_active_phase_discovery(
    h_init, ham_X, ham_ZZ,
    ham_cfg, ae_cfg, active_cfg, rng_key,
    checkpoint_path=None,
):
    """Top-level driver. Returns the run-result bundle (see module docstring).

    If `checkpoint_path` is given, the running bundle is pickled to that path
    (overwrite) after each outer iteration's retrain. Useful for crash safety
    on long runs: the last completed outer iteration is always recoverable.
    """
    rng_key, boot_key = jax.random.split(rng_key)

    # 1. INITIAL BOOTSTRAP
    print(
        f'[active] initial bootstrap at h={h_init:.4f}, '
        f'radius={active_cfg.bootstrap_radius_init}, '
        f'num_samples={active_cfg.num_samples_init}'
    )
    bootstrap = bootstrap_ae(
        h_center=h_init,
        radius=active_cfg.bootstrap_radius_init,
        num_samples=active_cfg.num_samples_init,
        ham_X=ham_X, ham_ZZ=ham_ZZ,
        ae_cfg=ae_cfg,
        rng_key=boot_key,
    )
    bootstrap_history = [float(h_init)]
    current_ae_params = bootstrap['ae_params']
    current_centroid = bootstrap['centroid']
    last_x_train_all = bootstrap['x_train']
    last_h_samples_all = bootstrap['h_samples']
    current_h = float(h_init)

    # Run-level history (one entry per Adam step + synthetic 'retrain' markers)
    h_per_step, loss_per_step, grad_per_step, event_per_step = [], [], [], []

    cfg_snapshot = {
        'h_init': float(h_init),
        'ham_cfg': asdict(ham_cfg),
        'ae_cfg': asdict(ae_cfg),
        'active_cfg': asdict(active_cfg),
    }

    def _pack():
        """Build the current results bundle (used for checkpoint + final return)."""
        return {
            'history': {
                'h_per_step': h_per_step,
                'loss_per_step': loss_per_step,
                'grad_per_step': grad_per_step,
                'event_per_step': event_per_step,
            },
            'bootstrap_history': bootstrap_history,
            'final_h': current_h,
            'final_ae_params': current_ae_params,
            'final_centroid': current_centroid,
            'final_x_train_all': last_x_train_all,
            'final_h_samples_all': last_h_samples_all,
            'cfg_snapshot': cfg_snapshot,
        }

    # 2. OUTER LOOP
    for outer_iter in range(active_cfg.max_outer_iters + 1):
        rng_key, inner_key = jax.random.split(rng_key)
        print(f'[active] outer_iter={outer_iter} inner block starting at h={current_h:.4f}')
        inner = _run_inner_block(
            current_h, current_ae_params, current_centroid,
            ham_X, ham_ZZ,
            ham_cfg, ae_cfg, active_cfg, inner_key,
        )
        print(
            f'[active]   inner exit: reason={inner["exit_reason"]}, '
            f'steps={len(inner["h_history"])}, h_end={inner["h"]:.4f}'
        )
        h_per_step.extend(inner['h_history'])
        loss_per_step.extend(inner['loss_history'])
        grad_per_step.extend(inner['grad_history'])
        event_per_step.extend(inner['event_history'])
        current_h = inner['h']

        if outer_iter == active_cfg.max_outer_iters:
            break

        # Retrain only when the inner block actually stalled. If it exited via
        # 'max_steps' the optimizer was still moving; the next inner block
        # picks up at current_h with the same AE.
        if inner['exit_reason'] != 'stall':
            print(f'[active]   exit_reason={inner["exit_reason"]} -> skip retrain')
            continue

        # 2d: retrain
        bootstrap_history.append(current_h)
        rng_key, retrain_key = jax.random.split(rng_key)
        print(
            f'[active]   retrain with {len(bootstrap_history)} centers: '
            f'{[f"{c:.3f}" for c in bootstrap_history]}'
        )
        retrain = retrain_with_history(
            bootstrap_history=bootstrap_history,
            radius=active_cfg.bootstrap_radius_retrain,
            num_samples_per_circle=active_cfg.num_samples_bootstrap,
            ham_X=ham_X, ham_ZZ=ham_ZZ,
            ae_cfg=ae_cfg,
            rng_key=retrain_key,
        )
        current_ae_params = retrain['ae_params']
        current_centroid = retrain['centroid']
        last_x_train_all = retrain['x_train_all']
        last_h_samples_all = retrain['h_samples_all']

        # Synthetic 'retrain' marker keeps per-step lists aligned
        h_per_step.append(current_h)
        loss_per_step.append(float('nan'))
        grad_per_step.append(float('nan'))
        event_per_step.append('retrain')

        if checkpoint_path is not None:
            save_pickle(_pack(), checkpoint_path)
            print(f'[active]   checkpoint -> {checkpoint_path}')

    return _pack()
