# Active Phase Discovery Workflow

Self-contained spec for the TFIM active-phase-discovery pipeline, written so it
can be ported to other models (z2gauge, etc.). Where TFIM-specific details
appear, a substitution note is given for porting.

---

## 1. High-level loop

```
1. INITIAL BOOTSTRAP at h_init:
   - Sample a circle of h-values uniformly in [h_init - r_init, h_init + r_init]
   - For each, solve for the ground state
   - Train a fresh AE on these ground states
   - Centroid = mean of normalized latents over training set
   - bootstrap_history = [h_init]

2. OUTER LOOP (up to max_outer_iters + 1 iterations):
   a. Inner Adam loop on h, minimizing L(h) = -‖z(h) - centroid‖ / D
   b. Inner exits on STALL (no movement for stall_window steps) or MAX_STEPS
   c. NaN events trigger a kick (revert + perturb + lr boost);
      3 failed kicks → RuntimeError
   d. If outer_iter == max_outer_iters: break
   e. RETRAIN-ON-STALL:
      - If inner exit_reason != 'stall', skip retrain (keep same AE, continue)
      - Otherwise:
        * bootstrap_history.append(current_h)
        * For each h_center in bootstrap_history, sample a tight circle of
          width r_retrain, solve for ground states
        * Concatenate all circles' ground states into x_train_all
        * Train a FRESH AE from scratch on x_train_all (no warm-start)
        * Centroid = mean of normalized latents over x_train_all

3. RETURN bundle (history of every step + final AE/centroid).
```

**Porting note (z2gauge)**: "ED ground state" → ITE ground state. Substitute the
appropriate solver in `sample_circle`.

---

## 2. Two-budget bootstrap

- `num_samples_init = 1000` — samples on the FIRST circle (initial bootstrap only)
- `num_samples_bootstrap = 500` — samples per circle on every retrain (applied
  to every center in `bootstrap_history`, including re-sampling the initial
  center)

Per-circle budget means: with `N` history centers, the retrain training set is
`N × 500`. Total cost grows linearly with history. The asymmetric `1000`-vs-`500`
exists only because the FIRST AE has to learn from a single circle.

---

## 3. AE architecture

```
Input        2^N  (= dim Hilbert space, problem-dependent)
  ↓ Dense → ReLU → Dropout(0.1)
Hidden       500
  ↓ Dense → (soft-normalize: z / sqrt(‖z‖² + ε))
Latent       10                ← bottleneck
  ↓ Dense → ReLU → Dropout(0.1)
Hidden       500
  ↓ Dense → (L2-normalize)
Output       2^N
```

- Symmetric MLP, `layer_widths = (D_in, 500, 10, 500, D_in)`
- Activation: ReLU between hidden layers
- Dropout `p = 0.1` (active during training; disabled at encode/eval time)
- Encoder output: **soft-normalize** `z = z̃ / sqrt(‖z̃‖² + ε)`, `ε = 1e-8`. Smooth
  at `‖z‖ → 0` (sidesteps the eigh-degeneracy NaN at h ≈ 0).
- Decoder output: strict L2-normalize
- Init: zero-mean Gaussian, scale 0.01

The latent dimension is set empirically. For TFIM, `latent_dim = 10` reproduces
the loss of `latent_dim = 20` with no penalty (effective rank is ~4–5; smaller
bottleneck concentrates variance into fewer PCs).

**Porting note**: `D_in` is the Hilbert dimension. For z2gauge on an N×M
plaquette lattice, `D_in = 2^(N×M)` (or whatever the basis dim is). Other layer
widths typically don't need to change.

---

## 4. Loss function

```python
def ae_loss(params, x, drop_p, key, center_coeff=1e-3):
    eps = 1e-12
    x = x / (‖x‖ + eps)                          # unit-norm input
    x_hat = autoencoder(params, x, drop_p, key)  # encode → decode → L2-norm

    overlap = ⟨x, x_hat⟩                          # per-sample dot product
    recon   = 1 − mean_batch(overlap²)            # quantum fidelity (sign-invariant)

    z       = encoder(params, x, drop_p=0, key)   # no-dropout latent
    var_pen = mean_dims(Var_batch(z_i))           # spread of each latent dim

    return recon + center_coeff * var_pen
```

**Two terms:**

- **Reconstruction (quantum fidelity)**: `1 − mean(⟨ψ|ψ̂⟩²)`. Squaring the
  overlap quotients out the unphysical `|ψ⟩ ↔ −|ψ⟩` sign indifference —
  critical for symmetry-broken phases where `|+⟩` and `|−⟩` are physically the
  same state. This is the natural Riemannian metric on projective Hilbert space.
  Minimum 0 (perfect overlap up to sign); maximum 1 (orthogonal).

- **Latent-variance penalty**: `center_coeff × mean_d Var(z_d)`. Default
  `1e-3`. Keeps the per-phase cluster from spreading out as a cheap
  reconstruction shortcut.

The two terms are summed with no weighting beyond the `center_coeff` multiplier.
The reconstruction dominates at training time; the variance penalty just keeps
the latent geometry well-behaved.

---

## 5. Inner-loop h-optimization

For each outer iteration, build an Adam optimizer and JIT a per-step update on:

```
L(h) = -‖z(h) - centroid‖₂ / D
```

where:
- `z(h) = fetch_latent(ae_params, ground_state(h), drop_p, key, normalize=True, eps=1e-8)`
- `D = latent_dim`
- Negative sign so Adam minimizes ⇒ maximizes latent distance from the centroid.

Per step, after computing `(h_new, loss, grad)`:

- Track `last_dir = h_new - h_prev` (used only by the NaN kick, not by stall
  detection)
- Track `ema_grad = (1 − α)·ema_grad + α·grad`, with `α = 1/stall_window`.
  This is an EMA of the gradient over the stall window.
- Track `consecutive_stall`: increments when `step ≥ stall_window`
  **AND** `|ema_grad| < stall_tol_grad`; resets otherwise.
- Track `consecutive_failed_kicks` for NaN handling.

**Why EMA(grad), not `|last_dir|`?** Dropout in `fetch_latent` injects per-step
noise (`|step_dir| ~ 1e-3`, `|grad| ~ 2e-2`) that swamps `param_tol_change`
even when h has converged, so a per-step magnitude check never fires. The EMA
averages the dropout noise out; the mean gradient direction does approach zero
near a local minimum, which is the right "stall" signal. `param_tol_change`
remains in the config as a vestigial field and is not consulted by the inner
loop.

**Exit conditions:**
- `consecutive_stall ≥ stall_window` → `exit_reason='stall'` (triggers retrain)
- Step `max_steps_block` reached → `exit_reason='max_steps'` (skip retrain,
  continue with same AE next iteration)

**Stall-only retrain**: the retrain in step 2(e) only fires on `exit_reason='stall'`.
If `'max_steps'`, the outer loop just continues — `current_h` carries over to the
next iteration with the same AE. `max_outer_iters` becomes a cap on outer
iterations, not retrains; actual retrains range from 0 to `max_outer_iters`.

---

## 6. NaN kick recovery

Per step, after computing `(h_new, loss, grad)`:

```
if any of {h_new, loss, grad} is not finite:
    if last_dir is None:
        kick = nan_jump_noise * Normal(0, 1)        # first NaN: random Gaussian
    else:
        kick = nan_jump_scale * last_dir            # subsequent: momentum overshoot
    h = h_prev_safe + kick
    opt_state = optax.adam(...).init(h)             # reset Adam moments (corrupted by NaN)
    boost_remaining = nan_lr_steps                  # next N steps use lr × nan_lr_mult
    consecutive_failed_kicks += 1
    if consecutive_failed_kicks >= 3:
        raise RuntimeError(f"Unrecoverable NaN at h={h_prev_safe}")
```

Defaults: `nan_jump_noise = 0.05`, `nan_jump_scale = 2.0`, `nan_lr_mult = 10.0`,
`nan_lr_steps = 5`.

---

## 7. Output bundle (returned by workflow)

```python
{
    'history': {
        'h_per_step':     list[float],
        'loss_per_step':  list[float],
        'grad_per_step':  list[float],
        'event_per_step': list[str],           # 'normal' | 'nan_kick' | 'retrain'
    },
    'bootstrap_history':   list[float],         # [h_init, h_stall1, h_stall2, ...]
    'final_h':             float,
    'final_ae_params':     params,
    'final_centroid':      array,
    'final_x_train_all':   array,               # for downstream PCA
    'final_h_samples_all': array,
    'cfg_snapshot':        {ham_cfg, ae_cfg, active_cfg, h_init, seed},
}
```

The `'retrain'` event tag is a **synthetic marker** appended between blocks
when a retrain happens (with `h=current_h, loss=NaN, grad=NaN`). Keeps
`len(event_per_step) == len(h_per_step)` and lets plotting read retrain
locations directly as `event == 'retrain'`.

---

## 8. Checkpoint + log

- `run_active_phase_discovery(..., checkpoint_path=...)`: after each retrain,
  the workflow pickles the current bundle to `checkpoint_path` (overwrite mode).
  Crash anywhere ⇒ last completed retrain is on disk.
- Main script tees stdout to a `.log` file (line-buffered) via a small inline
  `_Tee` class. Restored in a `finally` block.
- **Three timestamped outputs per run**:
  - `{prefix}_{ts}.pkl`        — final bundle
  - `{prefix}_{ts}_ckpt.pkl`   — checkpoint (overwrite each retrain)
  - `{prefix}_{ts}.log`        — full stdout transcript

---

## 9. Config knobs (defaults)

### HamConfig (TFIM-specific name; rename for z2gauge)

| Field | Default | Purpose |
|---|---|---|
| `N` | 10 | chain / lattice size |
| `J` | −1.0 | coupling constant |
| `h_init` | −0.4 | start point for h-optim |
| `seed` | 83948 | master RNG |
| `lr` | 0.01 | Adam lr on h |
| `max_steps_block` | 200 | inner-loop cap |
| `stall_window` | 25 | consecutive low-movement steps to declare stall; also the EMA window |
| `stall_tol_grad` | 5e-3 | threshold on \|EMA(grad)\| for stall detection |
| `stall_tol_loss` | 1e-4 | unused; vestigial |
| `param_tol_change` | 1e-4 | unused; vestigial (replaced by EMA-grad stall check) |
| `nan_lr_mult` | 10.0 | lr multiplier during NaN recovery |
| `nan_lr_steps` | 5 | duration of lr boost |

### AEConfig

| Field | Default | Purpose |
|---|---|---|
| `seed` | 83948 | AE weight init |
| `layer_widths` | (D_in, 500, 10, 500, D_in) | architecture; latent_dim = 10 |
| `dropout_p` | 0.1 | encoder/decoder dropout |
| `lr` | 1e-4 | AE Adam lr |
| `epochs` | 50000 | AE training epochs |
| `init_scale` | 0.01 | weight init magnitude |
| `center_coeff` | 1e-3 | latent variance penalty weight |

### ActiveConfig

| Field | Default | Purpose |
|---|---|---|
| `bootstrap_radius_init` | 0.3 | initial circle radius |
| `bootstrap_radius_retrain` | 0.3 | retrain circle radius (every circle uses this) |
| `num_samples_init` | 1000 | initial-circle sample count |
| `num_samples_bootstrap` | 500 | per-circle sample count on retrain |
| `max_outer_iters` | 5 | outer-loop cap |
| `nan_jump_scale` | 2.0 | NaN kick: momentum-overshoot multiplier |
| `nan_jump_noise` | 0.05 | NaN kick: first-NaN Gaussian scale |

---

## 10. File layout

```
<model>/scripts/
    configs/config.py                              # HamConfig, AEConfig, ActiveConfig
    hamiltonians/<model>.py                        # H builder + ground-state solver
    models/autoencoder.py                          # init_params, encoder (soft-normalize),
                                                   #   decoder, autoencoder, fetch_latent, ae_loss
    training/
        ae_train.py                                # train_autoencoder (threads center_coeff)
        bootstrap.py                               # sample_circle, train_ae_and_centroid,
                                                   #   bootstrap_ae, retrain_with_history
        optim_h.py                                 # _ae_latent_loss (soft-normalized)
    workflows/
        active_phase_discovery.py                  # run_active_phase_discovery + helpers
    main_active_phase.py                           # entry point: env vars → configs → run → pickle
    plot_active_phase_trajectory.py                # h vs step
    plot_active_phase_pca_trajectory.py            # latent PC1-PC2 with phase background
    analyze_phase_latent.py                        # standalone latent-space analysis
<model>/slurm/run_active_phase.sbatch              # cluster submission, H_INIT-overridable
<model>/data/                                      # run artifacts (gitignored *.pkl)
<model>/figures/                                   # plot outputs
```

---

## 11. Plot scripts

### `plot_active_phase_trajectory.py` — h vs step

- Reads a bundle pickle (CLI positional arg; glob-latest if omitted).
- Plots `h_per_step` vs step index (synthetic 'retrain' markers included in
  index — keeps positions readable directly).
- Per-event marker colors on top of the line:
  - `'normal'`     → `tab:blue` dot, size 2.5
  - `'retrain'`    → `tab:orange` dot, size 7 (stands out on the blue line)
  - `'nan_kick'`   → red `X`, size 8 (different shape from retrains)
- Phase boundary: dashed gray axhline at `h = phase_boundary` + small gray
  label hugging the line.
- SSB / PARA labels in **mediumorchid** (`#BA55D3`) at the right edge.
- figsize=(6.5, 4.5), dpi=300, `matplotlib.use('Agg')`.
- Output: `{figures}/active_phase_trajectory_{tag}_{ts}.pdf` (timestamp parsed
  from bundle filename so figure pairs with bundle).

### `plot_active_phase_pca_trajectory.py` — latent PC1-PC2

- Reads a bundle pickle (same CLI convention).
- Background sampling: 100 fixed `linspace` h-values per phase, ED-solve,
  encode through final AE, fit PCA(2) on the 200 background latents (PCA fit
  on background only — trajectory is a "visitor" projected onto these axes).
- Plot:
  - Background SSB scatter (`tab:blue`, alpha 0.45, small dots)
  - Background PARA scatter (`tab:red`, alpha 0.45, small dots)
  - Trajectory: subsampled to ≤40 points (filtering out 'retrain' markers, keeping
    NaN kicks). Connected `dimgray` line, dot markers with **alpha gradient
    0.2 → 1.0** so the time direction is visible without arrows.
  - **Start marker**: lime green square (s=120) at first trajectory point
  - **End marker**: black star (s=180) at last trajectory point
  - **Retrain markers**: `tab:orange` diamonds at trajectory positions
    closest to each `bootstrap_history[1:]` entry
  - **NaN kicks**: red `X` markers at any NaN-kick trajectory points
  - **Centroid**: black `+` at the projected `final_centroid`
- Axes labels include explained variance per PC: `'PC1 (XX%)'`, `'PC2 (XX%)'`.
- PCA computed via `numpy.linalg.svd` directly (project venv has no sklearn).

### `analyze_phase_latent.py` — diagnostic

Stand-alone analysis (not part of the active-phase pipeline). Samples
500 SSB + 500 PARA training points + 100+100 test points, trains a fresh AE
with the same hyperparameters, computes PCA(top-10), and plots:

- Histograms of PC1..PCk colored by phase
- PC1-PC2 scatter colored by h (single colormap across phases, with phase
  boundary marked on the colorbar)

Useful for sanity-checking that the loss + architecture produce a meaningful
latent geometry before running the full active-phase pipeline.

### `analyze_phase_latent_from_data.py` — combined-AE + trajectory overlay

Similar to `analyze_phase_latent.py`, but instead of sampling fresh data it
loads the pre-computed wavefunction pickles (`data/data_ssb_debiased.pkl` +
`data/data_para.pkl`) and trains the AE on the full union of those (1000+1000).
Then:

- Subsamples 500 SSB + 500 PARA latents for PCA / histograms / scatter so the
  scatter doesn't get visually saturated.
- Plots histograms of PC1..PC5 by phase.
- PC1-PC2 scatter colored by `h` with a single sequential colormap (default
  `Blues`) over the codebase-convention range `[-2, 0]`, with a vertical `h`
  colorbar.
- **One per-trajectory figure per bundle** when bundle paths are passed: the
  trajectory is ED-solved at each kept step, encoded with **this combined AE
  only** (the bundle's own `final_ae_params` is deliberately ignored), and
  projected onto the same PCA basis. Trajectory styling:
  - Time-graded line + markers via `LineCollection` (default colormap
    `Oranges_trunc`, light at start → dark at end).
  - Start and end squares color-matched to the trajectory endpoints.
  - Solid-triangle direction arrows every `ARROW_STRIDE` kept points
    (default 5; ~8 arrows per trajectory).
  - Horizontal step colorbar at the bottom labeled "step (0 → final)".
  - Boxed text annotation in the lower-right showing `g_0` and `g_final`
    for that trajectory.

**CLI per-bundle step truncation:** each positional bundle argument can take a
`:N` suffix, e.g. `path/to/bundle.pkl:300`, to truncate that trajectory to its
first N entries (counts include synthetic 'retrain' markers, matching the step
indexing used by `plot_active_phase_trajectory.py`). Without `:N` the full
trajectory is used. The truncation is encoded into the output filename as
`first{N}` so truncated vs full plots don't collide.

**Output filename disambiguation:** to keep figures from two different runs at
the same `h_init` (e.g., the same `g_0 = -1.5` with different seeds or dates)
from overwriting each other, the source bundle's own timestamp is embedded in
the output filename — `phase_latent_data_pc12_scatter_h{h_tag}_{bundle_ts}[_first{N}]_ep{N}_{ts}.pdf`.

Env vars: `EPOCHS` (override AE training epochs), `REUSE_AE=<path>` (skip
training; load a previously saved `phase_latent_data_ae_ep{N}_<ts>.pkl`),
`ARROW_STRIDE` (arrow density).

Outputs: `phase_latent_data_pc_histograms_ep{N}_<ts>.pdf`,
`phase_latent_data_pc12_scatter_h{tag}[_first{N}]_ep{N}_<ts>.pdf` (one per
bundle), plus a reusable artifact `data/phase_latent_data_ae_ep{N}_<ts>.pkl`.

Useful for visualizing where a single active-phase trajectory sits relative to
a phase-comprehensive latent space (the AE in the active-phase pipeline only
ever saw the local bootstrap circles, so it can't be used as that reference).

### `plot_combined_active_phase_trajectory.py` — multi-run h vs (normalized) step

Combined h-trajectory figure overlaying multiple active-phase runs on a shared
x-axis normalized to `[0, 1]` (so all runs end at the right edge of the plot
regardless of their actual step count). Configured by a hardcoded `RUNS` list
at the top of the script — each entry is `{path, max_steps, color, label}`.
Set `max_steps=None` to use the full trajectory; set an integer to truncate.

Each trajectory:
- Per-step dot scatter (no connecting line) in the run color.
- A filled square in the run color marks the start at `(x=0, h=h_init)`.
- A filled triangle in the run color marks the end at `(x=1, last plotted h)`.
- Retrain markers (synthetic `'retrain'` events) are drawn as black-edged stars
  in the run color.

Common annotations: dashed phase boundary at `g = -1`, mediumorchid SSB / PARA
labels at the right edge with their left edges aligned. Legend at lower-left
with 3 columns (tight `columnspacing` / `handletextpad`).

Output: `figures/combined_active_phase_trajectory_<ts>.pdf`.

Pairs naturally with `analyze_phase_latent_from_data.py`'s `:N` step
truncation: feed the same `(path, max_steps)` pairs to both scripts so the
h-space and PC-space figures show exactly the same step ranges for each run.

---

## 12. Verification

End-to-end smoke (small budget — `EPOCHS=200`, `BOOTSTRAP_N=20`):

```
H_INIT=-0.5 EPOCHS=200 INIT_N=20 BOOTSTRAP_N=15 MAX_OUTER=1 \
    python main_active_phase.py
```

Should:
- Print env-var-driven config
- Run initial bootstrap → outer loop (1-2 inner blocks)
- Write bundle + ckpt + log to `data/`
- Exit cleanly with summary line

Production budget (cluster):

```
H_INIT=-1.5 python main_active_phase.py    # uses defaults: EPOCHS=50000, etc.
```

Per outer iteration: ~30 min AE training (10000 epochs ≈ ~15 min on cluster
nodes; scale linearly). Full run with 5 retrains: ~3–5 hours wall-clock.

---

## 13. Porting to z2gauge — substitutions checklist

| TFIM piece | z2gauge equivalent |
|---|---|
| Hamiltonian: `H(h) = J·ΣZZ + h·ΣX`, J=−1 | Z2 plaquette Hamiltonian (already exists in `z2gauge/scripts/hamiltonians/`) |
| Solver: `gd_solver_ed` (ED via `jnp.linalg.eigh`) | ITE-based solver (already used in `z2gauge` to avoid `eigh` NaN — see memory `project_z2_ite.md`) |
| Phase boundary: `\|h\| = 1` | `\|h\| = 0.3` (per memory `project_z2gauge_phase_boundary.md`) |
| Sampling ranges (background analysis) | SSB → "deconfined" `h ∈ [−0.29, −0.01]`; PARA → "confined" `h ∈ [−1.0, −0.31]` |
| Data files | `z2gauge/data/data_ite_*.pkl`; **do not assume sign conventions match** (TFIM has a known sign quirk; check z2gauge similarly) |
| Hilbert dim | 2^(N_links) — substitute as the AE's `D_in` |
| Plot labels | Replace "SSB" / "PARA" with "deconfined" / "confined" or domain-appropriate names |

**Things that probably DON'T need to change**:
- AE architecture (`(D_in, 500, 10, 500, D_in)`, layer_widths is parametric)
- Loss function (fidelity + variance penalty — generic for wavefunctions)
- Outer-loop / stall / kick logic (model-agnostic)
- Output bundle structure (model-agnostic)
- Plot scripts' style and conventions

**Things that may need tuning**:
- `bootstrap_radius_init` / `bootstrap_radius_retrain` — should match the
  scale of the z2gauge `h` parameter (boundary at 0.3 vs 1.0 → may want
  smaller radii proportionally, e.g. 0.1 / 0.03)
- `ham_cfg.lr` (Adam lr on h) — may need re-tuning at z2gauge's smaller h scale
- `param_tol_change` — same scale dependence
- ITE solver is slower than ED → may want smaller `num_samples_init/bootstrap`
  per circle (e.g. 200 / 100) to keep wall-clock manageable

---

## 14. Memory references

- `feedback_no_cross_phase_loss.md` — never reintroduce a two-centroid loss
  using cross-phase data ("cheating")
- `feedback_tfim_active_phase_history.md` — per-circle budget + all-history
  retrain + stall-only retrain (the three load-bearing design choices)
- `project_tfim_encoder_norm.md` — soft-normalize z(h) via `eps=1e-8`
- `project_tfim_active_phase_design.md` — current pipeline summary
- `project_z2gauge_phase_boundary.md` — z2gauge boundary at `|h|=0.3`
- `project_z2_ite.md` — z2gauge uses ITE not ED
