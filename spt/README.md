# SPT cluster-chain backend for diffphase

An exactly solvable, dimension-tunable benchmark model added as a diffphase solver
backend. A generalized cluster (stabilizer) chain whose phase diagram is known in
closed form, so the autonomous discovery loop can be graded against an **analytic
ground truth**.

```
H(t) = - sum_{alpha=0}^{d-1} t_alpha * sum_i  Z_i X_{i+1}...X_{i+alpha-1} Z_{i+alpha}
        ( + optional kappa * sum_i Z_i Z_{i+2}, interacting control, off by default )
```

- `alpha=0`: transverse field `-t_0 sum_i X_i`
- `alpha=1`: Ising `-t_1 sum_i Z_i Z_{i+1}`
- `alpha=2`: cluster `-t_2 sum_i Z_i X_{i+1} Z_{i+2}`
- general `alpha`: `Z_i (X-string of length alpha-1) Z_{i+alpha}`

Opening `d` couplings gives a `d`-dimensional parameter space. Every term is a
product of an even number of `Z`s, so all commute with the global `Z2` parity
`P = prod_i X_i` (the fermion parity); we work in a fixed `P=+1` sector. In the
`Z` basis every term is real (no `Y`), so `H(t)` is real symmetric for all real
`t` (class BDI), the ground state is real, and the topological invariant is an
integer winding number.

## Layout
```
spt/scripts/
  analytic/cluster_exact.py     # the GRADER: winding, gap, boundary taxonomy, k*
  hamiltonians/projector.py     # frozen P-sector change-of-basis (off the autodiff tape)
  hamiltonians/cluster.py       # constant T_alpha; dense-eigh solver; custom-jvp ground state
  hamiltonians/observables.py   # rho0-routed observables + topological diagnostics
  models/autoencoder.py         # JAX feed-forward AE over the feature vector
  training/{ae_train,dataset,optim_t}.py
  workflows/active_phase_discovery.py   # discovery loop (vector t; xxz-style)
  main_active_phase.py          # single-start entry point
  benchmark.py                  # section-5 autonomy benchmark (K starts)
  tests/test_{analytic,solver,gradients}.py   # validation suite (sections 4.1-4.4)
```
Run tests from `spt/scripts/`: `python tests/test_analytic.py`, `test_solver.py`,
`test_gradients.py` (all assert + exit non-zero on failure; no pytest needed).

## Analytic labels (the ground truth)

Jordan-Wigner maps `H` to free Majorana fermions with Bloch symbol
`f(k) = sum_alpha t_alpha e^{i*alpha*k}`, equivalently the polynomial
`g(z) = sum_alpha t_alpha z^alpha` on the unit circle.

- **`winding(t)`** = number of roots of `g(z)` with `|z| < 1`, an integer
  `omega in {0,...,d-1}` (argument principle: the winding of `f(k)` about 0 counts
  the zeros of the analytic `g` enclosed by the unit circle). This is the exact
  phase label **at any L**, one root-find per call. Roots within `tol` of the
  circle are flagged on-circle (gapless / boundary); `tol=1e-6` tolerates numpy's
  ~`1e-8` repeated-root splitting.
- **Phases**: `t_0` dominant -> `omega=0` trivial paramagnet; `t_1` dominant ->
  `omega=1` Z2-broken; `t_2` dominant -> `omega=2` cluster SPT. The SPT has **no
  local order parameter** -- it is detected by string order and a degenerate
  entanglement spectrum (`hamiltonians/observables.py`), not a one-site expectation.
- **`gap_thermo(t)`** = `2*min_k |f(k)|`, the bulk gap that closes at boundaries.

### Boundary taxonomy (closed form)
`analytic/cluster_exact.boundary_crossings(path)` classifies each winding change by
the argument `k*` of the root crossing the unit circle:

| type | hyperplane | closing k* | central charge c | `|Δω|` |
|------|-----------|-----------|------|------|
| `z=+1` | `sum_alpha t_alpha = 0` | `0` | `1/2` | `1` |
| `z=-1` | `sum_alpha (-1)^alpha t_alpha = 0` | `pi` | `1/2` | `1` |
| `complex` | conjugate pair at `e^{+-i k*}` | `arg(root)` | `1` | `2` |

### Validation scope (section 4.1)
ED ground energy / in-sector gap match the closed-form free-fermion result to
machine precision for **d <= 2** and at every pure stabilizer point (all d): there
`E0 = -L`, in-sector gap `= 4` (a *pair* of quasiparticles must be excited to
preserve parity, so the in-sector gap is `2x` the bulk `2*min_k|f|`). The exact
finite-size **d >= 3 PBC** free-fermion *energy* carries parity / longer-range-
hopping corrections that are intentionally out of scope; for `d >= 3` the bulk gap
(`gap_thermo`) drives boundary detection and `winding` remains the exact label, so
benchmark grading is unaffected. Labels are cross-checked independently of the root
finder by the ED diagnostics (string order, entanglement entropy `S ~ omega*log2`,
entanglement-spectrum degeneracy) in `tests/test_solver.py`.

## Differentiability choices (the load-bearing ones)

- **Linear in `t`, constant terms.** `H_sector(t) = sum_alpha t_alpha T_alpha`
  with each `T_alpha` built once and frozen, so `dH/dt_alpha = T_alpha` and the
  energy gradient is exact Hellmann-Feynman, `dE0/dt_alpha = <psi0|T_alpha|psi0>`.
- **Frozen, t-independent P-sector projector.** `P` depends only on the model;
  the change-of-basis is built once in `projector.py`, outside any differentiated
  region. It never lands on the autodiff tape (which would inject a spurious
  connection term).
- **Dense `eigh`, never Krylov, on the differentiated path.** The fixed-sector
  matrix is small (`2^{L-1}`); dense `eigh` gives a correct analytic eigenvalue
  adjoint for free. We do **not** trace autodiff through Lanczos/`eigsh`.
- **Custom-jvp ground state** (`cluster.ground_state_vector`). The stabilizer
  Hamiltonian has a hugely degenerate EXCITED spectrum, and `jnp.linalg.eigh`'s
  eigenvector jvp builds the full `1/(e_i-e_j)` matrix over all pairs -> NaN even
  though only the ground vector is used. The hand-written adjoint uses only the
  ground-vs-excited resolvent `(e0-e_j)` (well defined whenever the ground state is
  non-degenerate, i.e. the in-sector gap is `O(1)` in a cell interior); an optional
  Lorentzian shift `eta` keeps it finite at a closing. With `eta=0` the gradient
  diverges AT a boundary by design -- that divergence is the detector signal.
- **State losses route through `rho0 = |psi0><psi0|`.** All observables are
  expectation values `Tr(rho0 O)` via a signed-permutation quadratic form (no dense
  `2^L` operators), so the eigensolver's arbitrary sign cancels and the AE feature
  vector is gauge invariant. For the differentiated entanglement feature we use the
  **Renyi-2** entropy `-log Tr(rho_A^2)` -- a smooth polynomial in `rho_A`, immune
  to the reduced-DM eigenvalue degeneracies that make a von-Neumann / SVD gradient
  NaN (von Neumann is kept as a forward-only diagnostic; the two agree at the pure
  points, `S ~ omega*log2`).

## Running

Single start (discovery from the trivial corner `e_0`):
```
cd spt/scripts
D=3 L=10 python main_active_phase.py
# smoke: D=3 L=10 EPOCHS=300 INIT_N=80 BOOTSTRAP_N=50 MAX_OUTER=2 python main_active_phase.py
```

Autonomy benchmark (K starts on the sphere, scored vs analytic labels):
```
D=3 K=4 L=10 python benchmark.py
D=8 K=16 L=10 python benchmark.py    # headline d=8 (expensive)
# env knobs: EPOCHS MINI_EPOCHS INIT_N BOOTSTRAP_N MAX_OUTER MAX_STEPS EPS SEED
```
PERFORMANCE: the fixed sector matrix is small (`2^{L-1}`), so JAX's default
all-core threading OVER-subscribes and slows the many tiny `eigh` calls. Cap it:
`OMP_NUM_THREADS=2 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false" ... python benchmark.py`
(this was ~5x faster in practice at L=10).
It writes `../data/spt_benchmark_*.{json,npz}` and a
`../figures/spt_benchmark_*_phases.png` (distinct phases vs cumulative solver
queries, active discovery vs random-search baseline), and reports the three section-5
items: (i) phases-vs-queries, (ii) detected boundary crossings vs analytic roots
with predicted `k*`, (iii) total cost vs the fair analytic sphere-grid count
(`~ (1/eps)^(d-1)`) and the random-search baseline.

## Optional interacting control (appendix)
`KAPPA != 0` adds `kappa * sum_i Z_i Z_{i+2}`, which is quartic under JW (genuinely
interacting -- no middle `X` to absorb the string). The free-fermion analytic
ground truth is then **relinquished**: `winding`/`gap` no longer apply, and
in-sector level crossings can appear at points that are not phase boundaries, so the
"smooth in every cell interior" guarantee weakens to "smooth except at isolated
within-phase crossings." Off by default; the phases persist by perturbative
stability for small `kappa`.
