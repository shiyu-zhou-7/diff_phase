# Equal-budget benchmarks at scale

Code and data behind Sec. IV of the manuscript ("Phase discovery at scale"):
the autonomous search measured against uniform random sampling at strictly
equal numbers of ground-state evaluations, on two free-fermion families.

Both families are solved **in their free-fermion representation** — closed-form
ground states, no diagonalisation at any step — which is what makes budgets of
3e5 queries with five seeds per point affordable.

## Accounting (identical for both methods)

One query = one ground state evaluated. The uniform baseline is drawn at
exactly the search's budget, from the rotation-invariant Gaussian measure on
the coupling sphere. Phase labels are computed afterwards, by us, for both
methods alike; nothing in the search objective ever sees a label.
Hyperparameters came from a successive-halving bracket run on seed families
disjoint from every reported number.

## `spt/` — generalized cluster chain, class BDI, L = 500, d = 20..200

The Jordan-Wigner image of the cluster chain is a Majorana bilinear
`H = -sum_a c_a sum_j i b_j a_{j+a}` with `c_0 = -t_0`, `c_{a>=1} = t_a`.
Note the sign at alpha = 0: `i b_j a_j = -sigma^x_j`, whereas
`i b_j a_{j+a}` reproduces the spin string for `a >= 1`. Verified term by term
in `scripts/jwcheck.py` (ratios -1, +1, +1, +1). This matches the
`flip alpha=0 sign` correction already in the main `spt/` tree.

- label: winding number = roots of `sum_a c_a z^a` inside the unit disc, from a
  companion matrix. Exact integer, no grid, no resolution error.
  (`scripts/cluster_ed_general.py: winding_label_phys_g`, which flips `t[0]`)
- state: in the antiperiodic sector the Majorana matrix is a twisted circulant,
  so the ground-state correlations are its polar factor,
  `g(r) = (1/L) sum_k [f(k)/|f(k)|] e^{-ikr}`. These L numbers are what the
  encoder receives. Verified against an explicit polar decomposition to 2e-14;
  `||g||_2 = 1` exactly, which is why the fidelity loss applies unchanged.

```
scripts/jed_final.py         production runs (SW_D, SW_BUDGET, SW_LAZY, SW_SEED)
scripts/jed_which.py         start decomposition, e_0 vs random start
scripts/unif_base.py|_one.py uniform baselines
scripts/cluster_ed_general.py  winding label + Majorana matrix
scripts/plot_main.py|plot_rarity.py   the standalone-report figures
slurm/run_*.sbatch           the array jobs as submitted
data/final_res/f_*.json      5 seeds x 4 budgets x 10 values of d  (keys: d, L, lazy, budget, start, seed, cov)
data/which_res/w_*.json      start-type runs, with the full sector sets
data/hist_res/h_*.json       sector histograms, 4e6 uniform draws per d
data/unif_all.json           consolidated uniform baselines
SPT_RESULTS.tex|.pdf         standalone working report
```

Headline: 5-25 more phases at equal budget; in 36 of the 40 (d, budget) cells
the search's worst seed beats the baseline's best realisation; the search at
1e4 solves exceeds uniform sampling at 3e5 (a measured 30x); at d = 140 it
reaches 21 sectors that 4e6 uniform draws never hit.

## `chern/` — generalized Qi-Wu-Zhang insulator, H = 16, 50 couplings

Two-band `H(k) = d(k).sigma`; the occupied-band Chern number is evaluated with
the Fukui-Hatsugai-Suzuki plaquette construction.

**Grid convergence matters here.** The FHS integer converges only once the
Berry flux per plaquette is resolved: at H = 16 a 96^2 grid agrees with 1536^2
on only about a third of points, and biases |C| toward zero. All numbers in the
manuscript use **768^2**, recomputed for both methods from stored trajectories
(`data/chern_res_M768/`); `data/chern_res/` holds the original 96^2 run
summaries for comparison.

```
scripts/chern_fast.py        cached-basis FHS evaluation
scripts/chern_hyperband.py   the search (Trial class) + successive-halving bracket
scripts/chern_sets.py|chern_validate*.py   held-out-seed validation, budget-matched baselines
scripts/chern_dump.py        re-runs the walks and dumps every paid theta
scripts/chern_relabel.py     relabels agent + baseline theta at 768^2 (and 96^2, for the shift)
scripts/chern_volref.py      1e6-sample sector-volume reference
scripts/chern_convsym.py     checks the residual label error is symmetric between methods
scripts/degree.py            C as a signed count on the product grid of zeros
scripts/complexity.py        hyperplane-arrangement bound on the realisable sign patterns
slurm/run_*.sbatch           the array jobs as submitted
data/chern_res_M768/relab_s<seed>_c<chunk>.json   converged relabels, 3e4-budget chunks (nested prefixes)
data/sector_volumes_M768.json                     volume reference at 768^2
chern_report.tex|.pdf        standalone working report
```

Headline: 207 vs 139 sectors at 3e5 evaluations (median of five fresh seeds),
margin growing from +21 at 3e4; pooled, 283 sectors against 160, with 123 found
only by the search and none found only by sampling; 259 of the 283 have no
known closed-form realisation in this family.

## Reproducing the manuscript figures

`make_benchmark_figs.py` builds Figs. 4 and 5 from the JSONs here. It expects
the data root to point at this folder; edit `G` at the top.

## Not included

- theta trajectory dumps (120 MB per seed) — with these, any future relabel is
  pure post-processing and no walk has to be re-run
- the autoencoder and training modules, which are imported from `spt/scripts`
  (`models.autoencoder`, `training.ae_train`); the `sys.path` lines at the top
  of the run scripts point at the cluster copy of that tree
