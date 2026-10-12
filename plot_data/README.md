# Manuscript plotting data

Numerical data and standalone plotting scripts for **Autonomous phase discovery**.
This package reproduces the six TFIM / Z2 gauge / XXZ demonstration panels and
the cluster-chain / Chern scaling and rarity figures. It contains plotting data,
not the complete simulation archive. No training, ground-state solver, model
checkpoint, or pickle file is needed to replot.

## Quick start

Download this directory, or check out the `plot-data` branch of `diff_phase`.
With Python 3.10 or later:

```sh
python -m pip install -r plot_data/requirements.txt
python plot_data/verify.py
python plot_data/plot_all.py
```

If you downloaded only this directory, omit `plot_data/` from those paths.
You may run the scripts from any working directory. To choose the output folder:

```sh
python plot_data/plot_all.py --output /path/to/figures
```

Outputs are individual vector PDFs and PNGs, `demonstrations.png` (the six-panel
overview), and the two benchmark figures. NumPy and Matplotlib are the only
plotting dependencies. Figure typography can vary slightly with Matplotlib and
font versions. The overview PNG is a convenience composition; its individual
panels are also available as vector PDFs. No generated figure overwrites the
paper repository. Edit the scripts in `scripts/` to change colors, fonts,
legends, marker sizes, or layout; edit `plot_all.py` for the overview layout.

For the original vector six-panel layout, optionally install Typst and run
(after the default plotting command):

```sh
typst compile --root plot_data plot_data/demonstrations.typ plot_data/figures/results_figure_grid_tfim_z2_xxz_2x3.pdf
```

This optional step uses the original panel dimensions and spacing; it is not
needed to access the data or generate the individual plots.

## Figure-to-data map

The numbering refers to the manuscript snapshot at commit `3f63fbe` in
`shiyu-zhou-7/diffphase_manuscript`; filenames remain useful if numbering changes.

| Figure / panel | Data | Replot script |
| --- | --- | --- |
| Demonstrations (Figure 3), TFIM parameter trajectory | `data/tfim_parameter.npz` | `scripts/tfim_parameter.py` |
| Demonstrations, Z2 parameter trajectory | `data/z2_parameter.npz` | `scripts/z2_parameter.py` |
| Demonstrations, XXZ parameter trajectory | `data/xxz_parameter.npz` | `scripts/xxz_parameter.py` |
| Demonstrations, TFIM latent PCA | `data/tfim_latent.npz` | `scripts/tfim_latent.py` |
| Demonstrations, Z2 latent PCA | `data/z2_latent.npz` | `scripts/z2_latent.py` |
| Demonstrations, XXZ latent PCA | `data/xxz_latent.npz` | `scripts/xxz_latent.py` |
| `results_figure_scale_2x1.pdf` | `data/benchmarks/` | `scripts/benchmarks.py` |
| `results_figure_chern_volume.pdf` | `data/benchmarks/` | `scripts/benchmarks.py` |

Schematic workflow/architecture diagrams are not empirical data and are outside
this package. Their sources remain in the manuscript repository.

## Reading the data yourself

```python
import numpy as np

d = np.load('plot_data/data/tfim_latent.npz', allow_pickle=False)
print(d.files)
xy = d['train_pca']          # (1000, 2), PC1 and PC2
phase = d['train_phase']     # Unicode labels, no Python objects
path = d['traj_pca']         # (41, 2), in temporal order

# Independently reproduce the projection from exported latent vectors.
reprojected = (d['traj_latents'] - d['z_mean']) @ d['pca_components'][:2].T
np.testing.assert_allclose(path, reprojected)
```

All NPZ files contain only numeric or Unicode arrays and can be loaded with
`allow_pickle=False`. The corresponding JSON file lists every array's shape and
dtype, axis labels/limits, source filenames, and SHA-256 fingerprints. Coordinates
are in the plotted Hamiltonian convention; the couplings are dimensionless in
the model's energy units. PCA coordinates are dimensionless latent coordinates.

### Parameter trajectories

For TFIM and Z2, `run{i}_values` contains the plotted coupling (TFIM: `g`; Z2:
`lambda`). The original experiment field is called `h` for both models; no extra
sign flip is applied on export. `run{i}_step` is a zero-based index into the
exported sequence; `run{i}_normalized_step` is `linspace(0,1,N)`.
`run{i}_event` preserves the original event strings and prepends `start`.
Stars mark `retrain` events. The initial parameter is explicitly prepended.
Each run is normalized separately, so equal horizontal coordinates across runs
do **not** imply equal numbers of solver calls.

- TFIM: runs `20260513_210852` (first 300 steps, g0=-0.5),
  `20260511_155044` (all steps, g0=-0.3), and `20260512_101827`
  (all steps, g0=-1.5). Exported lengths including the initial point: 301, 1206, 409.
- Z2: runs `20260518_115424` (first 300 steps, lambda0=-0.09),
  `20260518_115036` (first 450, lambda0=-0.10), and `20260604_092628`
  (first 800, lambda0=-0.75). Lengths: 301, 451, 801.
- XXZ: `prev_params` (1142 rows) and `run2_params` (490 rows) are successive
  trajectory segments. Their columns are **[Delta, h]**, while the plot uses
  **x=h, y=Delta**. The first segment includes the explicitly prepended
  initial point [-1.5,-0.3]. The resumed run is `20260505_235151` (job 437956).
  `nan_kick_origins` and `ae_spots` are the manual annotations used by the
  source figure; they are not claimed to be an exhaustive event log.
  `boundary_Delta` and `boundary_h` preserve the sampled phase-boundary spline;
  `boundary_anchors` preserves its digitized anchors. This background is an
  illustrative interpolation from the literature, not a new simulation result.

### Latent PCA panels

`train_pca` and `traj_pca` are the actual plotted coordinates, not wavefunctions
or unprocessed observations. `train_phase` gives the reference-cloud labels.
`var_ratio` contains fractions (multiply by 100 for percentages).
`pca_components` stores principal axes as rows, and `z_mean` is the fitted mean.
The PCA basis is fitted to the reference cloud, then applied to the trajectory.
The stored signs of the PCA axes are fixed to match the plotted convention.

| Model | Reference points | Trajectory points | Full latent dimension | PC1 / PC2 |
| --- | ---: | ---: | ---: | --- |
| TFIM | 1000 | 41 | 10 | 53.5% / 27.3% |
| Z2 | 1000 | 43 | 30 | 17.4% / 5.2% |
| XXZ | 1250 | 34 | 15 | 51.2% / 28.8% |

Full reference latents are `z_sub` (TFIM), `Z_all` (Z2), and `Z_train` (XXZ).
Full trajectory latents are always `traj_latents`. These small extra arrays
allow checking the projection or making a different PCA view without a model.
TFIM and Z2 also include selected original trajectory step indices and couplings.
The latent paths are subsampled and can start after the explicitly prepended
initial point shown in the parameter panel; see `trajectory_step`.
For XXZ, the packaged latent trajectory is the 34-point observation sequence in
the original all-phases data bundle; no unverified alignment to all 1632
parameter-trajectory rows is asserted.

The TFIM export uses cached AE/PCA artifact
`phase_latent_data_ae_ep1000_20260512_225654.pkl` and the g0=-0.3 run.
It re-evaluates only the 41 selected TFIM ground states locally to recover their
latents; collaborators never need to do that. Z2 uses the varied-seed AE and
cached trajectory latents for job 502930. XXZ uses the 15-dimensional all-phases
AE and data bundle `all_phases_latent_data_20260518_230716.pkl`.

### Scaling and rarity

Files in `data/benchmarks/` are copied unchanged from `scale_benchmarks/` at
code commit `41cdff6919624c3dbe1a3fd882b059895a67be00`:

- `final_res/`: 200 cluster-chain runs, 10 coupling dimensions x 4 budgets x
  5 seeds. `cov` is the number of distinct winding sectors observed.
- `unif_all.json`: five uniform-baseline coverage counts per dimension/budget.
  These are sorted realizations, so their list positions must not be treated
  as seed identities or paired with search seeds.
- `which_res/`, `hist_res/`: additional start-type sector sets and uniform
  histograms for the SPT rarity analysis (included for further plotting).
- `chern_res_M768/`: five Chern evaluation seeds (200--204), ten chunks each.
  Each chunk contains 30,000 paid points per method. Cumulative coverage is
  the **union** of `set_conv` over chunks, not the sum of chunk coverages.
  The older `set_old` uses the unconverged 96x96 grid and is not used in the
  headline plots. The final labels use 768x768 with residual discretization
  uncertainty documented in the original benchmark report.
- `sector_volumes_M768.json`: empirical sector probabilities from the
  one-million-sample Gaussian reference. An absent key means unobserved in
  that reference, not proven probability zero.

Coverage curves use medians with min--max bands over five realizations. The
rarity bars use means over five seeds. Query counts are evaluated Hamiltonian
points, not wall-clock runtime. The verification script checks the 36/40
separated SPT cells, the Chern endpoint medians 207/139, and pooled sets
283/160 (168 after adding the volume reference, leaving 115 search-only sectors).

## Provenance and limits

Exported on 2026-10-12 from the manuscript-producing local scripts. The code
checkout's base commit is `d2f7e4148b31d844769200661f8162542ec18412`; several
plotting scripts had local manuscript-style changes. Their exact content
fingerprints are retained in the per-panel JSON metadata; portable plotting
bodies are included here. `manifest.json` records all packaged data checksums.

`tools/export_local.py` documents the one-time extraction. It requires the
original private/local pickle files and JAX environment and is **not** part of
the collaborator quick start. It intercepts figure saving to export numerical
arrays and never retrains an AE. The original simulations, training data,
checkpoint files, private filesystem paths, and full wavefunctions are not
included. The package supports replotting and verification of the reported
summaries, not independently rerunning the entire scientific experiment.
