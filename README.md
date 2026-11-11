# Nonnegative matrix factorization with Wasserstein distance

Mutational signature extraction with NMF, where the usual KL (Poisson) loss is
replaced by an optimal-transport (Wasserstein) loss over the 96 SBS
trinucleotide channels.

A count matrix `X` (96 channels × samples) is factorised as `X ≈ W H`, with `W`
the signatures (96 × K) and `H` the exposures (K × samples). The KL loss treats
the 96 channels as unrelated bins. The OT loss uses a ground cost
`M` (96 × 96) that says how far apart two channels are, for example
`A[C>T]G` is closer to `A[C>T]A` than to `T[T>G]T`. A recovered signature that
puts mass on a neighbouring channel is then penalised less than one that puts
it on an unrelated channel.

## Repository layout

```
.
├── src/
│   ├── run_WNMF.py                      CLI: fit W-NMF and KL-NMF on a catalogue or on simulated data
│   ├── main_20261111.ipynb              same pipeline as a notebook, step by step
│   ├── create_ground_cost_matrix.ipynb  builds cost_matrix/M1.csv (weighted Hamming cost)
│   ├── create_SBS_graphs.ipynb          builds SBS_single_change_graph/graph_M1.csv (graph cost)
│   ├── cost_matrix/M1.csv               96 x 96 ground cost
│   ├── SBS_single_change_graph/graph_M1.csv  96 x 96 ground cost
│   ├── sampleInput.csv                  example 96 x 200 simulated count matrix
│   ├── config.py                        paths (repository root, COSMIC file) and random seed
│   └── wassersteinDL/                   Python port of the Rolet, Cuturi & Peyré MATLAB toolbox
├── data/                input catalogues, metadata, COSMIC reference signatures
└── install_env.sh       conda environment
```

## Datasets

All catalogues in `data/` are SBS96 count matrices: 96 rows (trinucleotide
contexts `5'[REF>ALT]3'`) and one column per sample.

| Cohort | Catalogue | Metadata | Samples | Unit |
|---|---|---|---|---|
| **TCGA** (whole exome) | `TCGA_WES_SBS96_catalog.csv` | `TCGA_WES_sample_metadata.csv` | 9,493 | tumour exome |
| **PCAWG** (whole genome) | `PCAWG_WGS_SBS96_catalog.csv` | `PCAWG_WGS_sample_metadata.csv`, `PCAWG_consortium_sample_sheet.csv` | 2,780 | tumour genome |
| **Kucab et al. 2019** | `Kucab2019_SBS96_subclone_catalogue.csv` | `Kucab2019_treatment_metadata.csv` | 324 | iPSC subclone exposed to a mutagen |

Other inputs:

- `data/COSMIC/COSMIC_v3.6_SBS_GRCh37.txt`: COSMIC v3.6 SBS reference
  signatures, used to simulate data and to fix the order of the 96 channels.

## Ground cost matrices

Two precomputed costs are stored as CSV files with the 96 SBS labels as row
and column names, both scaled to max 1. Any 96 × 96 matrix in the same format
can be passed to the CLI.

- `src/cost_matrix/M1.csv` (from `create_ground_cost_matrix.ipynb`): weighted
  Hamming distance on the three parts of a channel,
  `M = 1·[5' differs] + 2·[substitution differs] + 1·[3' differs]`.
- `src/SBS_single_change_graph/graph_M1.csv` (from `create_SBS_graphs.ipynb`):
  shortest-path distance on a graph whose nodes are the 96 channels. Two
  channels are joined by an edge of weight 1 when they differ in exactly one of
  the 5' base, REF base, ALT base or 3' base (for example `A[C>T]G`–`A[C>G]G`
  or `A[C>A]A`–`A[T>A]A`). The notebook also plots the graph, the neighbours of
  a channel and shortest paths between channels.

## Setup

```bash
bash install_env.sh          # creates the conda env "wdl" (Python 3.11)
conda activate wdl
conda install -c conda-forge pandas scikit-learn
```

`install_env.sh` installs NumPy, SciPy, matplotlib and Jupyter (plus PyTorch
and GNU Octave, which the current code does not use). The CLI and the notebooks
also need pandas and scikit-learn (KL-NMF baseline, MDS plots).

`src/config.py` sets `maindir` to an absolute path; change it to your clone of
the repository before running. Run everything from `src/`: the scripts import
`config` and `wassersteinDL` from the current directory, and the default paths
are relative to it.

## Running the CLI

```bash
cd src

# simulated data (10 COSMIC signatures, 200 samples), M1 cost
python run_WNMF.py -M cost_matrix/M1.csv -n sim_M1

# your own catalogue, graph cost, 5 signatures
python run_WNMF.py -i sampleInput.csv -M SBS_single_change_graph/graph_M1.csv -k 5 -n sample_graph
```

`run_WNMF.py`:

1. **Reads or simulates `X`.** With `--input`, it reads a count matrix with
   the SBS96 channels as the row index (a samples × 96 file is transposed
   automatically), reorders the rows to the COSMIC order and drops samples
   with no mutations. Without `--input`, it picks `k` random COSMIC
   signatures as `W_true`, draws sparse Dirichlet exposures for `N` samples
   (each signature active with probability 0.5, the first always active) and
   log-normal mutation burdens, and samples counts from a negative binomial
   with dispersion `r`. The choice of signatures is not seeded, so it changes
   between runs; the exposures and counts use `SEED` from `config.py`.
2. **Fits each method `--N-RUNS` times.** W-NMF normalises each sample to a
   probability vector and calls `wasserstein_DL` with entropy regularisation
   on both factors (seeds 100, 101, …). KL-NMF is scikit-learn `NMF` with
   `beta_loss="kullback-leibler"`, multiplicative updates and random
   initialisation on the raw counts; `W` columns are rescaled to sum to 1.
3. **Builds consensus signatures.** The signatures of all runs are pooled and
   clustered with PAM (k-medoids) on cosine distance; the `k` medoids are the
   consensus signatures, and the mean cosine to the medoid within each cluster
   is its cohesion.
4. **Validates (simulated data only, section 3.3 of the report).** Medoids are
   matched one-to-one to `W_true` with the Hungarian algorithm, and TP / FP /
   FN, precision, TPR and F1 are computed over cosine cutoffs from 0.70 to
   1.00.
5. **Refits exposures** with the consensus signatures held fixed: the W-NMF
   coefficient step, or the KL-NMF `H` update, scaled back to mutation counts.
   On simulated data these are compared with the true exposures (per-signature
   correlation and median relative error).
6. **Plots** an MDS map of the run signatures and medoids, and one SBS96 bar
   plot per signature: true signature above the two matched medoids for
   simulated data, or each W-NMF signature above its best KL-NMF match for
   real data.

### Options

| Option | Default | Meaning |
|---|---|---|
| `-M`, `--cost-matrix` | required | 96 × 96 ground cost CSV |
| `-i`, `--input` | none (simulate) | 96 × N count matrix CSV |
| `-k` | 10 | signatures to extract (and to simulate) |
| `-o`, `--outdir` | `./tmp_output/sig_plots` | output folder |
| `-n`, `--runName` | today, `YYYYMMDD` | subfolder of `--outdir` for this run |
| `--N` | 200 | simulated samples |
| `--r` | 50 | negative-binomial dispersion of the simulation |
| `--N-RUNS` | 10 | independent fits per method |
| `--gamma` | 0.02 | entropic regularisation of the Wasserstein loss |
| `--rhoL`, `--rhoD` | 0.05 | entropy weights on exposures and signatures |
| `--stop`, `--D-step-stop`, `--lambda-step-stop` | 1e-3, 5e-5, 5e-4 | stopping tolerances (outer loop, dictionary step, coefficient step) |
| `--alpha` | 0.5 | `alpha` option of `wasserstein_DL` |
| `--verbose` | 0 | solver verbosity |

One W-NMF fit on 96 × 200 takes roughly 15 s.

### Outputs

Written to `<outdir>/<runName>/`:

| File | Content |
|---|---|
| `WNMF_signatures.csv`, `KLNMF_signatures.csv` | consensus signatures, 96 × k |
| `WNMF_exposures.csv`, `KLNMF_exposures.csv` | refitted exposures in mutation counts, k × N |
| `MDS_signatures.png` | MDS map of run signatures (dots), medoids (stars) and true signatures (x) |
| `sbs_plots/*.png` | SBS96 profile plots, one file per signature |
| `X_simulated.csv`, `W_true.csv`, `H_true.csv` | simulated data and ground truth (simulation only) |
| `cosine_to_true.csv`, `TPR_curve.png` | cosine of each matched medoid and TPR over cutoffs (simulation only) |
| `exposure_eval.csv` | exposure correlation and median relative error (simulation only) |

On simulated data, signatures are named `WNMF_<COSMIC>` / `KLNMF_<COSMIC>`
after the true signature they are matched to; on real data they are
`WNMF_sig1`, `WNMF_sig2`, ….

## Notebooks

- `main_20261111.ipynb` runs the same pipeline interactively on simulated data:
  one test fit of each method with the convergence curve, then the validation,
  the exposure refit, the MDS map and the SBS96 plots. Its KL-NMF test fit
  (`fit_KLNMF`) runs one `nndsvda` start plus 10 random starts and keeps the
  lowest generalised KL divergence. The last cell writes `sampleInput.csv`.
- `create_ground_cost_matrix.ipynb` and `create_SBS_graphs.ipynb` build the
  two cost matrices described above.

## `src/wassersteinDL/`

A line-by-line NumPy/SciPy translation of the MATLAB code for Rolet, Cuturi &
Peyré (2016), with the same function names (one module per `.m` file).
`wasserstein_DL` alternates a coefficient step and a dictionary step, each
solved in the Fenchel dual, and solves

```
min_{λ, D}  W_γ(X, D λ) + ρ_L E(λ) + ρ_D E(D)
```

where `E` is the column-wise entropy; with `ρ_L, ρ_D > 0` this is NMF.
`_matlab.py` holds small helpers that mimic MATLAB built-ins, and `_external.py`
provides NumPy stand-ins for `fkmeans`, `pairwiseDistance` and `nnmf`, which the
MATLAB code calls but does not ship (used only by the `kmeans` and `euclidean`
initialisations).

## References

- A. Rolet, M. Cuturi, G. Peyré. *Fast Dictionary Learning with a Smoothed
  Wasserstein Loss.* AISTATS 2016.
  [PMLR v51](https://proceedings.mlr.press/v51/rolet16.html)
- J. E. Kucab et al. *A Compendium of Mutational Signatures of Environmental
  Agents.* Cell, 2019.
- COSMIC Mutational Signatures, v3.6.
  <https://cancer.sanger.ac.uk/signatures/>
