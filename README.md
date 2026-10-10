# tcsimpute

**Transcript-conditioned selection imputation of left-censored proteomics.**

In mass-spectrometry proteomics, many values are missing because the peptide fell below a detection
limit, so whether a value is observed depends on the value itself. `tcsimpute` imputes those values
from a paired transcriptome. Its core is a multimodal variational autoencoder whose protein
likelihood is a probit selection model:

- a learned detection limit, with shrunken per-protein deviations and a per-sample offset;
- a small missing-completely-at-random component;
- a per-protein term on the protein's own (cognate) transcript;
- closed-form conditional means for missing values, and importance-weighted predictive intervals.

The protein mean comes either from a linear regression on the cognate transcript and RNA principal
components (`model="linear"`, fast) or from a multimodal variational autoencoder (`model="vae"`, the
default), whose latent variable captures shared sample-level structure.

This repository accompanies the paper *tcsimpute: transcriptome-guided imputation of left-censored
proteomics with a probit selection model* (Pan, Lv and Wang, submitted) and contains everything needed
to reproduce it.

## Installation

```bash
pip install tcsimpute             # from PyPI, once released
pip install "git+https://github.com/xulinpan/graphOmics01"   # from GitHub
pip install -e ".[all]"           # from a clone, with plotting, CPTAC download and tests
```

Python 3.10 or newer is required, with PyTorch 2.1 or newer. A CPU build of PyTorch is enough; a CUDA
build is used when `device="cuda"` is passed. The comparators proDA, imputeLCMD (MinDet, MinProb, QRILC)
and missForest run from their R reference implementations, so the benchmark also needs R 4.3 or newer
with those packages (see `reproduce/README.md`); the imputer itself does not.

## Quick start

```python
import pandas as pd
from tcsimpute import TCSImputer

rna = pd.read_csv("rna.csv", index_col=0)          # samples x genes, log scale, complete
protein = pd.read_csv("protein.csv", index_col=0)  # samples x proteins, log scale, NaN = missing

imp = TCSImputer(limit_scale="raw").fit(rna, protein)     # or TCSImputer(model="linear", ...)
imputed = imp.transform(rna, protein)                       # observed values are kept as they are
lower, upper = imp.predict_interval(rna, protein, level=0.9)
imp.save("imputer.pt")                                      # reload with TCSImputer.load(...)
```

`examples/quickstart.py` runs end to end on simulated data in a few seconds.

From the command line:

```bash
tcsimpute --rna rna.csv --protein protein.csv --out imputed.csv --intervals pi [--model linear]
```

### Choosing options

- **`limit_scale`.** Use `"raw"` for log intensities, where one detection limit applies to all
  proteins. Use `"standardized"` for log ratios (for example TMT data relative to a reference),
  where proteins sit on different scales.
- **Cognate transcripts.** When both inputs are DataFrames, each protein is matched to the gene
  column with the same symbol. The part of a column name before `|` is compared, so `cptac` column
  labels work directly. Pass `cognate=` (one gene column index per protein, `-1` for none) to
  override.
- **Epoch selection.** `validation_fraction` (default 0.2) holds out samples. The epoch is chosen by
  the model's own objective on those samples, so the true values of missing entries are never
  needed.
- **`model`.** `"linear"` fits in seconds and was as accurate as the VAE in most of the paper's
  settings; it is a good default. `"vae"` was better when per-protein detection limits varied widely
  and on PDAC tumours.
- **Model settings.** Every field of `VAEConfig` can be passed as a keyword, e.g.
  `TCSImputer(latent_dim=8)`, `TCSImputer(device="cuda")` or `TCSImputer(likelihood="mar")`.
- **Is the selection model appropriate?** It assumes that missing values are mostly below a detection
  limit. If a protein that is missing in one run is usually quantified in other runs of the same
  sample (sporadic misses, common in DIA data), a missing-at-random model (`likelihood="mar"`) was
  more accurate in the paper. As a quick check, compute for each protein the AUC with which its
  transcript separates observed from missing samples: medians near 0.5 indicate missingness unrelated
  to abundance.

### What to expect

Hidden-entry RMSE in standardized units (mean over 20 grid points; from the paper). The CPTAC
scenarios use real tumour data with an imposed detection mechanism; ProCan has natural missingness and
is scored against the mean of the cell line's other runs.

| Setting | VAE | Linear | Best established comparator | Bayes oracle |
|---|---|---|---|---|
| Simulator | 0.990 | 0.950 | MinDet 1.298 | 0.689 |
| CPTAC PDAC, imposed mask | 1.256 | 1.268 | MinDet 1.542 | n/a |
| CPTAC LUAD, imposed mask | 1.002 | 1.002 | MinDet 1.334 | n/a |
| ProCan DIA, natural missingness | 1.146 | 1.149 | missForest 0.998 | n/a |

The ablations attribute the gains to the selection likelihood and the cognate-transcript term.
Transformer fusion and the contrastive term did not measurably help. Nominal 90% intervals covered
73–84% of hidden values (77–86% after the built-in conformal recalibration), so treat them as
approximate. On ProCan, transcript-aware imputers that assume missingness at random were more
accurate, because many DIA misses are sporadic rather than below a detection limit.

## Reproducing the paper

The scripts in `reproduce/` download the data, run the analyses that are not part of the benchmark
grid and build the paper's figures and tables; `reproduce/README.md` gives the full order. In short:

```bash
# 1. Data (network access; about 0.6 GB for ProCan plus the cptac package downloads)
python reproduce/fetch_cptac_cohort.py --cohort Pdac --out data_cptac
python reproduce/fetch_cptac_cohort.py --cohort Luad --out data_cptac
python reproduce/fetch_procan.py data_procan
python reproduce/export_procan_rna.py            # needs h5py

# 2. The revision grid: 4 primary, 13 sweep and 2 downstream scenarios, 3,084 runs
export TCSIMPUTE_RSCRIPT=/path/to/Rscript         # for proDA, imputeLCMD and missForest
tcsimpute-benchmark --grid revision --out runs_revision --workers 14 \
    --cptac-dir data_cptac --procan-dir data_procan \
    --scenarios SIM PDAC LUAD PROCAN DS-PDAC DS-LUAD sweeps
tcsimpute-scaling --out runs_revision/scaling.json

# 3. Summary with paired comparisons, Friedman/Nemenyi tests, and the extra analyses
tcsimpute-report --runs runs_revision
python reproduce/procan_strata.py
python reproduce/missingness_diagnostic.py
python reproduce/replicate_truth_sim.py
python reproduce/compact_summary.py
python reproduce/make_revision_assets.py --summary summary_compact.json \
    --scaling runs_revision/scaling.json --extra . --out assets
```

The full grid took about 6 hours with 14 worker processes on a 20-thread workstation, most of it
for missForest and the ProCan scenario. Use `tcsimpute-benchmark --out runs --smoke` for a few-minute check of every method.
The original submission's grid is still available with `--grid original`.

Each run writes `predictions.npz` and `result.json` under `runs/<scenario>/rep<r>_split<s>/<method>_seed<t>/`.
`result.json` holds the test metrics, a bootstrap confidence interval, diagnostics, a protocol
fingerprint, software versions and the runtime. Model selection never reads hidden truth, and test
truth is read once per run.

### Package layout

| Module | Contents |
|---|---|
| `tcsimpute.imputer` | `TCSImputer`, `match_cognates` (public API) |
| `tcsimpute.model` | `CensoredMultimodalVAE`, `VAEConfig`, training and prediction |
| `tcsimpute.metrics` | Hidden-entry metrics and bootstrap intervals |
| `tcsimpute.benchmark.data` | Simulator and its variants with Bayes oracle, CPTAC benchmark, frozen protocol |
| `tcsimpute.benchmark.natural` | ProCan natural-missingness benchmark with replicate-based truth |
| `tcsimpute.benchmark.downstream` | Differential-abundance and co-abundance metrics |
| `tcsimpute.benchmark.baselines` | Linear selection regression, Tobit, down-shifted normal, KNN, MIWAE, not-MIWAE, controls |
| `tcsimpute.benchmark.rbridge` | proDA, imputeLCMD and missForest through their R packages |
| `tcsimpute.benchmark.runner` | The run grids (`tcsimpute-benchmark`) |
| `tcsimpute.benchmark.report` | Paired comparisons and rank tests (`tcsimpute-report`) |
| `tcsimpute.benchmark.scaling` | Runtime scaling (`tcsimpute-scaling`) |
| `tcsimpute.benchmark.legacy_v1` | The original model, kept only as a comparator |

## Tests

```bash
pytest -q
```

The 37 tests check the closed-form conditional mean against Monte Carlo, that hidden training values
cannot reach any model input, and that predictions are deterministic. They also cover both model
variants of the public API, save/load, the command line, the ProCan and downstream benchmark code and
the rank tests.

## Citation

See `CITATION.cff`. Until the paper is published, please cite the repository and the preprint.

## License

MIT. See `LICENSE`.
