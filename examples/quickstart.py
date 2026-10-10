"""Quick start: impute left-censored proteins from a paired transcriptome.

Runs in well under a minute on a laptop CPU. It simulates a small cohort with the S2 generator from the
paper, fits the imputer on the observed values only, and compares the imputations with the hidden
truth.
"""

import numpy as np
import pandas as pd

from tcsimpute import TCSImputer
from tcsimpute.benchmark.data import S2Config, simulate_s2

# 1. Data: samples x genes RNA (complete) and samples x proteins with NaN for undetected values.
sim = simulate_s2(seed=0, config=S2Config(n_samples=150, n_genes=1000, n_proteins_generated=300,
                                          n_proteins=200, n_regulators=60))
genes = sim["gene_names"]
rna = pd.DataFrame(sim["rna"], index=sim["sample_ids"], columns=genes)
proteins = [f"{genes[c]}|protein" for c in sim["cognate"]]          # cognate = same gene symbol
truth = pd.DataFrame(sim["protein"], index=sim["sample_ids"], columns=proteins)
protein = truth.where(sim["observed"])                              # NaN = below detection
print(f"{protein.isna().to_numpy().mean():.1%} of protein values are missing")

# 2. Fit and impute. limit_scale="raw" suits log intensities; use "standardized" for log ratios.
imp = TCSImputer(limit_scale="raw", random_state=0).fit(rna, protein)
imputed = imp.transform(rna, protein)
lower, upper = imp.predict_interval(rna, protein, level=0.9)

# 3. Compare with the hidden truth (only possible because the data are simulated).
hidden = protein.isna().to_numpy()
err = (imputed.to_numpy() - truth.to_numpy())[hidden]
naive = (np.nanmean(protein.to_numpy(), axis=0)[None, :].repeat(len(protein), 0) - truth.to_numpy())[hidden]
cover = ((lower.to_numpy() <= truth.to_numpy()) & (truth.to_numpy() <= upper.to_numpy()))[hidden]
print(f"RMSE  imputer: {np.sqrt(np.mean(err ** 2)):.3f}   observed-mean fill: {np.sqrt(np.mean(naive ** 2)):.3f}")
print(f"bias  imputer: {err.mean():+.3f}   observed-mean fill: {naive.mean():+.3f}")
print(f"90% interval coverage of hidden values: {cover.mean():.2f}")
print("selected epoch:", imp.diagnostics_["selected_epoch"],
      "| active latent units:", imp.diagnostics_["active_units_train"])
