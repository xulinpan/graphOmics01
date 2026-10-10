"""Deployable diagnostic: within each protein, does the cognate transcript separate samples where the
protein is missing from samples where it is observed? Per-protein AUC (Mann-Whitney) of cognate RNA,
observed vs missing (0.5 = missingness unrelated to the transcript; >0.5 = missing samples have lower
transcript, as expected when missingness is abundance-dependent). Uses observed data and RNA only.
Also reports, where the truth of missing entries is known, their mean standardized value."""
import json

import numpy as np
from scipy.stats import mannwhitneyu

from tcsimpute.benchmark import runner as R

srcs = {"SIM": R.source_for("SIM", 201), "PDAC": R.source_for("PDAC", 11, cptac_dir="data_cptac"),
        "LUAD": R.source_for("LUAD", 11, cptac_dir="data_cptac"), "PROCAN": R.source_for("PROCAN", 1, procan_dir="data_procan")}
for k in ("logistic", "threshold", "batch", "mcar15", "widelimits"):
    srcs["SWEEP-" + k] = R.source_for("SWEEP-" + k, 301)
out = {}
for name, s in srcs.items():
    rna, obs, cog, prot = np.asarray(s["rna"]), np.asarray(s["observed"], bool), np.asarray(s["cognate"]), np.asarray(s["protein"], float)
    aucs, zmiss = [], []
    for j in range(obs.shape[1]):
        o = obs[:, j]
        if o.sum() < 5 or (~o).sum() < 5 or cog[j] < 0:
            continue
        x = rna[:, cog[j]]
        u = mannwhitneyu(x[o], x[~o]).statistic
        aucs.append(u / (o.sum() * (~o).sum()))
        mu, sd = prot[o, j].mean(), prot[o, j].std()
        t = prot[~o, j]
        t = t[np.isfinite(t)]
        if t.size and sd > 0:
            zmiss.append(((t - mu) / sd).mean())
    a = np.array(aucs)
    out[name] = {"proteins": int(a.size), "auc_median": float(np.median(a)), "auc_q25": float(np.percentile(a, 25)),
                 "auc_q75": float(np.percentile(a, 75)), "frac_auc_gt_0.6": float((a > 0.6).mean()),
                 "missing_truth_mean_z": float(np.mean(zmiss)) if zmiss else None}
    print(name, out[name])
json.dump(out, open("missingness_diagnostic.json", "w"), indent=1)
