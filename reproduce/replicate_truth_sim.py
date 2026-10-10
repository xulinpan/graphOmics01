"""Does scoring against a replicate-based truth (as in ProCan) change the ranking?

For each simulator grid point, five further 'runs' of every sample are simulated: x + N(0, 0.2^2)
technical noise, each detected with the simulator's probit mechanism (slope 0.5, MCAR 2%, global
limit d0, a fresh run offset N(0, 0.3^2)). The replicate truth of a hidden test entry is the mean of
its detected replicate values; entries never detected are unscored. Existing test predictions are
scored against (a) the true hidden values and (b) the replicate truth on its scored subset.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from tcsimpute.benchmark import runner as R

METHODS = ["oracle", "vae_full", "selreg", "abl_no_latent", "abl_mar_likelihood", "missforest_r", "knn_cog",
           "miwae_cog", "notmiwae_cog", "cognate_ols", "proda", "mindet_r", "observed_mean"]
acc = {m: {"true": [], "rep": [], "rep_bias": []} for m in METHODS}
zstats = []
for rep in R.REVISION_GRID["SIM"]["replicates"]:
    src = R.source_for("SIM", rep)
    d0 = src["mechanism"]["d0"]
    rng = np.random.default_rng(10_000 + rep)
    X = np.asarray(src["protein"])
    n, p = X.shape
    runs = X[None] + rng.normal(0, 0.2, (5, n, p))
    off = rng.normal(0, 0.3, (5, n, 1))
    det = rng.uniform(size=runs.shape) < 0.98 * norm.cdf((runs - d0 - off) / 0.5)
    cnt = det.sum(0)
    rep_truth = np.where(cnt > 0, (runs * det).sum(0) / np.maximum(cnt, 1), np.nan)
    for split in R.REVISION_GRID["SIM"]["splits"]:
        P = R.protocol_for("SIM", rep, split)
        te = P.rows("test")
        rt = (rep_truth[te] - P.protein_mean) / P.protein_scale
        hidden = ~P.observed[te]
        tr = P.truth["test"]
        ok = hidden & np.isfinite(rt)
        zstats.append([float(tr[hidden].mean()), float(tr[ok].mean()), float(rt[ok].mean()), float(ok.sum() / hidden.sum())])
        for m in METHODS:
            f = Path("runs_revision/SIM") / f"rep{rep}_split{split}" / f"{m}_seed1" / "predictions.npz"
            if not f.exists():
                continue
            pred = np.load(f)["prediction"]
            acc[m]["true"].append(np.sqrt(np.mean((pred - tr)[hidden] ** 2)))
            acc[m]["rep"].append(np.sqrt(np.mean((pred - rt)[ok] ** 2)))
            acc[m]["rep_bias"].append(np.mean((pred - rt)[ok]))
z = np.array(zstats).mean(0)
out = {"mean_z_all_hidden_true": z[0], "mean_z_scored_true": z[1], "mean_z_scored_replicate_truth": z[2],
       "scored_fraction": z[3],
       "methods": {m: {k: float(np.mean(v)) for k, v in d.items() if v} for m, d in acc.items()}}
json.dump(out, open("replicate_truth_sim.json", "w"), indent=1)
print({k: round(v, 3) for k, v in out.items() if k != "methods"})
for m, v in out["methods"].items():
    print(m.ljust(20), {k: round(x, 3) for k, x in v.items()})
