"""ProCan: hidden-entry error stratified by how often the protein was detected in the cell line's
other runs (low detection rate = near the detection limit; high = a sporadic miss).

Writes procan_strata.json: {method: {stratum: {n, rmse, bias}}} pooled over the 20 grid points.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

D = Path("data_procan")
RUNS = Path("runs_revision/PROCAN")
METHODS = ["vae_full", "selreg", "abl_no_latent", "abl_mar_likelihood", "proda", "missforest_r", "knn_cog",
           "notmiwae_cog", "miwae_cog", "mindet_r", "qrilc_r", "dsn", "cognate_ols", "observed_mean", "tobit"]
STRATA = [(0.0, 1 / 3, "rarely detected (<=1/3 of other runs)"), (1 / 3, 2 / 3, "sometimes (1/3-2/3)"),
          (2 / 3, 1.01, "usually detected (>2/3)")]

X = pd.read_csv(D / "ProCan-DepMapSanger_protein_matrix_6692_replicates.txt", sep="\t", index_col=0)
m = pd.read_csv(D / "ProCan-DepMapSanger_mapping_file_replicates.txt", sep="\t").set_index("Automatic_MS_filename").loc[X.index]
m = m[~m["SIDM"].astype(str).str.startswith("Control")]
runs = {s: list(g.index) for s, g in m.groupby("SIDM", sort=True)}

rate = {}
for rep in (1, 2, 3, 4):
    with np.load(D / f"cache_procan_rep{rep}.npz", allow_pickle=False) as f:
        lines, prots = f["sample_ids"], f["protein_names"]
    R = np.empty((len(lines), len(prots)))
    Xp = X[prots]
    for i, s in enumerate(lines):
        r = runs[s]
        k = (rep - 1) % len(r)
        others = [x for j, x in enumerate(r) if j != k]
        R[i] = Xp.loc[others].notna().mean(0).to_numpy()
    rate[rep] = (dict(zip(lines, range(len(lines)))), R)

acc = {mth: {lab: [] for *_, lab in STRATA} for mth in METHODS}
for d in sorted(RUNS.iterdir()):
    rep = int(d.name.split("_")[0][3:])
    pos, R = rate[rep]
    for mth in METHODS:
        f = d / f"{mth}_seed1" / "predictions.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=False)
        rows = np.array([pos[s] for s in z["sample_ids"]])
        truth, pred, hidden = z["truth"], z["prediction"], z["hidden"]
        ok = hidden & np.isfinite(truth)
        rr = R[rows]
        for lo, hi, lab in STRATA:
            sel = ok & (rr > lo - 1e-9) & (rr <= hi) if lo > 0 else ok & (rr <= hi)
            acc[mth][lab].append((pred[sel] - truth[sel]))
out = {}
for mth, d in acc.items():
    out[mth] = {}
    for lab, parts in d.items():
        if parts:
            e = np.concatenate(parts)
            out[mth][lab] = {"n": int(e.size), "rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(e.mean())}
Path("procan_strata.json").write_text(json.dumps(out, indent=1))
for mth in METHODS:
    print(mth.ljust(20), "  ".join(f"{v['rmse']:.3f}/{v['bias']:+.2f} (n={v['n']})" for v in out[mth].values()))
