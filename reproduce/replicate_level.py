"""Replicate-level paired comparisons for the primary scenarios (Supplementary Section S9).

Grid points that share a data replicate (a simulated data set, an imposed mask or a ProCan
primary-run choice) or a split are not independent. This script therefore summarises each paired
RMSE difference (method minus reference) at the level of data replicates: the mean over splits within
each replicate, the range of these replicate means, the number of replicates in which the reference
was better, and a two-stage cluster bootstrap interval (replicates resampled with replacement, then
splits within each resampled replicate). The naive bootstrap over grid points, which treats the 20
grid points as independent, is reported alongside for comparison.

The tables are written by make_revision_assets.py (Supplementary Tables S1-S4 and S18).

Usage: python replicate_level.py <runs_dir> <out_json>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCENARIOS = ("SIM", "PDAC", "LUAD", "PROCAN",
             # out-of-family detection masks on CPTAC (Supplementary Section S10); skipped when absent
             "ALT-PDAC-logit", "ALT-PDAC-sporadic", "ALT-LUAD-logit", "ALT-LUAD-sporadic")
REFERENCES = ("selreg", "vae_full")
B = 10_000


def load(runs):
    res = defaultdict(dict)                       # (scenario, method) -> {(rep, split): rmse}
    for sc in SCENARIOS:
        for f in (Path(runs) / sc).glob("rep*_split*/*_seed1/result.json"):
            r = json.loads(f.read_text())
            v = r.get("test", {}).get("rmse")
            if v is not None and not r.get("smoke"):
                res[(sc, r["method"])][(int(r["replicate"]), int(r["split_seed"]))] = float(v)
    return res


def compare(a, b, rng):
    keys = sorted(set(a) & set(b))
    reps = sorted({k[0] for k in keys})
    by_rep = {rp: np.array([a[k] - b[k] for k in keys if k[0] == rp]) for rp in reps}
    rep_means = np.array([by_rep[rp].mean() for rp in reps])
    flat = np.concatenate([by_rep[rp] for rp in reps])
    R = len(reps)
    # within-replicate resampled means, one independent draw per (bootstrap sample, slot): shape (R, B, R)
    M = np.stack([by_rep[rp][rng.integers(0, by_rep[rp].size, (B, R, by_rep[rp].size))].mean(-1) for rp in reps])
    pick = rng.integers(0, R, (B, R))
    boot = M[pick, np.arange(B)[:, None], np.arange(R)[None, :]].mean(1)
    naive = flat[rng.integers(0, flat.size, (B, flat.size))].mean(1)
    return {"n_replicates": len(reps), "n_points": int(flat.size), "mean_diff": float(flat.mean()),
            "replicate_means": [float(x) for x in rep_means], "replicate_min": float(rep_means.min()),
            "replicate_max": float(rep_means.max()), "replicates_reference_better": int((rep_means > 0).sum()),
            "points_reference_better": int((flat > 0).sum()),
            "cluster_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "naive_ci95": [float(np.percentile(naive, 2.5)), float(np.percentile(naive, 97.5))]}


def friedman_replicates(res):
    """Friedman test with data replicates (RMSE averaged over splits) as blocks, and Kendall's W."""
    from scipy.stats import friedmanchisquare
    out = {}
    for sc in SCENARIOS:
        methods = sorted(m for s, m in res if s == sc and m != "oracle")
        if not methods:
            continue
        reps = sorted(set.intersection(*[{k[0] for k in res[(sc, m)]} for m in methods]))
        X = np.array([[np.mean([v for k, v in res[(sc, m)].items() if k[0] == rp]) for m in methods] for rp in reps])
        stat, p = friedmanchisquare(*X.T)
        k = len(methods)
        out[sc] = {"n_blocks": len(reps), "n_methods": k, "chi2": float(stat), "p": float(p),
                   "kendall_w": float(stat / (len(reps) * (k - 1)))}
    return out


def _round(o):
    if isinstance(o, float):
        return float(f"{o:.5g}")
    if isinstance(o, dict):
        return {k: _round(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_round(v) for v in o]
    return o


def main():
    runs, out_json = sys.argv[1], Path(sys.argv[2])
    res = load(runs)
    rng = np.random.default_rng(2026)
    out = {"friedman_replicate_blocks": friedman_replicates(res)}
    for sc in SCENARIOS:
        methods = sorted({m for s, m in res if s == sc})
        if not methods:
            continue
        out[sc] = {}
        for ref in REFERENCES:
            if (sc, ref) not in res:
                continue
            out[sc][ref] = {m: compare(res[(sc, m)], res[(sc, ref)], rng)
                            for m in methods if m != ref and m != "oracle"}
    out_json.write_text(json.dumps(_round(out), separators=(",", ":")))
    print("wrote", out_json)


if __name__ == "__main__":
    main()
