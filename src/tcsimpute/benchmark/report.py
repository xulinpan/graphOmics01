"""Summarize the revision grid: per-method metrics, paired comparisons and rank tests.

Unit of replication: one grid point (data replicate x split; one training seed). For each
scenario and method the mean and SD over grid points of RMSE, bias, within-protein correlation
and interval coverage are reported. Each method is compared with two references, tcsimpute
(``vae_full``) and its linear variant (``selreg``), on matched grid points: mean paired RMSE
difference with a 95% bootstrap CI (10,000 resamples of grid points), Cohen's d_z, win count and
a Wilcoxon signed-rank test. Across methods, a Friedman test on RMSE (grid points as blocks) is
followed by the Nemenyi critical difference of mean ranks (alpha = 0.05). Grid points that share
data (same replicate or split) are not independent, so P-values are descriptive.

Usage: tcsimpute-report --runs runs --out runs/summary
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, studentized_range, wilcoxon

REFERENCES = ("vae_full", "selreg")
METRICS = ("rmse", "bias", "mae", "within_feature_r", "coverage90", "coverage_conformal",
           "coverage_pseudo", "coverage_level50", "coverage_level80", "coverage_level95",
           "interval_width90", "interval_width_conformal")
DOWNSTREAM = ("de_t_spearman", "de_precision", "de_recall", "de_effect_rmse", "coabundance_mae",
              "coabundance_spearman", "de_calls", "de_calls_reference")


def load(runs, include_smoke=False):
    rows = []
    for f in Path(runs).glob("*/*/*/result.json"):
        r = json.loads(f.read_text())
        if r.get("smoke") and not include_smoke:
            continue
        t = r["test"]
        row = {"scenario": r["scenario"], "point": f"{r['replicate']}_{r['split_seed']}",
               "method": r["method"], "seed": r["train_seed"], "runtime": r["runtime_seconds"]}
        row.update({k: t.get(k) for k in METRICS})
        for k, v in (t.get("downstream") or {}).items():
            row["ds_" + k] = v
        d = r.get("diagnostics") or {}
        row["active_units"] = d.get("active_units_train")
        row["selected_epoch"] = d.get("selected_epoch")
        for k, v in (r.get("inference_variants") or {}).items():
            row[f"variant_{k}_rmse"] = v.get("rmse")
        rows.append(row)
    return rows


def _mean_sd(vals):
    v = np.array([x for x in vals if x is not None], float)
    if v.size == 0:
        return None, None
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def bootstrap_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, (n, d.size))
    m = d[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def paired(rows_m, rows_ref, key="rmse"):
    a = {r["point"]: r[key] for r in rows_m if r[key] is not None}
    b = {r["point"]: r[key] for r in rows_ref if r[key] is not None}
    pts = sorted(set(a) & set(b))
    if len(pts) < 2:
        return None
    d = np.array([a[p] - b[p] for p in pts])
    out = {"n": len(pts), "mean_diff": float(d.mean()), "ci95": bootstrap_ci(d),
           "wins_reference": int((d > 0).sum()), "cohens_dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else None}
    try:
        out["wilcoxon_p"] = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        out["wilcoxon_p"] = None
    return out


def rank_tests(by_method, alpha=0.05):
    """Friedman + Nemenyi over methods present at every grid point (RMSE; lower is better)."""
    by_method = {m: rows for m, rows in by_method.items() if any(r["rmse"] is not None for r in rows)}
    points = None
    for rows in by_method.values():
        p = {r["point"] for r in rows if r["rmse"] is not None}
        points = p if points is None else points & p
    methods = [m for m, rows in by_method.items() if m != "oracle" and
               points and points <= {r["point"] for r in rows if r["rmse"] is not None}]
    if not points or len(methods) < 3 or len(points) < 3:
        return None
    pts = sorted(points)
    M = np.array([[{r["point"]: r["rmse"] for r in by_method[m]}[p] for m in methods] for p in pts])
    stat, pval = friedmanchisquare(*M.T)
    ranks = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)) + 1.0, 1, M)
    mean_ranks = ranks.mean(0)
    k, n = len(methods), len(pts)
    q = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = float(q * np.sqrt(k * (k + 1) / (6.0 * n)))
    return {"methods": methods, "n_points": n, "friedman_chi2": float(stat), "friedman_p": float(pval),
            "mean_ranks": {m: float(r) for m, r in zip(methods, mean_ranks)}, "nemenyi_cd": cd}


def summarize(rows):
    out = {}
    by_sc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_sc[r["scenario"]][r["method"]].append(r)
    for sc, by_method in sorted(by_sc.items()):
        res = {"methods": {}, "ranks": rank_tests(by_method)}
        for m, rs in sorted(by_method.items()):
            entry = {"n": len(rs), "runtime_median_s": float(np.median([r["runtime"] for r in rs]))}
            for k in METRICS + ("active_units", "selected_epoch", "variant_single_draw_rmse", "variant_mc50_rmse"):
                mu, sd = _mean_sd(r.get(k) for r in rs)
                if mu is not None:
                    entry[k] = mu
                    entry[k + "_sd"] = sd
            for k in DOWNSTREAM:
                mu, sd = _mean_sd(r.get("ds_" + k) for r in rs)
                if mu is not None:
                    entry["ds_" + k] = mu
                    entry["ds_" + k + "_sd"] = sd
            for ref in REFERENCES:
                if ref in by_method and m != ref:
                    entry[f"vs_{ref}"] = paired(rs, by_method[ref])
            res["methods"][m] = entry
        out[sc] = res
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tcsimpute-report", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--include-smoke", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    rows = load(a.runs, a.include_smoke)
    out = Path(a.out or Path(a.runs) / "summary")
    out.mkdir(parents=True, exist_ok=True)
    summ = summarize(rows)
    (out / "summary_revision.json").write_text(json.dumps(summ, indent=1))
    import csv
    keys = sorted({k for r in rows for k in r})
    with open(out / "runs_revision.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    for sc, res in summ.items():
        ms = res["methods"]
        best = sorted((v.get("rmse", np.inf), m) for m, v in ms.items() if v.get("rmse") is not None)[:5]
        print(f"{sc:16s} n={max(v['n'] for v in ms.values()):3d} best:", ", ".join(f"{m} {x:.3f}" for x, m in best))
    print("wrote", out / "summary_revision.json", "and", len(rows), "runs to runs_revision.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
