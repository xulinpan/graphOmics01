"""Aggregate runs_v2 results into tables (CSV/JSON/LaTeX) with paired comparisons.

Unit of replication: one (replicate, split, training seed) grid point. Methods without a training
seed contribute the same value to both seeds of a grid point. Paired tests compare each method
with the full model on matched grid points; points within a split share data, so p-values are
descriptive rather than confirmatory.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

LABELS = {
    "vae_full": "Censored-likelihood VAE (full)",
    "vae_full@single_draw": "\\quad inference: one latent draw",
    "vae_full@mc50": "\\quad inference: 50-draw average",
    "abl_zero_fill": "\\quad $-$ mask-aware encoder (zero fill)",
    "abl_mlp_fusion": "\\quad $-$ Transformer fusion (MLP)",
    "abl_no_contrastive": "\\quad $-$ contrastive term",
    "abl_no_skip": "\\quad $-$ cognate skip",
    "abl_no_latent": "\\quad $-$ latent (censored cognate regression)",
    "abl_mar_likelihood": "\\quad $-$ selection model (MAR likelihood)",
    "abl_hinge_penalty": "\\quad selection model $\\to$ v1 hinge",
    "abl_v1_elbo_scaling": "\\quad v1 ELBO scaling",
    "v1_reference": "v1 Transformer-VAE (as submitted)",
    "notmiwae": "not-MIWAE",
    "miwae": "MIWAE",
    "missforest": "missForest (RNA PCs + proteins)",
    "knn": "KNN (RNA PCs + proteins)",
    "qrilc": "QRILC",
    "minprob": "MinProb",
    "mindet": "MinDet",
    "dsn": "Down-shifted normal",
    "cognate_ols": "Cognate-transcript OLS",
    "observed_mean": "Observed training mean",
    "oracle": "Bayes oracle (known generator)",
}
ORDER = list(LABELS)


def load(runs):
    rows = []
    for path in sorted(Path(runs).glob("*/rep*_split*/*/result.json")):
        r = json.loads(path.read_text())
        if r.get("smoke"):
            continue
        rows.append(r)
        for name, metrics in r.get("inference_variants", {}).items():
            rows.append({**r, "method": f"{r['method']}@{name}", "test": metrics})
    return rows


def aggregate(rows, scenario):
    by = defaultdict(dict)
    seeds = sorted({r["train_seed"] for r in rows if r["scenario"] == scenario and r["method"] == "vae_full"})
    for r in rows:
        if r["scenario"] != scenario:
            continue
        key = (r["replicate"], r["split_seed"])
        by[r["method"]][(key, r["train_seed"])] = r
    # expand seedless methods to every seed
    for m, d in by.items():
        for (key, seed), r in list(d.items()):
            for s in seeds:
                d.setdefault((key, s), r)
    out = {}
    ref = by.get("vae_full", {})
    for m, d in by.items():
        keys = sorted(d)

        def col(metric):
            return np.array([np.nan if d[k]["test"].get(metric) is None else d[k]["test"][metric] for k in keys], float)
        rmse = col("rmse")
        wr = col("within_feature_r")
        entry = {"n_points": len(keys),
                 "rmse_mean": float(np.nanmean(rmse)),
                 "rmse_sd": float(np.nanstd(rmse, ddof=1)) if len(keys) > 1 else 0.0,
                 "within_r_mean": float(np.nanmean(wr)) if np.isfinite(wr).any() else None,
                 "bias_mean": float(np.nanmean(col("bias"))),
                 "mae_mean": float(np.nanmean(col("mae")))}
        cov = col("coverage90")
        if np.isfinite(cov).any():
            entry["coverage90_mean"] = float(np.nanmean(cov))
            entry["width90_mean"] = float(np.nanmean(col("interval_width90")))
        bv = col("boundary_violation_rate")
        if np.isfinite(bv).any():
            entry["boundary_violation_mean"] = float(np.nanmean(bv))
        common = [k for k in keys if k in ref]
        if m != "vae_full" and common:
            diff = np.array([d[k]["test"]["rmse"] - ref[k]["test"]["rmse"] for k in common])
            entry["paired_rmse_diff_vs_full_mean"] = float(diff.mean())
            entry["paired_wins_full"] = int((diff > 0).sum())
            entry["paired_n"] = len(diff)
            if len(diff) >= 5 and np.any(diff != 0):
                entry["wilcoxon_p"] = float(wilcoxon(diff).pvalue)
        diag = [d[k].get("diagnostics", {}) for k in keys]
        au = [x.get("active_units_train", x.get("active_units_test")) for x in diag]
        au = [a for a in au if a is not None]
        if au:
            entry["active_units_mean"] = float(np.mean(au))
        ep = [x.get("selected_epoch") for x in diag if x.get("selected_epoch")]
        if ep:
            entry["selected_epoch_median"] = float(np.median(ep))
        rt = [d[k].get("runtime_seconds") for k in keys if d[k].get("runtime_seconds") is not None]
        entry["runtime_median_s"] = float(np.median(rt)) if rt else None
        fps = {d[k]["protocol"]["fingerprint"] for k in keys}
        entry["n_protocols"] = len(fps)
        out[m] = entry
    return out


def fmt(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def latex_table(agg, scenario, caption, label, show_boundary=False):
    has_cov = any("coverage90_mean" in v for v in agg.values())
    cols = "lrrrr" + ("r" if has_cov else "") + ("r" if show_boundary else "") + "r"
    cols += "r"  # p-value column
    head = ["Method", "RMSE (SD)", "Within-$r$", "Bias", "$\\Delta$RMSE"]
    if has_cov:
        head.append("Cov.\\ 90\\%")
    if show_boundary:
        head.append("Above cutoff")
    head += ["Wins", "$p$"]
    lines = ["\\begin{table}[htbp]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{%spt}" % ("2.2" if show_boundary else "3.5"),
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{cols}}}", "\\toprule", " & ".join(head) + " \\\\", "\\midrule"]
    for m in ORDER:
        if m not in agg:
            continue
        v = agg[m]
        row = [LABELS[m], f"{fmt(v['rmse_mean'])} ({fmt(v['rmse_sd'])})", fmt(v["within_r_mean"]),
               f"{v['bias_mean']:+.3f}",
               "--" if m == "vae_full" else f"{v.get('paired_rmse_diff_vs_full_mean', float('nan')):+.3f}"]
        if has_cov:
            row.append(fmt(v.get("coverage90_mean")))
        if show_boundary:
            row.append(fmt(v.get("boundary_violation_mean")))
        row.append("--" if m == "vae_full" else f"{v.get('paired_wins_full', 0)}/{v.get('paired_n', 0)}")
        p = v.get("wilcoxon_p")
        row.append("--" if p is None else ("$<$0.001" if p < 0.001 else f"{p:.3f}"))
        row = ["0.000" if c in ("-0.000", "+0.000") else c for c in row]
        row = [("$-$" + c[1:]) if c.startswith("-") and c[1:2].isdigit() else c for c in row]
        lines.append(" & ".join(row) + " \\\\")
    note = ("\\par\\smallskip\\parbox{\\linewidth}{\\footnotesize Wins: grid points at which the full model had "
            "lower RMSE. $p$: two-sided Wilcoxon signed-rank test on paired RMSE differences (descriptive; grid "
            "points within a split share data). Within-$r$: mean within-protein Pearson correlation over hidden "
            "entries (``--'': constant within protein). Cov.\\ 90\\%: empirical coverage of 90\\% predictive "
            "intervals (likelihood-based variants only).}")
    lines += ["\\bottomrule", "\\end{tabular}", note, "\\end{table}"]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tcsimpute-summarize", description=__doc__)
    ap.add_argument("--runs", required=True, help="Directory passed to tcsimpute-benchmark --out.")
    a = ap.parse_args(argv)
    runs = Path(a.runs)
    rows = load(runs)
    if not rows:
        raise SystemExit(f"No completed (non-smoke) runs found under {runs}.")
    scenarios = sorted({r["scenario"] for r in rows})
    result = {sc: aggregate(rows, sc) for sc in scenarios}
    out = runs / "summary"
    out.mkdir(exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    with open(out / "summary.csv", "w") as f:
        keys = ["n_points", "rmse_mean", "rmse_sd", "within_r_mean", "bias_mean", "mae_mean",
                "coverage90_mean", "width90_mean", "boundary_violation_mean",
                "paired_rmse_diff_vs_full_mean", "paired_wins_full", "paired_n", "wilcoxon_p",
                "active_units_mean", "selected_epoch_median", "runtime_median_s", "n_protocols"]
        f.write("scenario,method," + ",".join(keys) + "\n")
        for sc, agg in result.items():
            for m in ORDER:
                if m in agg:
                    f.write(f"{sc},{m}," + ",".join("" if agg[m].get(k) is None else str(agg[m].get(k)) for k in keys) + "\n")
    caps = {
        "S1": ("S1: original synthetic fixture (deterministic raw cutoff); 3 splits $\\times$ 2 training seeds (6 grid points). Above cutoff: fraction of hidden predictions above the deterministic standardized cutoff.", "tab:s1", True),
        "S2": ("S2: nonlinear sparse many-to-many simulator with probabilistic detection; 3 data replicates $\\times$ 3 splits $\\times$ 2 training seeds (18 grid points). Hidden-entry test metrics on the training-standardized scale; means over grid points.", "tab:s2", False),
        "CPTAC": ("CPTAC PDAC semi-synthetic benchmark (145 tumours, 3{,}000 transcripts, 1{,}000 complete proteins, imposed probit detection); 3 splits $\\times$ 2 training seeds (6 grid points).", "tab:cptac", False),
    }
    tex = []
    for sc in [s for s in ("S2", "CPTAC", "S1") if s in scenarios]:
        cap, lab, bnd = caps.get(sc, (sc, "tab:" + sc.lower(), False))
        tex.append(latex_table(result[sc], sc, cap, lab, show_boundary=bnd))
    (out / "tables.tex").write_text("\n\n".join(tex) + "\n")
    for sc in scenarios:
        print(f"== {sc}")
        for m in ORDER:
            if m in result[sc]:
                v = result[sc][m]
                print(f"  {m:28s} rmse {v['rmse_mean']:.3f}+/-{v['rmse_sd']:.3f}  r {fmt(v['within_r_mean'])}  bias {v['bias_mean']:+.3f}"
                      f"  cov {fmt(v.get('coverage90_mean'))}  dRMSE {v.get('paired_rmse_diff_vs_full_mean', 0):+.3f}"
                      f"  wins {v.get('paired_wins_full', '-')}/{v.get('paired_n', '-')}  p {fmt(v.get('wilcoxon_p'), 4)}"
                      f"  AU {fmt(v.get('active_units_mean'), 1)}  n={v['n_points']}")


if __name__ == "__main__":
    main()
