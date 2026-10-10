"""Sensitivity of the ProCan results to proteins paired with another gene's transcript
(Supplementary Section S10).

ProCan proteins were matched to transcripts through the gene part of the UniProt entry name. For a
few proteins this part is the primary gene name of a different protein (TF_HUMAN is tissue factor,
F3, but TF is the transferrin gene), so the protein was paired with another gene's transcript. This
script rescores every ProCan run without those proteins, from the stored predictions, and reports
each method's mean RMSE and its paired difference from tcsimpute with and without them. The models
were not refitted, so the check covers scoring but not any effect of these proteins on fitting.

Usage: python procan_mismatch_sensitivity.py <runs_dir> <procan_dir> <uniprot_tsv> <out_json>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def wrong_gene_columns(names, uniprot):
    u = pd.read_csv(uniprot, sep="\t", dtype=str).fillna("")
    primary = dict(zip(u["Entry"], u["Gene Names (primary)"]))
    prim_set = set(u["Gene Names (primary)"]) - {""}
    out = []
    for j, col in enumerate(names):
        acc, entry = col.split(";")
        sym = entry.replace("_HUMAN", "")
        p = primary.get(acc.split("-")[0], "")
        if p and p != sym and sym in prim_set:
            out.append(j)
    return np.array(out, int)


def rmse(pred, truth, mask):
    e = (pred.astype(float) - truth.astype(float))[mask]
    return float(np.sqrt(np.mean(e ** 2)))


def main():
    runs, procan_dir, uniprot, out = Path(sys.argv[1]) / "PROCAN", Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    wrong = {}
    for r in (1, 2, 3, 4):
        with np.load(procan_dir / f"cache_procan_rep{r}.npz", allow_pickle=False) as f:
            names = [str(x) for x in f["protein_names"]]
        wrong[r] = wrong_gene_columns(names, uniprot)
    res = defaultdict(dict)
    for f in sorted(runs.glob("rep*_split*/*_seed1/predictions.npz")):
        method = f.parent.name.replace("_seed1", "")
        rep = int(f.parent.parent.name.split("_")[0][3:])
        with np.load(f, allow_pickle=False) as z:
            truth, pred, hidden = z["truth"], z["prediction"], z["hidden"]
        scored = hidden & np.isfinite(truth)
        keep = np.ones(truth.shape[1], bool)
        keep[wrong[rep]] = False
        res[method][f.parent.parent.name] = {"all": rmse(pred, truth, scored),
                                             "without": rmse(pred, truth, scored & keep[None, :]),
                                             "only": rmse(pred, truth, scored & ~keep[None, :])}
    summary = {}
    for m, d in res.items():
        summary[m] = {k: float(np.mean([v[k] for v in d.values()])) for k in ("all", "without", "only")}
        summary[m]["n_points"] = len(d)
        for ref in ("selreg", "vae_full"):
            if ref in res and m != ref:
                keys = sorted(set(d) & set(res[ref]))
                for k in ("all", "without"):
                    summary[m][f"diff_vs_{ref}_{k}"] = float(np.mean([d[g][k] - res[ref][g][k] for g in keys]))
    out.write_text(json.dumps({"wrong_gene_proteins": {r: len(v) for r, v in wrong.items()},
                               "methods": summary}, indent=1))
    for m in sorted(summary, key=lambda k: summary[k]["all"]):
        s = summary[m]
        print(f"{m:22s} all {s['all']:.3f} without {s['without']:.3f} only {s['only']:.3f} "
              f"d_lin {s.get('diff_vs_selreg_all', 0):+.3f}/{s.get('diff_vs_selreg_without', 0):+.3f}")


if __name__ == "__main__":
    main()
