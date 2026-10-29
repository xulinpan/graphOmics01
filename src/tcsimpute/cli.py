"""Command-line imputation from CSV files: ``tcsimpute --rna rna.csv --protein protein.csv --out imputed.csv``.

Both CSVs have samples as rows, a first column of sample IDs and features as columns. Empty cells in
the protein CSV are treated as missing. Cognate transcripts are matched by column name (the part
before any ``|``).
"""

import argparse
import json


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tcsimpute", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rna", required=True, help="Complete RNA matrix (log scale), samples x genes.")
    ap.add_argument("--protein", required=True, help="Protein matrix (log scale) with empty cells for missing.")
    ap.add_argument("--out", required=True, help="Output CSV with imputed proteins.")
    ap.add_argument("--model", choices=("linear", "vae"), default="linear",
                    help="'linear' (default) or the multimodal 'vae' variant.")
    ap.add_argument("--limit-scale", choices=("raw", "standardized"), default="raw",
                    help="'raw' for log intensities, 'standardized' for ratio data (default: raw).")
    ap.add_argument("--intervals", help="Prefix for 90%% interval CSVs (<prefix>_lower.csv, <prefix>_upper.csv).")
    ap.add_argument("--level", type=float, default=0.9)
    ap.add_argument("--validation-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-model", help="Optional path to save the fitted imputer.")
    a = ap.parse_args(argv)

    import pandas as pd
    from .imputer import TCSImputer
    rna = pd.read_csv(a.rna, index_col=0)
    prot = pd.read_csv(a.protein, index_col=0)
    common = rna.index.intersection(prot.index)
    if len(common) < len(prot.index):
        print(f"Using {len(common)} samples present in both files.")
    rna, prot = rna.loc[common], prot.loc[common]
    imp = TCSImputer(model=a.model, limit_scale=a.limit_scale, validation_fraction=a.validation_fraction,
                     random_state=a.seed).fit(rna, prot)
    imp.transform(rna, prot).to_csv(a.out)
    if a.intervals:
        lo, hi = imp.predict_interval(rna, prot, level=a.level)
        lo.to_csv(f"{a.intervals}_lower.csv")
        hi.to_csv(f"{a.intervals}_upper.csv")
    if a.save_model:
        imp.save(a.save_model)
    d = imp.diagnostics_
    keys = ("selected_epoch", "active_units_train", "detection_slope", "mcar_rate", "nll")
    print(json.dumps({k: d[k] for k in keys if k in d}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
