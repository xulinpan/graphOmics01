"""LaTeX table for the error-control experiment (Supplementary Table S15).
Usage: python inference_tables.py <inference_calibration.json> <out_dir>"""
import json
import sys
from pathlib import Path

LABELS = [("complete", "Complete data"), ("available_case", "Observed values only"), ("mindet", "MinDet"),
          ("tcsimpute_linear", "tcsimpute (linear), single"), ("tcsimpute_vae", "tcsimpute (VAE), single"),
          ("tcsimpute_linear_mi", "tcsimpute (linear), multiple")]


def f(v, d=3):
    if round(v, d) == 0:
        v = 0.0
    return f"{v:.{d}f}".replace("-", "$-$")


def main():
    S = json.load(open(sys.argv[1]))["summary"]
    out = Path(sys.argv[2])
    L = ["\\begin{table}[htbp]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
         "\\caption{Error control of Welch tests after imputation in repeated-mask experiments on CPTAC tumours "
         "(20 replicates per cohort; mean (SD) over replicates). In each replicate the tumours were split into two "
         "random halves, 10\\% of proteins were shifted by 0.5 within-protein SD in one half, and the "
         "intensity-scale detection mask was then imposed. Type-I error: fraction of unshifted proteins with at least "
         "10\\% missing values with $P<0.05$. FDR and power: Benjamini--Hochberg calls at 5\\% against the known "
         "shifted proteins. Coverage: 95\\% confidence intervals for the group difference of shifted proteins. "
         "Effect bias: mean error of the estimated difference relative to the true shift (negative: attenuated). "
         "Multiple imputation: 20 draws from each missing value's distribution given non-detection under the "
         "fitted linear model, combined by Rubin's rules; parameter uncertainty is not propagated.}",
         "\\label{tab:inference}",
         "\\begin{tabular}{llrrrrr}", "\\toprule",
         "Cohort & Method & Type-I error & FDR & Power & Coverage & Effect bias \\\\", "\\midrule"]
    for cohort, name in (("pdac", "PDAC"), ("luad", "LUAD")):
        c = S[cohort]["methods"]
        for k, lab in LABELS:
            e = c[k]
            L.append(" & ".join([f"{name} ($n={S[cohort]['n_samples']}$)" if k == "complete" else "", lab,
                                 f"{f(e['type1_missing10'])} ({f(e['type1_missing10_sd'])})",
                                 f"{f(e['fdr'])} ({f(e['fdr_sd'])})", f"{f(e['power'], 2)}",
                                 f"{f(e['ci_cover_effect'], 2)}", f"{f(e['effect_rel_bias'], 2)}"]) + " \\\\")
        L.append("\\addlinespace")
    L.pop()
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "tableS_inference.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", out / "tableS_inference.tex")


if __name__ == "__main__":
    main()
