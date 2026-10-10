"""LaTeX tables for the parameter-recovery study (Supplementary Tables S13 and S14).
Usage: python recovery_tables.py <parameter_recovery.json> <out_dir>"""
import json
import sys
from pathlib import Path

LABELS = {
    "base": "Base",
    "no_offsets": "No sample offsets",
    "limits_sd0.25": "Per-protein limits, SD 0.25",
    "limits_sd1": "Per-protein limits, SD 1",
    "slope0.25": "Sharp detection ($s=0.25$)",
    "slope1": "Shallow detection ($s=1$)",
    "weak_coupling": "Weak RNA coupling",
    "mcar0.15": "MCAR 15\\%",
    "sporadic": "Sporadic loss (mean 10\\%)",
}
ORDER = list(LABELS)


def f(v, d=2, sign=False):
    if v is None:
        return "--"
    if round(v, d) == 0:
        v = 0.0
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s.replace("-", "$-$")


def main():
    data = json.load(open(sys.argv[1]))
    out = Path(sys.argv[2])
    S = {(e["condition"], e["method"], e["prior_sd"]): e for e in data["summary"]}

    L = ["\\begin{table}[htbp]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{2.8pt}",
         "\\caption{Recovery of detection parameters on simulated data (10 data sets per condition; means). "
         "$\\Delta$limit: mean error of the per-protein detection limits on the raw scale; for the VAE it is given "
         "before and, in brackets, after adding the mean estimated sample offset, because only their sum enters "
         "the likelihood. $\\hat s$: detection slope on the raw scale (median over proteins; the model shares the "
         "slope on the standardized scale). The linear variant has no sample offsets, so its slope describes the "
         "offset-averaged curve (expected $\\sqrt{s^2+0.3^2}=0.58$ for $s=0.5$). $r_\\delta$: correlation of "
         "estimated with true sample offsets. RMSE: hidden test entries, standardized units, with the Bayes "
         "oracle in brackets. Base: $s=0.5$, MCAR 2\\%, sample offsets SD 0.3. Prior SD of per-protein limit "
         "deviations 0.25 (the default).}",
         "\\label{tab:recovery}",
         "\\begin{tabular}{lrrrrrrrrrrr}", "\\toprule",
         " & \\multicolumn{2}{c}{True} & \\multicolumn{3}{c}{Linear variant} & \\multicolumn{4}{c}{VAE variant} "
         "& \\multicolumn{2}{c}{RMSE} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(lr){7-10}\\cmidrule(lr){11-12}",
         "Condition & $s$ & $\\pi$ & $\\Delta$limit & $\\hat s$ & $\\hat\\pi$ & $\\Delta$limit & $\\hat s$ & "
         "$\\hat\\pi$ & $r_\\delta$ & Linear & VAE \\\\", "\\midrule"]
    for c in ORDER:
        lin, vae = S.get((c, "linear", 0.25)), S.get((c, "vae", 0.25))
        if not lin or not vae:
            continue
        orc = f" [{f(lin['oracle_rmse'])}]" if lin.get("oracle_rmse") is not None else ""
        L.append(" & ".join([
            LABELS[c], f(lin["true_slope"]), f(lin["true_pi"], 3),
            f(lin["limit_bias"], 2, True), f(lin["slope_raw_hat"]), f(lin["pi_hat"], 3),
            f(vae["limit_bias"], 2, True) + " [" + f(vae["limit_bias_effective"], 2, True) + "]",
            f(vae["slope_raw_hat"]), f(vae["pi_hat"], 3), f(vae.get("offset_recovery_r")),
            f(lin["rmse"]) + orc, f(vae["rmse"])]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "tableS_recovery.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    L = ["\\begin{table}[htbp]", "\\centering", "\\scriptsize",
         "\\caption{Sensitivity to the prior SD of per-protein limit deviations (standardized units; default "
         "0.25). Limit SD: spread of the per-protein detection limits on the raw scale, estimated and true; "
         "$r$: correlation of estimated with true per-protein limits (defined when the true limits vary). "
         "$\\hat\\pi$: estimated MCAR rate (true value in brackets). 10 data sets per row.}",
         "\\label{tab:priors}",
         "\\begin{tabular}{llrrrrr}", "\\toprule",
         "Condition & Variant & Prior SD & Limit SD (true) & $r$ & $\\hat\\pi$ (true) & RMSE \\\\", "\\midrule"]
    for c in ("base", "limits_sd1", "weak_coupling", "sporadic"):
        for m in ("linear", "vae"):
            for p in (0.1, 0.25, 1.0):
                e = S.get((c, m, p))
                if not e:
                    continue
                L.append(" & ".join([LABELS[c], "Linear" if m == "linear" else "VAE", f"{p:g}",
                                     f(e["limit_sd_hat"]) + " (" + f(e["limit_sd_true"]) + ")", f(e.get("limit_r")),
                                     f(e["pi_hat"], 3) + " (" + f(e["true_pi"], 3) + ")", f(e["rmse"])]) + " \\\\")
        L.append("\\addlinespace")
    if L[-1] == "\\addlinespace":
        L.pop()
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "tableS_priors.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", out / "tableS_recovery.tex", out / "tableS_priors.tex")


if __name__ == "__main__":
    main()
