"""Figures, tables and key numbers for the revised Bioinformatics submission.

Inputs: summary_revision.json (tcsimpute-report), scaling.json (tcsimpute-scaling).
Outputs in --out:
    fig1_model.{pdf,tif,jpg}        model schematic
    fig2_benchmark.{pdf,tif,jpg}    accuracy on SIM, PDAC, LUAD and ProCan
    fig3_robustness.{pdf,tif,jpg}   misspecification and sensitivity sweeps (excess RMSE heatmap)
    fig4_calib_downstream.{pdf,tif,jpg}  calibration, downstream analyses, runtime scaling
    table1.tex, table2.tex          main-text tables
    tableS_*.tex                    supplementary tables
    numbers.json                    every number quoted in the text, for checking
"""

import argparse
import json
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ labels and groups

LABEL = {
    "vae_full": "tcsimpute (VAE)", "selreg": "tcsimpute (linear)",
    "abl_no_latent": "VAE $-$ latent variable", "abl_no_skip": "VAE $-$ cognate term",
    "abl_mar_likelihood": "VAE, MAR likelihood", "abl_hinge_penalty": "VAE, hinge penalty",
    "abl_mlp_fusion": "VAE, MLP fusion", "abl_no_contrastive": "VAE $-$ contrastive term",
    "abl_zero_fill": "VAE $-$ mask input",
    "tobit": "Tobit regression", "proda": "proDA", "proda_design": "proDA (condition design)",
    "pirat_t": "Pirat-T", "pirat": "Pirat", "notmiwae_cog": "not-MIWAE", "notmiwae_cog16": "not-MIWAE",
    "miwae_cog": "MIWAE", "missforest_r": "missForest", "knn_cog": "KNN", "mindet_r": "MinDet",
    "minprob_r": "MinProb", "qrilc_r": "QRILC", "dsn": "Down-shifted normal",
    "cognate_ols": "Cognate OLS", "observed_mean": "Observed mean", "oracle": "Bayes oracle",
    "available_case": "No imputation",
}
PLAIN = {k: v.replace("$-$", "-") for k, v in LABEL.items()}
GROUP = {"vae_full": "tcs", "selreg": "tcs", "cognate_ols": "ctl", "observed_mean": "ctl", "oracle": "ctl",
         "available_case": "ctl"}
GROUP.update({k: "abl" for k in LABEL if k.startswith("abl_")})
COLORS = {"tcs": "#2a78d6", "abl": "#eb6834", "cmp": "#1baf7a", "ctl": "#8a8a85"}
MARKERS = {"tcs": "o", "abl": "s", "cmp": "^", "ctl": "D"}
GROUP_NAME = {"tcs": "tcsimpute", "abl": "Ablations of the VAE", "cmp": "Established imputers",
              "ctl": "Controls"}

MAIN_ORDER = ["selreg", "vae_full", "abl_no_latent", "abl_no_skip", "abl_mar_likelihood", "abl_hinge_penalty",
              "abl_mlp_fusion", "abl_no_contrastive", "abl_zero_fill", "tobit", "pirat_t", "pirat", "proda", "notmiwae_cog",
              "miwae_cog", "missforest_r", "knn_cog", "mindet_r", "minprob_r", "qrilc_r", "dsn",
              "cognate_ols", "observed_mean", "oracle"]
FIG2_ORDER = [m for m in MAIN_ORDER if not m.startswith("abl_")]
MAIN_SC = [("SIM", "Simulator\n(simulated data)"), ("PDAC", "CPTAC PDAC\n(imposed mask)"),
           ("LUAD", "CPTAC LUAD\n(imposed mask)"), ("PROCAN", "ProCan DIA\n(natural)")]


def hdr(t):
    return "\\shortstack{" + t.replace("\n", "\\\\") + "}"
SWEEP_ROWS = [("SWEEP-probit", "Probit (matched)"), ("SWEEP-logistic", "Logistic detection"),
              ("SWEEP-threshold", "Hard threshold"), ("SWEEP-widelimits", "Wide per-protein limits"),
              ("SWEEP-batch", "Batch-wise missingness"), ("SWEEP-mcar15", "15% MCAR"),
              ("SWEEP-t3", "Student-t noise"), ("SWEEP-miss10", "10% missing"),
              ("SWEEP-miss50", "50% missing"), ("SWEEP-n100", "n = 100"), ("SWEEP-n400", "n = 400"),
              ("SWEEP-weak", "Weak RNA coupling"), ("SWEEP-strong", "Strong RNA coupling")]
SWEEP_COLS = ["selreg", "vae_full", "abl_no_latent", "pirat_t", "proda", "notmiwae_cog16", "knn_cog", "mindet_r", "cognate_ols", "observed_mean"]
DS_ORDER = ["selreg", "vae_full", "abl_no_latent", "pirat_t", "pirat", "proda", "proda_design", "notmiwae_cog", "missforest_r", "knn_cog",
            "mindet_r", "qrilc_r", "dsn", "cognate_ols", "observed_mean", "available_case"]


def group(m):
    return GROUP.get(m, "cmp")


def f3(x, signed=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = f"{x:+.3f}" if signed else f"{x:.3f}"
    if s in ("+0.000", "-0.000"):
        s = "0.000"
    return s.replace("-", "$-$")


def f2(x):
    """Two decimals, signed, LaTeX minus (coarse intervals)."""
    t = f"{x:+.2f}"
    return ("$-$" + t[1:]) if t.startswith("-") else t


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#8a8a85", "axes.linewidth": 0.6,
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf")
    # journal artwork: RGB (no alpha), 600 dpi for combination figures, LZW-compressed TIFF
    import io
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=600, facecolor="white")
    buf.seek(0)
    Image.open(buf).convert("RGB").save(out / f"{name}.tif", dpi=(600, 600), compression="tiff_lzw")
    fig.savefig(out / f"{name}.jpg", dpi=150)


# ------------------------------------------------------------------ figure 1: schematic

def fig_model(out):
    plt = setup_mpl()
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(178 / 25.4, 3.0))
    ax.set_xlim(0, 178)
    ax.set_ylim(8, 84)
    ax.axis("off")

    def box(x, y, w, h, title, body, color, fill):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6",
                                    linewidth=0.8, edgecolor=color, facecolor=fill))
        ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="top", fontsize=7, fontweight="bold", color="#0b0b0b")
        ax.text(x + w / 2, y + h - 8.6, body, ha="center", va="top", fontsize=6.2, color="#52514e", linespacing=1.6)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#52514e", shrinkA=0, shrinkB=0))

    blue, blue_f = "#2a78d6", "#eef4fc"
    gray, gray_f = "#8a8a85", "#f6f6f4"
    org, org_f = "#eb6834", "#fdf0ea"
    box(2, 52, 34, 28, "Inputs", "RNA $r_i$ (complete)\nproteins $x_i$ with\nobservation mask $S_i$", gray, gray_f)
    box(44, 52, 40, 28, "Encoder (VAE variant)", "tokens for $r_i$ and $[x_i\\odot S_i,\\,S_i]$\none Transformer layer\n$q(z_i)$, sample offset $\\delta_i$", blue, blue_f)
    box(92, 52, 40, 28, "Protein mean", "VAE: $\\mu_{ij}=f_j(z_i)+\\beta_j r_{i,g(j)}$\nlinear: $\\mu_{ij}=\\alpha_j+\\beta_j r_{i,g(j)}$\n$\\qquad\\qquad+\\gamma_j^{\\top}\\mathrm{PC}(r_i)$", blue, blue_f)
    box(140, 52, 36, 28, "Detection model", "$P(S_{ij}{=}1\\mid x)=$\n$(1-\\pi)\\,\\Phi((x-\\tau_{ij})/s)$\n$\\tau_{ij}=t_j+\\delta_i\\rho_j$\nshrunken $t_j$, MCAR rate $\\pi$", org, org_f)
    box(44, 12, 62, 27, "Likelihood (fitted without hidden values)",
        "observed: $\\log\\mathcal{N}(x;\\mu,\\sigma^2)+\\log(1-\\pi)+\\log\\Phi((x-\\tau)/s)$\n"
        "missing: $\\log[\\pi+(1-\\pi)\\,\\Phi(a)]$\n"
        "$a=(\\tau-\\mu)/\\sqrt{\\sigma^2+s^2}$\n"
        "model selection on validation samples", org, org_f)
    box(114, 12, 62, 27, "Imputation and intervals",
        "$\\mathrm{E}[x\\mid S{=}0]=w\\mu+(1-w)\\,(\\mu-\\kappa\\,\\phi(a)/\\Phi(a))$\n"
        "$\\kappa=\\sigma^2/\\sqrt{\\sigma^2+s^2}$, $w$ = posterior MCAR probability\n"
        "approximate 90% intervals: weighted predictive\ndraws (+ optional conformal offset)", blue, blue_f)
    arrow(36, 66, 44, 66)
    arrow(84, 66, 92, 66)
    arrow(132, 66, 140, 66)
    arrow(112, 52, 82, 39.5)
    arrow(158, 52, 98, 39.5)
    arrow(106, 25.5, 114, 25.5)
    ax.annotate("", xy=(92, 60), xytext=(36, 56), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#8a8a85",
                ls=(0, (3, 2)), connectionstyle="arc3,rad=0.42", shrinkA=0, shrinkB=0))
    ax.text(19, 47, "linear variant:\nno encoder", ha="center", va="center", fontsize=5.8, color="#52514e", style="italic")
    fig.tight_layout(pad=0.2)
    save(fig, out, "fig1_model")
    plt.close(fig)


# ------------------------------------------------------------------ figure 2: benchmark

def fig_benchmark(summ, out):
    plt = setup_mpl()
    scs = [(s, t) for s, t in MAIN_SC if s in summ]
    fig, axes = plt.subplots(1, len(scs), figsize=(178 / 25.4, 3.9), sharey=True)
    axes = np.atleast_1d(axes)
    ypos = {m: i for i, m in enumerate(FIG2_ORDER)}
    for ax, (sc, title) in zip(axes, scs):
        ms = summ[sc]["methods"]
        ref = ms.get("selreg", {}).get("rmse")
        if ref is not None:
            ax.axvline(ref, color=COLORS["tcs"], lw=0.8, ls=(0, (3, 2)), zorder=0)
        vals = [ms[m]["rmse"] for m in FIG2_ORDER if ms.get(m, {}).get("rmse") is not None]
        cap = min(max(vals) + 0.1, 2.6)
        lo_x = min(vals) - 0.12
        for m in FIG2_ORDER:
            v = ms.get(m)
            if not v or v.get("rmse") is None:
                continue
            g = group(m)
            if v["rmse"] > cap:
                ax.plot(cap - 0.03, ypos[m], marker=">", ms=4.2, color=COLORS[g], mec="white", mew=0.6, zorder=3)
                ax.text(cap - 0.1, ypos[m], f"{v['rmse']:.2f}", ha="right", va="center", fontsize=6, color="#52514e")
                continue
            ax.errorbar(v["rmse"], ypos[m], xerr=v.get("rmse_sd", 0), fmt=MARKERS[g], ms=4.2, color=COLORS[g],
                        mec="white", mew=0.6, elinewidth=0.9, capsize=0, zorder=3)
        ax.set_xlim(lo_x, cap)
        ax.set_title(f"{'ABCD'[[s for s, _ in scs].index(sc)]}  {title}", fontsize=7.5, loc="left", fontweight="bold")
        ax.grid(axis="x", color="#e4e3dd", lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(FIG2_ORDER)))
    axes[0].set_yticklabels([PLAIN[m] for m in FIG2_ORDER])
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], color=COLORS[g], marker=MARKERS[g], ls="", ms=4.5, label=GROUP_NAME[g])
               for g in ("tcs", "cmp", "ctl")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.56, -0.005))
    fig.supxlabel("Hidden-entry test RMSE (training-standardized units; mean $\\pm$ SD over grid points)",
                  y=0.075, fontsize=7)
    fig.tight_layout(rect=(0, 0.09, 1, 0.98))
    save(fig, out, "fig2_benchmark")
    plt.close(fig)


# ------------------------------------------------------------------ figure 3: robustness heatmap

def sweep_matrix(summ):
    rows, R, O = [], [], []
    for sc, lab in SWEEP_ROWS:
        if sc not in summ:
            continue
        ms = summ[sc]["methods"]
        rows.append((sc, lab))
        R.append([ms.get(m, {}).get("rmse", np.nan) for m in SWEEP_COLS])
        O.append(ms.get("oracle", {}).get("rmse", np.nan))
    return rows, np.array(R, float), np.array(O, float)


def fig_robustness(summ, out):
    plt = setup_mpl()
    from matplotlib.colors import LinearSegmentedColormap
    rows, R, O = sweep_matrix(summ)
    if not rows:
        return
    best = np.nanmin(R, 1, keepdims=True)
    excess = R - best
    cmap = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, ax = plt.subplots(figsize=(178 / 25.4, 3.6))
    vmax = float(np.nanpercentile(excess, 95)) or 1.0
    im = ax.imshow(np.clip(excess, 0, vmax), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            if np.isfinite(R[i, j]):
                dark = excess[i, j] > 0.55 * vmax
                ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=6.2,
                        color="white" if dark else "#0b0b0b", fontweight="bold" if excess[i, j] == 0 else "normal")
    ax.set_xticks(range(len(SWEEP_COLS)))
    wrap = {"vae_full": "tcsimpute\n(VAE)", "selreg": "tcsimpute\n(linear)", "abl_no_latent": "VAE -\nlatent var.", "cognate_ols": "Cognate\nOLS",
            "observed_mean": "Observed\nmean"}
    ax.set_xticklabels([wrap.get(m, PLAIN[m]) for m in SWEEP_COLS], rotation=0, fontsize=6.5)
    ax.xaxis.tick_top()
    ylab = [f"{lab}  ({o:.2f})" if np.isfinite(o) else lab for (sc, lab), o in zip(rows, O)]
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(ylab)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cut = [i for i, (sc, _) in enumerate(rows) if sc == "SWEEP-t3"]
    if cut and cut[0] < len(rows) - 1:
        ax.axhline(cut[0] + 0.5, color="#52514e", lw=0.8)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("RMSE above the best method in the row", fontsize=6.5)
    cb.outline.set_visible(False)
    fig.tight_layout(pad=0.3)
    save(fig, out, "fig3_robustness")
    plt.close(fig)


# ------------------------------------------------------------------ figure 4: calibration, downstream, scaling

def fig_calib_downstream(summ, scaling, out):
    plt = setup_mpl()
    from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator
    fig = plt.figure(figsize=(178 / 25.4, 5.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1, 1], height_ratios=[1.15, 1], hspace=0.62, wspace=0.42)
    axB, axC = fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2])
    axA, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1:])
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    marks = ["o", "s", "^", "D"]
    # A calibration
    ax = axA
    ax.plot([0.45, 1], [0.45, 1], color="#8a8a85", lw=0.8, ls=(0, (3, 2)))
    for (sc, title), c, mk in zip(MAIN_SC, cols, marks):
        v = summ.get(sc, {}).get("methods", {}).get("vae_full")
        if not v:
            continue
        emp = [v.get(k) for k in ("coverage_level50", "coverage_level80", "coverage90", "coverage_level95")]
        if None in emp:
            continue
        ax.plot([0.5, 0.8, 0.9, 0.95], emp, color=c, lw=1.2, marker=mk, ms=3.5, label=title.split("\n")[0])
        if v.get("coverage_conformal") is not None:
            ax.plot([0.9], [v["coverage_conformal"]], marker=mk, ms=6, mfc="white", mec=c, mew=1.1, ls="")
        w = summ.get(sc, {}).get("methods", {}).get("selreg") or {}
        emp_l = [w.get(k) for k in ("coverage_level50", "coverage_level80", "coverage90", "coverage_level95")]
        if None not in emp_l:
            ax.plot([0.5, 0.8, 0.9, 0.95], emp_l, color=c, lw=0.9, ls=(0, (2, 1.5)), marker=mk, ms=2.6, mfc="white")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage\n(hidden test entries)")
    ax.legend(fontsize=6, loc="upper left", handlelength=1.4, title="solid: VAE; dashed: linear", title_fontsize=5.5)
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.3, 1.0)
    ax.grid(color="#e4e3dd", lw=0.5)
    ax.set_axisbelow(True)
    # D scaling
    ax = axD
    if scaling:
        res = scaling["results"]
        series = [("vae_full", "cpu", 1, "VAE, 1 CPU thread", cols[0], "o"),
                  ("vae_full", "cpu", 8, "VAE, 8 CPU threads", cols[1], "s"),
                  ("vae_full", "cuda", 8, "VAE, GPU", cols[2], "^"),
                  ("selreg", "cpu", 8, "Linear, 8 CPU threads", cols[3], "D")]
        xs = set()
        for meth, dev, th, lab, c, mk in series:
            pts = sorted((r["n_proteins"], r["seconds"]) for r in res if r["method"] == meth and r["device"] == dev
                         and r["threads"] == th and r["n_samples"] == 200)
            if pts:
                xs.update(p for p, _ in pts)
                ax.plot(*zip(*pts), color=c, marker=mk, ms=3.5, lw=1.2, label=lab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator(sorted(xs)))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.set_xlabel("Proteins (200 samples; genes = 1.6 x proteins + 200)")
        ax.set_ylabel("Seconds per fit\n(VAE: 50 epochs)")
        ax.legend(fontsize=6, loc="upper left", handlelength=1.4)
        ax.grid(color="#e4e3dd", lw=0.5, which="major")
        ax.set_axisbelow(True)
    # B, C downstream
    order = [m for m in DS_ORDER if any(m in summ.get(s, {}).get("methods", {}) for s in ("DS-PDAC", "DS-LUAD"))]
    y = {m: i for i, m in enumerate(order)}
    for ax, key, xlabel in ((axB, "ds_de_t_spearman", "Tumour vs normal: Spearman r of t statistics with complete data"),
                            (axC, "ds_coabundance_mae", "Co-abundance error")):
        for (sc, lab), c, mk, off in ((("DS-PDAC", "PDAC"), cols[0], "o", -0.17), (("DS-LUAD", "LUAD"), cols[1], "s", 0.17)):
            ms = summ.get(sc, {}).get("methods", {})
            first = True
            for m in order:
                v = ms.get(m, {})
                if v.get(key) is None:
                    continue
                ax.errorbar(v[key], y[m] + off, xerr=v.get(key + "_sd", 0), fmt=mk, ms=3.6, color=c, mec="white",
                            mew=0.5, elinewidth=0.8, capsize=0, label=lab if first else None)
                first = False
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([PLAIN[m] for m in order] if ax is axB else [])
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.grid(axis="x", color="#e4e3dd", lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel(xlabel)
    axB.legend(fontsize=6, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, handlelength=1)
    for ax, letter in ((axB, "A"), (axC, "B"), (axA, "C"), (axD, "D")):
        ax.text(-0.02, 1.06, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right")
    fig.subplots_adjust(left=0.19, right=0.985, top=0.93, bottom=0.08)
    save(fig, out, "fig4_calib_downstream")
    plt.close(fig)


# ------------------------------------------------------------------ tables

def table1(summ):
    scs = [(s, t) for s, t in MAIN_SC if s in summ]
    ranks = {s: (summ[s].get("ranks") or {}).get("mean_ranks", {}) for s, _ in scs}
    L = ["\\begin{table*}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{2.6pt}",
         "\\caption{Hidden-entry test RMSE (mean (SD) over grid points; training-standardized units) and mean "
         "rank across methods (lower is better; oracle excluded). Grid points: 20 per scenario "
         "(simulator: 5 data replicates $\\times$ 4 splits; CPTAC: 4 masks $\\times$ 5 splits; ProCan: 4 primary-run "
         "choices $\\times$ 5 splits). KNN, missForest, MIWAE and not-MIWAE receive RNA principal components and each "
         "protein's cognate transcript; Pirat-T receives each protein's cognate transcript, Pirat none. The CPTAC scenarios are semi-synthetic: a detection mask was imposed on complete "
         "TMT ratios. ProCan has natural missingness and is scored only on missing entries quantified in another run "
         "of the same cell line. Mean ranks are descriptive, because grid points sharing a data replicate or split are "
         "not independent. Paired differences with replicate-level win counts and ranges, and within-protein correlations, "
         "are in Supplementary Tables S1--S4 and S18.}",
         "\\label{tab:main}",
         "\\begin{tabular}{l" + "rr" * len(scs) + "}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{hdr(t)}}}" for _, t in scs) + " \\\\",
         "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(scs))),
         "Method & " + " & ".join("RMSE (SD) & Rank" for _ in scs) + " \\\\", "\\midrule"]
    for m in MAIN_ORDER:
        if not any(m in summ[s]["methods"] for s, _ in scs):
            continue
        cells = []
        for s, _ in scs:
            v = summ[s]["methods"].get(m)
            if v is None or v.get("rmse") is None:
                cells += ["--", "--"]
            else:
                r = ranks[s].get(m)
                cells += [f"{f3(v['rmse'])} ({f3(v['rmse_sd'])})", "--" if r is None else f"{r:.1f}"]
        lab = LABEL[m]
        if m.startswith("abl_"):
            lab = "\\quad " + lab
        L.append(" & ".join([lab] + cells) + " \\\\")
        if m in ("selreg", "abl_zero_fill", "dsn"):
            L.append("\\addlinespace")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(L) + "\n"


def table2(summ):
    keys = [("rmse", "RMSE"), ("ds_de_t_spearman", "$t$ corr."), ("ds_de_precision", "Prec."),
            ("ds_de_recall", "Recall"), ("ds_coabundance_mae", "Co-ab.")]
    scs = [("DS-PDAC", "PDAC"), ("DS-LUAD", "LUAD")]
    L = ["\\begin{table*}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{2.4pt}",
         "\\caption{Semi-synthetic downstream analyses: CPTAC tumours and normal adjacent tissues with an imposed "
         "intensity-scale detection mask (mean over 4 masks): imputation error and agreement with complete-data "
         "statistics, not a test of error control (see Supplementary Table S15). RMSE: all hidden entries of all "
         "samples (every sample was imputed; standardized units; Supplementary Table S23). Tumour-versus-normal Welch tests on imputed data are compared with the same tests "
         "on the complete data: Spearman correlation of $t$ statistics ($t$ corr.), and precision (Prec.) and recall of the BH 5\\% calls. "
         "Co-abundance error (Co-ab.): mean absolute difference between imputed and complete protein--protein correlations "
         "(proteins with $\\ge$10\\% missing). No imputation: tests and correlations on observed values only.}",
         "\\label{tab:downstream}",
         "\\begin{tabular}{l" + "rrrrr" * len(scs) + "}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{5}}{{c}}{{{t}}}" for _, t in scs) + " \\\\",
         "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}",
         "Method & " + " & ".join(" & ".join(k[1] for k in keys) for _ in scs) + " \\\\", "\\midrule"]
    for m in DS_ORDER:
        if not any(m in summ.get(s, {}).get("methods", {}) for s, _ in scs):
            continue
        cells = []
        for s, _ in scs:
            v = summ.get(s, {}).get("methods", {}).get(m, {})
            cells += [f3(v.get(k)) for k, _ in keys]
        L.append(" & ".join([LABEL[m]] + cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(L) + "\n"


def table_ds_accuracy(summ):
    """Imputation accuracy in the downstream scenarios (Supplementary Table S23)."""
    scs = [("DS-PDAC", "PDAC (145 tumours, 36 normals)"), ("DS-LUAD", "LUAD (111 tumours, 102 normals)")]
    L = ["\\begin{table}[htbp]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{2.6pt}",
         "\\caption{Imputation accuracy in the semi-synthetic downstream scenarios (CPTAC tumours and normal adjacent "
         "tissues; 4 imposed masks; every sample imputed after fitting on observed values with an 80/20 train/validation "
         "split). RMSE (SD over masks), bias and mean within-protein correlation $r_w$ over all hidden entries "
         "(standardized units). $\\Delta$: mean RMSE difference from tcsimpute (linear) (positive = linear variant "
         "better); Masks: masks in which the linear variant had the lower RMSE.}",
         "\\label{tab:dsacc}", "\\begin{tabular}{lrrrrrrrrrr}", "\\toprule",
         " & " + " & ".join(f"\\multicolumn{{5}}{{c}}{{{t}}}" for _, t in scs) + " \\\\",
         "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}",
         "Method & " + " & ".join("RMSE (SD) & Bias & $r_w$ & $\\Delta$ & Masks" for _ in scs) + " \\\\", "\\midrule"]
    for m in DS_ORDER:
        if m == "available_case":
            continue
        cells, seen = [], False
        for s, _ in scs:
            v = summ.get(s, {}).get("methods", {}).get(m)
            if not v or v.get("rmse") is None:
                cells += ["--"] * 5
                continue
            seen = True
            q = v.get("vs_selreg")
            if m == "vae_full":
                q = {"mean_diff": v["vs_selreg"]["mean_diff"], "wins_reference": v["vs_selreg"]["wins_reference"],
                     "n": v["vs_selreg"]["n"]}
            d = "--" if m == "selreg" or q is None else f3(q["mean_diff"], True)
            w = "--" if m == "selreg" or q is None else f"{q['wins_reference']}/{q['n']}"
            cells += [f"{f3(v['rmse'])} ({f3(v['rmse_sd'])})", f3(v.get("bias"), True), f3(v.get("within_feature_r")), d, w]
        if seen:
            L.append(" & ".join([LABEL[m]] + cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L) + "\n"


def table_full(summ, sc, title, label, rl=None):
    ms = summ[sc]["methods"]
    rv = (rl or {}).get(sc, {}).get("selreg", {})
    L = ["\\begin{table}[htbp]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{2.6pt}",
         f"\\caption{{{title}: all methods. RMSE (SD) over grid points; $\\Delta$: mean paired RMSE difference from "
         "tcsimpute (linear), the default (positive = worse than the linear variant), with an approximate 95\\% two-stage "
         "cluster bootstrap interval over data replicates (4--5 clusters; coarse); Reps: data replicates in which the linear "
         "variant had the lower mean RMSE (grid points in brackets); Supplementary Table S18 gives the replicate ranges; "
         "$r_w$: mean "
         "within-protein correlation of imputed and true hidden values; Cov.: empirical coverage of nominal 90\\% "
         f"intervals (after conformal adjustment in brackets).}}", f"\\label{{{label}}}",
         "\\begin{tabular}{lrrrrrr}", "\\toprule",
         "Method & RMSE (SD) & $\\Delta$ [95\\% CI] & Reps & $r_w$ & Bias & Cov. \\\\", "\\midrule"]
    for m in MAIN_ORDER:
        v = ms.get(m)
        if v is None or v.get("rmse") is None:
            continue
        p = v.get("vs_selreg")
        if m == "selreg" or p is None:
            dcell, wins, dz = "--", "--", "--"
        else:
            q = rv.get(m)
            if q is not None:
                dcell = f"{f3(q['mean_diff'], True)} [{f2(q['cluster_ci95'][0])}, {f2(q['cluster_ci95'][1])}]"
                wins = f"{q['replicates_reference_better']}/{q['n_replicates']} ({q['points_reference_better']})"
            else:
                dcell = f"{f3(p['mean_diff'], True)} [{f2(p['ci95'][0])}, {f2(p['ci95'][1])}]"
                wins = f"-- ({p['wins_reference']})"
            dz = "--" if p.get("cohens_dz") is None else f"{p['cohens_dz']:.1f}".replace("-", "$-$")
        cov = "--"
        if v.get("coverage90") is not None:
            cov = f"{v['coverage90']:.2f}"
            if v.get("coverage_conformal") is not None:
                cov += f" [{v['coverage_conformal']:.2f}]"
        L.append(" & ".join([PLAIN[m].replace("-", "$-$") if m.startswith("abl_") else LABEL[m],
                             f"{f3(v['rmse'])} ({f3(v['rmse_sd'])})", dcell, wins,
                             f3(v.get("within_feature_r")), f3(v.get("bias"), True), cov]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L) + "\n"


def table_replevel(rl):
    """Replicate-level paired differences against the linear variant (Supplementary Table S18)."""
    if not rl:
        return "% replicate_level.json not available\n"
    scs = [s for s, _ in MAIN_SC if s in rl]
    order = [m for m in MAIN_ORDER if m != "selreg" and not m.startswith("abl_")]

    g = lambda v: f3(v, True).replace("+", "")
    L = ["\\begin{table}[htbp]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
         "\\caption{Paired RMSE differences against tcsimpute (linear) at the level of data replicates. $\\Delta$: mean "
         "difference over grid points (method minus linear variant; positive = linear variant better). Reps: data replicates "
         "in which the linear variant had the lower mean RMSE. Range: smallest and largest replicate mean of the difference. "
         "Data replicates: 5 independently simulated data sets; 4 imposed masks on the same CPTAC cohort; 4 primary-run "
         "choices on the same ProCan cell lines (CPTAC and ProCan replicates share samples). Cluster-bootstrap intervals "
         "are in Tables S1--S4.}",
         "\\label{tab:replevel}",
         "\\begin{tabular}{lrrrrrr}", "\\toprule"]
    for k in range(0, len(scs), 2):
        pair = scs[k:k + 2]
        if k:
            L.append("\\midrule")
        L.append(" & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{dict(MAIN_SC)[s].replace(chr(10), ' ')}}}" for s in pair) + " \\\\")
        L.append("".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(pair))))
        L.append("Method & " + " & ".join("$\\Delta$ & Reps & Range" for _ in pair) + " \\\\")
        L.append("\\midrule")
        for m in order:
            if not any(m in rl[s].get("selreg", {}) for s in pair):
                continue
            cells = []
            for s in pair:
                q = rl[s].get("selreg", {}).get(m)
                if q is None:
                    cells += ["--", "--", "--"]
                    continue
                cells += [f"{g(q['mean_diff'])}",
                          f"{q['replicates_reference_better']}/{q['n_replicates']}",
                          f"{g(q['replicate_min'])} to {g(q['replicate_max'])}"]
            L.append(" & ".join([LABEL[m]] + cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L) + "\n"


def table_sweeps(summ):
    rows, R, O = sweep_matrix(summ)
    L = ["\\begin{table}[htbp]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{1.7pt}",
         "\\caption{Misspecification and sensitivity sweeps (simulator variants of Section S1; 10 data replicates each; one split). "
         "The sweeps ran the methods shown, not the full comparator set of Tables S1--S4. "
         "Mean hidden-entry test RMSE; the best method in each row, and any within 0.005 of it (treated as tied), is in bold. The Bayes oracle is available when the "
         "generating detection and noise models give a closed form or quadrature. Rows above the rule change the "
         "detection or noise model; rows below change the missing fraction, sample size or RNA--protein coupling.}",
         "\\label{tab:sweeps}",
         "\\begin{tabular}{l" + "r" * (len(SWEEP_COLS) + 1) + "}", "\\toprule",
         "Condition & " + " & ".join("\\shortstack[r]{" + ("VAE $-$\\\\latent" if m == "abl_no_latent" else PLAIN[m].replace(" ", "\\\\")) + "}" for m in SWEEP_COLS)
         + " & Oracle \\\\", "\\midrule"]
    for i, ((sc, lab), r, o) in enumerate(zip(rows, R, O)):
        best = np.nanmin(r)
        cells = [("\\textbf{%s}" % f3(x)) if np.isfinite(x) and x <= best + 0.005 else f3(x) for x in r]
        L.append(" & ".join([lab.replace("%", "\\%")] + cells + [f3(o) if np.isfinite(o) else "--"]) + " \\\\")
        if sc == "SWEEP-t3":
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L) + "\n"

ALT_SC = [("ALT-PDAC-logit", "PDAC, logistic"), ("ALT-PDAC-sporadic", "PDAC, sporadic"),
          ("ALT-LUAD-logit", "LUAD, logistic"), ("ALT-LUAD-sporadic", "LUAD, sporadic")]
ALT_ORDER = ["selreg", "vae_full", "abl_no_latent", "abl_mar_likelihood", "pirat_t", "proda", "notmiwae_cog",
             "missforest_r", "knn_cog", "mindet_r", "cognate_ols", "observed_mean"]


def table_alt(summ, rl):
    """Out-of-family detection masks on CPTAC (Supplementary Table S24)."""
    scs = [(s, t) for s, t in ALT_SC if s in summ]
    if not scs:
        return "% no ALT scenarios in summary\n"
    rl = rl or {}
    g = lambda v: f3(v, True).replace("+", "")
    L = ["\\begin{table}[htbp]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
         "\\caption{Imposed detection masks outside the model's family (CPTAC tumours; 4 masks $\\times$ 2 splits per "
         "scenario; Section~S13). Logistic: per-protein logistic detection curves whose limits and slopes vary between "
         "proteins, no sample offsets, 2\\% missingness completely at random. Sporadic: the probit mask with sample "
         "offsets, in which missingness completely at random is replaced by protein-specific random loss (fraction drawn "
         "from Beta(1,\\,4) for each protein). Hidden-entry test RMSE (standardized units); $\\Delta$: mean paired "
         "difference from tcsimpute (linear), the default (positive = linear variant better); Reps: masks in which the "
         "linear variant had the lower mean RMSE. The lowest RMSE in each column, and any within 0.005 of it, is in bold.}",
         "\\label{tab:alt}",
         "\\begin{tabular}{lrrrrrr}", "\\toprule"]
    for k in range(0, len(scs), 2):
        pair = scs[k:k + 2]
        if k:
            L.append("\\midrule")
        L.append(" & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{t}}}" for _, t in pair) + " \\\\")
        L.append("".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(pair))))
        L.append("Method & " + " & ".join("RMSE & $\\Delta$ & Reps" for _ in pair) + " \\\\")
        L.append("\\midrule")
        best = {s: min(v["rmse"] for v in summ[s]["methods"].values() if v.get("rmse") is not None) for s, _ in pair}
        for m in ALT_ORDER:
            if not any(m in summ[s]["methods"] for s, _ in pair):
                continue
            cells = []
            for s, _ in pair:
                v = summ[s]["methods"].get(m)
                if v is None or v.get("rmse") is None:
                    cells += ["--", "--", "--"]
                    continue
                r = f3(v["rmse"])
                if v["rmse"] <= best[s] + 0.005:
                    r = "\\textbf{" + r + "}"
                q = rl.get(s, {}).get("selreg", {}).get(m)
                if m == "selreg" or q is None:
                    cells += [r, "--", "--"]
                else:
                    cells += [r, g(q["mean_diff"]), f"{q['replicates_reference_better']}/{q['n_replicates']}"]
            L.append(" & ".join([LABEL[m]] + cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L) + "\n"


def table_procan_corrected(pc):
    """Corrected-matching ProCan rerun (Supplementary Table S25)."""
    if not pc:
        return "% procan_corrected.json not available\n"
    rows = [("observed_mean", "Observed mean"),
            ("no_rna_selection", "Selection, no transcripts"),
            ("rna_mar", "Transcripts, MAR likelihood"),
            ("selreg", "tcsimpute (linear)"),
            ("cognate_ols", "Cognate OLS"), ("knn_cog", "KNN"), ("mindet_r", "MinDet")]
    F, E = pc["panels"]["fixed"], pc["panels"]["expanded"]
    rng = lambda v: f"{v['rmse']:.3f} ({v['rmse_min']:.3f}--{v['rmse_max']:.3f})"
    sg = lambda x: f3(x, True)
    L = ["\\begin{table}[htbp]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
         "\\caption{ProCan rerun with corrected protein--transcript matching (Section~S14): the seven methods that completed all "
         "20 grid points (4 primary-run choices $\\times$ 5 splits) on both panels. RMSE: hidden-entry test RMSE, mean over grid "
         "points (range of the four primary-run-choice means). $\\Delta$ legacy: mean difference from the same method's main-benchmark "
         "predictions (Table~S4), rescored on the fixed panel's proteins. The first four rows form a factorial of the linear variant's components: "
         "transcripts in the mean model (no/yes) $\\times$ likelihood (MAR/selection). Below the table: factorial contrasts "
         "(mean RMSE difference; range over primary-run choices; choices with a positive difference). The second and third rows "
         "are the two control arms; `selection for MAR' replaces the MAR likelihood by the selection likelihood.}",
         "\\label{tab:procancorr}",
         "\\begin{tabular}{lrrrrr}", "\\toprule",
         " & \\multicolumn{3}{c}{Fixed panel (1{,}199--1{,}206 proteins)} & \\multicolumn{2}{c}{Expanded panel (1{,}500 proteins)} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}",
         "Method & RMSE (range) & Bias & $\\Delta$ legacy & RMSE (range) & Bias \\\\", "\\midrule"]
    for m, lab in rows:
        f, e = F["methods"][m], E["methods"][m]
        dl = F["minus_legacy"].get(m)
        L.append(" & ".join([lab, rng(f), sg(f["bias"]), "--" if dl is None else sg(dl["mean"]), rng(e), sg(e["bias"])]) + " \\\\")
        if m == "selreg":
            L.append("\\midrule")
    L += ["\\midrule", "\\multicolumn{6}{l}{\\textit{Factorial contrasts}} \\\\"]
    labs = [("selection_given_no_RNA", "Selection for MAR, no transcripts"),
            ("RNA_given_MAR", "Adding transcripts, MAR"),
            ("RNA_given_selection", "Adding transcripts, selection"),
            ("selection_given_RNA", "Selection for MAR, transcripts"),
            ("RMSE_interaction", "Interaction")]
    for k, lab in labs:
        a, b = F["factorial"][k], E["factorial"][k]
        cell = lambda v: f"{sg(v['mean'])} ({sg(v['min'])} to {sg(v['max'])}; {v['positive']}/{v['n']})"
        L.append(f"{lab} & \\multicolumn{{3}}{{r}}{{{cell(a)}}} & \\multicolumn{{2}}{{r}}{{{cell(b)}}} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L) + "\n"


def table_runtime(summ):
    scs = [s for s, _ in MAIN_SC if s in summ]
    L = ["\\begin{table}[htbp]", "\\centering", "\\scriptsize",
         "\\caption{Median wall time per job (seconds; one CPU thread for Python methods; R methods single-threaded), "
         "including model selection and test imputation. Pirat and Pirat-T are transductive and were fitted once per data "
         "replicate; their entry is the median fit time. Linear variant: runs without interval output; in brackets, the "
         "re-runs that also computed 90\\% and other predictive intervals (500 weighted draws per entry).}", "\\label{tab:runtime}",
         "\\begin{tabular}{l" + "r" * len(scs) + "}", "\\toprule",
         "Method & " + " & ".join(hdr(dict(MAIN_SC)[s]) for s in scs) + " \\\\", "\\midrule"]
    for m in MAIN_ORDER:
        if m == "oracle":
            continue
        vals = []
        for s in scs:
            mv = summ[s]["methods"].get(m, {})
            v = mv.get("r_fit_seconds_median") if m in ("pirat", "pirat_t") else mv.get("runtime_median_s")
            fmt_ = lambda x: f"{x:.1f}" if x < 100 else f"{x:.0f}"
            cell = "--" if v is None else fmt_(v)
            w = mv.get("runtime_median_s_with_intervals")
            if v is not None and w is not None:
                cell += f" [{fmt_(w)}]"
            vals.append(cell)
        L.append(" & ".join([LABEL[m]] + vals) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L) + "\n"


def table_scaling(scaling):
    if not scaling:
        return ""
    L = ["\\begin{table}[htbp]", "\\centering", "\\scriptsize",
         "\\caption{Runtime scaling on simulated data (genes = 1.6 $\\times$ proteins + 200). VAE: 50 training epochs "
         f"without early stopping. Hardware: {scaling['meta'].get('processor') or 'CPU'}"
         + (f"; GPU {scaling['meta']['cuda']}" if scaling['meta'].get('cuda') else "") + ".}",
         "\\label{tab:scaling}", "\\begin{tabular}{rrlrr}", "\\toprule",
         "Samples & Proteins & Model, device & Seconds & Peak GPU MB \\\\", "\\midrule"]
    for r in sorted(scaling["results"], key=lambda r: (r["n_samples"], r["n_proteins"], r["method"], r["device"], r["threads"])):
        name = ("VAE" if r["method"] == "vae_full" else "Linear") + ", " + (
            "GPU" if r["device"] == "cuda" else f"CPU {r['threads']} thread" + ("s" if r["threads"] > 1 else ""))
        L.append(f"{r['n_samples']} & {r['n_proteins']:,} & {name} & {r['seconds']:.1f} & "
                 f"{r.get('peak_gpu_mb', '--')} \\\\".replace(",", "{,}"))
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L) + "\n"


def table_strata(strata):
    if not strata:
        return ""
    order = ["vae_full", "selreg", "abl_no_latent", "abl_mar_likelihood", "pirat_t", "pirat", "missforest_r", "knn_cog", "miwae_cog",
             "notmiwae_cog", "proda", "cognate_ols", "mindet_r", "dsn", "qrilc_r", "observed_mean", "tobit"]
    labs = [("rare", "rarely"), ("some", "sometimes"), ("usual", "usually")]
    n = {k: strata["vae_full"][k]["n"] for k, _ in labs}
    L = ["\\begin{table}[htbp]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{4pt}",
         "\\caption{ProCan: hidden-entry test RMSE and bias, pooled over the 20 grid points and stratified by the fraction "
         "of the cell line's other runs in which the protein was quantified: rarely ($\\le$1/3; " + f"{n['rare']:,}"
         + " entries), sometimes (1/3--2/3; " + f"{n['some']:,}" + ") and usually ($>$2/3; " + f"{n['usual']:,}"
         + "). Entries never quantified in another run have no truth and are not scored. "
         "VAE, MAR likelihood: the tcsimpute VAE with the selection model replaced by a missing-at-random likelihood.}",
         "\\label{tab:strata}", "\\begin{tabular}{lrrrrrr}", "\\toprule",
         " & \\multicolumn{2}{c}{Rarely} & \\multicolumn{2}{c}{Sometimes} & \\multicolumn{2}{c}{Usually} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
         "Method & RMSE & Bias & RMSE & Bias & RMSE & Bias \\\\", "\\midrule"]
    for m in order:
        if m not in strata:
            continue
        cells = []
        for k, _ in labs:
            cells += [f3(strata[m][k]["rmse"]), f3(strata[m][k]["bias"], True)]
        L.append(" & ".join([PLAIN[m].replace("-", "$-$") if m.startswith("abl_") else LABEL[m]] + cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L) + "\n"


def table_reptruth(rt):
    if not rt:
        return ""
    order = ["oracle", "vae_full", "selreg", "abl_no_latent", "abl_mar_likelihood", "notmiwae_cog", "proda",
             "cognate_ols", "missforest_r", "mindet_r", "knn_cog", "miwae_cog", "observed_mean"]
    L = ["\\begin{table}[htbp]", "\\centering", "\\scriptsize",
         "\\caption{Simulator predictions rescored against a replicate-based truth built as in ProCan: five further runs "
         "per sample (technical SD 0.2, same probit detection, fresh run offsets); the truth of a hidden test entry is the "
         "mean of its detected replicate values, and entries never detected are unscored ("
         + f"{100 * rt['scored_fraction']:.0f}" + "\\% of hidden entries scored). Mean standardized value over entries, in the units used for scoring: all hidden entries "
         + f"{rt['mean_z_all_hidden_true']:.2f}".replace("-", "$-$") + "; scored entries, true values "
         + f"{rt['mean_z_scored_true']:.2f}".replace("-", "$-$") + "; scored entries, replicate truth "
         + f"{rt['mean_z_scored_replicate_truth']:.2f}".replace("-", "$-$") + ". Means over the 20 grid points.}",
         "\\label{tab:reptruth}", "\\begin{tabular}{lrrr}", "\\toprule",
         "Method & RMSE, true values & RMSE, replicate truth & Bias, replicate truth \\\\", "\\midrule"]
    for m in order:
        v = rt["methods"].get(m)
        if v:
            L.append(" & ".join([PLAIN[m].replace("-", "$-$") if m.startswith("abl_") else LABEL[m], f3(v["true"]),
                                 f3(v["rep"]), f3(v["rep_bias"], True)]) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L) + "\n"


def table_diag(dg):
    if not dg:
        return ""
    names = [("SIM", "Simulator (probit)"), ("SWEEP-logistic", "Simulator, logistic"),
             ("SWEEP-threshold", "Simulator, hard threshold"), ("SWEEP-widelimits", "Simulator, wide limits"),
             ("SWEEP-mcar15", "Simulator, 15\\% MCAR"), ("SWEEP-batch", "Simulator, batch-wise"),
             ("PDAC", "CPTAC PDAC (imposed)"), ("LUAD", "CPTAC LUAD (imposed)"), ("PROCAN", "ProCan DIA (natural)")]
    L = ["\\begin{table}[htbp]", "\\centering", "\\scriptsize",
         "\\caption{Missingness diagnostic computable from observed data: for each protein with at least five missing and "
         "five observed samples, the AUC with which its cognate transcript separates observed from missing samples "
         "(0.5: missingness unrelated to the transcript). Median (interquartile range) across proteins, one data replicate "
         "per scenario. Last column: mean true value of missing entries where it is known (ProCan: replicate-based "
         "truth of scored entries), standardized by each protein's observed mean and SD and averaged over proteins; it "
         "differs from the entry-weighted values of Supplementary Table S7.}", "\\label{tab:diag}", "\\begin{tabular}{lrr}", "\\toprule",
         "Scenario & AUC & Mean $z$ of missing \\\\", "\\midrule"]
    for k, lab in names:
        v = dg.get(k)
        if v:
            L.append(f"{lab} & {v['auc_median']:.2f} ({v['auc_q25']:.2f}--{v['auc_q75']:.2f}) & "
                     + f"{v['missing_truth_mean_z']:.2f}".replace("-", "$-$") + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ numbers for the text

def numbers(summ, scaling):
    out = {}
    for sc in summ:
        ms = summ[sc]["methods"]
        out[sc] = {m: {k: v[k] for k in ("rmse", "rmse_sd", "bias", "within_feature_r", "coverage90",
                                         "coverage_conformal", "coverage_level50", "coverage_level80",
                                         "coverage_level95", "coverage_pseudo", "runtime_median_s",
                                         "r_fit_seconds_median", "active_units",
                                         "ds_de_t_spearman", "ds_de_precision", "ds_de_recall",
                                         "ds_coabundance_mae", "ds_de_calls", "ds_de_calls_reference")
                       if v.get(k) is not None}
                   for m, v in ms.items()}
        for m, v in ms.items():
            for ref in ("vae_full", "selreg"):
                if v.get(f"vs_{ref}"):
                    out[sc][m][f"vs_{ref}"] = v[f"vs_{ref}"]
        cmp_ = [(v["rmse"], m) for m, v in ms.items()
                if v.get("rmse") is not None and group(m) == "cmp" and m != "tobit"]
        if cmp_:
            out[sc]["_best_established"] = min(cmp_)
        if summ[sc].get("ranks"):
            out[sc]["_ranks"] = summ[sc]["ranks"]
    if scaling:
        out["_scaling"] = scaling
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--scaling", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", default=None, help="directory with procan_strata.json, replicate_truth_sim.json, "
                                                   "missingness_diagnostic.json, replicate_level.json")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summ = json.loads(Path(a.summary).read_text(encoding="utf-8"))
    scaling = json.loads(Path(a.scaling).read_text(encoding="utf-8")) if a.scaling and Path(a.scaling).exists() else None
    w = lambda name, text: (out / name).write_text(text, encoding="utf-8")
    ex = Path(a.extra) if a.extra else None
    load = lambda n: json.loads((ex / n).read_text(encoding="utf-8")) if ex and (ex / n).exists() else None
    rl = load("replicate_level.json")
    w("table1.tex", table1(summ))
    w("table2.tex", table2(summ))
    supp = []
    for i, (sc, t) in enumerate(MAIN_SC, 1):
        if sc in summ:
            supp.append(table_full(summ, sc, {"SIM": "Simulator", "PDAC": "CPTAC PDAC (semi-synthetic, imposed mask)",
                                              "LUAD": "CPTAC LUAD (semi-synthetic, imposed mask)",
                                              "PROCAN": "ProCan DIA (natural missingness)"}[sc], f"tab:full{sc.lower()}", rl))
    w("tableS_full.tex", "\n".join(supp))
    w("tableS_sweeps.tex", table_sweeps(summ))
    w("tableS_dsacc.tex", table_ds_accuracy(summ))
    w("tableS_runtime.tex", table_runtime(summ))
    w("tableS_scaling.tex", table_scaling(scaling))
    w("tableS_replevel.tex", table_replevel(rl))
    w("tableS_alt.tex", table_alt(summ, rl))
    w("tableS_procancorr.tex", table_procan_corrected(load("procan_corrected.json")))
    w("tableS_strata.tex", table_strata(load("procan_strata.json")))
    w("tableS_reptruth.tex", table_reptruth(load("replicate_truth_sim.json")))
    w("tableS_diag.tex", table_diag(load("missingness_diagnostic.json")))
    w("numbers.json", json.dumps(numbers(summ, scaling), indent=1))
    fig_model(out)
    fig_benchmark(summ, out)
    fig_robustness(summ, out)
    fig_calib_downstream(summ, scaling, out)
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
