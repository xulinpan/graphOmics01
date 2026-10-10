"""Figure 1 of the paper: hidden-entry test RMSE by method and scenario (mean +/- SD across grid
points), read from the summary.json written by tcsimpute-summarize."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GROUPS = {  # group -> (color, marker, label); identity is carried by marker shape as well as hue
    "full": ("#2a78d6", "o", "Full model and inference variants"),
    "ablation": ("#eb6834", "s", "Ablations"),
    "comparator": ("#1baf7a", "^", "Earlier model and comparators"),
    "control": ("#8a8a85", "D", "Controls / oracle"),
}
METHODS = [
    ("vae_full", "Full model", "full"),
    ("vae_full@single_draw", "  one latent draw", "full"),
    ("vae_full@mc50", "  50-draw average", "full"),
    ("abl_zero_fill", "- mask-aware encoder", "ablation"),
    ("abl_mlp_fusion", "- Transformer fusion", "ablation"),
    ("abl_no_contrastive", "- contrastive term", "ablation"),
    ("abl_no_skip", "- cognate skip", "ablation"),
    ("abl_no_latent", "- latent", "ablation"),
    ("abl_mar_likelihood", "- selection model (MAR)", "ablation"),
    ("abl_hinge_penalty", "selection -> v1 hinge", "ablation"),
    ("abl_v1_elbo_scaling", "v1 ELBO scaling", "ablation"),
    ("v1_reference", "v1 model (submitted)", "comparator"),
    ("notmiwae", "not-MIWAE", "comparator"),
    ("miwae", "MIWAE", "comparator"),
    ("missforest", "missForest", "comparator"),
    ("knn", "KNN", "comparator"),
    ("qrilc", "QRILC", "comparator"),
    ("minprob", "MinProb", "comparator"),
    ("mindet", "MinDet", "comparator"),
    ("dsn", "Down-shifted normal", "comparator"),
    ("cognate_ols", "Cognate OLS", "control"),
    ("observed_mean", "Observed mean", "control"),
    ("oracle", "Bayes oracle", "control"),
]
TITLES = {"S1": "S1: v1 fixture", "S2": "S2: nonlinear simulator", "CPTAC": "CPTAC semi-synthetic"}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tcsimpute-figure", description=__doc__)
    ap.add_argument("--summary", required=True, help="summary.json written by tcsimpute-summarize.")
    ap.add_argument("--out", default="figure_rmse.pdf", help="Output PDF (a PNG is written alongside).")
    a = ap.parse_args(argv)
    summary = json.loads(Path(a.summary).read_text())
    scenarios = [s for s in ("S1", "S2", "CPTAC") if s in summary]
    plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8a85", "xtick.color": "#4a4a46", "ytick.color": "#4a4a46"})
    fig, axes = plt.subplots(1, len(scenarios), figsize=(7.2, 4.6), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    ypos = {m: i for i, (m, _, _) in enumerate(METHODS)}
    for ax, sc in zip(axes, scenarios):
        agg = summary[sc]
        ref = agg["vae_full"]["rmse_mean"]
        ax.axvline(ref, color="#2a78d6", lw=0.8, ls=(0, (3, 2)), zorder=0)
        for m, label, group in METHODS:
            if m not in agg:
                continue
            color, marker, _ = GROUPS[group]
            v = agg[m]
            ax.errorbar(v["rmse_mean"], ypos[m], xerr=v["rmse_sd"], fmt=marker, ms=4.5,
                        color=color, mec="white", mew=0.6, elinewidth=1.0, capsize=0, zorder=3)
        ax.set_title(TITLES[sc], fontsize=8, loc="left")
        ax.grid(axis="x", color="#e4e3dd", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(METHODS)))
    axes[0].set_yticklabels([lab for _, lab, _ in METHODS])
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], color=c, marker=mk, ls="", ms=5, label=lab) for c, mk, lab in GROUPS.values()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.55, -0.01))
    fig.supxlabel("Hidden-entry test RMSE (training-standardized units)", y=0.06, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(Path(a.out).with_suffix(".png"), dpi=200, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
