import json

S = json.load(open("runs_revision/summary/summary_revision.json"))
KEEP = {"n", "runtime_median_s", "rmse", "rmse_sd", "bias", "within_feature_r", "coverage90", "coverage_conformal",
        "coverage_level50", "coverage_level80", "coverage_level95", "interval_width90", "interval_width_conformal",
        "active_units", "ds_de_t_spearman", "ds_de_t_spearman_sd", "ds_de_precision", "ds_de_recall", "ds_de_effect_rmse",
        "ds_coabundance_mae", "ds_coabundance_mae_sd", "ds_de_calls", "ds_de_calls_reference"}


def rnd(x):
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, list):
        return [rnd(v) for v in x]
    if isinstance(x, dict):
        return {k: rnd(v) for k, v in x.items()}
    return x


SW = {"vae_full", "selreg", "abl_no_latent", "proda", "notmiwae_cog16", "knn_cog", "mindet_r", "cognate_ols",
      "observed_mean", "oracle"}
out = {}
for sc, res in S.items():
    ms = {}
    if sc.startswith("SWEEP"):
        for m, v in res["methods"].items():
            if m in SW:
                ms[m] = {k: v[k] for k in ("rmse", "bias", "coverage90", "coverage_conformal") if k in v}
        out[sc] = {"methods": ms, "ranks": None}
        continue
    for m, v in res["methods"].items():
        e = {k: v[k] for k in KEEP if k in v}
        for ref in ("vs_vae_full", "vs_selreg"):
            if v.get(ref):
                e[ref] = {k: v[ref][k] for k in ("n", "mean_diff", "ci95", "wins_reference", "cohens_dz")}
        ms[m] = e
    r = res.get("ranks")
    if r:
        r = {k: r[k] for k in ("n_points", "friedman_chi2", "friedman_p", "mean_ranks", "nemenyi_cd", "methods")}
    out[sc] = {"methods": ms, "ranks": r}
s = json.dumps(rnd(out), separators=(",", ":"))
open("summary_compact.json", "w").write(s)
print(len(s))
