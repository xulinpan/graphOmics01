"""Error control of differential-abundance tests after imputation (Supplementary Section S8).

Repeated-mask experiment on CPTAC tumours. In each replicate the tumours are split at random into two
groups of equal size, so that no protein differs between them, and a known shift of 0.5 within-protein
SD (random sign) is added to group B for a random 10% of proteins. An abundance-dependent detection
mask is then imposed on the shifted intensity-scale values, exactly as in the semi-synthetic benchmark,
and each method imputes the missing values. Welch tests compare the groups protein by protein.

Methods: complete data (reference); available-case tests on observed values; MinDet; tcsimpute
linear and VAE conditional means (single imputation); and tcsimpute linear multiple imputation
(M draws from each missing value's conditional distribution given non-detection, combined by
Rubin's rules with Barnard-Rubin degrees of freedom; parameter uncertainty is not propagated).

Usage: python inference_calibration.py --cptac-dir data_cptac --out inference --workers 14
"""
import argparse
import json
import os
from multiprocessing import get_context
from pathlib import Path

import numpy as np

N_REPLICATES = 20
EFFECT_FRACTION = 0.10
EFFECT_SD = 0.5
N_IMPUTATIONS = 20
ALPHA = 0.05
CPTAC_DIR = None


def welch(a, b):
    """Column-wise Welch statistics ignoring NaN: difference (b - a), its variance and df."""
    na, nb = np.sum(~np.isnan(a), 0), np.sum(~np.isnan(b), 0)
    ma, mb = np.nanmean(a, 0), np.nanmean(b, 0)
    va, vb = np.nanvar(a, 0, ddof=1) / na, np.nanvar(b, 0, ddof=1) / nb
    v = va + vb
    df = v ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    return mb - ma, v, df


def p_and_ci(d, v, df):
    from scipy.stats import t
    se = np.sqrt(v)
    p = 2 * t.sf(np.abs(d) / se, df)
    q = t.ppf(0.975, df)
    return p, d - q * se, d + q * se


def rubin(ds, vs, dfs):
    """Rubin's rules with the Barnard-Rubin small-sample degrees of freedom."""
    m = ds.shape[0]
    d = ds.mean(0)
    w = vs.mean(0)
    b = ds.var(0, ddof=1)
    tvar = w + (1 + 1 / m) * b
    lam = np.clip((1 + 1 / m) * b / tvar, 1e-12, 1 - 1e-12)
    nu_old = (m - 1) / lam ** 2
    nu_com = dfs.mean(0)
    nu_obs = (nu_com + 1) / (nu_com + 3) * nu_com * (1 - lam)
    return d, tvar, 1 / (1 / nu_old + 1 / nu_obs)


def bh(p, alpha=ALPHA):
    p = np.asarray(p)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    passed = np.flatnonzero(ranked <= alpha)
    calls = np.zeros(len(p), bool)
    if passed.size:
        calls[order[: passed.max() + 1]] = True
    return calls


def base_cohort(cohort):
    from tcsimpute.benchmark import data as D
    cfg = D.CPTACConfig(mask="intensity", include_normals=False)
    return D.load_cptac(CPTAC_DIR, 21, cfg, cohort=cohort)


def run_one(job):
    cohort, rep = job
    import torch
    torch.set_num_threads(1)
    from tcsimpute.benchmark import data as D
    from tcsimpute.benchmark import baselines as B
    from tcsimpute.model import VAEConfig, fit_vae, predict_split

    src = base_cohort(cohort)
    values = np.asarray(src["protein"], float)          # complete intensity-scale values
    n, p = values.shape
    sd_within = float(src["mechanism"]["within_protein_sd_median"])
    rng = np.random.default_rng(70_000 + 1000 * (cohort == "luad") + rep)
    group_b = np.zeros(n, bool)
    group_b[rng.permutation(n)[: n // 2]] = True
    effect = np.zeros(p)
    idx = rng.choice(p, int(round(EFFECT_FRACTION * p)), replace=False)
    effect[idx] = EFFECT_SD * values.std(0)[idx] * rng.choice([-1.0, 1.0], idx.size)
    shifted = values + group_b[:, None] * effect[None, :]

    offsets = rng.normal(0, 0.3, n) * sd_within
    slope = 0.5 * sd_within
    d0 = D.calibrate_detection_limit(shifted, offsets, slope, 0.02, 0.30)
    observed = rng.uniform(size=shifted.shape) < D.detection_probability(shifted, d0 + offsets[:, None], slope, 0.02)
    keep = (observed.mean(0) >= 0.3) & (observed[group_b].sum(0) >= 3) & (observed[~group_b].sum(0) >= 3)
    shifted, observed, effect = shifted[:, keep], observed[:, keep], effect[keep]
    miss_frac = 1 - observed.mean(0)
    sub = {"sample_ids": src["sample_ids"], "rna": src["rna"], "protein": shifted, "observed": observed,
           "cognate": src["cognate"][keep], "oracle_hidden_mean": None,
           "mechanism": {"kind": "inference_calibration", "replicate": rep}, "config": {}, "seed": rep}
    perm = rng.permutation(n)
    n_va = int(0.2 * n)
    split = {"train": perm[n_va:], "validation": perm[:n_va], "test": np.arange(n)}
    P = D.build_protocol(sub, split_seed=rep, name=f"inference-{cohort}-{rep}", split=split)
    allr = P.rows("test")
    to_raw = lambda z: np.asarray(z, float) * P.protein_scale + P.protein_mean

    imputed = {"complete": shifted, "available_case": np.where(observed, shifted, np.nan),
               "mindet": to_raw(B.min_det(P, allr))}
    fit = B.fit_selection_regression(P, limit_scale="raw", seed=rep)
    imputed["tcsimpute_linear"] = to_raw(B.selection_regression_predict(fit, P, allr))
    model, _ = fit_vae(P, VAEConfig(limit_scale="raw"), rep)
    imputed["tcsimpute_vae"] = to_raw(np.where(P.observed[allr], P.protein_input[allr], predict_split(model, P, "test")))

    # multiple imputation from the linear variant: draws of x | not detected, by importance resampling
    import torch as T
    with T.no_grad():
        X = B._tobit_design(P, fit["pcs"], allr)
        mu = (fit["alpha"] + (X * fit["beta"]).sum(-1)).numpy()
    sd, tau = fit["sd"].numpy()[None, :], fit["tau"].numpy()[None, :]
    s, pi = float(fit["s"]), float(fit["pi"])
    from scipy.stats import norm
    hidden = ~observed
    mi_d, mi_v, mi_df = [], [], []
    for m in range(N_IMPUTATIONS):
        cand = mu[None] + sd[None] * rng.standard_normal((50,) + mu.shape)           # 50 candidates per entry
        w = pi + (1 - pi) * norm.cdf((tau[None] - cand) / s)
        w = w / w.sum(0, keepdims=True)
        u = rng.uniform(size=mu.shape)
        pick = (np.cumsum(w, 0) < u[None]).sum(0).clip(max=49)
        draw = np.take_along_axis(cand, pick[None], 0)[0]
        z = np.where(hidden, draw, P.protein_input[allr])
        x = to_raw(z)
        d, v, df = welch(x[~group_b], x[group_b])
        mi_d.append(d), mi_v.append(v), mi_df.append(df)
    mi = rubin(np.array(mi_d), np.array(mi_v), np.array(mi_df))

    null, eff = effect == 0, effect != 0
    sel = miss_frac >= 0.10
    out = {"cohort": cohort, "replicate": rep, "n_samples": int(n), "n_proteins": int(keep.sum()),
           "n_effect": int(eff.sum()), "n_null_missing10": int((null & sel).sum()), "methods": {}}
    stats = {k: welch(v[~group_b], v[group_b]) for k, v in imputed.items()}
    stats["tcsimpute_linear_mi"] = mi
    for k, (d, v, df) in stats.items():
        pv, lo, hi = p_and_ci(d, v, df)
        calls = bh(pv)
        cover = (lo <= effect) & (effect <= hi)
        out["methods"][k] = {
            "type1_all": float((pv[null] < ALPHA).mean()),
            "type1_missing10": float((pv[null & sel] < ALPHA).mean()),
            "fdr": float((calls & null).sum() / max(calls.sum(), 1)),
            "power": float((calls & eff).mean() / eff.mean()),
            "power_missing10": float((calls & eff & sel).sum() / max((eff & sel).sum(), 1)),
            "ci_cover_null_missing10": float(cover[null & sel].mean()),
            "ci_cover_effect": float(cover[eff].mean()),
            "effect_rel_bias": float(np.mean((d[eff] - effect[eff]) * np.sign(effect[eff]))
                                     / np.mean(np.abs(effect[eff]))),
            "n_calls": int(calls.sum()),
        }
    return out


def summarize(fits):
    out = {}
    for cohort in sorted({f["cohort"] for f in fits}):
        rows = [f for f in fits if f["cohort"] == cohort]
        out[cohort] = {"n_replicates": len(rows), "methods": {}}
        for m in rows[0]["methods"]:
            e = {}
            for k in rows[0]["methods"][m]:
                v = np.array([r["methods"][m][k] for r in rows], float)
                e[k] = float(v.mean())
                e[k + "_sd"] = float(v.std())
            out[cohort]["methods"][m] = e
        out[cohort]["n_proteins"] = float(np.mean([r["n_proteins"] for r in rows]))
        out[cohort]["n_null_missing10"] = float(np.mean([r["n_null_missing10"] for r in rows]))
        out[cohort]["n_samples"] = rows[0]["n_samples"]
    return out


def _init(cptac_dir):
    global CPTAC_DIR
    CPTAC_DIR = cptac_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cptac-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--replicates", type=int, default=N_REPLICATES)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "fits").mkdir(parents=True, exist_ok=True)
    jobs = [(c, r) for c in ("pdac", "luad") for r in range(1, a.replicates + 1)]
    todo = [j for j in jobs if not (out / "fits" / ("%s_r%d.json" % j)).exists()]
    print(f"{len(jobs)} jobs, {len(todo)} to run", flush=True)
    with get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.cptac_dir,)) as pool:
        for i, res in enumerate(pool.imap_unordered(run_one, todo), 1):
            name = "%s_r%d.json" % (res["cohort"], res["replicate"])
            (out / "fits" / name).write_text(json.dumps(res))
            lin = res["methods"]["tcsimpute_linear"]
            print(f"[{i}/{len(todo)}] {name} linear type1={lin['type1_missing10']:.3f} fdr={lin['fdr']:.3f}", flush=True)
    fits = [json.loads(p.read_text()) for p in sorted((out / "fits").glob("*.json"))]
    (out / "inference_calibration.json").write_text(json.dumps({"fits": fits, "summary": summarize(fits)}, indent=1))
    print("done")


if __name__ == "__main__":
    main()
