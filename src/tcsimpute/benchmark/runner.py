"""Benchmark used in the paper: tcsimpute, its ablations and comparators across scenarios, data
replicates, splits and training seeds.

Two grids are defined:

* ``original`` - the first benchmark (S1, S2 and the within-protein CPTAC PDAC mask; 564 runs).
* ``revision`` - the benchmark of the revised paper:
    SIM            nonlinear many-to-many simulator, 5 data replicates x 4 splits (20 grid points)
    PDAC, LUAD     CPTAC cohorts with an intensity-scale detection mask, 4 masks x 5 splits (20 each)
    SWEEP-*        misspecification and sensitivity sweeps, 10 data replicates each
    PROCAN         natural missingness in DIA cell-line proteomics (see :mod:`.natural`),
                   4 primary-run choices x 5 splits
    DS-PDAC, DS-LUAD  downstream analyses (see :mod:`.downstream`): tumours and normals, every
                   sample imputed, 4 masks each

Each job fits one method on one frozen protocol and writes, under
<out>/<scenario>/rep<r>_split<s>/<method>_seed<t>/ :
    predictions.npz  - test predictions, truth, hidden mask (and inference variants / intervals)
    result.json      - test metrics, bootstrap CI, diagnostics, protocol fingerprint, runtime

Selection never reads hidden values: neural models select epochs (and, for MIWAE/not-MIWAE, the
latent dimension) by their own objective on validation samples; KNN selects k on a validation
pseudo-mask of observed entries; interval recalibration uses the same pseudo-mask. Test truth is
read once per job, after all choices are fixed. R comparators are transductive: they see every
sample's observed values and impute all samples.

Usage:
    tcsimpute-benchmark --grid revision --out runs --scenarios SIM PDAC LUAD --cptac-dir data_cptac
    tcsimpute-benchmark --grid revision --out runs --scenarios sweeps
    tcsimpute-benchmark --grid original --out runs_original --s1-rna S1.csv --cptac-dir data_cptac
"""

import argparse
import json
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

# Development replicate S2 seed 1 / split 11 was used for design-stage choices and is excluded.
ORIGINAL_GRID = {
    "S1": {"replicates": [42], "splits": [101, 202, 303], "train_seeds": [1, 2]},
    "S2": {"replicates": [101, 102, 103], "splits": [101, 202, 303], "train_seeds": [1, 2]},
    "CPTAC": {"replicates": [7], "splits": [101, 202, 303], "train_seeds": [1, 2]},
}

SWEEPS = {
    "SWEEP-probit": {},
    "SWEEP-logistic": {"mechanism": "logistic"},
    "SWEEP-threshold": {"mechanism": "threshold"},
    "SWEEP-widelimits": {"mechanism": "wide_limits"},
    "SWEEP-batch": {"mechanism": "batch", "target_missing": 0.40},
    "SWEEP-mcar15": {"mcar": 0.15},
    "SWEEP-t3": {"noise": "t3"},
    "SWEEP-miss10": {"target_missing": 0.125},
    "SWEEP-miss50": {"target_missing": 0.62, "n_proteins_generated": 1400},
    "SWEEP-n100": {"n_samples": 100},
    "SWEEP-n400": {"n_samples": 400},
    "SWEEP-weak": {"coupling": 0.5, "ptm_scale": 1.5},
    "SWEEP-strong": {"coupling": 1.5, "ptm_scale": 0.5},
}

_SPLITS5 = [101, 202, 303, 404, 505]
REVISION_GRID = {
    "SIM": {"replicates": [201, 202, 203, 204, 205], "splits": _SPLITS5[:4], "train_seeds": [1]},
    "PDAC": {"replicates": [11, 12, 13, 14], "splits": _SPLITS5, "train_seeds": [1]},
    "LUAD": {"replicates": [11, 12, 13, 14], "splits": _SPLITS5, "train_seeds": [1]},
    "PROCAN": {"replicates": [1, 2, 3, 4], "splits": _SPLITS5, "train_seeds": [1]},
    "DS-PDAC": {"replicates": [21, 22, 23, 24], "splits": [101], "train_seeds": [1]},
    "DS-LUAD": {"replicates": [21, 22, 23, 24], "splits": [101], "train_seeds": [1]},
    **{k: {"replicates": list(range(301, 311)), "splits": [101], "train_seeds": [1]} for k in SWEEPS},
    # second revision: CPTAC masks from mechanisms outside the model family (4 masks x 2 splits)
    **{f"ALT-{c}-{m}": {"replicates": [11, 12, 13, 14], "splits": [101, 202], "train_seeds": [1]}
       for c in ("PDAC", "LUAD") for m in ("logit", "sporadic")},
}
ALT_MECHANISM = {"logit": "logistic_protein", "sporadic": "sporadic"}

V2_VARIANTS = {
    "vae_full": {},
    "abl_zero_fill": {"mask_aware": False},
    "abl_mlp_fusion": {"fusion": "mlp"},
    "abl_no_contrastive": {"contrastive": 0.0},
    "abl_no_skip": {"skip": False},
    "abl_no_latent": {"use_latent": False},
    "abl_mar_likelihood": {"likelihood": "mar"},
    "abl_hinge_penalty": {"likelihood": "hinge"},
    "abl_v1_elbo_scaling": {"elbo": "v1"},
}
R_METHODS = {"proda": "proDA", "mindet_r": "MinDet", "minprob_r": "MinProb", "qrilc_r": "QRILC",
             "missforest_r": "missForest"}
SEEDLESS = {"oracle", "observed_mean", "mindet", "cognate_ols", "mindet_r", "tobit", "selreg", "proda",
            "available_case", "pirat", "pirat_t", "pirat_s", "proda_design"}
# Transductive R comparators whose imputation is cached per data replicate (added in the second
# revision): Pirat and Pirat-T (Etourneau et al. 2025) and proDA with the tumour/normal design.
TRANSDUCTIVE_CACHED = {"pirat", "pirat_t", "pirat_s", "proda_design"}

ORIGINAL_METHODS = (list(V2_VARIANTS) + ["v1_reference", "miwae", "notmiwae", "observed_mean",
                    "mindet", "minprob", "qrilc", "dsn", "cognate_ols", "knn", "missforest", "oracle"])
MAIN_METHODS = (["vae_full", "abl_no_latent", "abl_no_skip", "abl_mar_likelihood", "abl_hinge_penalty",
                 "abl_mlp_fusion", "abl_no_contrastive", "abl_zero_fill",
                 "selreg", "tobit", "proda", "pirat_t", "pirat", "mindet_r", "minprob_r", "qrilc_r", "dsn", "knn_cog",
                 "missforest_r", "miwae_cog", "notmiwae_cog", "cognate_ols", "observed_mean", "oracle"])
SWEEP_METHODS = ["vae_full", "abl_no_latent", "selreg", "proda", "pirat_t", "mindet_r", "knn_cog",
                 "notmiwae_cog16", "cognate_ols", "observed_mean", "oracle"]
ALT_METHODS = ["vae_full", "abl_no_latent", "abl_mar_likelihood", "selreg", "pirat_t", "proda",
               "mindet_r", "knn_cog", "missforest_r", "notmiwae_cog", "cognate_ols", "observed_mean"]
PROCAN_METHODS = [m for m in MAIN_METHODS if m != "oracle"]
DS_METHODS = ["vae_full", "abl_no_latent", "selreg", "proda", "proda_design", "pirat_t", "pirat", "mindet_r", "qrilc_r", "dsn", "knn_cog",
              "missforest_r", "notmiwae_cog", "cognate_ols", "observed_mean", "available_case"]


def limit_scale_for(scenario):
    # Declared a priori from the data type: intensity-like scales share a raw limit; the original
    # CPTAC benchmark imposed detection on per-protein standardized values.
    return "standardized" if scenario == "CPTAC" else "raw"


# ----------------------------------------------------------------------- protocol

_CACHE = {}


def source_for(scenario, replicate, s1_rna=None, cptac_dir=None, procan_dir=None):
    from . import data
    key = (scenario, replicate)
    if key in _CACHE:
        return _CACHE[key]
    if scenario == "S1":
        if s1_rna is None:
            raise ValueError("Scenario S1 needs --s1-rna (the original fixture's RNA CSV).")
        src = data.load_s1(s1_rna)
    elif scenario in ("S2", "SIM"):
        src = data.simulate_s2(replicate)
    elif scenario in SWEEPS:
        src = data.simulate_s2(replicate, replace(data.S2Config(), **SWEEPS[scenario]))
    elif scenario in ("CPTAC", "PDAC", "LUAD", "DS-PDAC", "DS-LUAD"):
        if cptac_dir is None:
            raise ValueError(f"Scenario {scenario} needs --cptac-dir.")
        cohort = "luad" if scenario.endswith("LUAD") else "pdac"
        cfg = data.CPTACConfig(mask="within" if scenario == "CPTAC" else "intensity",
                               include_normals=scenario.startswith("DS-"))
        src = _cached_cptac(Path(cptac_dir), scenario, cohort, replicate, cfg)
    elif scenario.startswith("ALT-"):
        if cptac_dir is None:
            raise ValueError(f"Scenario {scenario} needs --cptac-dir.")
        _, cohort, mech = scenario.split("-")
        cfg = data.CPTACConfig(mask="intensity", mechanism=ALT_MECHANISM[mech])
        src = _cached_cptac(Path(cptac_dir), scenario, cohort.lower(), replicate, cfg)
    elif scenario == "PROCAN":
        from .natural import load_procan
        if procan_dir is None:
            raise ValueError("Scenario PROCAN needs --procan-dir.")
        src = load_procan(procan_dir, replicate)
    else:
        raise ValueError(scenario)
    _CACHE[key] = src
    return src


def _cached_cptac(cptac_dir, scenario, cohort, replicate, cfg):
    from . import data
    tag = "" if scenario == "CPTAC" else "_" + scenario.lower().replace("-", "_")
    cache = cptac_dir / f"cache{tag}_mask{replicate}.npz"
    meta_path = cptac_dir / f"cache{tag}_mask{replicate}.json"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as f:
            src = {k: f[k] for k in f.files}
        src.update(json.loads(meta_path.read_text()))
        src["oracle_hidden_mean"] = None
        return src
    src = data.load_cptac(cptac_dir, replicate, cfg, cohort=cohort)
    arrays = {k: src[k] for k in ("sample_ids", "rna", "protein", "observed", "cognate",
                                  "gene_names", "protein_names", "is_normal")}
    np.savez_compressed(cache, **arrays)
    meta_path.write_text(json.dumps({k: src[k] for k in ("mechanism", "config", "seed")}))
    return src


def protocol_for(scenario, replicate, split_seed, s1_rna=None, cptac_dir=None, procan_dir=None):
    from .data import build_protocol
    src = source_for(scenario, replicate, s1_rna, cptac_dir, procan_dir)
    split = None
    if scenario.startswith("DS-"):
        # downstream scenarios impute every sample: 80/20 train/validation for model selection,
        # and the "test" rows are all samples (hidden values are never used for fitting).
        n = len(src["sample_ids"])
        perm = np.random.RandomState(split_seed).permutation(n)
        n_tr = int(0.8 * n)
        split = {"train": perm[:n_tr], "validation": perm[n_tr:], "test": np.arange(n)}
    return build_protocol(src, split_seed, f"{scenario}_rep{replicate}_split{split_seed}", log_rna=False,
                          split=split)


# ----------------------------------------------------------------------- one job

def fit_predict(P, scenario, method, train_seed, max_epochs=None, smoke=False, context=None):
    """Fit ``method`` on protocol ``P`` and return (test prediction, diagnostics, extras, interval)."""
    import torch
    from . import baselines as B
    from ..calibration import conformal_offset, selection_conformal_offset
    from ..model import VAEConfig, fit_vae, predict_split, interval_split, interval_arrays

    te = P.rows("test")
    rng = np.random.default_rng(train_seed)
    diag, extra, interval, extra_intervals = {}, {}, None, {}
    cog = method.endswith("_cog") or method.endswith("_cog16")

    if method in V2_VARIANTS:
        cfg = VAEConfig(limit_scale=limit_scale_for(scenario), **V2_VARIANTS[method])
        if max_epochs:
            cfg = replace(cfg, max_epochs=max_epochs)
        model, diag = fit_vae(P, cfg, train_seed)
        pred = predict_split(model, P, "test", "posterior_mean")
        if cfg.likelihood == "censored":
            interval = interval_split(model, P, "test", 0.9, seed=train_seed)
            # deployable recalibration on validation samples (no hidden values):
            # (a) selection-aware CQR on withheld observed entries (adopted);
            # (b) hidden-conditional CQR on the low-value pseudo-mask (reported for comparison).
            va = P.rows("validation")
            q_sel, cov_sel = selection_conformal_offset(model, P.rna[va], P.protein_input[va],
                                                        P.observed[va], 0.9, seed=train_seed)
            ps = P.pseudo_mask
            mask_v = P.observed[va] & ~ps
            lo_v, hi_v = interval_arrays(model, P.rna[va], np.where(mask_v, P.protein_input[va], 0.0),
                                         mask_v, 0.9, seed=train_seed)
            q_ps = conformal_offset(lo_v[ps], hi_v[ps], P.protein_input[va][ps], 0.9)
            diag.update({"conformal_offset90_selection": q_sel, "withheld_observed_coverage90": cov_sel,
                         "conformal_offset90_pseudo": q_ps})
            extra_intervals["conformal"] = (interval[0] - q_sel, interval[1] + q_sel)
            extra_intervals["pseudo"] = (interval[0] - q_ps, interval[1] + q_ps)
            for lev in (0.5, 0.8, 0.95):
                extra_intervals[f"level{int(lev * 100)}"] = interval_split(model, P, "test", lev, seed=train_seed)
        if method == "vae_full":
            extra["single_draw"] = predict_split(model, P, "test", "single_draw", seed=train_seed)
            extra["mc50"] = predict_split(model, P, "test", "mc", seed=train_seed)
    elif method == "v1_reference":
        pred, diag = run_v1_reference(P, scenario, train_seed, epochs=max_epochs or 200)
    elif method in ("miwae", "notmiwae", "miwae_cog", "notmiwae_cog", "notmiwae_cog16"):
        pcs = B.RNAPCs(P, seed=train_seed)
        kw = dict(missing_model=method.startswith("notmiwae"), epochs=max_epochs or 300, cognate=cog)
        if method in ("miwae_cog", "notmiwae_cog"):
            model, diag = B.fit_notmiwae_tuned(P, pcs, train_seed, **kw)
        else:
            model, diag = B.fit_notmiwae(P, pcs, train_seed, **kw)
        pred = B.notmiwae_predict(model, P, pcs, te, seed=train_seed, cognate=cog)
    elif method in ("observed_mean", "mindet", "minprob", "qrilc", "dsn"):
        fn = {"observed_mean": B.observed_mean, "mindet": B.min_det, "minprob": B.min_prob,
              "qrilc": B.qrilc, "dsn": B.down_shifted_normal}[method]
        pred = fn(P, te, rng=rng)
    elif method == "cognate_ols":
        pred = B.cognate_ols_predict(B.fit_cognate_ols(P), P, te)
    elif method == "tobit":
        fit = B.fit_tobit(P, seed=train_seed)
        pred = B.tobit_predict(fit, P, te)
    elif method == "selreg":
        fit = B.fit_selection_regression(P, limit_scale=limit_scale_for(scenario), seed=train_seed)
        diag = {"nll": fit["nll"], "slope": float(fit["s"]), "mcar": float(fit["pi"])}
        pred = B.selection_regression_predict(fit, P, te)
        # predictive intervals of the linear variant (added in the second revision; predictions unchanged)
        interval = B.selection_regression_interval(fit, P, te, 0.9, seed=train_seed)
        for lev in (0.5, 0.8, 0.95):
            extra_intervals[f"level{int(lev * 100)}"] = B.selection_regression_interval(fit, P, te, lev,
                                                                                          seed=train_seed)
        va = P.rows("validation")
        rng_w = np.random.default_rng(train_seed + 104729)
        withheld = (rng_w.uniform(size=P.observed[va].shape) < 0.2) & P.observed[va]
        lo_o, hi_o = B.selection_regression_interval(fit, P, va, 0.9, seed=train_seed, condition="observed")
        x_o = P.protein_input[va][withheld]
        q_sel = conformal_offset(lo_o[withheld], hi_o[withheld], x_o, 0.9)
        cov_sel = float(np.mean((x_o >= lo_o[withheld]) & (x_o <= hi_o[withheld])))
        lo_v, hi_v = B.selection_regression_interval(fit, P, va, 0.9, seed=train_seed)
        ps = P.pseudo_mask
        q_ps = conformal_offset(lo_v[ps], hi_v[ps], P.protein_input[va][ps], 0.9)
        diag.update({"conformal_offset90_selection": q_sel, "withheld_observed_coverage90": cov_sel,
                     "conformal_offset90_pseudo": q_ps})
        extra_intervals["conformal"] = (interval[0] - q_sel, interval[1] + q_sel)
        extra_intervals["pseudo"] = (interval[0] - q_ps, interval[1] + q_ps)
    elif method in ("pirat", "pirat_t", "pirat_s", "proda_design"):
        from .rbridge import r_impute
        ctx = context or {}
        raw = P.raw_protein_input()
        extra_files = {}
        if method == "pirat_t":
            cog = np.asarray(P.cognate)
            if np.any(cog < 0):
                raise ValueError("Pirat-T needs a cognate transcript for every protein.")
            extra_files["rna"] = np.asarray(ctx["rna_raw"], float)[:, cog]
        if method == "proda_design":
            if ctx.get("is_normal") is None:
                return None
            extra_files["groups"] = np.asarray(ctx["is_normal"], float)
        name = {"pirat": "Pirat", "pirat_t": "PiratT", "pirat_s": "PiratS", "proda_design": "proDA_design"}[method]
        out, info = r_impute(name, raw, seed=train_seed, extra=extra_files, cache=ctx.get("cache"))
        diag = {"r": info}
        pred = B._fill(P, te, B._to_std(out[te], P))
    elif method in ("knn", "knn_cog"):
        pcs = B.RNAPCs(P, seed=train_seed)
        va = P.rows("validation")
        pseudo_obs = P.observed[va] & ~P.pseudo_mask
        scores = {}
        for k in B.K_GRID:
            fitted = B.fit_knn(P, pcs, k, cognate=cog)
            vp = B.knn_predict(fitted, P, pcs, va, observed=pseudo_obs, cognate=cog)
            scores[k] = float(np.sqrt(np.mean((vp - P.protein_input[va])[P.pseudo_mask] ** 2)))
        best_k = min(scores, key=scores.get)
        diag = {"pseudo_mask_rmse": scores, "selected_k": best_k}
        pred = B.knn_predict(B.fit_knn(P, pcs, best_k, cognate=cog), P, pcs, te, cognate=cog)
    elif method == "missforest":
        pcs = B.RNAPCs(P, seed=train_seed)
        fitted = B.fit_missforest(P, pcs, train_seed, n_estimators=10 if smoke else 50,
                                  max_iter=2 if smoke else 5)
        pred = B.missforest_predict(fitted, P, pcs, te)
    elif method in R_METHODS:
        from .rbridge import r_impute
        raw = P.raw_protein_input()
        params = {}
        cov = None
        if method == "missforest_r":
            pcs = B.RNAPCs(P, seed=train_seed)
            cov = B.covariates(P, pcs, np.arange(len(P.sample_ids)), cognate=True)
            big = len(P.sample_ids) > 400          # ProCan: ~930 cell lines; lighter forest settings
            params = {"ntree": 20 if smoke else (25 if big else 50), "maxiter": 2 if smoke else (3 if big else 5)}
        out, info = r_impute(R_METHODS[method], raw, covariates=cov, seed=train_seed, **params)
        diag = {"r": info}
        pred = B._fill(P, te, B._to_std(out[te], P))
    elif method == "available_case":
        pred = np.where(P.observed[te], P.protein_input[te], np.nan).astype(np.float32)
    elif method == "oracle":
        if not P.oracle:
            return None
        pred = np.where(P.observed[te], P.protein_input[te], P.oracle["test"])
    else:
        raise ValueError(method)
    return pred, diag, extra, interval, extra_intervals


def run_job(scenario, replicate, split_seed, method, train_seed, out_root, smoke=False,
            s1_rna=None, cptac_dir=None, procan_dir=None):
    import torch
    torch.set_num_threads(1)
    from .. import metrics as M

    out = Path(out_root) / scenario / f"rep{replicate}_split{split_seed}" / f"{method}_seed{train_seed}"
    if (out / "result.json").exists():
        return str(out), "skipped"
    t0 = time.time()
    P = protocol_for(scenario, replicate, split_seed, s1_rna, cptac_dir, procan_dir)
    context = None
    if method in TRANSDUCTIVE_CACHED:
        src = source_for(scenario, replicate, s1_rna, cptac_dir, procan_dir)
        rna_raw = np.asarray(src["rna"], dtype=np.float64)
        context = {"rna_raw": rna_raw,
                   "is_normal": (np.asarray(src["is_normal"], float) if src.get("is_normal") is not None
                                 and scenario.startswith("DS-") else None),
                   "cache": Path(out_root) / "_rcache" / f"{scenario}_rep{replicate}_{method}.npz"}
    res = fit_predict(P, scenario, method, train_seed, max_epochs=15 if smoke else None, smoke=smoke,
                      context=context)
    if res is None:
        return str(out), "not applicable"
    pred, diag, extra, interval, extra_intervals = res
    out.mkdir(parents=True, exist_ok=True)
    te = P.rows("test")

    # ---- single read of test truth, after all choices are fixed
    truth = P.truth["test"]
    hidden = ~P.observed[te]
    lo, hi = interval if interval is not None else (None, None)
    test = {} if method == "available_case" else M.hidden_metrics(truth, pred, hidden, lo, hi)
    if scenario.startswith("DS-"):
        from .downstream import downstream_metrics
        src = source_for(scenario, replicate, s1_rna, cptac_dir, procan_dir)
        test["downstream"] = downstream_metrics(truth, pred, P.observed[te], np.asarray(src["is_normal"])[te])
    for name, (l2, h2) in extra_intervals.items():
        s = M.scored_mask(truth, hidden)
        test[f"coverage_{name}"] = float(np.mean((truth[s] >= l2[s]) & (truth[s] <= h2[s])))
        test[f"interval_width_{name}"] = float(np.mean(h2[s] - l2[s]))
    result = {
        "scenario": scenario, "replicate": replicate, "split_seed": split_seed,
        "method": method, "train_seed": train_seed,
        "test": test,
        "test_rmse_ci95_bootstrap_samples": (None if method == "available_case"
                                             else M.bootstrap_rmse(truth, pred, hidden, seed=train_seed)),
        "inference_variants": {k: M.hidden_metrics(truth, v, hidden) for k, v in extra.items()},
        "diagnostics": diag,
        "protocol": {"fingerprint": P.fingerprint(), **P.meta},
        "runtime_seconds": round(time.time() - t0, 2),
        "software": software(),
        "smoke": bool(smoke),
    }
    if P.meta["mechanism"]["kind"] == "deterministic_raw_cutoff":
        tau = (P.meta["mechanism"]["cutoff"] - P.protein_mean) / P.protein_scale
        result["test"]["boundary_violation_rate"] = M.boundary_violation(pred, hidden, tau)
    arrays = {"sample_ids": P.sample_ids[te], "truth": truth, "hidden": hidden, "prediction": pred}
    arrays.update({"prediction_" + k: v for k, v in extra.items()})
    if interval is not None:
        arrays.update({"interval90_lower": lo, "interval90_upper": hi})
    for name, (l2, h2) in extra_intervals.items():
        arrays.update({f"interval90_{name}_lower": l2, f"interval90_{name}_upper": h2})
    np.savez_compressed(out / "predictions.npz", **arrays)
    (out / "result.json").write_text(json.dumps(result, indent=2, default=_json_default))
    return str(out), "done"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def software():
    import scipy
    import sklearn
    import torch
    from .. import __version__
    return {"tcsimpute": __version__, "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
            "torch": str(torch.__version__), "platform": platform.platform()}


def run_v1_reference(P, scenario, seed, epochs=200):
    """The prototype architecture and loss as first written, with a deployable selection rule."""
    import copy
    import torch
    from .legacy_v1 import MultiOmicsTransformerVAE, state_of_the_art_loss
    torch.manual_seed(seed)
    tr, va, te = P.rows("train"), P.rows("validation"), P.rows("test")
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
    rna, prot, mask = t(P.rna), t(P.protein_input), t(P.observed)
    if P.meta["mechanism"]["kind"] == "deterministic_raw_cutoff":
        tau = (P.meta["mechanism"]["cutoff"] - P.protein_mean) / P.protein_scale
    else:
        tau = np.where(P.observed[tr], P.protein_input[tr], np.inf).min(0)
    thresholds = [0.0, t(tau)]
    model = MultiOmicsTransformerVAE([rna.shape[1], prot.shape[1]], embed_dim=32, num_heads=2,
                                     num_layers=1, latent_dim=16)
    adjs = [torch.eye(rna.shape[1]), torch.eye(prot.shape[1])]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed)
    ones = torch.ones_like(rna)

    def loss_on(idx, epoch):
        xs, ms = [rna[idx], prot[idx]], [ones[idx], mask[idx]]
        rec, mu, lv, ctx = model(xs, adjs, ms)
        return state_of_the_art_loss(rec, xs, ms, mu, lv, ctx, beta=min(0.1, epoch / 100.0),
                                     gamma=5.0, censor_thresholds=thresholds)[0]

    best, best_state, best_epoch, history = float("inf"), None, None, []
    for epoch in range(epochs):
        model.train()
        order = tr[torch.randperm(len(tr), generator=gen).numpy()]
        for start in range(0, len(order), 32):
            idx = order[start:start + 32]
            if len(idx) < 2:
                continue
            loss = loss_on(idx, epoch)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            model.eval()
            with torch.no_grad():
                v = float(loss_on(va, epoch))
            history.append({"epoch": epoch + 1, "validation_objective": v})
            if v < best:
                best, best_state, best_epoch = v, copy.deepcopy(model.state_dict()), epoch + 1
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        rec, mu, lv, _ = model([rna[te], prot[te]], adjs, [ones[te], mask[te]])  # one latent draw
        kl_dim = (0.5 * (mu ** 2 + lv.exp() - 1 - lv)).mean(0)
    pred = np.where(P.observed[te], P.protein_input[te], rec[1].numpy())
    return pred.astype(np.float32), {"selected_epoch": best_epoch, "validation_objective": best,
                                     "active_units_test": int((kl_dim > 0.01).sum()),
                                     "history": history}


# ----------------------------------------------------------------------- driver

def methods_for(grid, scenario):
    if grid == "original":
        return ORIGINAL_METHODS
    if scenario.startswith("SWEEP-"):
        return SWEEP_METHODS
    if scenario == "PROCAN":
        return PROCAN_METHODS
    if scenario.startswith("DS-"):
        return DS_METHODS
    if scenario.startswith("ALT-"):
        return ALT_METHODS
    return MAIN_METHODS


def jobs_for(grid, scenarios, methods=None, smoke=False, limit=0, only_splits=None):
    table = ORIGINAL_GRID if grid == "original" else REVISION_GRID
    jobs = []
    for sc in scenarios:
        g = table[sc]
        reps, splits, seeds = g["replicates"], g["splits"], g["train_seeds"]
        if only_splits:
            splits = [s for s in splits if s in only_splits] or splits[:1]
        if smoke or limit:
            reps, splits, seeds = reps[:1], splits[:max(1, limit)], seeds[:1]
        for r in reps:
            for s in splits:
                for m in (methods or methods_for(grid, sc)):
                    if m == "oracle" and not (sc in ("S2", "SIM") or sc in SWEEPS):
                        continue
                    if m == "available_case" and not sc.startswith("DS-"):
                        continue
                    for t in (seeds if m not in SEEDLESS else seeds[:1]):
                        jobs.append((sc, r, s, m, t))
    slow = ["pirat_s", "pirat_t", "pirat", "missforest_r", "missforest", "miwae_cog", "notmiwae_cog", "v1_reference", "notmiwae",
            "miwae", "notmiwae_cog16"]
    order = {m: i for i, m in enumerate(slow)}
    jobs.sort(key=lambda j: (order.get(j[3], 99), j[0]))
    return jobs


def expand_scenarios(names):
    out = []
    for n in names:
        out.extend(SWEEPS if n == "sweeps" else [n])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tcsimpute-benchmark", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", choices=["revision", "original"], default="revision")
    ap.add_argument("--scenarios", nargs="+", default=["SIM", "PDAC", "LUAD"],
                    help="Scenario names; 'sweeps' expands to all SWEEP-* scenarios.")
    ap.add_argument("--methods", nargs="+", default=None, help="Default: the grid's method list.")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", required=True, help="Output directory for run folders.")
    ap.add_argument("--s1-rna", help="RNA CSV of the original S1 fixture (needed for S1).")
    ap.add_argument("--cptac-dir", help="Directory with <cohort>_transcriptomics.csv / _proteomics.csv.")
    ap.add_argument("--procan-dir", help="Directory with the ProCan replicate matrix, mapping file and "
                                         "procan_transcriptomics.csv (see tcsimpute.benchmark.natural).")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Run only the first N splits of one replicate (debug).")
    ap.add_argument("--splits", type=int, nargs="+", default=None,
                    help="Run only these split seeds (e.g. one split per replicate first, so cached "
                         "transductive fits are reused by the remaining splits).")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.smoke:
        out = out / "_smoke"
    out.mkdir(parents=True, exist_ok=True)
    scenarios = expand_scenarios(a.scenarios)
    table = ORIGINAL_GRID if a.grid == "original" else REVISION_GRID
    # Build data caches once, before forking workers.
    for sc in scenarios:
        if sc in ("CPTAC", "PDAC", "LUAD", "DS-PDAC", "DS-LUAD") or sc.startswith("ALT-"):
            for r in table[sc]["replicates"]:
                source_for(sc, r, cptac_dir=a.cptac_dir)
        elif sc == "PROCAN":
            for r in table[sc]["replicates"]:
                source_for(sc, r, procan_dir=a.procan_dir)
    jobs = jobs_for(a.grid, scenarios, a.methods, a.smoke, a.limit, a.splits)
    print(f"{len(jobs)} jobs, {a.workers} workers -> {out}", flush=True)
    failures = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_job, *j, str(out), a.smoke, a.s1_rna, a.cptac_dir, a.procan_dir): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            j = futs[f]
            try:
                path, status = f.result()
                print(f"[{i}/{len(jobs)} {time.time() - t0:7.0f}s] {status}: {j}", flush=True)
            except Exception:
                failures.append({"job": j, "error": traceback.format_exc()})
                print(f"[{i}/{len(jobs)}] FAILED {j}\n{traceback.format_exc()}", flush=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out / f"failures_{stamp}.json").write_text(json.dumps(failures, indent=2, default=str))
    print(f"finished in {time.time() - t0:.0f}s; failures: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
