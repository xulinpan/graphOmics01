"""Parameter recovery and sensitivity of the probit detection model (Supplementary Section S5).

Data are simulated with the paper's simulator, whose detection process is
    P(observed_ij | x_ij) = (1 - pi_j) * Phi((x_ij - d0 - delta_i - e_j) / s)
with a global limit d0, sample offsets delta_i, per-protein deviations e_j, slope s and an MCAR
rate pi_j (one global value, except in the 'sporadic' condition). Both tcsimpute variants are fitted
and their detection parameters are compared with the generating values on the raw (log) scale.

Usage:  python parameter_recovery.py --out recovery --workers 14 [--replicates 10] [--smoke]
Writes one JSON per fit to <out>/fits/ and a summary to <out>/parameter_recovery.json.
"""
import argparse
import json
import os
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

import numpy as np

CONDITIONS = {
    # name: (S2Config overrides, extra sporadic rate spec or None)
    "base": ({}, None),
    "no_offsets": ({"sample_offset_sd": 0.0}, None),
    "limits_sd0.25": ({"mechanism": "wide_limits", "limit_spread_sd": 0.25}, None),
    "limits_sd1": ({"mechanism": "wide_limits", "limit_spread_sd": 1.0}, None),
    "slope0.25": ({"detection_slope": 0.25}, None),
    "slope1": ({"detection_slope": 1.0}, None),
    "weak_coupling": ({"coupling": 0.5, "ptm_scale": 1.5}, None),
    "mcar0.15": ({"mcar": 0.15}, None),
    "sporadic": ({"mcar": 0.0}, (1.0, 9.0)),   # protein-specific sporadic rate ~ Beta(1, 9), mean 0.10
}
SENSITIVITY_PRIORS = (0.1, 0.25, 1.0)          # prior SD of per-protein limit deviations (default 0.25)
SENSITIVITY_CONDITIONS = ("base", "limits_sd1", "weak_coupling", "sporadic")


def simulate(condition, replicate):
    """Simulate with the paper's generator, recording the true offsets and limit deviations."""
    from tcsimpute.benchmark import data as D
    overrides, sporadic = CONDITIONS[condition]
    cfg = replace(D.S2Config(), **overrides)
    captured = {}
    original = D._apply_mechanism

    def capture(y, offsets, c, rng):
        if c.mechanism == "wide_limits":
            state = rng.bit_generator.state
            dev = rng.normal(0, c.limit_spread_sd, y.shape[1])
            rng.bit_generator.state = state          # replay: the original draws the same deviations
        else:
            dev = np.zeros(y.shape[1])
        captured.update(offsets=np.array(offsets), dev=dev)
        return original(y, offsets, c, rng)

    D._apply_mechanism = capture
    try:
        src = D.simulate_s2(50_000 + replicate, cfg)
    finally:
        D._apply_mechanism = original
    keep = np.array([int(name[1:]) for name in src["protein_names"]])
    truth = {"d0": src["mechanism"]["d0"], "slope": cfg.detection_slope, "offsets": captured["offsets"],
             "dev": captured["dev"][keep]}
    if sporadic is not None:
        rng = np.random.default_rng(90_000 + replicate)
        pi_j = rng.beta(*sporadic, size=keep.size)
        drop = rng.uniform(size=src["observed"].shape) < pi_j[None, :]
        obs = src["observed"] & ~drop
        ok = obs.mean(0) >= cfg.min_observed_fraction
        src = dict(src, observed=obs[:, ok], protein=src["protein"][:, ok], cognate=src["cognate"][ok],
                   protein_names=src["protein_names"][ok], oracle_hidden_mean=None)
        truth["dev"], truth["pi_j"] = truth["dev"][ok], pi_j[ok]
        truth["pi"] = float(pi_j[ok].mean())
    else:
        truth["pi"] = cfg.mcar
    return src, truth


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def run_one(job):
    condition, replicate, method, prior = job
    import torch
    torch.set_num_threads(1)
    from tcsimpute.benchmark import data as D
    from tcsimpute.benchmark import baselines as B
    src, truth = simulate(condition, replicate)
    P = D.build_protocol(src, split_seed=replicate, name=f"recovery-{condition}-{replicate}")
    tr, te = P.rows("train"), P.rows("test")
    mu_hat, sd_hat = P.protein_mean, P.protein_scale
    out = {"condition": condition, "replicate": replicate, "method": method, "prior_sd": prior,
           "true_d0": truth["d0"], "true_slope": truth["slope"], "true_pi": truth["pi"]}

    if method == "linear":
        fit = B.fit_selection_regression(P, limit_scale="raw", seed=replicate, limit_deviation_sd=prior)
        tau = fit["tau"].numpy()
        s_std, pi_hat = float(fit["s"]), float(fit["pi"])
        alpha = fit["alpha"].numpy()
        pred = B.selection_regression_predict(fit, P, te)
        offsets_hat = None
    else:
        import torch as T
        from tcsimpute.model import VAEConfig, fit_vae, predict_split
        model, _ = fit_vae(P, VAEConfig(limit_scale="raw", limit_deviation_sd=prior), replicate)
        model.eval()
        with T.no_grad():
            tau = model.limit.cpu().numpy()
            s_std = float(T.exp(model.log_slope))
            pi_hat = float(T.sigmoid(model.mcar_logit))
            rows = np.concatenate([tr, te])
            dev = next(model.parameters()).device
            _, _, off, _ = model.encode(T.tensor(P.rna[rows], dtype=T.float32, device=dev),
                                        T.tensor(P.protein_input[rows], dtype=T.float32, device=dev),
                                        T.tensor(P.observed[rows], dtype=T.float32, device=dev))
            offsets_hat = (off.cpu().numpy() * float(np.median(sd_hat)))
        pred = predict_split(model, P, "test")
        alpha = None
        out["offset_recovery_r"] = _corr(offsets_hat, truth["offsets"][rows])
        out["offset_sd_hat"] = float(np.std(offsets_hat))
        out["offset_sd_true"] = float(np.std(truth["offsets"][rows]))
        out["offset_mean_hat"] = float(np.mean(offsets_hat))

    # raw-scale detection parameters
    limit_raw = tau * sd_hat + mu_hat                       # per-protein limit, raw scale
    true_limit = truth["d0"] + truth["dev"]
    out["limit_mean_hat"] = float(limit_raw.mean())
    out["limit_mean_true"] = float(true_limit.mean())
    out["limit_bias"] = float((limit_raw - true_limit).mean())
    out["limit_rmse"] = float(np.sqrt(((limit_raw - true_limit) ** 2).mean()))
    out["limit_r"] = _corr(limit_raw, true_limit)
    # Only the sum of the limit and the mean sample offset enters the likelihood, so the VAE's limit is
    # also reported after adding its mean estimated offset ("effective" limit).
    rows_all = np.concatenate([tr, te])
    shift_hat = out.get("offset_mean_hat", 0.0)
    shift_true = float(truth["offsets"][rows_all].mean())
    out["limit_bias_effective"] = float((limit_raw + shift_hat - true_limit - shift_true).mean())
    out["limit_sd_hat"] = float(limit_raw.std())
    out["limit_sd_true"] = float(true_limit.std())
    out["slope_raw_hat"] = float(s_std * np.median(sd_hat))  # slope is shared on the standardized scale
    out["slope_raw_hat_iqr"] = [float(v) for v in np.percentile(s_std * sd_hat, [25, 75])]
    out["pi_hat"] = pi_hat
    if "pi_j" in truth:
        out["pi_j_mean"] = float(truth["pi_j"].mean())

    # confounding between the limit and the protein mean level (linear variant, which has a mean intercept)
    full = src["protein"]
    true_level = full[tr].mean(0)
    if alpha is not None:
        level_hat = alpha * sd_hat + mu_hat
        out["level_bias"] = float((level_hat - true_level).mean())
        out["level_limit_error_r"] = _corr(level_hat - true_level, limit_raw - true_limit)

    # hidden test-entry accuracy (standardized units, as in the benchmark)
    hid = ~P.observed[te]
    err = (np.asarray(pred) - P.truth["test"])[hid]
    out["rmse"] = float(np.sqrt((err ** 2).mean()))
    out["bias"] = float(err.mean())
    if P.oracle:
        oerr = (P.oracle["test"] - P.truth["test"])[hid]
        out["oracle_rmse"] = float(np.sqrt((oerr ** 2).mean()))
    return out


def summarize(fits):
    keys = ["limit_bias", "limit_bias_effective", "offset_mean_hat", "limit_rmse", "limit_r", "limit_sd_hat",
            "limit_sd_true", "slope_raw_hat", "pi_hat", "true_pi", "true_slope", "offset_recovery_r", "offset_sd_hat", "offset_sd_true",
            "level_bias", "level_limit_error_r", "rmse", "bias", "oracle_rmse"]
    groups = {}
    for f in fits:
        groups.setdefault((f["condition"], f["method"], f["prior_sd"]), []).append(f)
    out = []
    for (c, m, p), rows in sorted(groups.items()):
        e = {"condition": c, "method": m, "prior_sd": p, "n": len(rows)}
        for k in keys:
            v = [r[k] for r in rows if r.get(k) is not None]
            if v:
                e[k] = float(np.mean(v))
                e[k + "_sd"] = float(np.std(v))
        out.append(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--replicates", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "fits").mkdir(parents=True, exist_ok=True)
    reps = range(1, (2 if a.smoke else a.replicates) + 1)
    conds = ["base", "sporadic"] if a.smoke else list(CONDITIONS)
    jobs = [(c, r, m, 0.25) for c in conds for r in reps for m in ("linear", "vae")]
    jobs += [(c, r, "linear", p) for c in SENSITIVITY_CONDITIONS if c in conds for r in reps
             for p in SENSITIVITY_PRIORS if p != 0.25]
    jobs += [("limits_sd1", r, "vae", 1.0) for r in reps if "limits_sd1" in conds]
    todo = [j for j in jobs if not (out / "fits" / ("%s_r%d_%s_p%s.json" % j)).exists()]
    print(f"{len(jobs)} jobs, {len(todo)} to run", flush=True)
    with get_context("spawn").Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_one, todo), 1):
            name = "%s_r%d_%s_p%s.json" % (res["condition"], res["replicate"], res["method"], res["prior_sd"])
            (out / "fits" / name).write_text(json.dumps(res))
            print(f"[{i}/{len(todo)}] {name} rmse={res['rmse']:.3f}", flush=True)
    fits = [json.loads(p.read_text()) for p in sorted((out / "fits").glob("*.json"))]
    (out / "parameter_recovery.json").write_text(json.dumps({"fits": fits, "summary": summarize(fits)}, indent=1))
    print("done")


if __name__ == "__main__":
    main()
