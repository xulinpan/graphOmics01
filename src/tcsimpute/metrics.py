"""Hidden-entry metrics on the frozen training-standardized protein scale."""

import numpy as np


def scored_mask(truth, hidden):
    """Hidden entries with a known value (natural-missingness benchmarks leave some unknown)."""
    return hidden & np.isfinite(truth)


def within_feature_r(truth, pred, hidden, min_entries=5):
    """Mean Pearson r across proteins, computed over each protein's hidden entries.

    Undefined (constant) features are counted separately and excluded from the mean, so a
    predictor that is constant within a protein has no within-feature correlation to report.
    """
    rs, undefined, eligible = [], 0, 0
    hidden = scored_mask(truth, hidden)
    for j in range(truth.shape[1]):
        k = hidden[:, j]
        if k.sum() < min_entries or truth[k, j].std() == 0:
            continue
        eligible += 1
        if pred[k, j].std() < 1e-10:
            undefined += 1
            continue
        rs.append(np.corrcoef(truth[k, j], pred[k, j])[0, 1])
    return (float(np.mean(rs)) if rs else None), eligible, undefined


def hidden_metrics(truth, pred, hidden, lower=None, upper=None):
    hidden = scored_mask(truth, hidden)
    t, p = truth[hidden], pred[hidden]
    if not (np.isfinite(t).all() and np.isfinite(p).all()):
        raise RuntimeError("Non-finite values in hidden-entry scoring.")
    wr, eligible, undefined = within_feature_r(truth, pred, hidden)
    out = {
        "hidden_entries": int(hidden.sum()),
        "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
        "mae": float(np.mean(np.abs(p - t))),
        "bias": float(np.mean(p - t)),
        "within_feature_r": wr,
        "within_feature_eligible": eligible,
        "within_feature_constant": undefined,
    }
    if lower is not None:
        lo, hi = lower[hidden], upper[hidden]
        out["coverage90"] = float(np.mean((t >= lo) & (t <= hi)))
        out["interval_width90"] = float(np.mean(hi - lo))
    return out


def bootstrap_rmse(truth, pred, hidden, n_boot=1000, seed=0):
    """Percentile CI resampling test samples (rows), keeping each sample's entries together."""
    rng = np.random.default_rng(seed)
    hidden = scored_mask(truth, hidden)
    se = np.where(hidden, (pred - np.nan_to_num(truth)) ** 2, 0.0).sum(1)
    cnt = hidden.sum(1)
    n = len(se)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(np.sqrt(se[idx].sum() / max(cnt[idx].sum(), 1)))
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def boundary_violation(pred, hidden, thresholds):
    """S1 only: fraction of hidden predictions above the deterministic standardized cutoff."""
    above = (pred > thresholds[None, :]) & hidden
    return float(above.sum() / max(hidden.sum(), 1))
