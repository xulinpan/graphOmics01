"""Downstream consequences of imputation, scored against the complete (pre-mask) data.

Used by the DS-PDAC and DS-LUAD scenarios: CPTAC tumours and normal adjacent tissues, an
intensity-scale detection mask, one fit per method on all samples (80/20 train/validation split
for model selection; every sample is imputed, which uses observed values only).

* Differential abundance, tumour vs normal: per-protein Welch t-tests on the complete data define
  the reference calls (Benjamini-Hochberg FDR 5%). The same tests on the imputed data are compared
  with them: Spearman correlation of t statistics, precision and recall of the calls, and the
  error in the standardized group difference.
* Co-abundance: Pearson correlations between proteins across all samples, among proteins with at
  least 10% missing values (where imputation matters). Reported: mean absolute error of the
  correlation coefficients and the Spearman correlation of the upper triangles.

``available_case`` is the no-imputation baseline: tests and correlations use observed values
only (pairwise-complete correlations).
"""

import numpy as np
from scipy.stats import spearmanr, ttest_ind


def _bh(p, alpha=0.05):
    p = np.asarray(p)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out < alpha


def _welch(X, is_normal):
    t, p = ttest_ind(X[~is_normal], X[is_normal], axis=0, equal_var=False, nan_policy="omit")
    t, p = np.asarray(t, float), np.asarray(p, float)
    return np.nan_to_num(t), np.where(np.isfinite(p), p, 1.0)


def _effect(X, is_normal):
    return np.nanmean(X[~is_normal], 0) - np.nanmean(X[is_normal], 0)


def _corr(X):
    import pandas as pd
    return pd.DataFrame(X).corr(min_periods=10).to_numpy()


def downstream_metrics(truth, pred, observed, is_normal, min_missing=0.10):
    """``truth``: complete standardized matrix; ``pred``: imputed (or NaN at hidden entries for the
    available-case baseline); ``observed``: mask; ``is_normal``: bool per sample."""
    is_normal = np.asarray(is_normal, bool)
    out = {"n_tumour": int((~is_normal).sum()), "n_normal": int(is_normal.sum())}
    t_true, p_true = _welch(truth, is_normal)
    t_imp, p_imp = _welch(pred, is_normal)
    sig_true, sig_imp = _bh(p_true), _bh(p_imp)
    tp = int((sig_true & sig_imp).sum())
    out.update({
        "de_t_spearman": float(spearmanr(t_true, t_imp).statistic),
        "de_calls_reference": int(sig_true.sum()), "de_calls": int(sig_imp.sum()),
        "de_precision": float(tp / max(sig_imp.sum(), 1)),
        "de_recall": float(tp / max(sig_true.sum(), 1)),
        "de_effect_rmse": float(np.sqrt(np.nanmean((_effect(truth, is_normal) - _effect(pred, is_normal)) ** 2))),
    })
    miss = 1.0 - observed.mean(0)
    sel = np.flatnonzero(miss >= min_missing)
    out["coabundance_proteins"] = int(sel.size)
    if sel.size >= 10:
        c_true, c_imp = _corr(truth[:, sel]), _corr(pred[:, sel])
        iu = np.triu_indices(sel.size, 1)
        a, b = c_true[iu], c_imp[iu]
        ok = np.isfinite(a) & np.isfinite(b)
        out.update({"coabundance_mae": float(np.mean(np.abs(a[ok] - b[ok]))),
                    "coabundance_spearman": float(spearmanr(a[ok], b[ok]).statistic),
                    "coabundance_pairs": int(ok.sum())})
    return out
