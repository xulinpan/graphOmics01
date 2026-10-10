"""Deployable interval recalibration by conformalized quantile regression on a pseudo-mask.

No hidden value is needed. Low observed validation entries are hidden at random, preferentially
the lowest ones (probability Phi((t - z) / 0.5), with t set so that about 10% of observed
entries are hidden), the model's intervals for them are computed as if they were missing, and the
CQR offset of Romano et al. (2019) is the conformal quantile of the conformity scores
E = max(lo - x, x - hi). Test intervals are widened (or narrowed) by that offset.

The offset is valid for the pseudo-mask mechanism; it transfers to truly missing entries to the
extent that the pseudo-mask resembles the real detection process. The benchmark reports
coverage before and after recalibration on truly hidden entries.
"""

import numpy as np
from scipy.stats import norm


def pseudo_mask(z, observed, fraction=0.10, slope=0.5, rng=None):
    """Boolean mask over ``observed`` entries of ``z`` (standardized values), low values preferred."""
    rng = np.random.default_rng(0) if rng is None else rng
    lo, hi = -10.0, 10.0
    zz = np.where(observed, z, 0.0)
    for _ in range(100):
        t = 0.5 * (lo + hi)
        frac = (norm.cdf((t - zz) / slope) * observed).sum() / max(observed.sum(), 1)
        lo, hi = (t, hi) if frac < fraction else (lo, t)
    return (rng.uniform(size=zz.shape) < norm.cdf((t - zz) / slope)) & observed


def conformal_offset(lower, upper, values, level=0.9):
    """CQR offset q such that [lower - q, upper + q] has finite-sample coverage >= level."""
    scores = np.maximum(lower - values, values - upper)
    n = scores.size
    if n == 0:
        return 0.0
    k = min(n, int(np.ceil((n + 1) * level)))
    return float(np.sort(scores)[k - 1])


def selection_conformal_offset(model, rna, protein, mask, level=0.9, seed=0, fraction=0.2):
    """CQR offset from withheld *observed* entries, scored under the model's predictive
    distribution for a detected value, p(x | z, observed).

    A random ``fraction`` of the observed entries in (validation) samples is withheld from the
    encoder; their intervals are computed conditional on detection and the conformity scores give
    the offset, which is then applied to intervals for undetected entries. Validity on undetected
    entries assumes the model's miscalibration is similar for detected and undetected values.
    Returns (offset, coverage of the uncorrected observed-conditional intervals).
    """
    from .model import interval_arrays
    rng = np.random.default_rng(seed + 104729)
    withheld = (rng.uniform(size=mask.shape) < fraction) & mask
    keep = mask & ~withheld
    lo, hi = interval_arrays(model, rna, np.where(keep, protein, 0.0), keep, level, seed=seed,
                             condition="observed")
    x = protein[withheld]
    q = conformal_offset(lo[withheld], hi[withheld], x, level)
    cov = float(np.mean((x >= lo[withheld]) & (x <= hi[withheld]))) if x.size else float("nan")
    return q, cov
