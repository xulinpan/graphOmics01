"""Data sources and frozen split protocol used in the paper.

Three scenarios share one protocol object:

* S1  - the v1 synthetic fixture (deterministic raw cutoff, P = R * eta), kept for continuity.
* S2  - a new simulator: latent programs, sparse many-to-many nonlinear RNA->protein map,
        protein-specific (non-transcriptional) factors, and probabilistic abundance-dependent
        detection with sample-specific limits plus a small MCAR component.
* CPTAC - semi-synthetic: real CPTAC PDAC tumour RNA and fully observed proteins, with the S2
        detection mechanism imposed so that the hidden truth is known.

The protocol splits samples first, then fits RNA scalers on training RNA and protein scalers on
observed training protein entries only. Hidden training truth is never stored. Validation and test
truth are stored separately for scoring.
"""

from dataclasses import dataclass, field, asdict
import hashlib
import json

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# detection mechanism (shared by S2 and the semi-synthetic CPTAC benchmark)
# ---------------------------------------------------------------------------

def detection_probability(y, d, s, mcar):
    """P(observed | y) = (1 - mcar) * Phi((y - d) / s); d broadcasts over samples."""
    return (1.0 - mcar) * norm.cdf((y - d) / s)


def calibrate_detection_limit(y, offsets, s, mcar, target_missing):
    """Bisection on the global limit d0 so that E[missing fraction] equals the target."""
    lo, hi = float(y.min()) - 10 * s, float(y.max()) + 10 * s
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        miss = 1.0 - detection_probability(y, mid + offsets[:, None], s, mcar).mean()
        if miss < target_missing:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def oracle_hidden_mean(m, v, d, s, mcar):
    """E[y | y hidden] for y ~ N(m, v) and P(hidden | y) = mcar + (1 - mcar) Phi((d - y)/s)."""
    root = np.sqrt(v + s ** 2)
    a = (d - m) / root
    log_phi_big = norm.logcdf(a)
    log_w_mcar = np.log(mcar) - np.logaddexp(np.log(mcar), np.log1p(-mcar) + log_phi_big)
    w_mcar = np.exp(log_w_mcar)
    mills = np.exp(norm.logpdf(a) - log_phi_big)
    censored_mean = m - (v / root) * mills
    return w_mcar * m + (1.0 - w_mcar) * censored_mean


# ---------------------------------------------------------------------------
# S2 simulator
# ---------------------------------------------------------------------------

@dataclass
class S2Config:
    n_samples: int = 200
    n_genes: int = 2000
    n_proteins_generated: int = 700
    n_proteins: int = 500
    n_programs: int = 8
    n_regulators: int = 100
    n_ptm_factors: int = 3
    target_missing: float = 0.36  # before feature filtering; realized ~0.30 after it
    detection_slope: float = 0.5
    mcar: float = 0.02
    sample_offset_sd: float = 0.3
    min_observed_fraction: float = 0.3
    # --- sensitivity / misspecification settings (defaults reproduce the original S2) ---
    mechanism: str = "probit"      # 'probit', 'logistic', 'threshold', 'wide_limits', 'batch'
    limit_spread_sd: float = 1.0   # per-protein limit SD for mechanism='wide_limits'
    batch_size: int = 10           # samples per acquisition batch for mechanism='batch'
    noise: str = "gaussian"        # 'gaussian' or 't3' (Student-t, 3 df, same variance)
    coupling: float = 1.0          # multiplies cis amplitudes and trans coefficients
    ptm_scale: float = 1.0         # multiplies RNA-independent protein-factor loadings


def simulate_s2(seed, config=None):
    """Return a dict with RNA, complete protein, mask, cognate map and oracle quantities.

    Everything is on a log2-like scale. The generating functions are recorded so that the
    Bayes-optimal hidden-entry predictor given RNA can be evaluated exactly.
    """
    c = config or S2Config()
    rng = np.random.default_rng(seed)
    n, g, k = c.n_samples, c.n_genes, c.n_programs
    z = rng.normal(size=(n, k))
    loadings = np.zeros((g, k))
    n_prog = rng.choice([0, 1, 2], size=g, p=[0.3, 0.5, 0.2])
    for j in range(g):
        if n_prog[j]:
            loadings[j, rng.choice(k, n_prog[j], replace=False)] = rng.normal(0, 0.7, n_prog[j])
    gene_base = rng.normal(6.0, 2.0, g)
    gene_sd = rng.uniform(0.3, 0.8, g)
    rna = gene_base + z @ loadings.T + rng.normal(size=(n, g)) * gene_sd
    rna_center, rna_spread = rna.mean(0), rna.std(0)
    xs = (rna - rna_center) / rna_spread

    pg = c.n_proteins_generated
    cognate = rng.choice(g, pg, replace=False)
    others = np.setdiff1d(np.arange(g), cognate)
    regulators = rng.choice(others, c.n_regulators, replace=False)
    cis_amp = rng.uniform(0.3, 1.2, pg) * rng.choice([1.0, -1.0], pg, p=[0.85, 0.15]) * c.coupling
    cis_rate = rng.uniform(0.5, 1.5, pg)
    n_trans = rng.poisson(1.5, pg)
    trans_idx = [rng.choice(regulators, t, replace=False) for t in n_trans]
    trans_coef = [rng.normal(0, 0.35, t) * c.coupling for t in n_trans]
    has_int = rng.uniform(size=pg) < 0.3
    int_partner = rng.choice(regulators, pg)
    int_coef = rng.normal(0, 0.3, pg) * has_int * c.coupling
    ptm_load = rng.normal(0, 0.4, (pg, c.n_ptm_factors)) * (rng.uniform(size=(pg, 1)) < 0.4) * c.ptm_scale
    noise_sd = rng.uniform(0.2, 0.4, pg)
    prot_base = rng.normal(22.0, 1.0, pg)

    mean = np.empty((n, pg))
    for p in range(pg):
        x_cog = xs[:, cognate[p]]
        mu = prot_base[p] + cis_amp[p] * np.tanh(cis_rate[p] * x_cog)
        if n_trans[p]:
            mu = mu + xs[:, trans_idx[p]] @ trans_coef[p]
        mu = mu + int_coef[p] * x_cog * xs[:, int_partner[p]]
        mean[:, p] = mu
    ptm = rng.normal(size=(n, c.n_ptm_factors))
    variance = (ptm_load ** 2).sum(1) + noise_sd ** 2
    if c.noise == "gaussian":
        eps = rng.normal(size=(n, pg))
    elif c.noise == "t3":
        eps = rng.standard_t(3, size=(n, pg)) / np.sqrt(3.0)   # unit variance
    else:
        raise ValueError(c.noise)
    protein = mean + ptm @ ptm_load.T + eps * noise_sd

    offsets = rng.normal(0, c.sample_offset_sd, n)
    observed, mech, d_entry, kind = _apply_mechanism(protein, offsets, c, rng)

    keep = np.flatnonzero(observed.mean(0) >= c.min_observed_fraction)[: c.n_proteins]
    if keep.size < c.n_proteins:
        raise RuntimeError("Too few proteins pass the observed-fraction filter.")
    oracle = None
    if c.noise == "gaussian" and kind is not None:
        m_k, v_k = mean[:, keep], np.broadcast_to(variance[keep][None, :], (n, keep.size))
        d_k = d_entry[:, keep]
        if kind == "probit":
            oracle = oracle_hidden_mean(m_k, v_k, d_k, c.detection_slope, c.mcar)
        elif kind == "threshold":
            oracle = oracle_hidden_mean(m_k, v_k, d_k, 1e-6, c.mcar)
        elif kind == "logistic":
            oracle = quadrature_hidden_mean(m_k, v_k, lambda y: c.mcar + (1 - c.mcar) * _sigmoid(
                (d_k[..., None] - y) / _logistic_scale(c.detection_slope)))
    return {
        "sample_ids": np.array(["S2_%04d" % i for i in range(n)]),
        "rna": rna, "protein": protein[:, keep], "observed": observed[:, keep],
        "cognate": cognate[keep],
        "gene_names": np.array(["G%04d" % j for j in range(g)]),
        "protein_names": np.array(["P%04d" % j for j in keep]),
        "oracle_hidden_mean": oracle,
        "mechanism": mech,
        "config": asdict(c), "seed": int(seed),
    }


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _logistic_scale(probit_slope):
    """Logistic scale with the same standard deviation as a probit curve of slope ``probit_slope``."""
    return probit_slope * np.sqrt(3.0) / np.pi


def quadrature_hidden_mean(m, v, hidden_prob, n_nodes=80):
    """E[y | hidden] for y ~ N(m, v) and P(hidden | y) = hidden_prob(y), by Gauss-Hermite quadrature.

    ``hidden_prob`` maps an array of shape m.shape + (n_nodes,) to probabilities.
    """
    x, w = np.polynomial.hermite.hermgauss(n_nodes)
    y = m[..., None] + np.sqrt(2.0 * v)[..., None] * x
    h = hidden_prob(y) * w
    return (h * y).sum(-1) / np.maximum(h.sum(-1), 1e-300)


def _calibrate(expected_missing, target, lo, hi):
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if expected_missing(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _apply_mechanism(y, offsets, c, rng):
    """Draw the observation mask. Returns (observed, mechanism record, per-entry limit, oracle kind)."""
    s, pi = c.detection_slope, c.mcar
    lo, hi = float(y.min()) - 10 * s, float(y.max()) + 10 * s
    n, p = y.shape
    if c.mechanism in ("probit", "wide_limits"):
        dev = rng.normal(0, c.limit_spread_sd, p) if c.mechanism == "wide_limits" else np.zeros(p)
        f = lambda d: 1 - detection_probability(y, d + offsets[:, None] + dev[None, :], s, pi).mean()
        d0 = _calibrate(f, c.target_missing, lo, hi)
        d_entry = d0 + offsets[:, None] + dev[None, :]
        observed = rng.uniform(size=y.shape) < detection_probability(y, d_entry, s, pi)
        kind = "probit"
    elif c.mechanism == "logistic":
        sl = _logistic_scale(s)
        prob = lambda d: (1 - pi) * _sigmoid((y - d - offsets[:, None]) / sl)
        d0 = _calibrate(lambda d: 1 - prob(d).mean(), c.target_missing, lo, hi)
        d_entry = d0 + offsets[:, None] + np.zeros((1, p))
        observed = rng.uniform(size=y.shape) < prob(d0)
        kind = "logistic"
    elif c.mechanism == "threshold":
        prob = lambda d: (1 - pi) * (y > d + offsets[:, None])
        d0 = _calibrate(lambda d: 1 - prob(d).mean(), c.target_missing, lo, hi)
        d_entry = d0 + offsets[:, None] + np.zeros((1, p))
        observed = rng.uniform(size=y.shape) < prob(d0)
        kind = "threshold"
    elif c.mechanism == "batch":
        # A protein is missed in a whole acquisition batch with a probit probability in the batch's
        # mean abundance (as for an isobaric plex), plus entry-level MCAR.
        batch = np.arange(n) // c.batch_size
        nb = batch.max() + 1
        ybar = np.stack([y[batch == b].mean(0) for b in range(nb)])
        boff = rng.normal(0, c.sample_offset_sd, nb)
        f = lambda d: 1 - ((1 - pi) * norm.cdf((ybar - d - boff[:, None]) / s)).mean()
        d0 = _calibrate(f, c.target_missing, lo, hi)
        batch_obs = rng.uniform(size=ybar.shape) < norm.cdf((ybar - d0 - boff[:, None]) / s)
        observed = batch_obs[batch] & (rng.uniform(size=y.shape) >= pi)
        d_entry, kind = np.full(y.shape, np.nan), None
    else:
        raise ValueError(c.mechanism)
    mech = {"kind": c.mechanism, "d0": float(d0), "slope": s, "mcar": pi,
            "sample_offset_sd": c.sample_offset_sd, "noise": c.noise, "coupling": c.coupling,
            "ptm_scale": c.ptm_scale}
    if c.mechanism == "wide_limits":
        mech["limit_spread_sd"] = c.limit_spread_sd
    if c.mechanism == "batch":
        mech["batch_size"] = c.batch_size
    return observed, mech, d_entry, kind


# ---------------------------------------------------------------------------
# S1: the v1 fixture rebuilt from its RNA CSV (identical generating rule)
# ---------------------------------------------------------------------------

def load_s1(rna_csv):
    import pandas as pd
    frame = pd.read_csv(rna_csv, dtype={"PATIENT_ID": str})
    ids = frame["PATIENT_ID"].to_numpy(dtype=str)
    names = list(frame.drop(columns="PATIENT_ID").columns[:2000])
    rna = frame[names].to_numpy(dtype=np.float32)
    order = np.argsort(ids, kind="stable")
    ids, rna = ids[order], rna[order].astype(np.float64)
    protein = rna[:, :500] * np.random.RandomState(42).normal(1.0, 0.1, (len(ids), 500))
    cutoff = float(np.exp(2.0 + norm.ppf(0.3)))
    return {
        "sample_ids": ids, "rna": rna, "protein": protein, "observed": protein >= cutoff,
        "cognate": np.arange(500), "gene_names": np.array(names),
        "protein_names": np.array(["PROT_%d" % j for j in range(1, 501)]),
        "oracle_hidden_mean": None,
        "mechanism": {"kind": "deterministic_raw_cutoff", "cutoff": cutoff},
        "config": {"source": "v1 fixture", "rna_csv": str(rna_csv)}, "seed": 42,
    }


# ---------------------------------------------------------------------------
# CPTAC PDAC semi-synthetic benchmark
# ---------------------------------------------------------------------------

@dataclass
class CPTACConfig:
    n_proteins: int = 1000
    n_genes: int = 3000
    min_mean_expression: float = 1.0
    target_missing: float = 0.30
    detection_slope: float = 0.5
    mcar: float = 0.02
    sample_offset_sd: float = 0.3
    selection_seed: int = 2026
    # 'within': the original benchmark (limit on each protein's cohort z-score, so every protein
    #           loses its own lowest values);
    # 'intensity': one limit per sample on an intensity-like scale. Each protein's log-ratio
    #           profile is shifted by a protein-level baseline abundance b_j, drawn with SD
    #           baseline_sd_factor x (median within-protein SD) and correlated (baseline_rna_corr)
    #           with the mean level of its transcript, as protein and mRNA abundance are across genes.
    mask: str = "within"
    baseline_sd_factor: float = 1.5
    baseline_rna_corr: float = 0.5
    baseline_offset: float = 20.0      # cosmetic: puts the intensity-like scale near typical log2 values
    oversample: float = 1.4            # proteins drawn before the observed-fraction filter ('intensity')
    min_observed_fraction: float = 0.3
    include_normals: bool = False      # keep normal adjacent tissue (".N" samples) for downstream tests


def load_cptac(data_dir, mask_seed, config=None, cohort="pdac"):
    """Tumours (optionally plus normals) with both modalities; complete proteins with a cognate
    transcript; an imposed detection mechanism so that the hidden truth is known.

    Feature selection uses the complete cohort's RNA variance and protein completeness, which
    involve no hidden values (proteins are complete before masking). With ``mask='intensity'``
    the filter on the observed fraction after masking uses only the observation pattern.
    """
    import pandas as pd
    from pathlib import Path
    c = config or CPTACConfig()
    data_dir = Path(data_dir)
    rna = pd.read_csv(data_dir / f"{cohort}_transcriptomics.csv", index_col=0)
    prot = pd.read_csv(data_dir / f"{cohort}_proteomics.csv", index_col=0)
    shared = set(rna.index) & set(prot.index)
    samples = sorted(x for x in shared if c.include_normals or not x.endswith(".N"))
    rna, prot = rna.loc[samples], prot.loc[samples]
    rna_sym = np.array([col.split("|")[0] for col in rna.columns])
    prot_sym = np.array([col.split("|")[0] for col in prot.columns])
    rna_mean = rna.mean(0).to_numpy()
    expressed = rna_mean >= c.min_mean_expression
    # one transcript column per symbol: the most variable expressed one
    variances = rna.var(0).to_numpy()
    best = {}
    for j in np.flatnonzero(expressed):
        sym = rna_sym[j]
        if sym not in best or variances[j] > variances[best[sym]]:
            best[sym] = j
    complete = prot.notna().all(0).to_numpy()
    counts = pd.Series(prot_sym).value_counts()
    unique = np.array([counts[s] == 1 for s in prot_sym])
    eligible = np.flatnonzero(complete & unique & np.array([s in best for s in prot_sym]))
    rng = np.random.default_rng(c.selection_seed)
    n_draw = c.n_proteins if c.mask == "within" else min(len(eligible), int(round(c.n_proteins * c.oversample)))
    chosen = np.sort(rng.choice(eligible, n_draw, replace=False))
    protein = prot.iloc[:, chosen].to_numpy(dtype=np.float64)

    mrng = np.random.default_rng(mask_seed)
    offsets = mrng.normal(0, c.sample_offset_sd, len(samples))
    mech = {"kind": f"imposed_probit_{c.mask}", "slope": c.detection_slope, "mcar": c.mcar,
            "sample_offset_sd": c.sample_offset_sd, "mask_seed": int(mask_seed), "cohort": cohort}
    if c.mask == "within":
        zscore = (protein - protein.mean(0)) / protein.std(0)
        d0 = calibrate_detection_limit(zscore, offsets, c.detection_slope, c.mcar, c.target_missing)
        p_obs = detection_probability(zscore, d0 + offsets[:, None], c.detection_slope, c.mcar)
        observed = mrng.uniform(size=p_obs.shape) < p_obs
        values = protein
    elif c.mask == "intensity":
        sd_within = float(np.median(protein.std(0)))
        cog_mean = np.array([rna_mean[best[prot_sym[j]]] for j in chosen])
        zc = (cog_mean - cog_mean.mean()) / cog_mean.std()
        brng = np.random.default_rng(c.selection_seed + 1)
        u = c.baseline_rna_corr * zc + np.sqrt(1 - c.baseline_rna_corr ** 2) * brng.normal(size=len(chosen))
        baseline = c.baseline_offset + c.baseline_sd_factor * sd_within * u
        values = protein - protein.mean(0) + baseline          # intensity-like scale
        # detection on the intensity scale, shared slope in units of the within-protein SD
        slope = c.detection_slope * sd_within
        d0 = calibrate_detection_limit(values, offsets * sd_within, slope, c.mcar, c.target_missing)
        p_obs = detection_probability(values, d0 + offsets[:, None] * sd_within, slope, c.mcar)
        observed = mrng.uniform(size=p_obs.shape) < p_obs
        keep = np.flatnonzero(observed.mean(0) >= c.min_observed_fraction)
        if keep.size < c.n_proteins:
            raise RuntimeError(f"Only {keep.size} proteins pass the observed-fraction filter.")
        keep = np.sort(np.random.default_rng(c.selection_seed + 2).choice(keep, c.n_proteins, replace=False))
        chosen, values, observed = chosen[keep], values[:, keep], observed[:, keep]
        mech.update({"baseline_sd": c.baseline_sd_factor * sd_within, "baseline_rna_corr": c.baseline_rna_corr,
                     "slope_intensity_units": slope, "within_protein_sd_median": sd_within})
    else:
        raise ValueError(c.mask)
    mech["d0"] = float(d0)
    cog_cols = [best[prot_sym[j]] for j in chosen]
    rest = [j for j in best.values() if j not in set(cog_cols)]
    rest = sorted(rest, key=lambda j: -variances[j])[: max(0, c.n_genes - len(cog_cols))]
    gene_cols = list(cog_cols) + rest
    col_pos = {j: i for i, j in enumerate(gene_cols)}
    rna_values = rna.iloc[:, gene_cols].to_numpy(dtype=np.float64)
    return {
        "sample_ids": np.array(samples), "rna": rna_values, "protein": values,
        "observed": observed,
        "cognate": np.array([col_pos[j] for j in cog_cols]),
        "gene_names": np.array([rna.columns[j] for j in gene_cols]),
        "protein_names": np.array([prot.columns[j] for j in chosen]),
        "is_normal": np.array([x.endswith(".N") for x in samples]),
        "oracle_hidden_mean": None,
        "mechanism": mech,
        "config": asdict(c), "seed": int(mask_seed),
    }


# ---------------------------------------------------------------------------
# frozen split protocol
# ---------------------------------------------------------------------------

SPLITS = ("train", "validation", "test")


@dataclass
class Protocol:
    name: str
    sample_ids: np.ndarray
    split: dict
    rna: np.ndarray            # standardized with training RNA moments (all samples)
    protein_input: np.ndarray  # standardized observed values, 0 at hidden entries
    observed: np.ndarray       # bool, True = observed
    protein_mean: np.ndarray
    protein_scale: np.ndarray
    cognate: np.ndarray
    truth: dict = field(repr=False)          # validation/test standardized complete protein
    oracle: dict = field(repr=False)         # validation/test standardized oracle (S2 only)
    pseudo_mask: np.ndarray = None           # validation observed entries hidden for tuning
    meta: dict = None

    def rows(self, split):
        return self.split[split]

    def raw_protein_input(self):
        """Observed values on the original (log) scale, NaN at hidden entries."""
        raw = self.protein_input * self.protein_scale + self.protein_mean
        return np.where(self.observed, raw, np.nan)

    def fingerprint(self):
        h = hashlib.sha256(json.dumps(self.meta, sort_keys=True, default=str).encode())
        for arr in (self.rna, self.protein_input, self.observed, self.protein_mean,
                    self.protein_scale, *(self.split[s] for s in SPLITS)):
            h.update(np.ascontiguousarray(arr).tobytes())
        for s in ("validation", "test"):
            h.update(np.ascontiguousarray(self.truth[s]).tobytes())
        return h.hexdigest()


def build_protocol(source, split_seed, name, fractions=(0.6, 0.2), pseudo_fraction=0.10,
                   log_rna=False, split=None):
    """Freeze one split. ``split`` (dict of index arrays) overrides the random 60/20/20 split."""
    ids = np.asarray(source["sample_ids"])
    rna = np.asarray(source["rna"], dtype=np.float64)
    if log_rna:
        rna = np.log2(rna + 1.0)
    protein = np.asarray(source["protein"], dtype=np.float64)
    observed = np.asarray(source["observed"], dtype=bool)
    n = len(ids)
    if split is None:
        perm = np.random.RandomState(split_seed).permutation(n)
        n_tr, n_va = int(n * fractions[0]), int(n * fractions[1])
        split = {"train": np.sort(perm[:n_tr]), "validation": np.sort(perm[n_tr:n_tr + n_va]),
                 "test": np.sort(perm[n_tr + n_va:])}
    else:
        split = {k: np.sort(np.asarray(v)) for k, v in split.items()}
    tr = split["train"]
    r_mean, r_sd = rna[tr].mean(0), rna[tr].std(0)
    r_sd[r_sd < 1e-12] = 1.0
    obs_tr = observed[tr]
    counts = obs_tr.sum(0)
    if np.any(counts < 2):
        raise ValueError("A protein has fewer than two observed training values.")
    p_mean = np.where(obs_tr, protein[tr], 0).sum(0) / counts
    p_sd = np.sqrt(np.where(obs_tr, (protein[tr] - p_mean) ** 2, 0).sum(0) / counts)
    p_sd[p_sd < 1e-12] = 1.0
    z_prot = (protein - p_mean) / p_sd
    truth = {s: z_prot[split[s]].astype(np.float32) for s in ("validation", "test")}
    oracle = {}
    if source.get("oracle_hidden_mean") is not None:
        o = (source["oracle_hidden_mean"] - p_mean) / p_sd
        oracle = {s: o[split[s]].astype(np.float32) for s in ("validation", "test")}
    # Deployable tuning set: low-abundance-weighted subset of OBSERVED validation entries.
    from ..calibration import pseudo_mask as _pseudo
    va = split["validation"]
    pseudo = _pseudo(z_prot[va], observed[va], pseudo_fraction, 0.5, np.random.default_rng(split_seed + 7919))
    meta = {"name": name, "split_seed": int(split_seed), "fractions": list(fractions),
            "n_samples": n, "n_genes": int(rna.shape[1]), "n_proteins": int(protein.shape[1]),
            "hidden_fraction": {s: float(1 - observed[split[s]].mean()) for s in SPLITS},
            "hidden_entries": {s: int((~observed[split[s]]).sum()) for s in SPLITS},
            "pseudo_mask_entries": int(pseudo.sum()), "log_rna": bool(log_rna),
            "mechanism": source["mechanism"], "source_config": source["config"],
            "source_seed": source["seed"]}
    return Protocol(
        name=name, sample_ids=ids, split=split,
        rna=((rna - r_mean) / r_sd).astype(np.float32),
        protein_input=np.where(observed, z_prot, 0.0).astype(np.float32),
        observed=observed, protein_mean=p_mean, protein_scale=p_sd,
        cognate=np.asarray(source["cognate"]), truth=truth, oracle=oracle,
        pseudo_mask=pseudo, meta=meta)
