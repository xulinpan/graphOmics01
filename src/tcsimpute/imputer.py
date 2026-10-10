"""High-level, scikit-learn-style interface for transcriptome-guided protein imputation."""

from dataclasses import asdict, fields, replace
import warnings

import numpy as np
import torch

from .model import VAEConfig, interval_arrays, predict_arrays, train_model

__all__ = ["TCSImputer", "match_cognates"]


def match_cognates(protein_names, gene_names, sep="|"):
    """Map each protein to the column index of its cognate transcript (-1 if none).

    Names are compared on the part before ``sep`` (e.g. ``"TP53|ENSP..."`` -> ``"TP53"``), so the
    column labels returned by the ``cptac`` package can be passed directly. If a symbol occurs
    more than once among the genes, the first occurrence is used.
    """
    first = {}
    for j, name in enumerate(gene_names):
        first.setdefault(str(name).split(sep)[0], j)
    return np.array([first.get(str(name).split(sep)[0], -1) for name in protein_names], dtype=int)


def _as_array(x, name):
    try:
        import pandas as pd
        if isinstance(x, pd.DataFrame):
            return x.to_numpy(dtype=np.float64), x.index, list(x.columns)
    except ImportError:  # pragma: no cover - pandas is optional
        pass
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional (samples x features).")
    return arr, None, None


class _Data:
    """Minimal stand-in for the benchmark protocol, used by the linear variant."""

    def __init__(self, rna, protein_input, observed, protein_mean, protein_scale, cognate, train=None):
        self.rna, self.protein_input, self.observed = rna, protein_input, observed
        self.protein_mean, self.protein_scale, self.cognate = protein_mean, protein_scale, cognate
        self._train = train

    def rows(self, split):
        return self._train if split == "train" else np.arange(len(self.rna))


class TCSImputer:
    """Transcript-conditioned selection imputer for left-censored proteomics.

    Imputes protein values that are missing because they fall below a detection limit, using a
    paired RNA matrix. Missing protein values are given as ``NaN``; RNA must be complete. Inputs
    should be on a log scale (e.g. log2 intensities or log2 ratios). Two variants share the
    probit-selection likelihood: ``model="vae"`` (a multimodal variational autoencoder whose
    latent variable captures shared sample-level structure) and ``model="linear"`` (an
    encoder-free regression of each protein on its cognate transcript and RNA principal
    components, fitted by L-BFGS).

    Parameters
    ----------
    model : {"vae", "linear"}
        Which variant to fit.
    limit_scale : {"raw", "standardized"}
        ``"raw"`` assumes one detection limit on the original log scale, as for log intensities.
        ``"standardized"`` assumes a common limit in per-protein standardized units, as for
        ratio data whose proteins are on different reference scales.
    validation_fraction : float
        Fraction of samples held out to select the training epoch. The selection uses the model's
        own objective on those samples and never needs the true values of missing entries.
    random_state : int
        Seed for the split, initialization and training.
    config : VAEConfig, optional
        Full model configuration. Keyword arguments that match ``VAEConfig`` fields override it,
        e.g. ``TCSImputer(latent_dim=8, use_latent=False)``.

    Attributes
    ----------
    model_ : CensoredMultimodalVAE
    diagnostics_ : dict
        Selected epoch, active latent units, detection parameters and the training history.
    """

    def __init__(self, limit_scale="raw", validation_fraction=0.2, random_state=0, config=None,
                 model="vae", **config_overrides):
        if model not in ("vae", "linear"):
            raise ValueError("model must be 'vae' or 'linear'.")
        self.model = model
        valid = {f.name for f in fields(VAEConfig)}
        unknown = set(config_overrides) - valid
        if unknown:
            raise TypeError(f"Unknown configuration fields: {sorted(unknown)}")
        base = config or VAEConfig()
        self.config = replace(base, limit_scale=limit_scale, **config_overrides)
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1).")
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    # ------------------------------------------------------------------ fitting
    def fit(self, rna, protein, cognate=None):
        """Fit on paired matrices.

        Parameters
        ----------
        rna : array-like or DataFrame, shape (n_samples, n_genes), complete.
        protein : array-like or DataFrame, shape (n_samples, n_proteins), ``NaN`` = missing.
        cognate : array-like of int, shape (n_proteins,), optional
            Column of ``rna`` holding each protein's own transcript, or -1. If omitted and both
            inputs are DataFrames, it is derived with :func:`match_cognates`; otherwise the
            cognate term is disabled.
        """
        R, _, gene_names = _as_array(rna, "rna")
        P, _, protein_names = _as_array(protein, "protein")
        if R.shape[0] != P.shape[0]:
            raise ValueError("rna and protein must have the same number of samples (rows).")
        if not np.isfinite(R).all():
            raise ValueError("rna must be complete and finite.")
        if np.isinf(P).any():
            raise ValueError("protein must contain finite values or NaN.")
        if cognate is None:
            if gene_names is not None and protein_names is not None:
                cognate = match_cognates(protein_names, gene_names)
            else:
                cognate = np.full(P.shape[1], -1)
        cognate = np.asarray(cognate, dtype=int)
        if cognate.shape != (P.shape[1],) or cognate.max(initial=-1) >= R.shape[1]:
            raise ValueError("cognate must have one entry per protein, each < n_genes or -1.")
        if (cognate < 0).all() and self.config.skip:
            warnings.warn("No cognate transcripts given; the cognate term is disabled.", stacklevel=2)

        n = R.shape[0]
        perm = np.random.RandomState(self.random_state).permutation(n)
        n_val = max(1, int(round(n * self.validation_fraction)))
        val_idx, train_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])
        observed = np.isfinite(P)
        obs_tr = observed[train_idx]
        counts = obs_tr.sum(0)
        if np.any(counts < 2):
            bad = np.flatnonzero(counts < 2)[:10].tolist()
            raise ValueError(f"Proteins need at least two observed training values; failing columns: {bad}")

        self.rna_mean_ = R[train_idx].mean(0)
        self.rna_scale_ = R[train_idx].std(0)
        self.rna_scale_[self.rna_scale_ < 1e-12] = 1.0
        Ptr = np.where(obs_tr, P[train_idx], 0.0)
        self.protein_mean_ = Ptr.sum(0) / counts
        self.protein_scale_ = np.sqrt(np.where(obs_tr, (P[train_idx] - self.protein_mean_) ** 2, 0.0).sum(0) / counts)
        self.protein_scale_[self.protein_scale_ < 1e-12] = 1.0
        self.cognate_ = cognate
        self.n_genes_, self.n_proteins_ = R.shape[1], P.shape[1]
        self.train_index_, self.validation_index_ = train_idx, val_idx

        rna_z, prot_z, mask = self._standardize(R, P)
        if self.model == "linear":
            from .benchmark.baselines import fit_selection_regression
            data = _Data(rna_z, prot_z, mask, self.protein_mean_, self.protein_scale_, cognate,
                         np.sort(np.concatenate([train_idx, val_idx])))
            self.model_ = fit_selection_regression(data, limit_scale=self.config.limit_scale,
                                                   seed=self.random_state)
            self.diagnostics_ = {"nll": self.model_["nll"], "detection_slope": float(self.model_["s"]),
                                 "mcar_rate": float(self.model_["pi"])}
            return self
        self.model_, self.diagnostics_ = train_model(
            rna_z, prot_z, mask, cognate, self.protein_mean_, self.protein_scale_,
            train_idx, val_idx, self.config, self.random_state)
        return self

    def _linear_moments(self, rna_z, prot_z, mask):
        from .benchmark.baselines import _tobit_design
        data = _Data(rna_z, prot_z, mask, self.protein_mean_, self.protein_scale_, self.cognate_)
        X = _tobit_design(data, self.model_["pcs"], np.arange(len(rna_z)))
        mu = (self.model_["alpha"] + (X * self.model_["beta"]).sum(-1)).numpy()
        return mu, (self.model_["sd"] ** 2).numpy()

    def _standardize(self, R, P):
        rna_z = ((R - self.rna_mean_) / self.rna_scale_).astype(np.float32)
        mask = np.isfinite(P)
        prot_z = np.where(mask, (P - self.protein_mean_) / self.protein_scale_, 0.0).astype(np.float32)
        return rna_z, prot_z, mask

    def _check_fitted(self):
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit() before using the imputer.")

    def _prepare(self, rna, protein):
        self._check_fitted()
        R, idx, _ = _as_array(rna, "rna")
        P, pidx, pcols = _as_array(protein, "protein")
        if R.shape[1] != self.n_genes_ or P.shape[1] != self.n_proteins_:
            raise ValueError("Feature dimensions differ from the fitted data.")
        if R.shape[0] != P.shape[0]:
            raise ValueError("rna and protein must have the same number of samples (rows).")
        return R, P, (pidx if pidx is not None else idx), pcols

    def _wrap(self, values, index, columns):
        if columns is None:
            return values
        import pandas as pd
        return pd.DataFrame(values, index=index, columns=columns)

    # --------------------------------------------------------------- prediction
    def transform(self, rna, protein, inference="posterior_mean", seed=0):
        """Return ``protein`` with missing entries imputed, on the original scale.

        ``inference`` is ``"posterior_mean"`` (deterministic, default), ``"single_draw"`` or
        ``"mc"`` (average over 50 latent draws). Observed values are returned unchanged.
        """
        R, P, index, columns = self._prepare(rna, protein)
        rna_z, prot_z, mask = self._standardize(R, P)
        if self.model == "linear":
            from .benchmark.baselines import selection_regression_predict
            data = _Data(rna_z, prot_z, mask, self.protein_mean_, self.protein_scale_, self.cognate_)
            z = selection_regression_predict(self.model_, data, np.arange(len(rna_z)))
        else:
            z = predict_arrays(self.model_, rna_z, prot_z, mask, inference, seed)
        out = np.where(mask, P, z * self.protein_scale_ + self.protein_mean_)
        return self._wrap(out, index, columns)

    def fit_transform(self, rna, protein, cognate=None, **kwargs):
        return self.fit(rna, protein, cognate).transform(rna, protein, **kwargs)

    def predict_interval(self, rna, protein, level=0.9, seed=0):
        """Lower and upper predictive bounds for missing entries, on the original scale.

        For missing entries the bounds are importance-weighted quantiles of the predictive
        distribution conditional on the entry being undetected. Observed entries get their
        observed value as both bounds. Nominal coverage is approximate (see the paper's
        calibration analysis).
        """
        R, P, index, columns = self._prepare(rna, protein)
        rna_z, prot_z, mask = self._standardize(R, P)
        if self.model == "linear":
            lo, hi = self._linear_interval(rna_z, prot_z, mask, level, seed)
        else:
            lo, hi = interval_arrays(self.model_, rna_z, prot_z, mask, level, seed)
        lo = np.where(mask, P, lo * self.protein_scale_ + self.protein_mean_)
        hi = np.where(mask, P, hi * self.protein_scale_ + self.protein_mean_)
        return self._wrap(lo, index, columns), self._wrap(hi, index, columns)

    def _linear_interval(self, rna_z, prot_z, mask, level, seed, n_draws=500):
        from scipy.stats import norm
        mu, var = self._linear_moments(rna_z, prot_z, mask)
        tau, s, pi = (self.model_[k].numpy() for k in ("tau", "s", "pi"))
        rng = np.random.default_rng(seed)
        lo, hi = np.empty_like(mu), np.empty_like(mu)
        q = ((1 - level) / 2, (1 + level) / 2)
        for i in range(mu.shape[0]):
            x = mu[i] + np.sqrt(var) * rng.standard_normal((n_draws, mu.shape[1]))
            w = pi + (1 - pi) * norm.cdf((tau - x) / s)
            order = np.argsort(x, 0)
            xs, ws = np.take_along_axis(x, order, 0), np.take_along_axis(w, order, 0)
            cw = np.cumsum(ws, 0) / ws.sum(0)
            lo[i] = xs[np.argmax(cw >= q[0], 0), np.arange(mu.shape[1])]
            hi[i] = xs[np.argmax(cw >= q[1], 0), np.arange(mu.shape[1])]
        return lo, hi

    # ---------------------------------------------------------------- persistence
    _STATE = ("rna_mean_", "rna_scale_", "protein_mean_", "protein_scale_", "cognate_",
              "n_genes_", "n_proteins_", "train_index_", "validation_index_", "diagnostics_")

    def save(self, path):
        """Save the fitted imputer (weights, scalers and configuration) with ``torch.save``."""
        self._check_fitted()
        torch.save({"config": asdict(self.config), "validation_fraction": self.validation_fraction,
                    "random_state": self.random_state, "variant": self.model,
                    "state": {k: getattr(self, k) for k in self._STATE},
                    "model": self.model_ if self.model == "linear" else self.model_.state_dict()}, path)

    @classmethod
    def load(cls, path):
        from .model import CensoredMultimodalVAE
        blob = torch.load(path, weights_only=False, map_location="cpu")
        cfg = VAEConfig(**blob["config"])
        obj = cls(limit_scale=cfg.limit_scale, validation_fraction=blob["validation_fraction"],
                  random_state=blob["random_state"], config=cfg, model=blob.get("variant", "vae"))
        for k, v in blob["state"].items():
            setattr(obj, k, v)
        if obj.model == "linear":
            obj.model_ = blob["model"]
            return obj
        model = CensoredMultimodalVAE(obj.n_genes_, obj.n_proteins_, obj.cognate_, obj.protein_mean_,
                                      obj.protein_scale_, np.zeros(obj.n_proteins_),
                                      len(obj.train_index_), cfg)
        model.load_state_dict(blob["model"])
        model.eval()
        obj.model_ = model
        return obj
