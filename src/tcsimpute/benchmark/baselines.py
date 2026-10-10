"""Comparators (fitted on training rows only unless marked transductive; tuning on validation).

Single-sample left-censoring imputers (per sample on the native log scale): MinDet, MinProb,
QRILC (Python re-implementations; the revision runs the imputeLCMD reference code through
:mod:`.rbridge`) and the down-shifted normal (DSN) of Perseus.
Multivariate: KNN and missForest-style iterative random forests. With ``cognate=True`` the
design is [RNA PCs, each protein's cognate transcript, proteins], so these methods see the same
transcript information as tcsimpute's cognate term.
Deep generative: MIWAE (MAR) and not-MIWAE with a self-masking logistic missingness model.
Censored regression: per-protein Tobit regression on [cognate transcript, RNA PCs] with the
censoring point at the lowest observed training value.
Controls: observed-training mean, cognate-transcript OLS on observed pairs, Bayes oracle (S2).
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm, truncnorm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer

K_GRID = (3, 5, 10, 20)
N_PCS = 20


def _to_std(raw, protocol):
    return (raw - protocol.protein_mean) / protocol.protein_scale


def _fill(protocol, rows, pred_std, observed=None):
    observed = protocol.observed[rows] if observed is None else observed
    return np.where(observed, protocol.protein_input[rows], pred_std).astype(np.float32)


# ------------------------------------------------------------ per-sample methods

def min_det(protocol, rows, q=0.01, observed=None, rng=None):
    observed = protocol.observed[rows] if observed is None else observed
    raw = protocol.protein_input[rows] * protocol.protein_scale + protocol.protein_mean
    raw = np.where(observed, raw, np.nan)
    floor = np.nanquantile(raw, q, axis=1, keepdims=True)
    return _fill(protocol, rows, _to_std(np.broadcast_to(floor, raw.shape), protocol), observed)


def _training_protein_sd(protocol):
    tr = protocol.rows("train")
    raw = protocol.raw_protein_input()[tr]
    return float(np.nanmedian(np.nanstd(raw, axis=0)))


def min_prob(protocol, rows, q=0.01, tune_sigma=1.0, observed=None, rng=None):
    observed = protocol.observed[rows] if observed is None else observed
    raw = protocol.protein_input[rows] * protocol.protein_scale + protocol.protein_mean
    raw = np.where(observed, raw, np.nan)
    floor = np.nanquantile(raw, q, axis=1, keepdims=True)
    sd = _training_protein_sd(protocol) * tune_sigma
    draw = floor + sd * rng.standard_normal(raw.shape)
    return _fill(protocol, rows, _to_std(draw, protocol), observed)


def qrilc(protocol, rows, tune_sigma=1.0, observed=None, rng=None):
    """Quantile-regression imputation of left-censored data (Lazar et al.; imputeLCMD)."""
    observed = protocol.observed[rows] if observed is None else observed
    raw = protocol.protein_input[rows] * protocol.protein_scale + protocol.protein_mean
    out = np.empty_like(raw)
    for i in range(raw.shape[0]):
        x = raw[i, observed[i]]
        f = 1.0 - observed[i].mean()
        probs = np.linspace(0.001, 0.99, 100)
        qx = np.quantile(x, probs)
        qn = norm.ppf(f + (1.0 - f) * probs)
        slope, intercept = np.polyfit(qn, qx, 1)
        mu, sd = intercept, abs(slope) * tune_sigma
        upper = (norm.ppf(min(f + 0.001, 0.999)) * abs(slope) + intercept - mu) / max(sd, 1e-8)
        out[i] = truncnorm.rvs(-np.inf, upper, loc=mu, scale=max(sd, 1e-8), size=raw.shape[1],
                               random_state=rng)
    return _fill(protocol, rows, _to_std(out, protocol), observed)


def down_shifted_normal(protocol, rows, width=0.3, shift=1.8, observed=None, rng=None):
    observed = protocol.observed[rows] if observed is None else observed
    raw = protocol.protein_input[rows] * protocol.protein_scale + protocol.protein_mean
    raw = np.where(observed, raw, np.nan)
    m, s = np.nanmean(raw, 1, keepdims=True), np.nanstd(raw, 1, keepdims=True)
    draw = m - shift * s + width * s * rng.standard_normal(raw.shape)
    return _fill(protocol, rows, _to_std(draw, protocol), observed)


# ------------------------------------------------------------ multivariate methods

class RNAPCs:
    def __init__(self, protocol, n=N_PCS, seed=0):
        tr = protocol.rows("train")
        self.pca = PCA(n_components=n, random_state=seed).fit(protocol.rna[tr])
        scores = self.pca.transform(protocol.rna[tr])
        self.sd = scores.std(0)

    def __call__(self, rna):
        return self.pca.transform(rna) / self.sd


def covariates(protocol, pcs, rows, cognate=False):
    """Complete covariates: RNA PCs, plus each protein's cognate transcript if ``cognate``."""
    parts = [pcs(protocol.rna[rows])]
    if cognate:
        cog = np.asarray(protocol.cognate)
        x = protocol.rna[rows][:, np.clip(cog, 0, None)]
        parts.append(np.where(cog[None, :] >= 0, x, 0.0))
    return np.concatenate(parts, 1)


def _design(protocol, pcs, rows, observed=None, cognate=False):
    observed = protocol.observed[rows] if observed is None else observed
    prot = np.where(observed, protocol.protein_input[rows], np.nan)
    return np.concatenate([covariates(protocol, pcs, rows, cognate), prot], 1)


def _n_cov(protocol, cognate):
    return N_PCS + (protocol.protein_input.shape[1] if cognate else 0)


def fit_knn(protocol, pcs, k, cognate=False):
    tr = protocol.rows("train")
    return KNNImputer(n_neighbors=k, weights="uniform").fit(_design(protocol, pcs, tr, cognate=cognate))


def knn_predict(model, protocol, pcs, rows, observed=None, cognate=False):
    observed = protocol.observed[rows] if observed is None else observed
    out = model.transform(_design(protocol, pcs, rows, observed, cognate))[:, _n_cov(protocol, cognate):]
    return _fill(protocol, rows, out, observed)


def fit_missforest(protocol, pcs, seed, n_estimators=50, max_iter=5, cognate=False):
    tr = protocol.rows("train")
    rf = RandomForestRegressor(n_estimators=n_estimators, max_features="sqrt", n_jobs=1,
                               random_state=seed)
    imp = IterativeImputer(estimator=rf, max_iter=max_iter, initial_strategy="mean",
                           random_state=seed, skip_complete=True)
    return imp.fit(_design(protocol, pcs, tr, cognate=cognate))


def missforest_predict(model, protocol, pcs, rows, observed=None, cognate=False):
    observed = protocol.observed[rows] if observed is None else observed
    out = model.transform(_design(protocol, pcs, rows, observed, cognate))[:, _n_cov(protocol, cognate):]
    return _fill(protocol, rows, out, observed)


# ------------------------------------------------------------ censored (Tobit) regression

def fit_tobit(protocol, n_pcs=10, ridge=1e-2, max_iter=500, seed=0):
    """Per-protein Tobit regression of the standardized protein on [cognate transcript, RNA PCs].

    Each protein is left-censored at its lowest observed training value c_j: an observed value
    contributes log N(x; mu, sigma^2), a hidden one log Phi((c_j - mu) / sigma). All proteins are
    fitted jointly (independent parameters) by L-BFGS with a small ridge penalty on slopes.
    """
    tr = protocol.rows("train")
    pcs = RNAPCs(protocol, n=n_pcs, seed=seed)
    X = _tobit_design(protocol, pcs, tr)                        # [n, P, k]
    y = torch.tensor(protocol.protein_input[tr], dtype=torch.float32)
    ob = torch.tensor(protocol.observed[tr], dtype=torch.float32)
    c = torch.tensor(np.where(protocol.observed[tr], protocol.protein_input[tr], np.inf).min(0),
                     dtype=torch.float32)
    P, k = X.shape[1], X.shape[2]
    beta = torch.zeros(P, k, requires_grad=True)
    alpha = torch.zeros(P, requires_grad=True)
    log_sd = torch.zeros(P, requires_grad=True)
    opt = torch.optim.LBFGS([beta, alpha, log_sd], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        mu = alpha + (X * beta).sum(-1)
        sd = torch.exp(log_sd)
        ll_obs = -0.5 * ((y - mu) / sd) ** 2 - log_sd
        ll_hid = torch.special.log_ndtr((c - mu) / sd)
        loss = -(ob * ll_obs + (1 - ob) * ll_hid).sum() / len(tr) + ridge * (beta ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return {"pcs": pcs, "beta": beta.detach(), "alpha": alpha.detach(), "sd": torch.exp(log_sd).detach(),
            "c": c}


def fit_selection_regression(protocol, limit_scale="raw", n_pcs=10, ridge=1e-2, max_iter=1000,
                             seed=0, limit_deviation_sd=0.25):
    """Linear probit-selection regression: tcsimpute's detection model without the encoder.

    mean_ij = alpha_j + beta_j' [cognate transcript, RNA PCs]; P(observed | x) =
    (1 - pi) Phi((x - tau_j) / s) with tau_j a global limit (raw or standardized scale, as for
    tcsimpute) plus shrunken per-protein deviations. Fitted jointly by L-BFGS.
    """
    tr = protocol.rows("train")
    torch.manual_seed(seed)
    pcs = RNAPCs(protocol, n=n_pcs, seed=seed)
    X = _tobit_design(protocol, pcs, tr)
    y = torch.tensor(protocol.protein_input[tr], dtype=torch.float32)
    ob = torch.tensor(protocol.observed[tr], dtype=torch.float32)
    mu_hat = torch.tensor(protocol.protein_mean, dtype=torch.float32)
    sd_hat = torch.tensor(protocol.protein_scale, dtype=torch.float32)
    lowest = np.where(protocol.observed[tr], protocol.protein_input[tr], np.inf).min(0)
    if limit_scale == "raw":
        d_init = float(np.median(lowest * protocol.protein_scale + protocol.protein_mean))
    else:
        d_init = float(np.median(lowest))
    P, k = X.shape[1], X.shape[2]
    beta = torch.zeros(P, k, requires_grad=True)
    alpha = torch.zeros(P, requires_grad=True)
    log_sd = torch.zeros(P, requires_grad=True)
    D = torch.tensor(d_init, requires_grad=True)
    dev = torch.zeros(P, requires_grad=True)
    log_s = torch.tensor(math.log(0.5), requires_grad=True)
    logit_pi = torch.tensor(math.log(0.02 / 0.98), requires_grad=True)
    params = [beta, alpha, log_sd, D, dev, log_s, logit_pi]

    def tau():
        base = (D - mu_hat) / sd_hat if limit_scale == "raw" else D.expand(P)
        return base + dev

    def loss_fn():
        mu = alpha + (X * beta).sum(-1)
        sd, s = torch.exp(log_sd), torch.exp(log_s)
        lp, l1p = F.logsigmoid(logit_pi), F.logsigmoid(-logit_pi)
        t = tau()
        ll_obs = -0.5 * ((y - mu) / sd) ** 2 - log_sd + l1p + torch.special.log_ndtr((y - t) / s)
        ll_hid = torch.logaddexp(lp.expand_as(mu), l1p + torch.special.log_ndtr((t - mu) / torch.sqrt(sd ** 2 + s ** 2)))
        nll = -(ob * ll_obs + (1 - ob) * ll_hid).sum() / len(tr)
        return nll + ridge * (beta ** 2).sum() + 0.5 * ((dev / limit_deviation_sd) ** 2).sum() / len(tr)

    opt = torch.optim.LBFGS(params, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return {"pcs": pcs, "beta": beta.clone(), "alpha": alpha.clone(), "sd": torch.exp(log_sd),
                "tau": tau().clone(), "s": torch.exp(log_s), "pi": torch.sigmoid(logit_pi),
                "nll": float(loss_fn())}


def selection_regression_predict(fit, protocol, rows, observed=None):
    X = _tobit_design(protocol, fit["pcs"], rows)
    mu = fit["alpha"] + (X * fit["beta"]).sum(-1)
    var = fit["sd"] ** 2
    root = torch.sqrt(var + fit["s"] ** 2)
    a = (fit["tau"] - mu) / root
    log_cdf = torch.special.log_ndtr(a)
    pi = fit["pi"]
    w = pi / (pi + (1 - pi) * torch.exp(log_cdf))
    mills = torch.exp(-0.5 * a ** 2 - 0.5 * math.log(2 * math.pi) - log_cdf)
    pred = (w * mu + (1 - w) * (mu - var / root * mills)).numpy()
    return _fill(protocol, rows, pred, observed)


def _tobit_design(protocol, pcs, rows):
    cog = np.asarray(protocol.cognate)
    xc = np.where(cog[None, :] >= 0, protocol.rna[rows][:, np.clip(cog, 0, None)], 0.0)
    z = pcs(protocol.rna[rows])
    X = np.concatenate([xc[:, :, None], np.broadcast_to(z[:, None, :], (len(rows), len(cog), z.shape[1]))], 2)
    return torch.tensor(X, dtype=torch.float32)


def tobit_predict(fit, protocol, rows, observed=None):
    X = _tobit_design(protocol, fit["pcs"], rows)
    mu = fit["alpha"] + (X * fit["beta"]).sum(-1)
    sd = fit["sd"]
    a = (fit["c"] - mu) / sd
    mills = torch.exp(-0.5 * a ** 2 - 0.5 * math.log(2 * math.pi) - torch.special.log_ndtr(a))
    pred = (mu - sd * mills).numpy()
    return _fill(protocol, rows, pred, observed)


# ------------------------------------------------------------ controls

def observed_mean(protocol, rows, observed=None, rng=None):
    return _fill(protocol, rows, np.zeros_like(protocol.protein_input[rows]), observed)


def fit_cognate_ols(protocol):
    tr = protocol.rows("train")
    x = protocol.rna[tr][:, protocol.cognate]
    y, ob = protocol.protein_input[tr], protocol.observed[tr]
    slopes, intercepts = np.zeros(y.shape[1]), np.zeros(y.shape[1])
    for j in range(y.shape[1]):
        xj, yj = x[ob[:, j], j], y[ob[:, j], j]
        xc = xj - xj.mean()
        ss = xc @ xc
        slopes[j] = 0.0 if ss < 1e-12 else xc @ (yj - yj.mean()) / ss
        intercepts[j] = yj.mean() - slopes[j] * xj.mean()
    return slopes, intercepts


def cognate_ols_predict(fit, protocol, rows, observed=None):
    slopes, intercepts = fit
    pred = protocol.rna[rows][:, protocol.cognate] * slopes + intercepts
    return _fill(protocol, rows, pred, observed)


# ------------------------------------------------------------ MIWAE / not-MIWAE

class NotMIWAE(nn.Module):
    """Ipsen, Mattei & Frellsen (ICLR 2021) with a self-masking logistic missing model on the
    protein columns; ``missing_model=False`` gives MIWAE (Mattei & Frellsen, ICML 2019)."""

    def __init__(self, n_cov, n_prot, latent=16, hidden=128, missing_model=True):
        super().__init__()
        d = n_cov + n_prot
        self.n_cov, self.missing_model = n_cov, missing_model
        self.enc = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 2 * latent))
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, d))
        self.log_sd = nn.Parameter(torch.zeros(d))
        self.w = nn.Parameter(torch.zeros(n_prot))
        self.b = nn.Parameter(torch.full((n_prot,), -1.0))
        self.latent = latent

    def log_weights(self, x, s, k):
        """x: [B, D] zero-filled; s: [B, D] mask. Returns log w [K, B] and samples."""
        h = self.enc(x * s)
        mu, log_sd = h[:, :self.latent], h[:, self.latent:].clamp(-8, 4)
        eps = torch.randn((k,) + mu.shape, device=x.device)
        z = mu + torch.exp(log_sd) * eps
        mean = self.dec(z)
        sd = torch.exp(self.log_sd)
        x_samp = mean + sd * torch.randn_like(mean)
        x_full = s * x + (1 - s) * x_samp
        log_px = (s * (-0.5 * ((x - mean) / sd) ** 2 - self.log_sd - 0.5 * math.log(2 * math.pi))).sum(-1)
        log_pz = (-0.5 * z ** 2).sum(-1)
        log_qz = (-0.5 * eps ** 2 - log_sd).sum(-1)
        lw = log_px + log_pz - log_qz
        if self.missing_model:
            xp = x_full[..., self.n_cov:]
            sp = s[:, self.n_cov:]
            logits = F.softplus(self.w) * (xp - self.b)
            lw = lw + (sp * F.logsigmoid(logits) + (1 - sp) * F.logsigmoid(-logits)).sum(-1)
        return lw, x_full

    def bound(self, x, s, k):
        lw, _ = self.log_weights(x, s, k)
        return (torch.logsumexp(lw, 0) - math.log(k)).mean()

    @torch.no_grad()
    def impute(self, x, s, k=1000):
        lw, x_full = self.log_weights(x, s, k)
        w = torch.softmax(lw, 0).unsqueeze(-1)
        return (w * x_full).sum(0)


def _miwae_tensors(protocol, pcs, rows, cognate):
    cov = covariates(protocol, pcs, rows, cognate)
    x = np.concatenate([cov, protocol.protein_input[rows]], 1)
    s = np.concatenate([np.ones_like(cov), protocol.observed[rows]], 1)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(s, dtype=torch.float32)


def fit_notmiwae(protocol, pcs, seed, missing_model=True, k=20, epochs=300, eval_every=5,
                 patience=60, lr=1e-3, batch=32, cognate=False, latent=16):
    torch.manual_seed(seed)
    tr, va = protocol.rows("train"), protocol.rows("validation")
    x_tr, s_tr = _miwae_tensors(protocol, pcs, tr, cognate)
    x_va, s_va = _miwae_tensors(protocol, pcs, va, cognate)
    model = NotMIWAE(_n_cov(protocol, cognate), protocol.protein_input.shape[1], latent=latent,
                     missing_model=missing_model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, best_epoch, since, history = -float("inf"), None, None, 0, []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(tr), generator=gen)
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            loss = -model.bound(x_tr[idx], s_tr[idx], k)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % eval_every == 0:
            model.eval()
            with torch.no_grad():
                v = float(model.bound(x_va, s_va, k))
            history.append({"epoch": epoch + 1, "validation_bound": v})
            if v > best:
                best, best_state, best_epoch, since = v, copy.deepcopy(model.state_dict()), epoch + 1, 0
            else:
                since += eval_every
                if since >= patience:
                    break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"selected_epoch": best_epoch, "validation_bound": best, "latent": latent,
                   "history": history}


def fit_notmiwae_tuned(protocol, pcs, seed, latents=(8, 16, 32), **kw):
    """Select the latent dimension by the validation importance-weighted bound (no hidden values)."""
    best = None
    scores = {}
    for d in latents:
        model, diag = fit_notmiwae(protocol, pcs, seed, latent=d, **kw)
        scores[d] = diag["validation_bound"]
        if best is None or diag["validation_bound"] > best[1]["validation_bound"]:
            best = (model, diag)
    best[1]["latent_scores"] = scores
    return best


def notmiwae_predict(model, protocol, pcs, rows, seed=0, cognate=False):
    torch.manual_seed(seed)
    x, s = _miwae_tensors(protocol, pcs, rows, cognate)
    chunk = max(1, int(2e7 // (1000 * x.shape[1])))      # keep the 1,000 importance draws in memory
    out = torch.cat([model.impute(x[i:i + chunk], s[i:i + chunk]) for i in range(0, len(x), chunk)])
    return _fill(protocol, rows, out.numpy()[:, _n_cov(protocol, cognate):])
