"""Censored-likelihood (probit-selection) multimodal VAE.

Every component is switchable so that it can be ablated:

* ``elbo='correct'``   reconstruction summed over features and KL summed over latent
                       coordinates, both per sample, then averaged over the batch.
                       ``elbo='v1'`` reproduces v1's mismatched reduction (per-entry mean
                       reconstruction, batch-summed KL, beta ramped to 0.1).
* ``mask_aware``       the protein encoder receives [values, mask]; otherwise zero-filled values.
* ``likelihood``       'censored' : Gaussian decoder with a probit selection model
                                    P(obs | x) = (1 - pi) Phi((x - tau_ij)/s),
                                    tau_ij = t_j + delta_i * r_j: a global limit (raw or
                                    standardized scale) plus shrunk per-protein deviations, and an
                                    amortized sample offset delta_i.
                                    The hidden-entry likelihood and E[x | hidden, z] are closed form.
                       'mar'      : Gaussian on observed entries only (missingness ignored).
                       'hinge'    : 'mar' plus v1's squared-hinge penalty at a fixed per-feature limit.
* ``inference``        'posterior_mean' (deterministic plug-in), 'single_draw' (v1 behaviour), or
                       'mc' (average of conditional expectations over K latent draws).
* ``fusion``           'transformer' over two modality tokens, or a concatenation 'mlp'.
* ``contrastive``      weight of symmetric InfoNCE between pre-fusion modality tokens (0 = off).
* ``skip``             per-protein linear term on its cognate transcript, added to the decoder mean.
"""

from dataclasses import dataclass, asdict
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import log_ndtr

LOG2PI = math.log(2 * math.pi)


@dataclass
class VAEConfig:
    latent_dim: int = 16
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 1
    ff_dim: int = 128
    dropout: float = 0.1
    decoder_hidden: int = 128
    mask_aware: bool = True
    fusion: str = "transformer"
    contrastive: float = 0.1
    skip: bool = True
    use_latent: bool = True         # False: decoder mean = per-feature intercept (+ skip); no KL
    likelihood: str = "censored"
    elbo: str = "correct"
    beta_warmup_epochs: int = 20
    hinge_gamma: float = 5.0
    offset_prior_sd: float = 1.0
    limit_scale: str = "raw"        # 'raw': one global limit on the original log scale (intensity data)
                                    # 'standardized': one global limit in per-protein standardized units
    limit_deviation_sd: float = 0.25  # prior SD of per-protein deviations from the global limit
    lr: float = 1e-3
    weight_decay: float = 1e-5
    fast_lr: float = 1e-2           # per-feature scalars: skip weights, log-SDs, detection limits
    batch_size: int = 32
    max_epochs: int = 300
    eval_every: int = 5
    patience: int = 60
    selection: str = "protein_objective"  # validation criterion: 'objective' (full loss) or
                                          # 'protein_objective' (protein terms of the ELBO only)
    device: str = "cpu"             # 'cpu', 'cuda' or 'auto' (cuda when available)


def _randn(shape, generator, device):
    """Standard normal draws from a CPU generator (reproducible on any device), moved to ``device``."""
    return torch.randn(tuple(shape), generator=generator).to(device)


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _log_normal(x, m, log_sd):
    return -0.5 * ((x - m) * torch.exp(-log_sd)) ** 2 - log_sd - 0.5 * LOG2PI


class CensoredMultimodalVAE(nn.Module):
    def __init__(self, n_genes, n_proteins, cognate, protein_mean, protein_scale, init_limit,
                 n_train, config):
        super().__init__()
        c = self.config = config
        e = c.embed_dim
        self.n_genes, self.n_proteins = n_genes, n_proteins
        self.rna_token = nn.Sequential(nn.Linear(n_genes, e), nn.LayerNorm(e), nn.GELU())
        p_in = 2 * n_proteins if c.mask_aware else n_proteins
        self.prot_token = nn.Sequential(nn.Linear(p_in, e), nn.LayerNorm(e), nn.GELU())
        if c.fusion == "transformer":
            self.modality_emb = nn.Parameter(torch.zeros(2, e))
            layer = nn.TransformerEncoderLayer(e, c.num_heads, c.ff_dim, c.dropout,
                                               batch_first=True, norm_first=True)
            self.fuse = nn.TransformerEncoder(layer, c.num_layers, enable_nested_tensor=False)
        elif c.fusion == "mlp":
            self.fuse = nn.Sequential(nn.Linear(2 * e, 2 * e), nn.GELU(), nn.Dropout(c.dropout),
                                      nn.Linear(2 * e, 2 * e))
        else:
            raise ValueError(c.fusion)
        self.fc_mu = nn.Linear(2 * e, c.latent_dim)
        self.fc_logvar = nn.Linear(2 * e, c.latent_dim)
        self.fc_offset = nn.Linear(2 * e, 1)
        h = c.decoder_hidden
        self.decoder = nn.Sequential(nn.Linear(c.latent_dim, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.rna_out = nn.Linear(h, n_genes)
        self.prot_out = nn.Linear(h, n_proteins)
        self.rna_log_sd = nn.Parameter(torch.zeros(n_genes))
        self.prot_log_sd = nn.Parameter(torch.full((n_proteins,), -0.5))
        cognate = torch.as_tensor(np.asarray(cognate), dtype=torch.long)
        self.register_buffer("cognate", cognate.clamp(min=0))
        self.register_buffer("has_cognate", (cognate >= 0).float())
        self.skip_weight = nn.Parameter(torch.zeros(n_proteins))
        mu = torch.as_tensor(protein_mean, dtype=torch.float32)
        sd = torch.as_tensor(protein_scale, dtype=torch.float32)
        init_limit = torch.as_tensor(init_limit, dtype=torch.float32)
        self.register_buffer("p_mu", mu)
        self.register_buffer("p_sd", sd)
        self.n_train = n_train
        if c.limit_scale == "raw":
            # tau_j = (D - mu_j) / sd_j ; sample offsets are raw-scale shifts.
            d0 = float(torch.median(init_limit * sd + mu))
            self.register_buffer("offset_scale", torch.median(sd) / sd)
        elif c.limit_scale == "standardized":
            d0 = float(torch.median(init_limit))
            self.register_buffer("offset_scale", torch.ones_like(sd))
        else:
            raise ValueError(c.limit_scale)
        self.global_limit = nn.Parameter(torch.tensor(d0))
        self.limit_dev = nn.Parameter(torch.zeros(n_proteins))
        self.register_buffer("hinge_limit", init_limit.clone())  # fixed for the hinge variant
        self.log_slope = nn.Parameter(torch.tensor(math.log(0.5)))
        self.mcar_logit = nn.Parameter(torch.tensor(math.log(0.02 / 0.98)))

    # ----------------------------------------------------------------- encoder
    def encode(self, rna, prot, mask):
        t_r = self.rna_token(rna)
        x = torch.cat([prot * mask, mask], 1) if self.config.mask_aware else prot * mask
        t_p = self.prot_token(x)
        if self.config.fusion == "transformer":
            tokens = torch.stack([t_r, t_p], 1) + self.modality_emb
            fused = self.fuse(tokens).flatten(1)
        else:
            fused = self.fuse(torch.cat([t_r, t_p], 1))
        mu, logvar = self.fc_mu(fused), self.fc_logvar(fused).clamp(-10, 6)
        offset = self.fc_offset(fused).squeeze(1)
        return mu, logvar, offset, (t_r, t_p)

    def decode(self, z, rna):
        if not self.config.use_latent:
            z = torch.zeros_like(z)
        h = self.decoder(z)
        m_r = self.rna_out(h)
        m_p = self.prot_out(h)
        if self.config.skip:
            m_p = m_p + self.skip_weight * self.has_cognate * rna[..., self.cognate]
        return m_r, m_p

    @property
    def limit(self):
        if self.config.limit_scale == "raw":
            base = (self.global_limit - self.p_mu) / self.p_sd
        else:
            base = self.global_limit.expand_as(self.p_mu)
        return base + self.limit_dev

    def detection(self, offset):
        tau = self.limit + offset.unsqueeze(-1) * self.offset_scale
        slope = torch.exp(self.log_slope)
        log_pi = F.logsigmoid(self.mcar_logit)
        log_1mpi = F.logsigmoid(-self.mcar_logit)
        return tau, slope, log_pi, log_1mpi

    # -------------------------------------------------------------- likelihood
    def protein_loglik(self, x, mask, m_p, offset):
        """Per-entry log-likelihood terms (observed and hidden) for the chosen model."""
        lsd = self.prot_log_sd
        obs = mask * _log_normal(x, m_p, lsd)
        if self.config.likelihood != "censored":
            return obs, torch.zeros_like(obs)
        tau, s, log_pi, log_1mpi = self.detection(offset)
        obs = obs + mask * (log_1mpi + log_ndtr((x - tau) / s))
        root = torch.sqrt(torch.exp(2 * lsd) + s ** 2)
        hid = (1 - mask) * torch.logaddexp(log_pi.expand_as(m_p), log_1mpi + log_ndtr((tau - m_p) / root))
        return obs, hid

    def objective(self, rna, prot, mask, z_sample=True, epoch=0):
        """Returns (loss, parts). Loss is per-feature normalized; ratios match the ELBO."""
        c = self.config
        mu, logvar, offset, (t_r, t_p) = self.encode(rna, prot, mask)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if z_sample else mu
        m_r, m_p = self.decode(z, rna)
        if not c.use_latent:
            mu, logvar = torch.zeros_like(mu), torch.zeros_like(logvar)
        ll_r = _log_normal(rna, m_r, self.rna_log_sd)
        ll_obs, ll_hid = self.protein_loglik(prot, mask, m_p, offset)
        kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)
        off_pen = 0.5 * (offset / c.offset_prior_sd) ** 2 if c.likelihood == "censored" else 0 * offset
        hinge = ((1 - mask) * F.relu(m_p - self.hinge_limit) ** 2)
        n_feat = self.n_genes + self.n_proteins
        if c.elbo == "correct":
            beta = min(1.0, (epoch + 1) / max(1, c.beta_warmup_epochs))
            nll = -(ll_r.sum(1) + ll_obs.sum(1) + ll_hid.sum(1))
            per_sample = nll + beta * kl + off_pen
            dev_pen = 0.5 * (self.limit_dev / c.limit_deviation_sd) ** 2
            loss = (per_sample.mean() + dev_pen.sum() / self.n_train) / n_feat
            if c.likelihood == "hinge":
                loss = loss + c.hinge_gamma * hinge.sum() / (1 - mask).sum().clamp(min=1)
        elif c.elbo == "v1":
            beta = min(0.1, epoch / 100.0)
            rec = -ll_r.mean() - ll_obs.sum() / mask.sum().clamp(min=1)
            rec = rec - ll_hid.sum() / (1 - mask).sum().clamp(min=1)
            dev_pen = 0.5 * (self.limit_dev / c.limit_deviation_sd) ** 2
            loss = rec + beta * kl.sum() + off_pen.mean() + dev_pen.mean()
            if c.likelihood == "hinge":
                loss = loss + c.hinge_gamma * hinge.sum() / (1 - mask).sum().clamp(min=1)
        else:
            raise ValueError(c.elbo)
        cont = torch.zeros((), device=rna.device)
        if c.contrastive > 0 and rna.shape[0] > 1:
            a, b = F.normalize(t_r, dim=1), F.normalize(t_p, dim=1)
            logits = a @ b.T / 0.1
            labels = torch.arange(a.shape[0], device=a.device)
            cont = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
            loss = loss + c.contrastive * cont
        prot_nll = -(ll_obs.sum(1) + ll_hid.sum(1)).mean() / self.n_proteins
        parts = {"protein_nll_per_feature": float(prot_nll.detach()),
                 "kl_mean": float(kl.mean().detach()),
                 "kl_per_dim": (0.5 * (mu ** 2 + logvar.exp() - 1 - logvar)).mean(0).detach(),
                 "contrastive": float(cont.detach())}
        return loss, parts

    # -------------------------------------------------------------- prediction
    def _hidden_expectation(self, m_p, offset):
        if self.config.likelihood != "censored":
            return m_p
        tau, s, log_pi, log_1mpi = self.detection(offset)
        var = torch.exp(2 * self.prot_log_sd)
        root = torch.sqrt(var + s ** 2)
        a = (tau - m_p) / root
        log_cdf = log_ndtr(a)
        log_w_mcar = log_pi - torch.logaddexp(log_pi.expand_as(a), log_1mpi + log_cdf)
        w = torch.exp(log_w_mcar)
        log_pdf = -0.5 * a ** 2 - 0.5 * LOG2PI
        mills = torch.exp(log_pdf - log_cdf)
        return w * m_p + (1 - w) * (m_p - var / root * mills)

    @torch.no_grad()
    def predict(self, rna, prot, mask, inference="posterior_mean", n_draws=50, generator=None):
        self.eval()
        mu, logvar, offset, _ = self.encode(rna, prot, mask)
        if inference == "posterior_mean":
            zs = [mu]
        elif inference == "single_draw":
            zs = [mu + torch.exp(0.5 * logvar) * _randn(mu.shape, generator, mu.device)]
        elif inference == "mc":
            zs = [mu + torch.exp(0.5 * logvar) * _randn(mu.shape, generator, mu.device)
                  for _ in range(n_draws)]
        else:
            raise ValueError(inference)
        preds = [self._hidden_expectation(self.decode(z, rna)[1], offset) for z in zs]
        pred = torch.stack(preds).mean(0)
        return torch.where(mask.bool(), prot, pred)

    @torch.no_grad()
    def predictive_interval(self, rna, prot, mask, level=0.9, n_z=25, n_x=20, generator=None,
                            condition="hidden"):
        """Weighted-sample quantiles of p(x | hidden, data) (censored likelihood) or of the
        Gaussian predictive (other likelihoods), integrating over q(z).

        ``condition='observed'`` weights draws by P(observed | x) instead, giving the model's
        predictive distribution for an entry known to be detected (used for recalibration)."""
        self.eval()
        mu, logvar, offset, _ = self.encode(rna, prot, mask)
        sd = torch.exp(self.prot_log_sd)
        tau, s, log_pi, log_1mpi = self.detection(offset)
        samples, weights = [], []
        for _ in range(n_z):
            z = mu + torch.exp(0.5 * logvar) * _randn(mu.shape, generator, mu.device)
            m_p = self.decode(z, rna)[1]
            x = m_p.unsqueeze(0) + sd * _randn((n_x,) + tuple(m_p.shape), generator, mu.device)
            if self.config.likelihood == "censored" and condition == "hidden":
                lw = torch.logaddexp(log_pi.expand_as(x), log_1mpi + log_ndtr((tau - x) / s))
            elif self.config.likelihood == "censored" and condition == "observed":
                lw = log_1mpi + log_ndtr((x - tau) / s)
            else:
                lw = torch.zeros_like(x)
            samples.append(x)
            weights.append(lw)
        x = torch.cat(samples)               # [S, B, P]
        lw = torch.cat(weights)
        w = torch.softmax(lw, 0)
        order = torch.argsort(x, 0)
        xs = torch.gather(x, 0, order)
        cw = torch.cumsum(torch.gather(w, 0, order), 0)
        lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
        lo_idx = (cw < lo_q).sum(0, keepdim=True).clamp(max=x.shape[0] - 1)
        hi_idx = (cw < hi_q).sum(0, keepdim=True).clamp(max=x.shape[0] - 1)
        return torch.gather(xs, 0, lo_idx)[0], torch.gather(xs, 0, hi_idx)[0]


# ------------------------------------------------------------------ training

def train_model(rna, protein, mask, cognate, protein_mean, protein_scale, train_idx, val_idx,
                config, seed):
    """Fit the model on ``train_idx`` and select the epoch on ``val_idx``.

    ``rna`` (n x g) is standardized RNA, ``protein`` (n x p) is standardized protein with zeros at
    hidden entries, and ``mask`` (n x p) is True where a protein value is observed.
    ``protein_mean``/``protein_scale`` are the observed-training moments used for standardization.
    The epoch is selected by the model's own objective on the validation samples at the posterior
    mean, which needs no hidden truth.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, va = np.asarray(train_idx), np.asarray(val_idx)
    device = resolve_device(config.device)
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    rna, prot, mask_np = t(rna), t(protein), np.asarray(mask, dtype=bool)
    mask = t(mask_np)
    obs_tr = mask_np[tr]
    x_tr = np.where(obs_tr, np.asarray(protein)[tr], np.inf)
    init_limit = x_tr.min(0)  # lowest observed standardized training value per protein
    model = CensoredMultimodalVAE(rna.shape[1], prot.shape[1], cognate, protein_mean,
                                  protein_scale, init_limit, len(tr), config).to(device)
    fast_names = {"skip_weight", "rna_log_sd", "prot_log_sd", "global_limit", "limit_dev",
                  "log_slope", "mcar_logit"}
    fast = [p for n, p in model.named_parameters() if n in fast_names]
    slow = [p for n, p in model.named_parameters() if n not in fast_names]
    opt = torch.optim.Adam([{"params": slow, "lr": config.lr, "weight_decay": config.weight_decay},
                            {"params": fast, "lr": config.fast_lr, "weight_decay": 0.0}])
    gen = torch.Generator().manual_seed(seed)
    best, best_epoch, best_state, history, since = float("inf"), None, None, [], 0
    for epoch in range(config.max_epochs):
        model.train()
        order = tr[torch.randperm(len(tr), generator=gen).numpy()]
        for start in range(0, len(order), config.batch_size):
            idx = order[start:start + config.batch_size]
            if len(idx) < 2:
                continue
            loss, _ = model.objective(rna[idx], prot[idx], mask[idx], z_sample=True, epoch=epoch)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % config.eval_every == 0 or epoch + 1 == config.max_epochs:
            model.eval()
            with torch.no_grad():
                vloss, parts = model.objective(rna[va], prot[va], mask[va], z_sample=False,
                                               epoch=config.max_epochs)
            v = float(vloss) if config.selection == "objective" else parts["protein_nll_per_feature"]
            history.append({"epoch": epoch + 1, "validation_objective": float(vloss),
                            "validation_protein_nll": parts["protein_nll_per_feature"],
                            "kl_mean": parts["kl_mean"],
                            "active_units": int((parts["kl_per_dim"] > 0.01).sum())})
            if v < best:
                best, best_epoch, best_state, since = v, epoch + 1, copy.deepcopy(model.state_dict()), 0
            else:
                since += config.eval_every
                if since >= config.patience:
                    break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, parts = model.objective(rna[tr], prot[tr], mask[tr], z_sample=False, epoch=config.max_epochs)
    diagnostics = {
        "selected_epoch": best_epoch, "validation_objective": best,
        "epochs_run": history[-1]["epoch"],
        "kl_per_dim_train": [float(x) for x in parts["kl_per_dim"]],
        "active_units_train": int((parts["kl_per_dim"] > 0.01).sum()),
        "detection_slope": float(torch.exp(model.log_slope).detach()),
        "global_limit": float(model.global_limit.detach()),
        "mcar_rate": float(torch.sigmoid(model.mcar_logit).detach()),
        "skip_weight_mean_abs": float(model.skip_weight.abs().mean().detach()),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "device": str(device),
        "history": history, "config": asdict(config),
    }
    return model, diagnostics


def fit_vae(protocol, config, seed):
    """Benchmark entry point: train on a frozen :class:`~tcsimpute.benchmark.data.Protocol`."""
    return train_model(protocol.rna, protocol.protein_input, protocol.observed, protocol.cognate,
                       protocol.protein_mean, protocol.protein_scale, protocol.rows("train"),
                       protocol.rows("validation"), config, seed)


def _model_device(model):
    return next(model.parameters()).device


def predict_arrays(model, rna, protein, mask, inference="posterior_mean", seed=0):
    """Standardized imputations for standardized ``rna``/``protein`` and boolean ``mask``."""
    dev = _model_device(model)
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)
    gen = torch.Generator().manual_seed(seed)
    out = model.predict(t(rna), t(protein), t(mask), inference=inference, generator=gen)
    return out.cpu().numpy()


def interval_arrays(model, rna, protein, mask, level=0.9, seed=0, condition="hidden", max_values=2e7):
    """Standardized lower/upper predictive bounds for every entry (observed entries included).

    Samples are processed in row blocks so that the 500 predictive draws per entry stay within
    about ``max_values`` floats; rows are independent given the fitted model.
    """
    dev = _model_device(model)
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)
    gen = torch.Generator().manual_seed(seed)
    n, p = np.shape(protein)
    block = max(1, int(max_values // (500 * max(p, 1))))
    los, his = [], []
    for i in range(0, n, block):
        sl = slice(i, i + block)
        lo, hi = model.predictive_interval(t(rna[sl]), t(protein[sl]), t(mask[sl]), level=level,
                                           generator=gen, condition=condition)
        los.append(lo.cpu().numpy())
        his.append(hi.cpu().numpy())
    return np.concatenate(los), np.concatenate(his)


def predict_split(model, protocol, split, inference="posterior_mean", seed=0):
    rows = protocol.rows(split)
    return predict_arrays(model, protocol.rna[rows], protocol.protein_input[rows],
                          protocol.observed[rows], inference, seed)


def interval_split(model, protocol, split, level=0.9, seed=0):
    rows = protocol.rows(split)
    return interval_arrays(model, protocol.rna[rows], protocol.protein_input[rows],
                           protocol.observed[rows], level, seed)
