"""Tests for the benchmark code: closed forms, information boundaries and model wiring."""

import numpy as np
import pytest
import torch
from scipy.stats import norm

from tcsimpute.benchmark import baselines as B
from tcsimpute.benchmark.data import S2Config, build_protocol, oracle_hidden_mean, simulate_s2
from tcsimpute.metrics import hidden_metrics, within_feature_r
from tcsimpute.model import CensoredMultimodalVAE, VAEConfig, fit_vae, predict_split

SMALL = S2Config(n_samples=60, n_genes=120, n_proteins_generated=60, n_proteins=40, n_regulators=20)


@pytest.fixture(scope="module")
def proto():
    return build_protocol(simulate_s2(5, SMALL), 3, "test")


def test_oracle_matches_monte_carlo():
    rng = np.random.default_rng(0)
    m, v, d, s, pi = 1.0, 0.7, 0.4, 0.5, 0.05
    y = rng.normal(m, np.sqrt(v), 2_000_000)
    hidden = rng.uniform(size=y.size) < pi + (1 - pi) * norm.cdf((d - y) / s)
    assert abs(y[hidden].mean() - oracle_hidden_mean(m, v, d, s, pi)) < 5e-3


def test_model_hidden_expectation_matches_oracle_formula():
    cfg = VAEConfig(likelihood="censored", limit_scale="standardized")
    model = CensoredMultimodalVAE(3, 4, [0, 1, 2, -1], np.zeros(4), np.ones(4), np.zeros(4), 10, cfg)
    with torch.no_grad():
        model.prot_log_sd.fill_(0.5 * np.log(0.7))
        model.global_limit.fill_(0.4)
        model.log_slope.fill_(np.log(0.5))
        model.mcar_logit.fill_(np.log(0.05 / 0.95))
        m = torch.full((1, 4), 1.0)
        e = model._hidden_expectation(m, torch.zeros(1)).numpy()
    assert np.allclose(e, oracle_hidden_mean(1.0, 0.7, 0.4, 0.5, 0.05), atol=1e-5)


def test_protocol_scalers_use_observed_training_entries_only(proto):
    tr = proto.rows("train")
    ob = proto.observed[tr]
    raw = proto.raw_protein_input()[tr]
    assert np.allclose(np.nanmean(raw, 0), proto.protein_mean)
    assert np.allclose(np.nanstd(raw, 0), proto.protein_scale)
    assert np.all(proto.protein_input[~proto.observed] == 0)
    assert set(proto.truth) == {"validation", "test"}


def test_hidden_training_truth_does_not_reach_the_protocol():
    src = simulate_s2(5, SMALL)
    a = build_protocol(src, 3, "a")
    tr = a.rows("train")
    perturbed = dict(src)
    perturbed["protein"] = src["protein"].copy()
    hid = np.zeros_like(src["observed"])
    hid[tr] = ~src["observed"][tr]
    perturbed["protein"][hid] += 100.0
    b = build_protocol(perturbed, 3, "b")
    for name in ("rna", "protein_input", "protein_mean", "protein_scale"):
        assert np.array_equal(getattr(a, name), getattr(b, name))


def test_pseudo_mask_only_hides_observed_validation_entries(proto):
    va = proto.rows("validation")
    assert proto.pseudo_mask.shape == proto.observed[va].shape
    assert not np.any(proto.pseudo_mask & ~proto.observed[va])
    assert proto.pseudo_mask.sum() > 0


def test_fit_predict_keeps_observed_and_is_deterministic(proto):
    cfg = VAEConfig(max_epochs=10, eval_every=5, limit_scale="raw")
    model, diag = fit_vae(proto, cfg, 1)
    p1 = predict_split(model, proto, "test")
    p2 = predict_split(model, proto, "test")
    te = proto.rows("test")
    assert np.array_equal(p1, p2)
    assert np.allclose(p1[proto.observed[te]], proto.protein_input[te][proto.observed[te]])
    assert diag["selected_epoch"] in (5, 10)
    assert "kl_per_dim_train" in diag


@pytest.mark.parametrize("variant", [dict(mask_aware=False), dict(fusion="mlp"), dict(contrastive=0.0),
                                     dict(skip=False), dict(use_latent=False), dict(likelihood="mar"),
                                     dict(likelihood="hinge"), dict(elbo="v1")])
def test_ablation_variants_run(proto, variant):
    model, _ = fit_vae(proto, VAEConfig(max_epochs=5, eval_every=5, **variant), 1)
    assert np.isfinite(predict_split(model, proto, "test")).all()


def test_per_sample_baselines_preserve_observed(proto):
    te = proto.rows("test")
    rng = np.random.default_rng(0)
    for fn in (B.min_det, B.min_prob, B.qrilc, B.down_shifted_normal, B.observed_mean):
        out = fn(proto, te, rng=rng)
        assert np.allclose(out[proto.observed[te]], proto.protein_input[te][proto.observed[te]])
        assert np.isfinite(out).all()


def test_within_feature_r_excludes_constant_predictors():
    truth = np.random.default_rng(0).normal(size=(20, 3))
    hidden = np.ones_like(truth, dtype=bool)
    r, eligible, constant = within_feature_r(truth, np.zeros_like(truth), hidden)
    assert r is None and eligible == 3 and constant == 3
    m = hidden_metrics(truth, truth, hidden)
    assert m["rmse"] == 0 and abs(m["within_feature_r"] - 1) < 1e-12
