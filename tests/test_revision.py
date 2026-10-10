"""Tests for the revision benchmark: natural-missingness loader, downstream metrics, selection
regression, misspecified simulators and the report statistics."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tcsimpute.benchmark import baselines as B
from tcsimpute.benchmark.data import S2Config, build_protocol, simulate_s2
from tcsimpute.benchmark.downstream import downstream_metrics
from tcsimpute.benchmark.report import paired, rank_tests
from tcsimpute import metrics as M

SMALL = S2Config(n_samples=120, n_genes=300, n_proteins_generated=90, n_proteins=60, n_regulators=20)


@pytest.fixture()
def procan_dir(tmp_path):
    rng = np.random.default_rng(0)
    lines = [f"SIDM{i:05d}" for i in range(60)]
    genes = [f"G{j}" for j in range(200)]
    R = rng.normal(size=(60, 200))
    pd.DataFrame(R, index=pd.Index(lines, name="SIDM"), columns=genes).to_csv(tmp_path / "procan_transcriptomics.csv")
    prots = [f"Q{j};G{j}_HUMAN" for j in range(80)]
    rows, maps = [], []
    for i, s in enumerate(lines):
        for r in range(3):
            x = 5 + R[i, :80] + rng.normal(0, 0.3, 80)
            rows.append(np.where(rng.uniform(size=80) < 1 / (1 + np.exp(-(x - 4.6) / 0.4)), x, np.nan))
            maps.append({"Automatic_MS_filename": f"run_{s}_{r}", "SIDM": s})
    pd.DataFrame(rows, index=[m["Automatic_MS_filename"] for m in maps], columns=prots).to_csv(
        tmp_path / "ProCan-DepMapSanger_protein_matrix_6692_replicates.txt", sep="\t")
    pd.DataFrame(maps).to_csv(tmp_path / "ProCan-DepMapSanger_mapping_file_replicates.txt", sep="\t", index=False)
    return tmp_path


def test_procan_truth_comes_from_other_runs(procan_dir):
    from tcsimpute.benchmark.natural import ProCanConfig, load_procan
    src = load_procan(procan_dir, 2, ProCanConfig(n_proteins=40, n_genes=100))
    obs, prot = src["observed"], src["protein"]
    assert np.isfinite(prot[obs]).all()                      # primary run values where observed
    assert np.isnan(prot[~obs]).any() and np.isfinite(prot[~obs]).any()   # some hidden truth unknown
    assert src["mechanism"]["primary_run_index"] == 1
    P = build_protocol(src, 101, "procan")
    hidden = ~P.observed[P.rows("test")]
    pred = np.where(hidden, 0.0, P.protein_input[P.rows("test")])
    m = M.hidden_metrics(P.truth["test"], pred, hidden)      # unknown truth is skipped, not NaN
    assert np.isfinite(m["rmse"]) and m["hidden_entries"] < hidden.sum()


def test_downstream_perfect_imputation():
    rng = np.random.default_rng(1)
    is_normal = np.arange(60) < 20
    X = rng.normal(size=(60, 40))
    X[~is_normal, :10] += 1.5
    observed = rng.uniform(size=X.shape) > 0.3
    d = downstream_metrics(X, X.copy(), observed, is_normal)
    assert d["de_t_spearman"] == pytest.approx(1.0)
    assert d["de_precision"] == 1.0 and d["de_recall"] == 1.0 and d["de_effect_rmse"] == pytest.approx(0.0)
    assert d["coabundance_mae"] == pytest.approx(0.0)
    available = np.where(observed, X, np.nan)
    d2 = downstream_metrics(X, available, observed, is_normal)
    assert 0 < d2["de_t_spearman"] < 1


def test_selection_regression_beats_observed_mean():
    src = simulate_s2(5, SMALL)
    P = build_protocol(src, 101, "s")
    te = P.rows("test")
    hidden = ~P.observed[te]
    fit = B.fit_selection_regression(P, limit_scale="raw", seed=0, max_iter=300)
    sel = M.hidden_metrics(P.truth["test"], B.selection_regression_predict(fit, P, te), hidden)
    naive = M.hidden_metrics(P.truth["test"], B.observed_mean(P, te), hidden)
    assert sel["rmse"] < naive["rmse"] and abs(sel["bias"]) < abs(naive["bias"])


@pytest.mark.parametrize("override", [{"mechanism": "logistic"}, {"mechanism": "threshold"},
                                      {"mechanism": "wide_limits"}, {"mechanism": "batch", "target_missing": 0.4},
                                      {"noise": "t3"}, {"mcar": 0.15}])
def test_misspecified_simulators(override):
    src = simulate_s2(7, replace(SMALL, **override))
    miss = 1 - src["observed"].mean()
    assert 0.1 < miss < 0.5
    assert src["mechanism"]["kind"] == override.get("mechanism", "probit")


def test_report_paired_and_rank_tests():
    pts = [f"{r}_{s}" for r in range(4) for s in range(3)]
    mk = lambda m, base: [{"point": p, "rmse": base + 0.01 * i, "method": m} for i, p in enumerate(pts)]
    by = {"vae_full": mk("vae_full", 1.0), "selreg": mk("selreg", 0.9), "knn": mk("knn", 1.5)}
    d = paired(by["knn"], by["vae_full"])
    assert d["n"] == 12 and d["mean_diff"] == pytest.approx(0.5) and d["wins_reference"] == 12
    assert d["ci95"][0] <= 0.5 <= d["ci95"][1]
    rt = rank_tests(by)
    assert rt["mean_ranks"]["selreg"] == 1.0 and rt["mean_ranks"]["knn"] == 3.0 and rt["friedman_p"] < 0.01
