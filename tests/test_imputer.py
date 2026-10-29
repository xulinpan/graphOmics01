"""Tests for the public TCSImputer API and the command-line interface."""

import numpy as np
import pandas as pd
import pytest

from tcsimpute import TCSImputer, match_cognates
from tcsimpute.benchmark.data import S2Config, simulate_s2

SMALL = S2Config(n_samples=80, n_genes=150, n_proteins_generated=60, n_proteins=40, n_regulators=20)
FAST = dict(model="vae", max_epochs=10, eval_every=5)


@pytest.fixture(scope="module")
def frames():
    sim = simulate_s2(3, SMALL)
    ids = sim["sample_ids"]
    rna = pd.DataFrame(sim["rna"], index=ids, columns=sim["gene_names"])
    # name each protein after its cognate gene, as real data would
    names = [f"{sim['gene_names'][c]}|PROT{j}" for j, c in enumerate(sim["cognate"])]
    full = pd.DataFrame(sim["protein"], index=ids, columns=names)
    observed = full.where(sim["observed"])
    return rna, observed, full, sim


def test_match_cognates_uses_symbol_before_separator():
    idx = match_cognates(["A|x", "C|y", "Z|z"], ["A|g1", "B", "C|g3"])
    assert idx.tolist() == [0, 2, -1]


def test_fit_transform_keeps_observed_and_fills_missing(frames):
    rna, observed, full, sim = frames
    imp = TCSImputer(random_state=1, **FAST).fit(rna, observed)
    out = imp.transform(rna, observed)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == list(observed.columns) and (out.index == observed.index).all()
    mask = observed.notna().to_numpy()
    assert np.array_equal(out.to_numpy()[mask], observed.to_numpy()[mask])
    assert np.isfinite(out.to_numpy()).all()
    assert (imp.cognate_ == sim["cognate"]).all()   # derived from column names


def test_imputations_sit_below_observed_means(frames):
    """Under left censoring, imputed values should on average lie below each protein's observed mean."""
    rna, observed, full, _ = frames
    imp = TCSImputer(random_state=1, **FAST).fit(rna, observed)
    out = imp.transform(rna, observed).to_numpy()
    hidden = observed.isna().to_numpy()
    shift = (out - imp.protein_mean_)[hidden]
    assert shift.mean() < 0


def test_transform_is_deterministic_and_works_on_new_samples(frames):
    rna, observed, _, _ = frames
    imp = TCSImputer(random_state=2, **FAST).fit(rna.iloc[:60], observed.iloc[:60])
    a = imp.transform(rna.iloc[60:], observed.iloc[60:])
    b = imp.transform(rna.iloc[60:], observed.iloc[60:])
    assert a.equals(b) and len(a) == 20


def test_arrays_in_arrays_out(frames):
    rna, observed, _, sim = frames
    imp = TCSImputer(random_state=0, **FAST).fit(rna.to_numpy(), observed.to_numpy(), cognate=sim["cognate"])
    out = imp.transform(rna.to_numpy(), observed.to_numpy())
    assert isinstance(out, np.ndarray) and out.shape == observed.shape


def test_intervals_bracket_imputations(frames):
    rna, observed, _, _ = frames
    imp = TCSImputer(random_state=1, **FAST).fit(rna, observed)
    mean = imp.transform(rna, observed).to_numpy()
    lo, hi = (x.to_numpy() for x in imp.predict_interval(rna, observed))
    hidden = observed.isna().to_numpy()
    assert (lo[hidden] <= hi[hidden]).all()
    inside = (lo[hidden] <= mean[hidden]) & (mean[hidden] <= hi[hidden])
    assert inside.mean() > 0.95
    assert np.array_equal(lo[~hidden], observed.to_numpy()[~hidden])


def test_save_and_load_roundtrip(frames, tmp_path):
    rna, observed, _, _ = frames
    imp = TCSImputer(random_state=1, **FAST).fit(rna, observed)
    path = tmp_path / "imp.pt"
    imp.save(path)
    again = TCSImputer.load(path)
    assert again.transform(rna, observed).equals(imp.transform(rna, observed))


def test_input_validation(frames):
    rna, observed, _, _ = frames
    with pytest.raises(TypeError):
        TCSImputer(not_a_field=1)
    bad = rna.copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        TCSImputer(**FAST).fit(bad, observed)
    with pytest.raises(ValueError):
        TCSImputer(**FAST).fit(rna.iloc[:10], observed)
    sparse = observed.copy()
    sparse.iloc[:, 0] = np.nan
    with pytest.raises(ValueError):
        TCSImputer(**FAST).fit(rna, sparse)
    with pytest.raises(RuntimeError):
        TCSImputer().transform(rna, observed)


def test_ablation_switches_pass_through(frames):
    rna, observed, _, _ = frames
    imp = TCSImputer(use_latent=False, likelihood="censored", **FAST).fit(rna, observed)
    assert imp.diagnostics_["active_units_train"] == 0


def test_cli_roundtrip(frames, tmp_path):
    from tcsimpute.cli import main
    rna, observed, _, _ = frames
    rna.to_csv(tmp_path / "rna.csv")
    observed.to_csv(tmp_path / "prot.csv")
    # a short run: patch the default config through the environment is not needed; just run it
    rc = main(["--rna", str(tmp_path / "rna.csv"), "--protein", str(tmp_path / "prot.csv"),
               "--out", str(tmp_path / "out.csv"), "--intervals", str(tmp_path / "pi")])
    assert rc == 0
    out = pd.read_csv(tmp_path / "out.csv", index_col=0)
    assert out.shape == observed.shape and np.isfinite(out.to_numpy()).all()
    assert (tmp_path / "pi_lower.csv").exists() and (tmp_path / "pi_upper.csv").exists()
    rc = main(["--rna", str(tmp_path / "rna.csv"), "--protein", str(tmp_path / "prot.csv"),
               "--out", str(tmp_path / "out_lin.csv"), "--model", "linear"])
    assert rc == 0 and pd.read_csv(tmp_path / "out_lin.csv", index_col=0).shape == observed.shape


def test_linear_variant_fit_transform_intervals_and_roundtrip(frames, tmp_path):
    rna, observed, full, _ = frames
    imp = TCSImputer(model="linear", random_state=0).fit(rna, observed)
    out = imp.transform(rna, observed)
    mask = observed.notna().to_numpy()
    assert np.array_equal(out.to_numpy()[mask], observed.to_numpy()[mask]) and np.isfinite(out.to_numpy()).all()
    hidden = ~mask
    err_lin = np.sqrt(np.mean((out.to_numpy() - full.to_numpy())[hidden] ** 2))
    naive = np.sqrt(np.mean((np.nanmean(observed.to_numpy(), 0)[None, :] - full.to_numpy())[hidden] ** 2))
    assert err_lin < naive
    lo, hi = (x.to_numpy() for x in imp.predict_interval(rna, observed))
    assert (lo[hidden] <= out.to_numpy()[hidden] + 1e-6).mean() > 0.95 and (hi[hidden] >= lo[hidden]).all()
    imp.save(tmp_path / "lin.pt")
    again = TCSImputer.load(tmp_path / "lin.pt")
    assert again.transform(rna, observed).equals(out)
    with pytest.raises(ValueError):
        TCSImputer(model="tree")


def test_default_model_is_linear():
    assert TCSImputer().model == "linear"
