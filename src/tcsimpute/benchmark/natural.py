"""Natural-missingness benchmark on ProCan-DepMapSanger DIA proteomics (Goncalves et al. 2022).

Each cell line was acquired in about six MS runs. For data replicate ``r`` the ``r``-th run of
every cell line (in file order, cycling) is the *primary* run: its values are what the imputers
see, and its missing entries are the entries to impute. The ground truth for a missing entry is
the mean of that cell line's other runs where the protein was quantified; entries never
quantified in any other run have no known truth and are not scored. Missingness is therefore the
instrument's own, not imposed, and the scored entries are exactly those near the detection limit
(observed in some runs, missed in the primary run).

Transcriptomics are the RNA view of the multi-omics object released with the study
(``MultiOmics_DIANN.hdf5``), exported to ``procan_transcriptomics.csv`` (cell lines x genes,
centred per gene). Proteins are matched to transcripts by the gene part of the UniProt entry name
(``P37108;SRP14_HUMAN`` -> ``SRP14``).

Prepare ``procan_dir`` with:
    ProCan-DepMapSanger_protein_matrix_6692_replicates.txt
    ProCan-DepMapSanger_mapping_file_replicates.txt
    procan_transcriptomics.csv
"""

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

MATRIX = "ProCan-DepMapSanger_protein_matrix_6692_replicates.txt"
MAPPING = "ProCan-DepMapSanger_mapping_file_replicates.txt"
RNA = "procan_transcriptomics.csv"


@dataclass
class ProCanConfig:
    n_proteins: int = 1500
    n_genes: int = 3000
    min_observed_fraction: float = 0.30   # in the primary runs (an observation-pattern filter)
    max_observed_fraction: float = 0.97   # keep proteins that actually have missing values
    min_runs: int = 2
    selection_seed: int = 2026


def load_procan(procan_dir, replicate, config=None):
    import pandas as pd
    c = config or ProCanConfig()
    d = Path(procan_dir)
    cache = d / f"cache_procan_rep{replicate}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as f:
            src = {k: f[k] for k in f.files}
        import json
        src.update(json.loads((d / f"cache_procan_rep{replicate}.json").read_text()))
        src["oracle_hidden_mean"] = None
        return src

    X = pd.read_csv(d / MATRIX, sep="\t", index_col=0)
    m = pd.read_csv(d / MAPPING, sep="\t").set_index("Automatic_MS_filename").loc[X.index]
    m = m[~m["SIDM"].astype(str).str.startswith("Control")]
    rna = pd.read_csv(d / RNA, index_col=0)
    rna = rna[rna.notna().all(axis=1)]
    runs = {sidm: list(g.index) for sidm, g in m.groupby("SIDM", sort=True)}
    lines = sorted(s for s, r in runs.items() if len(r) >= c.min_runs and s in rna.index)

    primary_runs, other_runs = [], []
    for s in lines:
        r = runs[s]
        k = (replicate - 1) % len(r)
        primary_runs.append(r[k])
        other_runs.append([x for i, x in enumerate(r) if i != k])
    A = X.loc[primary_runs].to_numpy(dtype=np.float64)
    truth_other = np.vstack([X.loc[o].mean(axis=0, skipna=True).to_numpy(dtype=np.float64) for o in other_runs])

    sym = np.array([col.split(";")[1].replace("_HUMAN", "") if ";" in col else col for col in X.columns])
    gene_names = np.array(rna.columns, dtype=str)
    gene_pos = {g: i for i, g in enumerate(gene_names)}
    obs_frac = np.isfinite(A).mean(0)
    counts = pd.Series(sym).value_counts()
    eligible = np.flatnonzero((obs_frac >= c.min_observed_fraction) & (obs_frac <= c.max_observed_fraction)
                              & np.array([s in gene_pos and counts[s] == 1 for s in sym]))
    rng = np.random.default_rng(c.selection_seed)
    chosen = np.sort(rng.choice(eligible, min(c.n_proteins, eligible.size), replace=False))

    observed = np.isfinite(A[:, chosen])
    # complete matrix: primary values where observed, otherwise the other runs' mean (NaN = unknown)
    protein = np.where(observed, A[:, chosen], truth_other[:, chosen])

    R = rna.loc[lines].to_numpy(dtype=np.float64)
    cog_cols = [gene_pos[sym[j]] for j in chosen]
    var = R.var(0)
    rest = [j for j in np.argsort(-var) if j not in set(cog_cols)][: max(0, c.n_genes - len(cog_cols))]
    gene_cols = list(cog_cols) + list(rest)
    col_pos = {j: i for i, j in enumerate(gene_cols)}
    hidden = ~observed
    mech = {"kind": "natural_dia", "primary_run_index": int(replicate - 1),
            "missing_fraction": float(hidden.mean()),
            "scored_fraction_of_missing": float(np.isfinite(protein[hidden]).mean()),
            "abundance_missingness_spearman": float(pd.Series(np.nanmean(A[:, chosen], 0)).corr(
                pd.Series(hidden.mean(0)), method="spearman"))}
    src = {
        "sample_ids": np.array(lines), "rna": R[:, gene_cols], "protein": protein,
        "observed": observed, "cognate": np.array([col_pos[j] for j in cog_cols]),
        "gene_names": gene_names[gene_cols], "protein_names": np.array(X.columns[chosen], dtype=str),
        "oracle_hidden_mean": None, "mechanism": mech, "config": asdict(c), "seed": int(replicate),
    }
    import json
    np.savez_compressed(cache, **{k: src[k] for k in ("sample_ids", "rna", "protein", "observed", "cognate",
                                                      "gene_names", "protein_names")})
    (d / f"cache_procan_rep{replicate}.json").write_text(json.dumps({k: src[k] for k in ("mechanism", "config", "seed")}))
    return src
