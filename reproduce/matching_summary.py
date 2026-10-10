"""Protein-transcript matching and feature filtering in the CPTAC and ProCan benchmarks
(Supplementary Section S10).

For each data set this script retraces the selection made by ``load_cptac`` and ``load_procan``
and counts what is kept and dropped at each step: protein entries without a usable gene symbol,
symbols shared by several protein entries (isoforms or duplicate entries), proteins without an
expressed transcript, incomplete proteins (CPTAC), the observed-fraction filters and the random
draws. It also compares the CPTAC proteins that are complete in the original data with those
that are not, to show what the complete-protein selection represents, and checks the ProCan
matching by UniProt entry name against the primary gene names in UniProt.

Usage:
    python matching_summary.py --cptac-dir data_cptac --procan-dir data_procan \
        --uniprot uniprot_human_reviewed_genes.tsv --out matching_summary.json

The UniProt table is the TSV returned by
https://rest.uniprot.org/uniprotkb/stream?format=tsv&fields=accession,id,gene_primary,gene_synonym&query=(organism_id:9606)%20AND%20(reviewed:true)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tcsimpute.benchmark.data import CPTACConfig, load_cptac
from tcsimpute.benchmark.natural import MAPPING, MATRIX, RNA, ProCanConfig

CPTAC_MASKS = (11, 12, 13, 14)
DS_MASKS = (21, 22, 23, 24)
PROCAN_REPS = (1, 2, 3, 4)


def q(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return {"n": int(x.size), "median": float(np.median(x)), "q25": float(np.quantile(x, 0.25)),
            "q75": float(np.quantile(x, 0.75))}


def spearman_cols(a, b):
    """Spearman correlation of matching columns of a and b over rows where a is finite (>= 10 rows)."""
    out = pd.DataFrame(a).corrwith(pd.DataFrame(np.where(np.isfinite(a), b, np.nan)), method="spearman").to_numpy().copy()
    out[np.isfinite(a).sum(0) < 10] = np.nan
    return out


def plex_blocks(missing):
    """Groups of samples with (almost) identical missingness over incomplete proteins.

    In TMT data a protein that is not identified in a plex is missing for every sample of that plex,
    so samples of a plex share their missingness pattern. Samples are linked when they agree on more
    than 99% of incomplete proteins; the connected components estimate the plexes. Returns the number
    and sizes of the groups and the fraction of missing entries that lie in protein x group blocks
    missing for every sample of a group of at least two samples.
    """
    from scipy.sparse.csgraph import connected_components
    inc = missing.any(0) & ~missing.all(0)
    Mi = missing[:, inc].astype(float)
    agree = (Mi @ Mi.T + (1 - Mi) @ (1 - Mi).T) / Mi.shape[1]
    k, lab = connected_components(agree > 0.99, directed=False)
    sizes = np.bincount(lab)
    block = sum(missing[lab == g][:, missing[lab == g].all(0)].sum() for g in range(k) if sizes[g] >= 2)
    return {"groups": int(k), "singletons": int((sizes == 1).sum()), "group_size_min": int(sizes.min()),
            "group_size_max": int(sizes.max()),
            "missing_in_whole_group_blocks": float(block / max(missing.sum(), 1))}


def cptac(data_dir, cohort, include_normals):
    c = CPTACConfig(mask="intensity", include_normals=include_normals)
    d = Path(data_dir)
    rna = pd.read_csv(d / f"{cohort}_transcriptomics.csv", index_col=0)
    prot = pd.read_csv(d / f"{cohort}_proteomics.csv", index_col=0)
    shared = set(rna.index) & set(prot.index)
    samples = sorted(x for x in shared if include_normals or not x.endswith(".N"))
    n_prot_samples = int(sum(include_normals or not x.endswith(".N") for x in prot.index))
    n_rna_samples = int(sum(include_normals or not x.endswith(".N") for x in rna.index))
    rna, prot = rna.loc[samples], prot.loc[samples]
    rna_sym = np.array([col.split("|")[0] for col in rna.columns])
    prot_sym = np.array([col.split("|")[0] for col in prot.columns])
    prot_ids = np.array([col.split("|")[1] if "|" in col else "" for col in prot.columns])
    rna_mean = rna.mean(axis=0).to_numpy()
    expressed = rna_mean >= c.min_mean_expression
    rna_counts = pd.Series(rna_sym).value_counts()
    expressed_syms = set(rna_sym[expressed])
    counts = pd.Series(prot_sym).value_counts()
    blank = np.array([s in ("", "nan") for s in prot_sym])
    unique = np.array([counts[s] == 1 for s in prot_sym]) & ~blank
    in_rna = np.array([s in set(rna_sym) for s in prot_sym])
    has_expr = np.array([s in expressed_syms for s in prot_sym])
    complete = prot.notna().all(axis=0).to_numpy()
    eligible = complete & unique & has_expr
    matched = unique & has_expr

    # what the complete-protein selection represents: complete vs incomplete matched proteins
    P = prot.to_numpy(dtype=float)
    var = rna.var(axis=0).to_numpy()
    best = {}
    for j in np.flatnonzero(expressed):
        s = rna_sym[j]
        if s not in best or var[j] > var[best[s]]:
            best[s] = j
    idx = np.flatnonzero(matched)
    cog = np.array([best[prot_sym[j]] for j in idx])
    R = rna.to_numpy(dtype=float)
    miss = np.isnan(P[:, idx]).mean(0)
    rho = spearman_cols(P[:, idx], R[:, cog])
    sd = np.nanstd(P[:, idx], 0)
    comp = complete[idx]
    # samples with many missing proteins (plex-level structure of the original missingness)
    per_sample_missing = np.isnan(P[:, idx]).mean(1)
    rho_abund_missing = pd.Series(rna_mean[cog]).corr(pd.Series(miss), method="spearman")

    # final protein sets: read from the run caches written by the benchmark if present, else recomputed
    finals, n_pass = [], []
    tag = ("ds_" if include_normals else "") + cohort
    for seed in (DS_MASKS if include_normals else CPTAC_MASKS):
        cache = d / f"cache_{tag}_mask{seed}.npz"
        if cache.exists():
            with np.load(cache, allow_pickle=False) as f:
                names = [str(x) for x in f["protein_names"]]
        else:
            names = [str(x) for x in load_cptac(data_dir, seed, c, cohort=cohort)["protein_names"]]
        finals.append(set(names))
        n_pass.append(len(names))
    inter = set.intersection(*finals)
    union = set.union(*finals)
    out = {
        "cohort": cohort, "include_normals": include_normals,
        "samples": {"proteomics": n_prot_samples, "transcriptomics": n_rna_samples, "both": len(samples)},
        "protein_entries": int(len(prot_sym)),
        "protein_id_has_separator": int(sum((";" in x) or ("," in x) for x in prot_ids)),
        "protein_blank_symbol": int(blank.sum()),
        "protein_unique_symbols": int((counts == 1).sum()),
        "protein_symbols_shared": int(((counts > 1) & ~counts.index.isin(["", "nan"])).sum()),
        "protein_entries_shared_symbol": int((~unique & ~blank).sum()),
        "rna_entries": int(len(rna_sym)), "rna_symbols": int(rna_counts.size),
        "rna_symbols_multiple_entries": int((rna_counts > 1).sum()),
        "rna_entries_expressed": int(expressed.sum()), "rna_symbols_expressed": len(expressed_syms),
        "unique_protein_symbol_in_rna": int((unique & in_rna).sum()),
        "unique_protein_no_rna_symbol": int((unique & ~in_rna).sum()),
        "unique_protein_rna_not_expressed": int((unique & in_rna & ~has_expr).sum()),
        "matched": int(matched.sum()),
        "complete_all_entries": int(complete.sum()),
        "matched_complete": int(eligible.sum()),
        "matched_incomplete": int((matched & ~complete).sum()),
        "drawn": int(min(eligible.sum(), round(c.n_proteins * c.oversample))),
        "final_per_mask": n_pass, "final_overlap_all_masks": len(inter), "final_union": len(union),
        "original_missing_fraction": float(np.isnan(P).mean()),
        "matched_missing_fraction": float(np.isnan(P[:, idx]).mean()),
        "incomplete_missing_fraction": q(miss[~comp]),
        "incomplete_over_50pct_missing": int((miss[~comp] > 0.5).sum()),
        "cognate_rna_mean": {"complete": q(rna_mean[cog][comp]), "incomplete": q(rna_mean[cog][~comp])},
        "rna_protein_spearman": {"complete": q(rho[comp]), "incomplete": q(rho[~comp])},
        "protein_sd": {"complete": q(sd[comp]), "incomplete": q(sd[~comp])},
        "spearman_rna_mean_vs_missing_fraction": float(rho_abund_missing),
        "per_sample_missing_fraction": q(per_sample_missing),
        "samples_over_10pct_missing": int((per_sample_missing > 0.10).sum()),
        "plex_structure": plex_blocks(np.isnan(P)),
    }
    return out


def procan(procan_dir, uniprot):
    c = ProCanConfig()
    d = Path(procan_dir)
    X = pd.read_csv(d / MATRIX, sep="\t", index_col=0)
    m_all = pd.read_csv(d / MAPPING, sep="\t").set_index("Automatic_MS_filename").loc[X.index]
    m = m_all[~m_all["SIDM"].astype(str).str.startswith("Control")]
    rna_all = pd.read_csv(d / RNA, index_col=0)
    rna = rna_all[rna_all.notna().all(axis=1)]
    runs = {s: list(g.index) for s, g in m.groupby("SIDM", sort=True)}
    lines = sorted(s for s, r in runs.items() if len(r) >= c.min_runs and s in rna.index)
    cols = pd.Series(X.columns)
    acc = cols.str.split(";").str[0]
    sym = np.array([col.split(";")[1].replace("_HUMAN", "") if ";" in col else col for col in X.columns])
    gene_set = set(rna.columns.astype(str))
    counts = pd.Series(sym).value_counts()
    in_rna = np.array([s in gene_set for s in sym])
    unique = np.array([counts[s] == 1 for s in sym])

    # UniProt check of the entry-name matching
    u = pd.read_csv(uniprot, sep="\t", dtype=str).fillna("")
    primary = dict(zip(u["Entry"], u["Gene Names (primary)"]))
    acc_base = acc.str.replace(r"-\d+$", "", regex=True)
    prim = np.array([primary.get(a, "") for a in acc_base])
    found = prim != ""
    agree = found & (prim == sym)
    prim_counts = pd.Series(prim[found]).value_counts()
    matched_by_name = unique & in_rna
    wrong = matched_by_name & found & (prim != sym)
    # the entry name is the primary gene name of another UniProt entry: the transcript is another gene's
    prim_set = set(u["Gene Names (primary)"]) - {""}
    wrong_gene = wrong & np.array([s in prim_set for s in sym])
    alias = wrong & ~wrong_gene               # an outdated symbol of the same gene
    missed = ~in_rna & found & np.array([p in gene_set for p in prim])
    # matching through UniProt instead: primary gene, else a synonym, found among the RNA genes
    syn = dict(zip(u["Entry"], u["Gene Names (synonym)"]))
    ugene = []
    for a_, p_ in zip(acc_base, prim):
        g_ = p_ if p_ in gene_set else next((x for x in syn.get(a_, "").split() if x in gene_set), "")
        ugene.append(g_)
    ugene = np.array(ugene)
    ucounts = pd.Series(ugene[ugene != ""]).value_counts()
    uniprot_matched = np.array([g_ != "" and ucounts[g_] == 1 for g_ in ugene])

    reps = {}
    chosen_sets = []
    for r in PROCAN_REPS:
        primary_runs = [runs[s][(r - 1) % len(runs[s])] for s in lines]
        A = X.loc[primary_runs].to_numpy(dtype=np.float64)
        obs = np.isfinite(A).mean(0)
        band = (obs >= c.min_observed_fraction) & (obs <= c.max_observed_fraction)
        elig = band & matched_by_name
        extra = band & uniprot_matched & ~matched_by_name
        level = np.nanmean(A, 0)
        n_ch = int(min(c.n_proteins, elig.sum()))
        chosen_sets.append(set(np.flatnonzero(elig)) if n_ch == elig.sum() else None)
        cache = d / f"cache_procan_rep{r}.npz"
        cache_match = None
        if cache.exists() and chosen_sets[-1] is not None:
            with np.load(cache, allow_pickle=False) as f:
                cache_match = set(str(x) for x in f["protein_names"]) == set(cols[np.flatnonzero(elig)])
        reps[r] = {"observed_below_030": int((obs < c.min_observed_fraction).sum()),
                   "observed_above_097": int((obs > c.max_observed_fraction).sum()),
                   "observed_in_band": int(band.sum()), "eligible": int(elig.sum()), "chosen": n_ch,
                   "chosen_wrong_symbol": int((elig & wrong).sum()) if n_ch == elig.sum() else None,
                   "chosen_wrong_gene": int((elig & wrong_gene).sum()) if n_ch == elig.sum() else None,
                   "chosen_outdated_symbol": int((elig & alias).sum()) if n_ch == elig.sum() else None,
                   "in_band_uniprot_only": int(extra.sum()),
                   "observed_fraction": {"chosen": q(obs[elig]), "uniprot_only": q(obs[extra])},
                   "mean_log_intensity": {"chosen": q(level[elig]), "uniprot_only": q(level[extra])},
                   "missing_fraction_chosen": float(np.isnan(A[:, elig]).mean()),
                   "matches_run_cache": cache_match}
    sets = [s for s in chosen_sets if s is not None]
    examples = [f"{sym[j]} (UniProt gene {prim[j]})" for j in np.flatnonzero(wrong)[:12]]
    missed_examples = [f"{sym[j]} -> {prim[j]}" for j in np.flatnonzero(missed)[:12]]
    return {
        "runs": int(len(X)), "control_runs": int(len(m_all) - len(m)),
        "cell_lines": int(len(runs)), "cell_lines_with_rna": int(sum(s in rna.index for s in runs)),
        "rna_lines_total": int(len(rna_all)), "rna_lines_complete": int(len(rna)),
        "cell_lines_used": len(lines), "rna_genes": int(rna.shape[1]),
        "protein_entries": int(len(cols)),
        "entries_single_accession": int((cols.str.count(";") == 1).sum()),
        "entries_isoform_accession": int(acc.str.contains(r"-\d+$").sum()),
        "entry_name_symbols_shared": int((counts > 1).sum()),
        "entries_shared_entry_name": int((~unique).sum()),
        "entry_name_in_rna": int(in_rna.sum()), "entry_name_not_in_rna": int((~in_rna).sum()),
        "matched_by_entry_name": int(matched_by_name.sum()),
        "uniprot_reviewed_found": int(found.sum()),
        "entry_name_equals_primary_gene": int(agree.sum()),
        "uniprot_primary_genes_shared": int((prim_counts > 1).sum()),
        "matched_wrong_symbol": int(wrong.sum()), "wrong_examples": examples,
        "matched_wrong_gene": int(wrong_gene.sum()),
        "wrong_gene_examples": [f"{sym[j]} (UniProt gene {prim[j]})" for j in np.flatnonzero(wrong_gene)],
        "matched_outdated_symbol": int(alias.sum()),
        "uniprot_matched_unique": int(uniprot_matched.sum()),
        "missed_matches": int(missed.sum()), "missed_examples": missed_examples,
        "replicates": reps,
        "chosen_overlap_all_replicates": len(set.intersection(*sets)) if sets else None,
        "chosen_union": len(set.union(*sets)) if sets else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cptac-dir", required=True)
    ap.add_argument("--procan-dir", required=True)
    ap.add_argument("--uniprot", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"procan": procan(a.procan_dir, a.uniprot)}
    print(json.dumps(out["procan"], indent=1), flush=True)
    for cohort in ("pdac", "luad"):
        for normals in (False, True):
            key = f"{cohort}{'_ds' if normals else ''}"
            out[key] = cptac(a.cptac_dir, cohort, normals)
            print(key, json.dumps(out[key]), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
