"""LaTeX tables for protein-transcript matching and filtering (Supplementary Tables S19-S21).
Usage: python matching_tables.py <matching_summary.json> <procan_mismatch.json> <out_dir>"""
import json
import sys
from pathlib import Path


def n(v):
    return "--" if v is None else f"{v:,}".replace(",", "{,}")


def f(v, d=2):
    return "--" if v is None else f"{v:.{d}f}".replace("-", "$-$")


def med(e, d=2):
    if not e:
        return "--"
    return f"{f(e['median'], d)} ({f(e['q25'], d)}--{f(e['q75'], d)})"


def cptac_table(S):
    cols = [("pdac", "PDAC"), ("luad", "LUAD"), ("pdac_ds", "PDAC (DS)"), ("luad_ds", "LUAD (DS)")]
    C = [S[k] for k, _ in cols]
    rows = [
        ("Samples with both modalities", [c["samples"]["both"] for c in C]),
        ("Protein entries (symbol $|$ Ensembl protein)", [c["protein_entries"] for c in C]),
        ("\\quad entries listing several proteins", [c["protein_id_has_separator"] for c in C]),
        ("\\quad entries without a gene symbol", [c["protein_blank_symbol"] for c in C]),
        ("\\quad symbols with several entries (excluded)", [c["protein_symbols_shared"] for c in C]),
        ("\\quad entries with a shared symbol (excluded)", [c["protein_entries_shared_symbol"] for c in C]),
        ("Transcript entries (symbol $|$ Ensembl gene)", [c["rna_entries"] for c in C]),
        ("\\quad symbols with several entries", [c["rna_symbols_multiple_entries"] for c in C]),
        ("\\quad expressed (mean $\\ge1$)", [c["rna_entries_expressed"] for c in C]),
        ("Unique protein symbols without a transcript", [c["unique_protein_no_rna_symbol"] for c in C]),
        ("Unique protein symbols, transcript not expressed", [c["unique_protein_rna_not_expressed"] for c in C]),
        ("Matched proteins", [c["matched"] for c in C]),
        ("\\quad quantified in every sample (eligible)", [c["matched_complete"] for c in C]),
        ("\\quad with missing values (excluded)", [c["matched_incomplete"] for c in C]),
        ("Drawn at random (seed 2026)", [c["drawn"] for c in C]),
        ("Kept after the observed-fraction filter, per mask", [c["final_per_mask"][0] for c in C]),
        ("\\quad kept under all four masks", [c["final_overlap_all_masks"] for c in C]),
        ("\\quad kept under at least one mask", [c["final_union"] for c in C]),
    ]
    L = ["\\begin{table}[htbp]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
         "\\caption{Matching of CPTAC proteins to transcripts and protein selection. Proteins (UMich) and "
         "transcripts (BCM) were matched by gene symbol. A symbol shared by several protein entries (different "
         "Ensembl protein identifiers of one gene, i.e.\\ protein isoforms) was excluded rather than resolved. "
         "For each symbol the most variable expressed transcript entry was used. DS: downstream scenarios, which "
         "include normal adjacent tissues, so completeness was required over tumours and normals.}",
         "\\label{tab:matchcptac}",
         "\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule",
         "Step & " + " & ".join(t for _, t in cols) + " \\\\", "\\midrule"]
    for lab, vals in rows:
        L.append(lab + " & " + " & ".join(n(v) for v in vals) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L) + "\n"


def complete_table(S):
    cols = [("pdac", "PDAC"), ("luad", "LUAD")]
    L = ["\\begin{table}[htbp]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
         "\\caption{What the CPTAC complete-protein selection represents: matched proteins quantified in every "
         "tumour (complete, eligible for the benchmark) against matched proteins with missing values in the "
         "original data (incomplete, excluded). Medians (interquartile ranges) over proteins. Cognate RNA: mean "
         "log$_2$ expression of the matched transcript. RNA--protein $\\rho$: Spearman correlation of protein "
         "log ratio and cognate transcript over tumours in which the protein was quantified. Plex groups: groups of "
         "tumours with identical missingness over incomplete proteins (agreement $>99\\%$), which correspond to "
         "TMT plexes.}",
         "\\label{tab:completesel}",
         "\\begin{tabular}{lllll}", "\\toprule",
         " & \\multicolumn{2}{c}{PDAC} & \\multicolumn{2}{c}{LUAD} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
         " & Complete & Incomplete & Complete & Incomplete \\\\", "\\midrule"]
    C = [S[k] for k, _ in cols]
    L.append("Proteins & " + " & ".join(f"{n(c['matched_complete'])} & {n(c['matched_incomplete'])}" for c in C) + " \\\\")
    L.append("Missing fraction & " + " & ".join(f"0 & {med(c['incomplete_missing_fraction'])}" for c in C) + " \\\\")
    L.append("Cognate RNA & " + " & ".join(f"{med(c['cognate_rna_mean']['complete'], 1)} & "
                                             f"{med(c['cognate_rna_mean']['incomplete'], 1)}" for c in C) + " \\\\")
    L.append("RNA--protein $\\rho$ & " + " & ".join(f"{med(c['rna_protein_spearman']['complete'])} & "
                                                     f"{med(c['rna_protein_spearman']['incomplete'])}" for c in C) + " \\\\")
    L.append("Protein SD & " + " & ".join(f"{med(c['protein_sd']['complete'])} & "
                                           f"{med(c['protein_sd']['incomplete'])}" for c in C) + " \\\\")
    L.append("\\midrule")
    L.append("\\multicolumn{5}{l}{\\textit{Missingness in the original data}} \\\\")
    L.append("Missing entries, all proteins & " + " & ".join(f"\\multicolumn{{2}}{{l}}{{{100 * c['original_missing_fraction']:.1f}\\%}}" for c in C) + " \\\\")
    L.append("Plex groups (sizes) & " + " & ".join(
        f"\\multicolumn{{2}}{{l}}{{{c['plex_structure']['groups']} ({c['plex_structure']['group_size_min']}--"
        f"{c['plex_structure']['group_size_max']} tumours)}}" for c in C) + " \\\\")
    L.append("\\quad of which in whole-plex blocks & " + " & ".join(
        f"\\multicolumn{{2}}{{l}}{{{100 * c['plex_structure']['missing_in_whole_group_blocks']:.1f}\\%}}" for c in C) + " \\\\")
    L.append("$\\rho$(cognate RNA, missing fraction), matched & " + " & ".join(
        f"\\multicolumn{{2}}{{l}}{{{f(c['spearman_rna_mean_vs_missing_fraction'])}}}" for c in C) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L) + "\n"


def procan_table(S, M):
    p = S["procan"]
    R = p["replicates"]
    ms = M["methods"]
    big = max(abs(ms[m]["without"] - ms[m]["all"]) for m in ms)
    dl = max(abs(ms[m].get("diff_vs_selreg_without", 0) - ms[m].get("diff_vs_selreg_all", 0)) for m in ms)
    reps = ["1", "2", "3", "4"]
    L = ["\\begin{table}[htbp]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
         "\\caption{Matching of ProCan proteins to transcripts and protein selection. Proteins were matched to "
         "transcripts through the gene part of the UniProt entry name (\\texttt{P37108;SRP14\\_HUMAN} $\\to$ "
         "SRP14). The check uses the primary gene names of reviewed human UniProt entries (downloaded 10 October "
         "2026). Wrong gene: the entry-name gene part is the primary gene name of a different protein, so the "
         "protein was paired with that protein's transcript. Outdated symbol: the gene part is an earlier symbol "
         "of the correct gene that the RNA data still use. Matchable through UniProt: proteins whose primary gene "
         "name, or a synonym, is a unique RNA gene. Rescoring every run without the wrong-gene proteins (models not "
         f"refitted) changed each method's mean RMSE by at most {big:.3f} and each paired difference from the linear "
         f"variant by at most {dl:.3f}.}}",
         "\\label{tab:matchprocan}",
         "\\begin{tabular}{lrrrr}", "\\toprule",
         "\\multicolumn{5}{l}{\\textit{Data}} \\\\",
         f"MS runs in the replicate matrix (control runs) & \\multicolumn{{4}}{{l}}{{{n(p['runs'])} ({n(p['control_runs'])})}} \\\\",
         f"Cell lines; with complete RNA and $\\ge2$ runs & \\multicolumn{{4}}{{l}}{{{n(p['cell_lines'])}; {n(p['cell_lines_used'])}}} \\\\",
         f"RNA genes & \\multicolumn{{4}}{{l}}{{{n(p['rna_genes'])}}} \\\\",
         f"Protein entries (all single accessions; isoform accessions) & \\multicolumn{{4}}{{l}}{{{n(p['protein_entries'])} ({n(p['entries_isoform_accession'])})}} \\\\",
         "\\midrule", "\\multicolumn{5}{l}{\\textit{Matching}} \\\\",
         f"Entry-name gene part found among RNA genes (matched) & \\multicolumn{{4}}{{l}}{{{n(p['matched_by_entry_name'])}}} \\\\",
         f"\\quad not found (excluded) & \\multicolumn{{4}}{{l}}{{{n(p['entry_name_not_in_rna'])}}} \\\\",
         f"\\quad matched to a wrong gene & \\multicolumn{{4}}{{l}}{{{n(p['matched_wrong_gene'])}}} \\\\",
         f"\\quad matched through an outdated symbol & \\multicolumn{{4}}{{l}}{{{n(p['matched_outdated_symbol'])}}} \\\\",
         f"Matchable through UniProt gene names & \\multicolumn{{4}}{{l}}{{{n(p['uniprot_matched_unique'])}}} \\\\",
         "\\midrule", "Data replicate (primary run) & 1 & 2 & 3 & 4 \\\\", "\\midrule",
         "Observed in $<30\\%$ of primary runs & " + " & ".join(n(R[r]["observed_below_030"]) for r in reps) + " \\\\",
         "Observed in $>97\\%$ of primary runs & " + " & ".join(n(R[r]["observed_above_097"]) for r in reps) + " \\\\",
         "Observed in 30--97\\% & " + " & ".join(n(R[r]["observed_in_band"]) for r in reps) + " \\\\",
         "\\quad and matched (all used) & " + " & ".join(n(R[r]["chosen"]) for r in reps) + " \\\\",
         "\\quad\\quad of which wrong gene & " + " & ".join(n(R[r]["chosen_wrong_gene"]) for r in reps) + " \\\\",
         "\\quad and matchable only through UniProt & " + " & ".join(n(R[r]["in_band_uniprot_only"]) for r in reps) + " \\\\",
         "Observed fraction, used (median) & " + " & ".join(f(R[r]["observed_fraction"]["chosen"]["median"]) for r in reps) + " \\\\",
         "Observed fraction, UniProt only (median) & " + " & ".join(f(R[r]["observed_fraction"]["uniprot_only"]["median"]) for r in reps) + " \\\\",
         f"Proteins used in all four / any replicate & \\multicolumn{{4}}{{l}}{{{n(p['chosen_overlap_all_replicates'])} / {n(p['chosen_union'])}}} \\\\",
         ]
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L) + "\n"


def main():
    S = json.loads(Path(sys.argv[1]).read_text())
    M = json.loads(Path(sys.argv[2]).read_text())
    out = Path(sys.argv[3])
    (out / "tableS_matching.tex").write_text(cptac_table(S) + "\n" + complete_table(S) + "\n" + procan_table(S, M),
                                             encoding="utf-8")
    print("wrote", out / "tableS_matching.tex")


if __name__ == "__main__":
    main()
