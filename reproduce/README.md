# Reproducing the paper

Run everything from one working directory; the scripts expect `data_cptac/`, `data_procan/` and
`runs_revision/` there. Install the package with the reproduction extras first
(`pip install -e ".[reproduce]"`, which adds matplotlib, cptac, pytest, requests and h5py), and R 4.5
for the R comparators. The benchmark runs recorded tcsimpute 0.1.0 or, for the runs added in version 0.2.2, 0.2.1 (the code released as 0.2.2 before its version number was raised), Python 3.12.15, NumPy 2.4.4,
SciPy 1.18.1, scikit-learn 1.9.1, PyTorch 2.11.0, pandas 3.0.6, cptac 1.5.14, R 4.5.0, imputeLCMD 2.1,
missForest 1.5, proDA 1.22.1 and Pirat 1.2.6 (Supplementary Table S12).

| Step | Command | Output |
|---|---|---|
| R comparators | `Rscript reproduce/install_r_packages.R` | proDA, imputeLCMD, missForest and Pirat (Bioconductor; Pirat creates its own Python environment with basilisk on first use) in the R user library |
| CPTAC data | `python reproduce/fetch_cptac_cohort.py --cohort Pdac --out data_cptac` (and `--cohort Luad`) | `<cohort>_transcriptomics.csv`, `<cohort>_proteomics.csv`, provenance with SHA-256 |
| ProCan data | `python reproduce/fetch_procan.py data_procan` | figshare 10.6084/m9.figshare.19345397 files (about 0.6 GB) |
| ProCan RNA | `python reproduce/export_procan_rna.py` (needs `h5py`) | `data_procan/procan_transcriptomics.csv` |
| Benchmark | `tcsimpute-benchmark --grid revision --out runs_revision --cptac-dir data_cptac --procan-dir data_procan --scenarios SIM PDAC LUAD PROCAN DS-PDAC DS-LUAD sweeps ALT-PDAC-logit ALT-PDAC-sporadic ALT-LUAD-logit ALT-LUAD-sporadic` | 3,782 run folders |
| Scaling | `tcsimpute-scaling --out runs_revision/scaling.json` | runtime table (uses a GPU if present) |
| Summary | `tcsimpute-report --runs runs_revision` | `runs_revision/summary/summary_revision.json` and CSV |
| ProCan strata | `python reproduce/procan_strata.py` | `procan_strata.json` (Supplementary Table S6) |
| Replicate truth | `python reproduce/replicate_truth_sim.py` | `replicate_truth_sim.json` (Table S7) |
| Diagnostic | `python reproduce/missingness_diagnostic.py` | `missingness_diagnostic.json` (Table S8) |
| Parameter recovery | `python reproduce/parameter_recovery.py --out recovery --workers 14` then `python reproduce/recovery_tables.py recovery/parameter_recovery.json assets` | `parameter_recovery.json`, Supplementary Tables S13–S14 |
| Error control | `python reproduce/inference_calibration.py --cptac-dir data_cptac --out inference --workers 12` then `python reproduce/inference_tables.py inference/inference_calibration.json assets` | `inference_calibration.json`, Supplementary Table S15 |
| Replicate-level comparisons | `python reproduce/replicate_level.py runs_revision replicate_level.json` | `replicate_level.json` (read by `make_revision_assets.py`), Supplementary Tables S1-S4 and S18 |
| UniProt gene names | `curl -o uniprot_human_reviewed_genes.tsv "https://rest.uniprot.org/uniprotkb/stream?format=tsv&fields=accession,id,gene_primary,gene_synonym&query=%28organism_id%3A9606%29%20AND%20%28reviewed%3Atrue%29"` | reviewed human UniProt entries (the paper used the release downloaded on 10 October 2026) |
| Matching and filtering | `python reproduce/matching_summary.py --cptac-dir data_cptac --procan-dir data_procan --uniprot uniprot_human_reviewed_genes.tsv --out matching_summary.json`, `python reproduce/procan_mismatch_sensitivity.py runs_revision data_procan uniprot_human_reviewed_genes.tsv procan_mismatch.json`, then `python reproduce/matching_tables.py matching_summary.json procan_mismatch.json assets` | Supplementary Tables S19-S21 (UniProt table: reviewed human entries, fields accession, id, gene_primary, gene_synonym, from rest.uniprot.org) |
| Compact summary | `python reproduce/compact_summary.py` | `summary_compact.json` |
| Figures and tables | `python reproduce/make_revision_assets.py --summary summary_compact.json --scaling runs_revision/scaling.json --extra . --out assets` | Figures 1–4 (PDF, TIFF, JPEG) and all LaTeX tables |

Set `TCSIMPUTE_RSCRIPT` to the `Rscript` executable if it is not on the `PATH`. The R comparators
run with one thread per worker; set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` when running many
workers.
