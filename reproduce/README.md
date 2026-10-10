# Reproducing the paper

Run everything from one working directory; the scripts expect `data_cptac/`, `data_procan/` and
`runs_revision/` there. Install the package first (`pip install -e ".[all]"`).

| Step | Command | Output |
|---|---|---|
| R comparators | `Rscript reproduce/install_r_packages.R` | proDA, imputeLCMD, missForest in the R user library |
| CPTAC data | `python reproduce/fetch_cptac_cohort.py --cohort Pdac --out data_cptac` (and `--cohort Luad`) | `<cohort>_transcriptomics.csv`, `<cohort>_proteomics.csv`, provenance with SHA-256 |
| ProCan data | `python reproduce/fetch_procan.py data_procan` | figshare 10.6084/m9.figshare.19345397 files (about 0.6 GB) |
| ProCan RNA | `python reproduce/export_procan_rna.py` (needs `h5py`) | `data_procan/procan_transcriptomics.csv` |
| Benchmark | `tcsimpute-benchmark --grid revision --out runs_revision --cptac-dir data_cptac --procan-dir data_procan --scenarios SIM PDAC LUAD PROCAN DS-PDAC DS-LUAD sweeps` | 3,084 run folders |
| Scaling | `tcsimpute-scaling --out runs_revision/scaling.json` | runtime table (uses a GPU if present) |
| Summary | `tcsimpute-report --runs runs_revision` | `runs_revision/summary/summary_revision.json` and CSV |
| ProCan strata | `python reproduce/procan_strata.py` | `procan_strata.json` (Supplementary Table S6) |
| Replicate truth | `python reproduce/replicate_truth_sim.py` | `replicate_truth_sim.json` (Table S7) |
| Diagnostic | `python reproduce/missingness_diagnostic.py` | `missingness_diagnostic.json` (Table S8) |
| Parameter recovery | `python reproduce/parameter_recovery.py --out recovery --workers 14` then `python reproduce/recovery_tables.py recovery/parameter_recovery.json assets` | `parameter_recovery.json`, Supplementary Tables S13–S14 |
| Error control | `python reproduce/inference_calibration.py --cptac-dir data_cptac --out inference --workers 12` then `python reproduce/inference_tables.py inference/inference_calibration.json assets` | `inference_calibration.json`, Supplementary Table S15 |
| Compact summary | `python reproduce/compact_summary.py` | `summary_compact.json` |
| Figures and tables | `python reproduce/make_revision_assets.py --summary summary_compact.json --scaling runs_revision/scaling.json --extra . --out assets` | Figures 1–4 (PDF, TIFF, JPEG) and all LaTeX tables |

Set `TCSIMPUTE_RSCRIPT` to the `Rscript` executable if it is not on the `PATH`. The R comparators
run with one thread per worker; set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` when running many
workers.
