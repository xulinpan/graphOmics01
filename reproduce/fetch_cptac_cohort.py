"""Export one CPTAC cohort's transcriptomics and proteomics (cptac package) with SHA-256 provenance."""
import argparse, hashlib, json, platform
from importlib.metadata import version
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--cohort", required=True)          # e.g. Luad, Ccrcc, Ucec
ap.add_argument("--out", required=True)
ap.add_argument("--rna-source", default="bcm")
ap.add_argument("--protein-source", default="umich")
ap.add_argument("--list-only", action="store_true")
a = ap.parse_args()
import cptac, pandas as pd
ds = getattr(cptac, a.cohort)()
print(ds.list_data_sources())
if a.list_only:
    raise SystemExit(0)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
tag = a.cohort.lower()
info = {}
for name, frame in (("transcriptomics", ds.get_transcriptomics(a.rna_source)),
                    ("proteomics", ds.get_proteomics(a.protein_source))):
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = ["|".join(str(x) for x in col) for col in frame.columns]
    frame.index = frame.index.astype(str); frame.index.name = "Patient_ID"
    path = out / f"{tag}_{name}.csv"
    frame.to_csv(path)
    info[name] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "shape": list(frame.shape), "missing_fraction": float(frame.isna().to_numpy().mean()),
                  "value_range": [float(frame.min().min()), float(frame.max().max())]}
    print(name, frame.shape, info[name]["value_range"])
meta = {"cohort": f"CPTAC {a.cohort} (cptac.{a.cohort})", "rna_source": a.rna_source,
        "protein_source": a.protein_source, "cptac_version": version("cptac"),
        "pandas": pd.__version__, "python": platform.python_version(), "files": info}
(out / f"{tag}_provenance.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(json.dumps(meta, indent=2))
