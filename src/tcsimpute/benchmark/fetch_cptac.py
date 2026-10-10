"""Download CPTAC PDAC transcriptomics and proteomics with the ``cptac`` package and export them,
with SHA-256 provenance, for the CPTAC benchmark. Requires ``pip install tcsimpute[cptac]``."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tcsimpute-fetch-cptac", description=__doc__)
    ap.add_argument("--out", required=True, help="Output directory (pass it to tcsimpute-benchmark --cptac-dir).")
    ap.add_argument("--rna-source", default="bcm")
    ap.add_argument("--protein-source", default="umich")
    ap.add_argument("--list-only", action="store_true", help="Only list the available data sources.")
    a = ap.parse_args(argv)
    import cptac
    import pandas as pd
    pdac = cptac.Pdac()
    print(pdac.list_data_sources())
    if a.list_only:
        return 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    info = {}
    for name, frame in (("transcriptomics", pdac.get_transcriptomics(a.rna_source)),
                        ("proteomics", pdac.get_proteomics(a.protein_source))):
        frame = frame.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = ["|".join(str(x) for x in col) for col in frame.columns]
        frame.index = frame.index.astype(str)
        frame.index.name = "Patient_ID"
        path = out / f"pdac_{name}.csv"
        frame.to_csv(path)
        info[name] = {"file": path.name, "sha256": _sha256(path), "shape": list(frame.shape),
                      "missing_fraction": float(frame.isna().to_numpy().mean())}
        print(name, frame.shape)
    meta = {"cohort": "CPTAC PDAC (cptac.Pdac)", "rna_source": a.rna_source,
            "protein_source": a.protein_source, "cptac_version": version("cptac"),
            "pandas": pd.__version__, "python": platform.python_version(), "files": info}
    (out / "provenance.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
