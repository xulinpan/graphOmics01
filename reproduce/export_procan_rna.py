"""Export the transcriptomics view of MultiOmics_DIANN.hdf5 (Goncalves et al. 2022, figshare
10.6084/m9.figshare.19345397) to CSV: cell lines (SIDM) x genes. Run with the .venv_io interpreter."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

d = Path("data_procan")
f = h5py.File(d / "MultiOmics_DIANN.hdf5", "r")
dec = lambda a: [x.decode() for x in a]
genes = dec(f["features/transcriptomics"][:])
frames = []
for g in dec(f["groups/groups"][:]):
    X = f[f"data/transcriptomics/{g}"][:]
    frames.append(pd.DataFrame(X, index=dec(f[f"samples/{g}"][:]), columns=genes))
rna = pd.concat(frames)
rna.index.name = "SIDM"
out = d / "procan_transcriptomics.csv"
rna.to_csv(out, float_format="%.5g")
miss_rows = rna.isna().all(1)
print("shape", rna.shape, "samples without RNA", int(miss_rows.sum()), "missing frac", float(rna.isna().to_numpy().mean()))
print("value range", np.nanmin(rna.to_numpy()), np.nanmax(rna.to_numpy()))
print("per-gene mean (first 5)", rna.mean(0).head().round(3).to_dict())
print("per-gene sd median", float(np.nanmedian(rna.std(0))))
print("index head", list(rna.index[:5]), "genes head", genes[:5])
h = hashlib.sha256(out.read_bytes()).hexdigest()
(d / "procan_transcriptomics.json").write_text(json.dumps({"source": "MultiOmics_DIANN.hdf5 data/transcriptomics",
    "shape": list(rna.shape), "sha256": h}, indent=2))
