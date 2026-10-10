"""Run reference R implementations (imputeLCMD, missForest, proDA) on a raw-scale matrix.

These methods are transductive: they receive every sample's observed values (never a hidden
value) and impute all of them; the benchmark scores the test rows only. Set the environment
variable ``TCSIMPUTE_RSCRIPT`` to the Rscript executable if it is not on the PATH.
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).with_name("rmethods.R")


def rscript():
    return os.environ.get("TCSIMPUTE_RSCRIPT", shutil.which("Rscript") or "Rscript")


def r_impute(method, X_raw, covariates=None, seed=0, timeout=4 * 3600, **params):
    """Impute ``X_raw`` (samples x proteins, NaN = missing) with an R method; returns (matrix, info)."""
    with tempfile.TemporaryDirectory(prefix="tcs_r_") as d:
        d = Path(d)
        np.savetxt(d / "protein.csv", np.asarray(X_raw, float), delimiter=",", fmt="%.10g")
        if covariates is not None:
            np.savetxt(d / "covariates.csv", np.asarray(covariates, float), delimiter=",", fmt="%.10g")
        (d / "params.json").write_text(json.dumps({"method": method, "seed": int(seed), **params}))
        proc = subprocess.run([rscript(), "--vanilla", str(SCRIPT), str(d)], capture_output=True,
                              text=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"R {method} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")
        out = np.loadtxt(d / "imputed.csv", delimiter=",", ndmin=2)
        info = json.loads((d / "info.json").read_text())
    if out.shape != np.shape(X_raw):
        raise RuntimeError(f"R {method} returned shape {out.shape}, expected {np.shape(X_raw)}")
    return out, info
