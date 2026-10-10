"""Runtime scaling of tcsimpute with the number of proteins and samples.

Data come from the S2 simulator at the requested size (genes = 1.6 x proteins + 200). The VAE is
trained for a fixed 50 epochs (no early stopping) so that the time per epoch is comparable across
sizes; the linear selection regression (``selreg``) is fitted to convergence. Devices: CPU with
1 and 8 threads, and CUDA when available.

Usage: tcsimpute-scaling --out runs/scaling.json
"""

import argparse
import json
import platform
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

SIZES = [(200, 500), (200, 2000), (200, 5000), (200, 10000), (1000, 2000), (1000, 10000)]


def run(sizes=SIZES, epochs=50, seed=1):
    import torch
    from . import baselines as B
    from .data import S2Config, build_protocol, simulate_s2
    from ..model import VAEConfig, fit_vae
    devices = [("cpu", 1), ("cpu", 8)] + ([("cuda", 8)] if torch.cuda.is_available() else [])
    results = []
    for n, p in sizes:
        cfg = replace(S2Config(), n_samples=n, n_proteins=p, n_proteins_generated=int(1.4 * p),
                      n_genes=int(1.6 * p) + 200)
        t0 = time.time()
        src = simulate_s2(seed, cfg)
        P = build_protocol(src, 101, f"scale_n{n}_p{p}")
        sim_s = time.time() - t0
        for dev, threads in devices:
            torch.set_num_threads(threads)
            vc = VAEConfig(max_epochs=epochs, eval_every=epochs, patience=10 * epochs, device=dev)
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t0 = time.time()
            fit_vae(P, vc, seed)
            if dev == "cuda":
                torch.cuda.synchronize()
            secs = time.time() - t0
            rec = {"method": "vae_full", "n_samples": n, "n_proteins": p, "n_genes": cfg.n_genes,
                   "device": dev, "threads": threads, "epochs": epochs, "seconds": round(secs, 2),
                   "seconds_per_epoch": round(secs / epochs, 4)}
            if dev == "cuda":
                rec["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
            results.append(rec)
            print(rec, flush=True)
        torch.set_num_threads(8)
        t0 = time.time()
        B.fit_selection_regression(P, limit_scale="raw", seed=seed)
        rec = {"method": "selreg", "n_samples": n, "n_proteins": p, "n_genes": cfg.n_genes, "device": "cpu",
               "threads": 8, "seconds": round(time.time() - t0, 2), "simulate_seconds": round(sim_s, 2)}
        results.append(rec)
        print(rec, flush=True)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tcsimpute-scaling", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    a = ap.parse_args(argv)
    import torch
    res = run(epochs=a.epochs)
    meta = {"platform": platform.platform(), "processor": platform.processor(), "torch": torch.__version__,
            "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    Path(a.out).write_text(json.dumps({"meta": meta, "results": res}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
