"""Summarise the audited corrected-matching ProCan subset into procan_corrected.json (Supplementary Table S25).

Usage: python reproduce/procan_corrected_summary.py <procan_corrected/scripts_and_metadata/audited_completed_subset.json> [out.json]
The audited subset and its runs are in the Zenodo deposit of the benchmark run outputs.
"""
import hashlib
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("procan_corrected.json")
raw = src.read_bytes()
d = json.loads(raw)
r4 = lambda x: None if x is None else round(float(x), 4)
out = {"source": "procan_corrected_2026-10-11/audited_completed_subset.json", "source_sha256": hashlib.sha256(raw).hexdigest(),
       "checked_utc": d["checked_utc"], "records": len(d["artifact_manifest"]), "panels": {}}
for pan, pv in d["panels"].items():
    P = {"methods": {}, "minus_legacy": {}, "factorial": {}}
    for m, s in pv["summary"].items():
        P["methods"][m] = {"rmse": r4(s["rmse"]["mean"]), "rmse_min": r4(s["rmse"]["run_choice_min"]),
                           "rmse_max": r4(s["rmse"]["run_choice_max"]), "bias": r4(s["bias"]["mean"]),
                           "n_points": s["rmse"]["n_points"]}
    for m, v in (pv.get("corrected_minus_matched_legacy") or {}).items():
        P["minus_legacy"][m] = {"mean": r4(v["mean_diff"]), "min": r4(v["replicate_min"]), "max": r4(v["replicate_max"])}
    for m, v in (pv.get("factorial_contrasts") or {}).items():
        P["factorial"][m] = {"mean": r4(v["mean_diff"]), "min": r4(v["replicate_min"]), "max": r4(v["replicate_max"]),
                             "positive": v["replicates_reference_better"], "n": v["n_replicates"]}
    out["panels"][pan] = P
out_path.write_text(json.dumps(out, separators=(",", ":")))
print("wrote", out_path)
