"""Download ProCan-DepMapSanger DIA protein matrices (figshare 10.6084/m9.figshare.19345397) with SHA-256 provenance."""
import hashlib, json, sys, time
from pathlib import Path
import requests

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data_procan")
out.mkdir(parents=True, exist_ok=True)
FILES = {
    "ProCan-DepMapSanger_mapping_file_averaged.txt": "https://ndownloader.figshare.com/files/34411133",
    "ProCan-DepMapSanger_mapping_file_replicates.txt": "https://ndownloader.figshare.com/files/34411136",
    "ProCan-DepMapSanger_protein_matrix_6692_averaged.txt": "https://ndownloader.figshare.com/files/34411151",
    "ProCan-DepMapSanger_protein_matrix_6692_replicates.txt": "https://ndownloader.figshare.com/files/34411169",
    "MultiOmics_DIANN.hdf5": "https://ndownloader.figshare.com/files/34411016",
}
prov = {}
for name, url in FILES.items():
    path = out / name
    if not path.exists():
        t0 = time.time()
        with requests.get(url, stream=True, timeout=120, allow_redirects=True) as r:
            r.raise_for_status()
            tmp = path.with_suffix(path.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            tmp.rename(path)
        print(f"downloaded {name} in {time.time() - t0:.0f}s", flush=True)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    prov[name] = {"url": url, "bytes": path.stat().st_size, "sha256": h.hexdigest()}
    print(name, prov[name]["bytes"], flush=True)
(out / "provenance.json").write_text(json.dumps(prov, indent=2))
print("done")
