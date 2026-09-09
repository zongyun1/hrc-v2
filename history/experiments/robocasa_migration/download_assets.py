"""Download official asset archives needed by sampled scenes, with resumable curl."""
import hashlib
import argparse
import json
import subprocess
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
assets = root / "external/robocasa/robocasa/models/assets"
links = json.loads((assets / "box_links/box_links_assets.json").read_text())
cache = root / "outputs/robocasa_migration/downloads"
cache.mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--groups", nargs="+", choices=list(links),
                    default=["textures", "fixtures_lightwheel", "objaverse", "objects_lightwheel"])
args = parser.parse_args()
for key in args.groups:
    parent = assets / "objects" if key in ("objaverse", "objects_lightwheel", "aigen_objs") else assets
    marker = cache / f"{key}.json"
    if marker.exists() and json.loads(marker.read_text()).get("parent") == str(parent):
        print("Already extracted", key, flush=True)
        continue
    url = links[key].replace("/s/", "/shared/static/") + ".zip"
    archive = cache / f"{key}.zip"
    print("DOWNLOAD", key, url, flush=True)
    if not archive.exists() or not zipfile.is_zipfile(archive):
        subprocess.run(["curl", "-fL", "--retry", "3", "--connect-timeout", "30", "-C", "-", url, "-o", str(archive)], check=True)
    parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        print("EXTRACT", key, len(z.infolist()), z.namelist()[:5], flush=True)
        for info in z.infolist():
            target = (parent / info.filename).resolve()
            if not target.is_relative_to(parent.resolve()):
                raise ValueError(f"Unsafe archive path: {info.filename}")
        z.extractall(parent)
    digest = hashlib.sha256()
    with archive.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    marker.write_text(json.dumps({"url": url, "sha256": digest.hexdigest(), "parent": str(parent)}, indent=2))
