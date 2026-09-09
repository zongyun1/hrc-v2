"""Fetch pinned upstream checkouts without modifying an existing different checkout."""
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sources = json.loads((Path(__file__).parent / "sources.json").read_text())
for name in ("robocasa", "robosuite"):
    spec = sources[name]
    dest = root / "external" / name
    if not dest.exists():
        subprocess.run(["git", "clone", "--depth", "1", spec["url"], str(dest)], check=True)
        actual = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
        if actual != spec["commit"]:
            subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", spec["commit"]], check=True)
            subprocess.run(["git", "-C", str(dest), "checkout", "--detach", spec["commit"]], check=True)
    actual = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
    if actual != spec["commit"]:
        raise RuntimeError(f"Existing {name} checkout differs: {actual}")
    print(name, actual, flush=True)
