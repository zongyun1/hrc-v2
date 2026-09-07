from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LATEST_PYTHON = Path(
    os.environ.get(
        "LATEST_GENESIS_PY",
        REPO_ROOT / ".venv-genesis-latest" / "bin" / "python",
    )
)


@pytest.mark.skipif(
    os.environ.get("GENESIS_HR_BENCH_LATEST_GENESIS_POUR_WATER_SMOKE") != "1",
    reason="set GENESIS_HR_BENCH_LATEST_GENESIS_POUR_WATER_SMOKE=1 to run Genesis pour-water smoke test",
)
def test_latest_genesis_pour_water_no_avatar_collect(tmp_path: Path) -> None:
    if not LATEST_PYTHON.is_file():
        pytest.skip(f"latest Genesis Python not found: {LATEST_PYTHON}")

    env = os.environ.copy()
    env.setdefault("GENESIS_BACKEND", "gpu")
    env.setdefault("GENESIS_SOFTWARE_RENDER", "0")
    env.pop("PYOPENGL_PLATFORM", None)

    cmd = [
        str(LATEST_PYTHON),
        str(REPO_ROOT / "scripts" / "collect.py"),
        "--task",
        "pour_water",
        "--config",
        str(REPO_ROOT / "config" / "latest_genesis" / "pour_water_no_avatar.yml"),
        "--episodes",
        "1",
        "--save-dir",
        str(tmp_path / "pour_water"),
        "--start-seed",
        "0",
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=700,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout
