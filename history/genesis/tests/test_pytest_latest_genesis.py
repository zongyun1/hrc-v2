import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
DEFAULT_LATEST_PY = REPO / ".venv-genesis-latest" / "bin" / "python"


@pytest.mark.skipif(
    os.environ.get("GENESIS_HR_BENCH_LATEST_GENESIS_SMOKE") != "1",
    reason="set GENESIS_HR_BENCH_LATEST_GENESIS_SMOKE=1 to run latest Genesis smoke",
)
def test_latest_genesis_robot_only_smoke():
    latest_py = Path(os.environ.get("LATEST_GENESIS_PY", DEFAULT_LATEST_PY))
    if not latest_py.is_file():
        pytest.skip(f"latest Genesis Python not found: {latest_py}")

    cmd = [
        sys.executable,
        str(REPO / "scripts" / "check_latest_genesis_integration.py"),
        "--python",
        str(latest_py),
        "--robots",
        "franka",
        "--tasks",
        "blocks_ranking_rgb",
        "--steps",
        "1",
    ]
    env = os.environ.copy()
    env.setdefault("GENESIS_BACKEND", "gpu")
    env.setdefault("GENESIS_SOFTWARE_RENDER", "0")
    env.pop("PYOPENGL_PLATFORM", None)
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=700,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout
