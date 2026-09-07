"""Run latest-Genesis integration smoke checks.

This is intentionally a subprocess harness. It runs the benchmark code under
the latest Genesis environment while allowing the repo's normal Python
environment to remain unchanged.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_LATEST_PY = REPO / ".venv-genesis-latest" / "bin" / "python"


def _latest_env() -> dict[str, str]:
    env = dict(os.environ)
    tmp = Path(env.get("TMPDIR", "/tmp"))
    env.setdefault("GENESIS_BACKEND", "cpu")
    env.setdefault("GENESIS_SOFTWARE_RENDER", "1")
    env.setdefault("NUMBA_CACHE_DIR", str(tmp / "genesis_hr_bench_numba_cache"))
    env.setdefault("MPLCONFIGDIR", str(tmp / "genesis_hr_bench_mpl_cache"))
    env.setdefault("XDG_CACHE_HOME", str(tmp / "genesis_hr_bench_xdg_cache"))
    return env


def _run(label: str, cmd: list[str], timeout: float | None, env: dict[str, str]) -> bool:
    print(f"=== {label} ===", flush=True)
    print(" ".join(cmd), flush=True)
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if stdout:
            print(stdout, end="" if stdout.endswith("\n") else "\n", flush=True)
        print(f"TIMEOUT {label} after {timeout}s", flush=True)
        return False

    elapsed = time.monotonic() - started
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)
    status = "PASS" if proc.returncode == 0 else "FAIL"
    print(f"{status} {label} elapsed={elapsed:.2f}s returncode={proc.returncode}", flush=True)
    return proc.returncode == 0


def _version_cmd(py: Path) -> list[str]:
    code = (
        "import importlib.metadata as md\n"
        "for name in ('genesis-world', 'gs-nyx', 'gs-nyx-plugin'):\n"
        "    try:\n"
        "        print(f'{name}=={md.version(name)}')\n"
        "    except md.PackageNotFoundError:\n"
        "        print(f'{name}=<missing>')\n"
    )
    return [str(py), "-c", code]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(os.environ.get("LATEST_GENESIS_PY", DEFAULT_LATEST_PY)),
        help="Python executable for the latest Genesis environment.",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        default=["franka"],
        help="Robots for scripts/probe_latest_genesis_robot_only.py.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["blocks_ranking_rgb", "put_object_cabinet"],
        help="Robot-only task probes for scripts/probe_latest_genesis_robot_tasks.py.",
    )
    parser.add_argument("--full-robot-sweep", action="store_true")
    parser.add_argument("--full-task-sweep", action="store_true")
    parser.add_argument("--skip-task-sweep", action="store_true")
    parser.add_argument("--task-timeout", type=float, default=120.0)
    parser.add_argument("--robot-timeout", type=float, default=120.0)
    parser.add_argument("--steps", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.python.is_file():
        print(f"missing latest Genesis Python: {args.python}", file=sys.stderr)
        return 2

    env = _latest_env()
    failures: list[str] = []

    checks = [
        ("package versions", _version_cmd(args.python), 30.0),
    ]

    robot_cmd = [
        str(args.python),
        "scripts/probe_latest_genesis_robot_only.py",
    ]
    if env.get("GENESIS_SOFTWARE_RENDER", "").strip() == "1":
        robot_cmd.append("--software-render")
    if not args.full_robot_sweep:
        robot_cmd.extend(["--robots", *args.robots])
    checks.append(("direct robot smoke", robot_cmd, args.robot_timeout))

    if not args.skip_task_sweep:
        task_cmd = [
            str(args.python),
            "scripts/probe_latest_genesis_robot_tasks.py",
            "--timeout",
            str(args.task_timeout),
            "--steps",
            str(args.steps),
        ]
        if not args.full_task_sweep:
            task_cmd.extend(["--tasks", *args.tasks])
        checks.append(("robot-only task smoke", task_cmd, args.task_timeout * max(1, len(args.tasks)) + 30.0))

    for label, cmd, timeout in checks:
        if not _run(label, cmd, timeout, env):
            failures.append(label)

    if failures:
        print("FAILED_CHECKS " + " ".join(failures), flush=True)
        return 1
    print("LATEST_GENESIS_INTEGRATION_SMOKE_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
