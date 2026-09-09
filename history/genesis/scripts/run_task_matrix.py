"""Run one scripted episode for every registered benchmark task.

The matrix is intentionally serial: Genesis owns a large GPU scene and
parallel processes make results less useful on a single-GPU host. Each task
gets an isolated output directory and log, and failures do not stop the run.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def registered_tasks() -> list[str]:
    source = (ROOT / "envs/tasks/__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "TASK_MAP" for target in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            raise RuntimeError("TASK_MAP is not a literal dictionary")
        return [key.value for key in node.value.keys if isinstance(key, ast.Constant)]
    raise RuntimeError("TASK_MAP not found")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", action="append", help="Run only this task (repeatable).")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds per task.")
    parser.add_argument("--save-root", default="outputs/task_matrix")
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip frame rendering/encoding while preserving the full rollout.",
    )
    parser.add_argument(
        "--video-stride",
        type=int,
        default=None,
        help="Capture one video frame every N simulation steps.",
    )
    args = parser.parse_args()

    tasks = args.task or registered_tasks()
    save_root = (ROOT / args.save_root).resolve()
    save_root.mkdir(parents=True, exist_ok=True)
    summary = []
    python = sys.executable
    env = os.environ.copy()
    env.setdefault("GENESIS_BACKEND", "gpu")
    env.setdefault("GENESIS_SOFTWARE_RENDER", "0")

    for index, task in enumerate(tasks, 1):
        task_dir = save_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        log_path = task_dir / "run.log"
        command = [
            python, "scripts/collect.py", "--task", task,
            "--episodes", str(args.episodes), "--start-seed", "0",
            "--save-dir", str(task_dir),
        ]
        if args.no_video:
            command.append("--no-video")
        elif args.video_stride is not None:
            command.extend(["--video-stride", str(max(1, args.video_stride))])
        started = time.time()
        status = "failed"
        returncode = None
        error = None
        print(f"[{index}/{len(tasks)}] START {task}", flush=True)
        try:
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command, cwd=ROOT, env=env, stdout=log,
                    stderr=subprocess.STDOUT, timeout=args.timeout,
                )
            returncode = result.returncode
            status = "passed" if returncode == 0 and any(task_dir.glob("seed_*.pkl")) else "failed"
        except subprocess.TimeoutExpired:
            error = f"timeout after {args.timeout}s"
            status = "timeout"
        except Exception as exc:  # keep the matrix moving
            error = f"{type(exc).__name__}: {exc}"
        record = {
            "task": task, "status": status, "returncode": returncode,
            "elapsed_s": round(time.time() - started, 1),
            "log": str(log_path), "error": error,
        }
        summary.append(record)
        print(f"[{index}/{len(tasks)}] {status.upper()} {task} ({record['elapsed_s']}s)", flush=True)

    summary_path = save_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    passed = sum(item["status"] == "passed" for item in summary)
    print(f"DONE passed={passed} failed={len(summary) - passed} summary={summary_path}", flush=True)
    return 0 if passed == len(summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
