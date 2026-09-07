#!/usr/bin/env python3
"""Validate reset-only initialization for an eval-set manifest slice."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        path = path / "manifest.jsonl"
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_config(episode: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(episode.get("config_overrides", {}))
    for key in ("renderer", "setting", "action_type", "action_substeps"):
        if key in episode:
            cfg[key] = episode[key]
    return cfg


def validate_one(episode: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    result = {
        "id": episode["id"],
        "eval_set": episode.get("eval_set"),
        "category": episode.get("category"),
        "task": episode["task"],
        "seed": int(episode["seed"]),
        "episode_index": episode.get("episode_index"),
        "ok": False,
        "elapsed_s": None,
        "error": None,
    }
    task = None
    try:
        from envs.tasks import resolve_task_class

        _, TaskClass = resolve_task_class(episode["task"], no_human=False)
        task = TaskClass(build_config(episode))
        task.reset(seed=int(episode["seed"]))
        result["ok"] = True
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        result["elapsed_s"] = time.time() - t0
        del task
    return result


def validate_one_in_subprocess(
    eval_set: Path,
    absolute_index: int,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"worker_{absolute_index:06d}.json"
    log_path = output_dir / f"worker_{absolute_index:06d}.log"
    cmd = [
        sys.executable,
        __file__,
        "--eval-set",
        str(eval_set),
        "--worker-index",
        str(absolute_index),
        "--single-result-path",
        str(result_path),
    ]
    with log_path.open("w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))

    episodes = load_eval_set(eval_set)
    episode = episodes[absolute_index]
    return {
        "id": episode["id"],
        "eval_set": episode.get("eval_set"),
        "category": episode.get("category"),
        "task": episode["task"],
        "seed": int(episode["seed"]),
        "episode_index": episode.get("episode_index"),
        "ok": False,
        "elapsed_s": None,
        "error": (
            f"worker subprocess exited with code {proc.returncode}; "
            f"log={log_path}"
        ),
    }


def write_outputs(output_dir: Path, results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(results)
    ok = sum(1 for row in results if row.get("ok"))
    by_task: dict[str, dict[str, Any]] = {}
    for row in results:
        item = by_task.setdefault(row["task"], {"total": 0, "ok": 0})
        item["total"] += 1
        item["ok"] += int(bool(row.get("ok")))
    summary = {
        "total": total,
        "ok": ok,
        "failures": total - ok,
        "by_task": by_task,
    }
    (output_dir / "results.jsonl").write_text(
        "".join(json.dumps(row, default=str) + "\n" for row in results),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"init validation: {ok}/{total} ok; failures={total - ok}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--subprocess-per-episode",
        action="store_true",
        help="Run each reset in a fresh child process to isolate Genesis segfaults.",
    )
    parser.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-result-path", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is None and args.worker_index is None:
        raise SystemExit("--output-dir is required unless --worker-index is used")
    all_episodes = load_eval_set(args.eval_set)
    if args.worker_index is not None:
        row = validate_one(all_episodes[int(args.worker_index)])
        if args.single_result_path is not None:
            args.single_result_path.parent.mkdir(parents=True, exist_ok=True)
            args.single_result_path.write_text(json.dumps(row, default=str), encoding="utf-8")
        print(json.dumps(row, default=str), flush=True)
        raise SystemExit(0 if row["ok"] else 1)

    episodes = all_episodes
    start = max(0, int(args.start_index))
    end = args.end_index if args.end_index is None else max(start, int(args.end_index))
    indexed_episodes = list(enumerate(episodes[start:end], start=start))
    if args.limit is not None:
        indexed_episodes = indexed_episodes[: max(0, int(args.limit))]
    episodes = [episode for _idx, episode in indexed_episodes]

    print(
        f"Validating {len(episodes)} initializations from {args.eval_set} "
        f"(start_index={start}, end_index={end})",
        flush=True,
    )
    results = []
    worker_dir = args.output_dir / "worker_logs"
    for i, (absolute_index, episode) in enumerate(indexed_episodes, start=1):
        print(
            f"[{i}/{len(episodes)}] {episode['task']} seed={episode['seed']}",
            flush=True,
        )
        if args.subprocess_per_episode:
            row = validate_one_in_subprocess(args.eval_set, absolute_index, worker_dir)
        else:
            row = validate_one(episode)
        results.append(row)
        status = "OK" if row["ok"] else "FAIL"
        elapsed = row.get("elapsed_s")
        elapsed_s = f"{elapsed:.1f}s" if elapsed is not None else "unknown"
        print(f"  {status} elapsed={elapsed_s}", flush=True)
        if row.get("error"):
            print(row["error"], flush=True)
        write_outputs(args.output_dir, results)


if __name__ == "__main__":
    main()
