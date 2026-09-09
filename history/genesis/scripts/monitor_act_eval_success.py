#!/usr/bin/env python3
"""Write a compact ACT eval success-rate dashboard.

The eval wrapper only writes results.json after a whole multi-seed job
finishes. This monitor also parses job logs so partially completed seeds show
up while long video evals are still running.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = REPO_ROOT / "runs" / "act" / "eval"
JOB_LOG_ROOT = REPO_ROOT / "job_logs"
OUT_PATH = EVAL_ROOT / "success_rate_latest.md"
RECENT_SECONDS = 12 * 60 * 60

EP_RE = re.compile(r"Episode\s+\d+/\d+\s+\(seed=(\d+)\)")
JOB_RE = re.compile(r"act_eval_(\d+)\.log$")
HEADER_SEEDS_RE = re.compile(r"\bseeds:\s+(\d+)-(\d+)\b")
RATE_RE = re.compile(r"Success rate:\s+(\d+)/(\d+)\s+=\s+([0-9.]+)%")
SAFE_RE = re.compile(r"safe-distance:\s+min=([+\-0-9.]+|—)\s*cm")
COLL_RE = re.compile(r"avatar-collision:\s+(\d+)/(\d+)\s+checks hit")


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def _parse_job_log(path: Path) -> dict:
    text = _read_text(path)
    job_match = JOB_RE.search(path.name)
    job_id = job_match.group(1) if job_match else ""
    task = ""
    output = ""
    episodes_total = None
    header_episodes_total = None
    start_seed = None
    current_seed = None
    current_collision = None
    current_safe_distance_m = None
    seeds: list[dict] = []

    for raw_line in text.splitlines():
        line = raw_line.replace("\r", "")
        if line.startswith("=== ") and " task: " in line:
            task = line.split(" task: ", 1)[1].split(" ", 1)[0]
            header_seeds = HEADER_SEEDS_RE.search(line)
            if header_seeds:
                start_seed = int(header_seeds.group(1))
                end_seed = int(header_seeds.group(2))
                header_episodes_total = max(0, end_seed - start_seed + 1)
                episodes_total = header_episodes_total
        elif line.startswith("episodes:"):
            if header_episodes_total is not None:
                continue
            try:
                episodes_total = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("start_seed:"):
            try:
                start_seed = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("output:"):
            output = line.split(":", 1)[1].strip()
        else:
            if "[run_vla_eval] running eval:" in line:
                try:
                    argv = shlex.split(line.split("running eval:", 1)[1].strip())
                except ValueError:
                    argv = []
                for idx, token in enumerate(argv):
                    if token == "--task" and idx + 1 < len(argv):
                        task = argv[idx + 1]
                    elif token == "--video-dir" and idx + 1 < len(argv):
                        output = argv[idx + 1]
                    elif token == "--episodes" and idx + 1 < len(argv):
                        if header_episodes_total is not None:
                            continue
                        try:
                            episodes_total = int(argv[idx + 1])
                        except ValueError:
                            pass
                    elif token == "--start-seed" and idx + 1 < len(argv):
                        try:
                            start_seed = int(argv[idx + 1])
                        except ValueError:
                            pass
            ep = EP_RE.search(line)
            if ep:
                current_seed = int(ep.group(1))
                current_collision = None
                current_safe_distance_m = None
                continue
            coll = COLL_RE.search(line)
            if coll:
                current_collision = {
                    "n_collisions": int(coll.group(1)),
                    "n_checks": int(coll.group(2)),
                }
                continue
            safe = SAFE_RE.search(line)
            if safe:
                token = safe.group(1)
                current_safe_distance_m = None if token == "—" else float(token) / 100.0
                continue
            if "SUCCESS" in line or "FAIL" in line:
                if current_seed is not None:
                    seeds.append({
                        "seed": current_seed,
                        "success": "SUCCESS" in line,
                        "line": line.strip(),
                        "avatar_collision": current_collision,
                        "safe_distance_m": current_safe_distance_m,
                    })
                    current_seed = None
                    current_collision = None
                    current_safe_distance_m = None

    rate = RATE_RE.search(text)
    completed_successes = sum(1 for item in seeds if item["success"])
    completed = len(seeds)
    if rate:
        completed_successes = int(rate.group(1))
        completed = int(rate.group(2))
        if episodes_total is None:
            episodes_total = completed

    scheduled_seeds = None
    if episodes_total is not None and start_seed is not None:
        scheduled_seeds = set(range(start_seed, start_seed + episodes_total))

    return {
        "job_id": job_id,
        "task": task,
        "output": output,
        "episodes_total": episodes_total,
        "scheduled_seeds": scheduled_seeds,
        "completed": completed,
        "successes": completed_successes,
        "rate": completed_successes / completed if completed else None,
        "seeds": seeds,
        "clean_success_safe_distance_sum_m": _clean_success_safe_distance_sum(seeds),
        "clean_success_safe_distance_count": _clean_success_safe_distance_count(seeds),
        "clean_success_safe_distance_avg_m": _clean_success_safe_distance_avg(seeds),
        "collision_episodes": _collision_episodes(seeds),
        "log": path,
        "mtime": path.stat().st_mtime if path.exists() else 0,
    }


def _is_clean_success(item: dict) -> bool:
    if not bool(item.get("success")):
        return False
    coll = item.get("avatar_collision") or {}
    return int(coll.get("n_collisions") or 0) == 0


def _clean_success_safe_distances(seeds: list[dict]) -> list[float]:
    vals = [
        float(item["safe_distance_m"])
        for item in seeds
        if _is_clean_success(item) and item.get("safe_distance_m") is not None
    ]
    return vals


def _clean_success_safe_distance_sum(seeds: list[dict]) -> float:
    return sum(_clean_success_safe_distances(seeds))


def _clean_success_safe_distance_count(seeds: list[dict]) -> int:
    return len(_clean_success_safe_distances(seeds))


def _clean_success_safe_distance_avg(seeds: list[dict]) -> float | None:
    vals = _clean_success_safe_distances(seeds)
    return sum(vals) / len(vals) if vals else None


def _collision_episodes(seeds: list[dict]) -> int:
    n = 0
    for item in seeds:
        coll = item.get("avatar_collision") or {}
        if int(coll.get("n_collisions") or 0) > 0:
            n += 1
    return n


def _completed_results() -> list[dict]:
    rows = []
    cutoff = time.time() - RECENT_SECONDS
    paths = sorted(EVAL_ROOT.glob("*/results.json")) + sorted(EVAL_ROOT.glob("*/results_seed_*.json"))
    grouped = {}
    for path in paths:
        if path.stat().st_mtime < cutoff:
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        output = _rel(path.parent)
        task = data.get("task", path.parent.name)
        group = grouped.setdefault(
            (task, output),
            {
                "task": task,
                "output": output,
                "episodes_total": 0,
                "completed": 0,
                "successes": 0,
                "rate": None,
                "videos": 0,
                "mtime": 0,
                "clean_success_safe_distance_sum_m": 0.0,
                "clean_success_safe_distance_count": 0,
                "clean_success_safe_distance_avg_m": None,
                "collision_episodes": 0,
            },
        )
        episodes = int(data.get("episodes") or 0)
        group["episodes_total"] += episodes
        group["completed"] += episodes
        if data.get("success_rate") is not None:
            group["successes"] += int(round(float(data["success_rate"]) * episodes))
        ep_results = data.get("episode_results") or []
        clean_safe_vals = []
        for item in ep_results:
            coll = item.get("avatar_collision") or {}
            safe = item.get("safe_distance") or {}
            if (
                bool(item.get("success"))
                and int(coll.get("n_collisions") or 0) == 0
                and safe.get("safe_distance_m") is not None
            ):
                clean_safe_vals.append(float(safe["safe_distance_m"]))
        group["clean_success_safe_distance_sum_m"] += sum(clean_safe_vals)
        group["clean_success_safe_distance_count"] += len(clean_safe_vals)
        group["collision_episodes"] += sum(
            1
            for item in ep_results
            if int((item.get("avatar_collision") or {}).get("n_collisions") or 0) > 0
        )
        group["mtime"] = max(group["mtime"], path.stat().st_mtime)
    for group in grouped.values():
        group["videos"] = len(sorted((REPO_ROOT / group["output"]).glob("*.mp4")))
        if group["completed"]:
            group["rate"] = group["successes"] / group["completed"]
        safe_n = int(group.get("clean_success_safe_distance_count") or 0)
        if safe_n:
            group["clean_success_safe_distance_avg_m"] = (
                float(group.get("clean_success_safe_distance_sum_m") or 0.0) / safe_n
            )
        rows.append(group)
    rows.sort(key=lambda item: item["mtime"], reverse=True)
    return rows


def _active_and_recent_logs() -> list[dict]:
    rows = []
    cutoff = time.time() - RECENT_SECONDS
    active_job_ids = _active_eval_job_ids()
    for path in sorted(JOB_LOG_ROOT.glob("act_eval_*.log")):
        if path.stat().st_mtime < cutoff:
            continue
        row = _parse_job_log(path)
        if not row["task"] and not row["output"]:
            continue
        if row.get("job_id") not in active_job_ids and not row.get("completed"):
            continue
        output = row.get("output")
        if output and not (REPO_ROOT / output).exists():
            continue
        rows.append(row)
    rows.sort(key=lambda item: item["mtime"], reverse=True)
    return rows


def _active_eval_job_ids() -> set[str]:
    proc = subprocess.run(
        ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%i %j"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    active = set()
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "act_eval":
            active.add(parts[0])
    return active


def _fmt_rate(row: dict) -> str:
    rate = row.get("rate")
    if rate is None:
        return "-"
    return f"{100.0 * float(rate):.1f}%"


def _fmt_safe_distance(row: dict) -> str:
    val = row.get("clean_success_safe_distance_avg_m")
    if val is None:
        return "-"
    return f"{100.0 * float(val):+.1f} cm"


def _latest_per_task(rows: list[dict]) -> list[dict]:
    latest = {}
    for row in sorted(rows, key=lambda item: item.get("mtime", 0), reverse=True):
        task = row.get("task") or row.get("output") or row.get("job_id") or "-"
        if task not in latest:
            latest[task] = row
    return list(latest.values())


def _group_logs_by_output(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        key = row.get("output") or row.get("job_id") or row.get("task") or "-"
        group = grouped.setdefault(
            key,
            {
                "job_id": [],
                "task": row.get("task", ""),
                "output": row.get("output", ""),
                "episodes_total": 0,
                "completed": 0,
                "successes": 0,
                "rate": None,
                "mtime": 0,
                "scheduled_seeds": set(),
                "completed_seeds": {},
                "clean_success_safe_distance_sum_m": 0.0,
                "clean_success_safe_distance_count": 0,
                "clean_success_safe_distance_avg_m": None,
                "collision_episodes": 0,
                "clean_success_safe_distance_by_seed": {},
                "collision_by_seed": {},
            },
        )
        if row.get("job_id"):
            group["job_id"].append(row["job_id"])
        group["task"] = group["task"] or row.get("task", "")
        group["output"] = group["output"] or row.get("output", "")
        if row.get("scheduled_seeds"):
            group["scheduled_seeds"].update(row["scheduled_seeds"])
        elif row.get("episodes_total") is not None:
            group["episodes_total"] += int(row["episodes_total"])
        for item in row.get("seeds", []):
            group["completed_seeds"][item["seed"]] = bool(item["success"])
            coll = item.get("avatar_collision") or {}
            has_collision = int(coll.get("n_collisions") or 0) > 0
            group["collision_by_seed"][item["seed"]] = has_collision
            if (
                bool(item.get("success"))
                and not has_collision
                and item.get("safe_distance_m") is not None
            ):
                group["clean_success_safe_distance_by_seed"][item["seed"]] = float(item["safe_distance_m"])
        if not row.get("seeds"):
            group["completed"] += int(row.get("completed", 0))
            group["successes"] += int(row.get("successes", 0))
            group["clean_success_safe_distance_sum_m"] += float(
                row.get("clean_success_safe_distance_sum_m") or 0.0
            )
            group["clean_success_safe_distance_count"] += int(
                row.get("clean_success_safe_distance_count") or 0
            )
            group["collision_episodes"] += int(row.get("collision_episodes", 0))
        group["mtime"] = max(group["mtime"], row.get("mtime", 0))

    out = []
    for group in grouped.values():
        if group["scheduled_seeds"]:
            group["episodes_total"] = len(group["scheduled_seeds"])
        if group["completed_seeds"]:
            group["completed"] = len(group["completed_seeds"])
            group["successes"] = sum(1 for success in group["completed_seeds"].values() if success)
        if group["clean_success_safe_distance_by_seed"]:
            group["clean_success_safe_distance_sum_m"] = sum(
                group["clean_success_safe_distance_by_seed"].values()
            )
            group["clean_success_safe_distance_count"] = len(
                group["clean_success_safe_distance_by_seed"]
            )
        safe_n = int(group.get("clean_success_safe_distance_count") or 0)
        if safe_n:
            group["clean_success_safe_distance_avg_m"] = (
                float(group.get("clean_success_safe_distance_sum_m") or 0.0) / safe_n
            )
        if group["collision_by_seed"]:
            group["collision_episodes"] = sum(1 for hit in group["collision_by_seed"].values() if hit)
        if group["completed"]:
            group["rate"] = group["successes"] / group["completed"]
        group.pop("scheduled_seeds", None)
        group.pop("completed_seeds", None)
        group.pop("clean_success_safe_distance_by_seed", None)
        group.pop("collision_by_seed", None)
        group["job_id"] = ",".join(group["job_id"][:3]) + (
            f"+{len(group['job_id']) - 3}" if len(group["job_id"]) > 3 else ""
        )
        out.append(group)
    out.sort(key=lambda item: item.get("mtime", 0), reverse=True)
    return out


def main() -> None:
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    log_rows = _latest_per_task(_group_logs_by_output(_active_and_recent_logs()))
    result_rows = _latest_per_task(_completed_results())
    newest_by_task = {}
    for row in [*log_rows, *result_rows]:
        task = row.get("task")
        if not task:
            continue
        prev = newest_by_task.get(task)
        if prev is None or row.get("mtime", 0) > prev.get("mtime", 0):
            newest_by_task[task] = row
    newest_outputs = {
        row.get("output") for row in newest_by_task.values() if row.get("output")
    }
    log_rows = [row for row in log_rows if row.get("output") in newest_outputs]
    result_rows = [row for row in result_rows if row.get("output") in newest_outputs]
    active_outputs = {row.get("output") for row in log_rows if row.get("output")}
    result_rows = [row for row in result_rows if row.get("output") not in active_outputs]

    lines = [
        "# ACT Eval Success Rates",
        "",
        f"Updated: `{now}`",
        "",
        "Only the newest recent eval row per task is shown.",
        "",
        "## Active / Recent Eval Jobs",
        "",
        "| Job | Task | Completed | Success | Rate | Avg Safe Dist (Clean Success) | Collision Eps | Output |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in log_rows[:20]:
        total = row.get("episodes_total")
        completed = row["completed"]
        completed_s = f"{completed}/{total}" if total else str(completed)
        output = row.get("output") or "-"
        lines.append(
            f"| `{row['job_id']}` | `{row['task'] or '-'}` | {completed_s} | "
            f"{row['successes']} | {_fmt_rate(row)} | {_fmt_safe_distance(row)} | "
            f"{row.get('collision_episodes', 0)} | `{output}` |"
        )
    if not log_rows:
        lines.append("| - | - | - | - | - | - | - | - |")

    lines += [
        "",
        "## Latest Completed Results",
        "",
        "| Task | Episodes | Rate | Avg Safe Dist (Clean Success) | Collision Eps | Videos | Output |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in result_rows[:30]:
        lines.append(
            f"| `{row['task']}` | {row.get('episodes_total', '-')} | "
            f"{_fmt_rate(row)} | {_fmt_safe_distance(row)} | "
            f"{row.get('collision_episodes', 0)} | {row.get('videos', 0)} | `{row['output']}` |"
        )
    if not result_rows:
        lines.append("| - | - | - | - | - | - | - |")

    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {_rel(OUT_PATH)}")


if __name__ == "__main__":
    main()
