#!/usr/bin/env python3
"""Validate per-task scene-init YAMLs and build combined eval-set YAML."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parent
PER_TASK = ROOT / "per_task"
MANIFEST = ROOT / "manifest.jsonl"
EXPECTED_PER_TASK = 100


def load_manifest_rows() -> list[dict]:
    rows = []
    with MANIFEST.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    manifest_rows = load_manifest_rows()
    by_task: dict[str, list[dict]] = {}
    task_order: list[str] = []
    for row in manifest_rows:
        task = row["task"]
        if task not in by_task:
            task_order.append(task)
            by_task[task] = []
        by_task[task].append(row)

    all_episodes = []
    bad = []
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tasks": {},
    }

    for task in task_order:
        path = PER_TASK / f"{task}.yaml"
        if not path.exists():
            bad.append({"task": task, "error": "missing per-task yaml"})
            summary["tasks"][task] = {"episodes": 0, "valid": 0, "bad": EXPECTED_PER_TASK}
            continue
        data = yaml.safe_load(path.read_text()) or {}
        episodes = list(data.get("episodes") or [])
        invalid = [
            i for i, ep in enumerate(episodes)
            if "init" not in ep or "init_error" in ep
        ]
        if len(episodes) != EXPECTED_PER_TASK or invalid:
            bad.append({
                "task": task,
                "episodes": len(episodes),
                "invalid_indices": invalid[:20],
                "invalid_count": len(invalid),
            })
        summary["tasks"][task] = {
            "episodes": len(episodes),
            "valid": len(episodes) - len(invalid),
            "bad": len(invalid),
            "path": str(path.relative_to(ROOT)),
        }
        all_episodes.extend(episodes)

    combined = {
        "schema_version": "genesis_hr_bench.scene_init_set.v1",
        "scene_init_schema_version": "genesis_hr_bench.scene_init.v1",
        "eval_set": "eval_v2_100_scene_init",
        "source_manifest": str(MANIFEST.relative_to(ROOT)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "episodes": all_episodes,
    }
    (ROOT / "scene_init.yaml").write_text(yaml.safe_dump(combined, sort_keys=False))
    (ROOT / "validation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (ROOT / "validation_errors.json").write_text(json.dumps(bad, indent=2, sort_keys=True))

    print(f"tasks={len(task_order)} episodes={len(all_episodes)} bad_tasks={len(bad)}")
    if bad:
        print(f"validation failed: {ROOT / 'validation_errors.json'}")
        return 1
    print(f"wrote {ROOT / 'scene_init.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
