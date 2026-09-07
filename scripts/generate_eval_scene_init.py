#!/usr/bin/env python3
"""Generate deterministic scene-init files for eval.

The output records concrete table/avatar/object initialization state for each
episode.  Eval can replay it with:

    python scripts/eval.py --task <task> --scene-init-file <output.yaml> ...
    python scripts/eval_vla.py --task <task> --scene-init-file <output.yaml> ...
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.scene_init import SCHEMA_VERSION, capture_scene_init, jsonable
from envs.tasks import TASK_MAP, TASK_ALIASES, resolve_task_class


def _load_yaml(path: str | os.PathLike | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_default_config() -> dict:
    path = Path(__file__).resolve().parent.parent / "config" / "default.yml"
    return _load_yaml(path) if path.exists() else {}


def _merge_config(*configs: dict) -> dict:
    out = {}
    for cfg in configs:
        for key, value in (cfg or {}).items():
            if isinstance(value, dict) and isinstance(out.get(key), dict):
                out[key] = _merge_config(out[key], value)
            else:
                out[key] = deepcopy(value)
    return out


def _manifest_path(path: str | os.PathLike) -> Path:
    p = Path(path)
    if p.is_dir():
        jsonl = p / "manifest.jsonl"
        yaml_path = p / "manifest.yaml"
        if jsonl.exists():
            return jsonl
        if yaml_path.exists():
            return yaml_path
    return p


def _load_manifest(path: str | os.PathLike) -> list[dict]:
    p = _manifest_path(path)
    if p.suffix == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("episodes", []))


def _task_list(args: argparse.Namespace) -> list[str]:
    tasks = []
    for item in args.task or []:
        tasks.extend(x.strip() for x in item.split(",") if x.strip())
    if args.all_tasks:
        tasks.extend(sorted([*TASK_MAP.keys(), *TASK_ALIASES.keys()]))
    seen = set()
    out = []
    for task in tasks:
        if task not in seen:
            seen.add(task)
            out.append(task)
    return out


def _episode_specs(args: argparse.Namespace) -> list[dict]:
    if args.manifest:
        rows = _load_manifest(args.manifest)
        tasks = set(_task_list(args))
        if tasks:
            rows = [row for row in rows if row.get("task") in tasks]
        if args.first_per_task:
            first = []
            seen = set()
            for row in rows:
                task = row.get("task")
                if task in seen:
                    continue
                seen.add(task)
                first.append(row)
            rows = first
        if args.limit is not None:
            rows = rows[: max(0, int(args.limit))]
        return rows

    tasks = _task_list(args)
    if not tasks:
        raise ValueError("Provide --task TASK, --all-tasks, or --manifest")
    rows = []
    for task in tasks:
        for i in range(args.episodes):
            seed = int(args.start_seed) + i
            rows.append({
                "id": f"scene_init__{task}__ep_{i:03d}__seed_{seed}",
                "task": task,
                "seed": seed,
                "episode_index": i,
            })
    return rows


def _episode_config(base_config: dict, row: dict, args: argparse.Namespace) -> dict:
    cfg = _merge_config(base_config, row.get("config_overrides") or {})
    cfg["show_viewer"] = False
    cfg["skip_reset_obs"] = bool(args.skip_reset_obs)
    cfg.setdefault("side_video", False)
    cfg.setdefault("save_video", False)
    cfg.setdefault("vla_recording", {})
    cfg["vla_recording"]["enabled"] = False
    if args.renderer:
        cfg["renderer"] = args.renderer
    if args.randomize_avatar:
        cfg["randomize_avatar"] = True
    if args.no_randomize_avatar:
        cfg["randomize_avatar"] = False
    if args.randomize_table:
        cfg["randomize_table"] = True
        cfg["debug_table_randomization"] = True
    if args.table_variant_name:
        cfg["randomize_table"] = True
        cfg["debug_table_randomization"] = True
        cfg["table_variant_name"] = args.table_variant_name
    if args.no_randomize_table:
        cfg["randomize_table"] = False
    if args.table_random_objects:
        cfg["table_random_objects"] = True
        cfg["task_irrelevant_objects"] = True
    if args.no_table_random_objects:
        cfg["table_random_objects"] = False
        cfg["task_irrelevant_objects"] = False
    if args.hard:
        cfg["hard"] = True
        cfg["difficulty"] = "hard"
    return cfg


def generate(args: argparse.Namespace) -> dict:
    base_config = _merge_config(
        _load_default_config(),
        _load_yaml(args.config),
        json.loads(args.config_json) if args.config_json else {},
    )
    rows = _episode_specs(args)
    out_episodes = []

    for index, row in enumerate(rows):
        task_name = row["task"]
        seed = int(row.get("seed", args.start_seed + index))
        print(f"[{index + 1}/{len(rows)}] task={task_name} seed={seed}", flush=True)
        resolved_task_name, TaskClass = resolve_task_class(
            task_name,
            no_human=bool((row.get("config_overrides") or {}).get("no_human", False)),
        )
        cfg = _episode_config(base_config, row, args)
        try:
            task = TaskClass(cfg)
            print("  reset: begin", flush=True)
            task.reset(seed=seed)
            print("  reset: done; capture: begin", flush=True)
            init = capture_scene_init(task, episode_id=row.get("id"), seed=seed)
            print("  capture: done", flush=True)
            episode = deepcopy(row)
            episode["task"] = resolved_task_name
            episode["init"] = init
            out_episodes.append(jsonable(episode))
        except Exception:
            err = traceback.format_exc()
            if not args.keep_failures:
                print(err, flush=True)
                raise
            episode = deepcopy(row)
            episode["task"] = resolved_task_name
            episode["init_error"] = err
            out_episodes.append(jsonable(episode))
            print(err, flush=True)

    return {
        "schema_version": "genesis_hr_bench.scene_init_set.v1",
        "scene_init_schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_manifest": str(args.manifest) if args.manifest else None,
        "episodes": out_episodes,
    }


def write_output(data: dict, path: str | os.PathLike) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".jsonl":
        with out.open("w", encoding="utf-8") as f:
            for episode in data["episodes"]:
                f.write(json.dumps(episode, sort_keys=True) + "\n")
    elif out.suffix == ".json":
        with out.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
    else:
        with out.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=None,
                        help="Existing eval manifest dir/YAML/JSONL to materialize.")
    parser.add_argument("--task", action="append", default=[],
                        help="Task name or comma-separated task names. Repeatable.")
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of manifest episodes after task filtering.")
    parser.add_argument("--first-per-task", action="store_true",
                        help="When --manifest is set, keep only the first episode "
                             "for each task after optional task filtering.")
    parser.add_argument("--config", default=None, help="Base task config YAML.")
    parser.add_argument("--config-json", default=None,
                        help="Inline JSON config overrides.")
    parser.add_argument("--renderer", choices=["rasterizer", "raytracer"], default=None)
    parser.add_argument("--randomize-avatar", action="store_true")
    parser.add_argument("--no-randomize-avatar", action="store_true")
    parser.add_argument("--randomize-table", action="store_true")
    parser.add_argument("--no-randomize-table", action="store_true")
    parser.add_argument("--table-variant-name", default=None)
    parser.add_argument("--table-random-objects", "--task-irrelevant-objects",
                        dest="table_random_objects", action="store_true")
    parser.add_argument("--no-table-random-objects", "--no-task-irrelevant-objects",
                        dest="no_table_random_objects", action="store_true")
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--skip-reset-obs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-failures", action="store_true")
    parser.add_argument("--output", required=True,
                        help="Output .yaml, .json, or .jsonl scene-init file.")
    args = parser.parse_args()

    if args.renderer == "raytracer":
        os.environ.setdefault("GENESIS_BACKEND", "gpu")

    data = generate(args)
    write_output(data, args.output)
    print(f"wrote {len(data['episodes'])} scene-init episodes -> {args.output}")


if __name__ == "__main__":
    main()
