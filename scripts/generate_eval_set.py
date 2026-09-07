#!/usr/bin/env python3
"""Generate initialization-only eval-set manifests.

The eval set intentionally stores task initialization specs only: task name,
seed, category, and deterministic environment defaults. It does not store
rollouts, observations, actions, videos, or success labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "genesis_hr_bench.eval_set.v1"
DEFAULT_NAME = "eval_v0_smoke"

DEFAULTS: dict[str, Any] = {
    "renderer": "rasterizer",
    "setting": "sim",
    "action_type": "ee",
    "action_substeps": 25,
    "max_steps": 300,
    "config_overrides": {
        "eval_mode": True,
        "randomize_avatar": False,
        "randomize_table": False,
        "track_avatar_collision": True,
        "save_video": False,
        "vla_recording": {"enabled": False},
        "neutral_avatar_strict_layout": True,
        "neutral_avatar_region_candidates": 32,
    },
}

BASE_EPISODES: list[dict[str, Any]] = [
    {"category": "intent", "task": "take_from_human_easy", "seed": 1},
    {"category": "intent", "task": "deliver_to_human_easy", "seed": 135},
    {"category": "intent", "task": "take_from_human_safety", "seed": 0},
    {"category": "intent", "task": "stamp_documents", "seed": 1},
    {"category": "intent", "task": "pour_water", "seed": 0},
    {"category": "intent", "task": "oil_bottle_recovery", "seed": 38},
    {"category": "intent", "task": "open_microwave", "seed": 30},
    {"category": "intent", "task": "factory_inspect_pack", "seed": 1},
    {"category": "assist", "task": "categorize_cooperative", "seed": 21},
    {"category": "assist", "task": "put_object_cabinet_assist", "seed": 5},
    {"category": "assist", "task": "stack_bowls_three_assist", "seed": 3},
    {"category": "assist", "task": "place_bread_in_basket_assist", "seed": 9},
    {"category": "assist", "task": "dump_bin_assist", "seed": 0},
    {"category": "assist", "task": "place_burger_fries_assist", "seed": 9},
    {"category": "assist", "task": "place_dual_shoes_assist", "seed": 0},
    {"category": "assist", "task": "place_food_in_skillet_assist", "seed": 8},
    {"category": "assist", "task": "blocks_ranking_rgb_assist", "seed": 2},
    {"category": "assist", "task": "blocks_ranking_size_assist", "seed": 10},
    {"category": "interrupt", "task": "categorize_interrupt", "seed": 2},
    {"category": "interrupt", "task": "put_object_cabinet_interrupt", "seed": 2},
    {"category": "interrupt", "task": "stack_bowls_three_interrupt", "seed": 5},
    {"category": "interrupt", "task": "place_bread_in_basket_interrupt", "seed": 0},
    {"category": "interrupt", "task": "dump_bin_interrupt", "seed": 7},
    {"category": "interrupt", "task": "place_burger_fries_interrupt", "seed": 1},
    {"category": "interrupt", "task": "place_dual_shoes_interrupt", "seed": 1},
    {"category": "interrupt", "task": "place_food_in_skillet_interrupt", "seed": 2},
    {"category": "interrupt", "task": "blocks_ranking_rgb_interrupt", "seed": 9},
    {"category": "interrupt", "task": "blocks_ranking_size_interrupt", "seed": 5},
    {"category": "neutral", "task": "categorize_neutral", "seed": 0},
    {"category": "neutral", "task": "put_object_cabinet_neutral", "seed": 11},
    {"category": "neutral", "task": "stack_bowls_three_neutral", "seed": 1},
    {"category": "neutral", "task": "place_bread_in_basket_neutral", "seed": 15},
    {"category": "neutral", "task": "dump_bin_neutral", "seed": 5},
    {"category": "neutral", "task": "place_burger_fries_neutral", "seed": 28},
    {"category": "neutral", "task": "place_dual_shoes_neutral", "seed": 4},
    {"category": "neutral", "task": "place_food_in_skillet_neutral", "seed": 38},
    {"category": "neutral", "task": "blocks_ranking_rgb_neutral", "seed": 20},
    {"category": "neutral", "task": "blocks_ranking_size_neutral", "seed": 61},
]


def episode_id(eval_set_name: str, episode: dict[str, Any]) -> str:
    if "episode_index" in episode:
        return (
            f"{eval_set_name}__{episode['category']}__{episode['task']}"
            f"__ep_{episode['episode_index']:03d}__seed_{episode['seed']}"
        )
    return f"{eval_set_name}__{episode['category']}__{episode['task']}__seed_{episode['seed']}"


def expand_episodes(episodes_per_task: int, seed_offset: int) -> list[dict[str, Any]]:
    if episodes_per_task == 1 and seed_offset == 0:
        return [dict(episode) for episode in BASE_EPISODES]

    episodes = []
    for base in BASE_EPISODES:
        for episode_index in range(episodes_per_task):
            episode = dict(base)
            episode["episode_index"] = episode_index
            episode["seed"] = seed_offset + episode_index
            episodes.append(episode)
    return episodes


def with_ids(
    eval_set_name: str,
    episodes_per_task: int,
    seed_offset: int,
) -> list[dict[str, Any]]:
    return [
        {"id": episode_id(eval_set_name, episode), **episode}
        for episode in expand_episodes(episodes_per_task, seed_offset)
    ]


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def dump_yaml_mapping(lines: list[str], mapping: dict[str, Any], indent: int) -> None:
    prefix = " " * indent
    for key, value in mapping.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            dump_yaml_mapping(lines, value, indent + 2)
        else:
            lines.append(f"{prefix}{key}: {yaml_scalar(value)}")


def build_yaml(eval_set_name: str, episodes_per_task: int, seed_offset: int) -> str:
    description = (
        "One deterministic initialization per registered task, for quick policy evaluation smoke tests."
        if episodes_per_task == 1
        else f"{episodes_per_task} deterministic initializations per registered task."
    )
    lines: list[str] = [
        f"schema_version: {SCHEMA_VERSION}",
        f"name: {eval_set_name}",
        f"description: {description}",
        "contains_rollouts: false",
        f"episodes_per_task: {episodes_per_task}",
        f"seed_offset: {seed_offset}",
        "defaults:",
    ]
    dump_yaml_mapping(lines, DEFAULTS, indent=2)
    lines.append("episodes:")
    for episode in with_ids(eval_set_name, episodes_per_task, seed_offset):
        lines.extend(
            [
                f"  - id: {episode['id']}",
                f"    category: {episode['category']}",
                f"    task: {episode['task']}",
                f"    seed: {episode['seed']}",
            ]
        )
        if "episode_index" in episode:
            lines.append(f"    episode_index: {episode['episode_index']}")
    return "\n".join(lines) + "\n"


def build_jsonl(eval_set_name: str, episodes_per_task: int, seed_offset: int) -> str:
    records = []
    for episode in with_ids(eval_set_name, episodes_per_task, seed_offset):
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "eval_set": eval_set_name,
                **episode,
                **DEFAULTS,
            }
        )
    return "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)


def build_readme(eval_set_name: str, episodes_per_task: int) -> str:
    return f"""# {eval_set_name}

This eval set stores deterministic episode initializations only. It does not
contain expert rollouts, videos, actions, observations, or success labels.

It contains {episodes_per_task} initialization specs per task.

Each episode is defined by:

- `task`: registered task name resolved by `envs.tasks.resolve_task_class`
- `seed`: seed passed to `task.reset(seed=seed)`
- `category`: high-level grouping for reporting
- defaults plus any episode-local fields: deterministic environment knobs that
  should be applied during inference

The policy under evaluation should run from these initial states and produce
its own trajectory. Any reference videos outside this directory are visual
debug artifacts, not part of the eval set.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=DEFAULT_NAME, help="Eval set name.")
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=1,
        help="Number of initialization specs to generate for each task.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="First seed to use when --episodes-per-task is greater than 1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_sets") / DEFAULT_NAME,
        help="Directory to write README.md, manifest.yaml, and manifest.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes_per_task <= 0:
        raise ValueError("--episodes-per-task must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = expand_episodes(args.episodes_per_task, args.seed_offset)
    (args.output_dir / "README.md").write_text(
        build_readme(args.name, args.episodes_per_task),
        encoding="utf-8",
    )
    (args.output_dir / "manifest.yaml").write_text(
        build_yaml(args.name, args.episodes_per_task, args.seed_offset),
        encoding="utf-8",
    )
    (args.output_dir / "manifest.jsonl").write_text(
        build_jsonl(args.name, args.episodes_per_task, args.seed_offset),
        encoding="utf-8",
    )
    print(f"Wrote {len(episodes)} initialization specs to {args.output_dir}")


if __name__ == "__main__":
    main()
