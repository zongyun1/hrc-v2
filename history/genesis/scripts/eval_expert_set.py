#!/usr/bin/env python3
"""Evaluate scripted expert baselines on an initialization-only eval set."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from multiprocessing import get_context
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.tasks import resolve_task_class


POLICIES = ("expert_full_state", "expert")


def load_eval_set(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        path = path / "manifest.jsonl"
    episodes = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


# Optional CLI renderer override (e.g. "batch" for the Madrona batch renderer).
# Set once in main() before workers are forked, so ProcessPoolExecutor workers
# inherit it. None = use whatever each episode's manifest specifies.
_RENDERER_OVERRIDE: str | None = None


def build_config(episode: dict[str, Any], policy: str) -> dict[str, Any]:
    cfg = deepcopy(episode.get("config_overrides", {}))
    for key in ("renderer", "setting", "action_type", "action_substeps"):
        if key in episode:
            cfg[key] = episode[key]
    if _RENDERER_OVERRIDE:
        cfg["renderer"] = _RENDERER_OVERRIDE
    if policy == "expert_full_state":
        cfg["eval_mode"] = False
    return cfg


def resolve_for_policy(task_name: str, policy: str):
    resolved_name, task_cls = resolve_task_class(task_name, no_human=False)
    return resolved_name, task_cls, False, None


def _unprivileged_parent_play_once(task_cls):
    """Return a parent play_once for blind interrupt expert fallback, if any."""
    for cls in task_cls.__mro__[1:]:
        if cls.__name__ == "BaseTask":
            break
        if cls.__module__.endswith("inspect_motion_mixin"):
            continue
        play_once = cls.__dict__.get("play_once")
        if play_once is not None:
            return play_once
    return None


def policy_play_mode(task_name: str, task_cls, policy: str) -> tuple[str, object | None]:
    if policy != "expert":
        return "play_once", None
    if hasattr(task_cls, "play_blind_once"):
        return "play_blind_once", None
    if task_name.endswith("_interrupt") or task_name.endswith("_assist"):
        parent_play = _unprivileged_parent_play_once(task_cls)
        if parent_play is not None:
            return "parent_play_once", parent_play
    return "play_once", None


def configure_parent_rollout_eval(cfg: dict, task_name: str, play_mode: str) -> None:
    if play_mode not in {"parent_play_once", "play_blind_once"}:
        return
    if not task_name.endswith("_interrupt"):
        return
    if task_name == "dump_bin_interrupt" and play_mode == "play_blind_once":
        cfg.setdefault("eval_trigger_step_min", 0)
        cfg.setdefault("eval_trigger_step_max", 0)
        cfg.setdefault("eval_robot_retreat_policy_steps", 0)
        cfg.setdefault("eval_glide_policy_steps", 0)
    else:
        cfg.setdefault("eval_trigger_step_min", 5)
        cfg.setdefault("eval_trigger_step_max", 25)
    cfg.setdefault("eval_interrupt_force_at_trigger", True)


def run_with_eval_step_driver(task, fn, cfg: dict) -> bool:
    """Drive eval-mode avatar hooks during scripted parent rollouts.

    Parent play_once implementations call step_sim directly, not take_action.
    For expert interrupt/assist fallbacks we still need eval-mode avatar logic
    to tick at policy-step cadence.
    """
    orig_step_sim = task.step_sim
    tick_interval = max(1, int(cfg.get("eval_parent_rollout_tick_interval", 5)))
    counter = {"sim": 0, "policy": 0}

    def driven_step_sim():
        orig_step_sim()
        counter["sim"] += 1
        if counter["sim"] % tick_interval != 0:
            return
        if not bool(getattr(task, "config", {}).get("eval_mode", False)):
            return
        if getattr(task, "avatar", None) is None:
            return
        if hasattr(task, "_eval_step_avatar"):
            task._eval_step_avatar()
        elif hasattr(task, "_eval_at_step"):
            counter["policy"] += 1
            task._eval_at_step(counter["policy"])

    task.step_sim = driven_step_sim
    try:
        return bool(fn())
    finally:
        task.step_sim = orig_step_sim


def run_episode(
    episode: dict[str, Any],
    policy: str,
    output_dir: Path,
    save_video: bool,
) -> dict[str, Any]:
    task_name = episode["task"]
    seed = int(episode["seed"])
    cfg = build_config(episode, policy)
    cfg["eval_seed"] = seed
    resolved_name, task_cls, no_human_applied, no_human_note = resolve_for_policy(task_name, policy)
    play_mode, parent_play_once = policy_play_mode(task_name, task_cls, policy)
    if play_mode in {"play_blind_once", "parent_play_once"}:
        cfg["eval_mode"] = True
        configure_parent_rollout_eval(cfg, task_name, play_mode)
    elif policy == "expert":
        # Fallback experts call play_once(); keep that path internally
        # consistent with scripted calibration instead of mixing it with
        # eval-mode avatar state.
        cfg["eval_mode"] = False

    if no_human_applied:
        cfg["no_human"] = True
        cfg["randomize_avatar"] = False
        cfg["track_avatar_collision"] = False
        cfg["avatar_retreat_enabled"] = False
        cfg["eval_mode"] = False

    video_path = None
    if save_video:
        video_dir = output_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"{episode['id']}__{policy}.mp4"

    t0 = time.time()
    result = {
        "id": episode["id"],
        "eval_set": episode.get("eval_set"),
        "category": episode.get("category"),
        "task": task_name,
        "resolved_task": resolved_name,
        "seed": seed,
        "policy": policy,
        "no_human_applied": no_human_applied,
        "no_human_note": no_human_note,
        "success": False,
        "play_success": False,
        "elapsed_s": None,
        "error": None,
        "metrics": {},
        "video": str(video_path) if video_path else None,
    }

    task = None
    try:
        task = task_cls(cfg)
        task.reset(seed=seed)
        if video_path is not None:
            task.start_video(str(video_path))
        if play_mode == "play_blind_once":
            play_success = bool(task.play_blind_once())
        elif play_mode == "parent_play_once":
            play_success = run_with_eval_step_driver(
                task, lambda: parent_play_once(task), cfg,
            )
        else:
            play_success = bool(task.play_once())
        if not play_success:
            task.plan_success = False
        metrics = task.evaluate()
        if video_path is not None:
            task.save_video()
        result["play_success"] = play_success
        result["success"] = bool(metrics.get("success", False))
        result["metrics"] = metrics
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        result["elapsed_s"] = time.time() - t0
        del task
    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    success = sum(1 for r in results if r.get("success"))
    by_category = {}
    by_task = {}
    for key, bucket in (("category", by_category), ("task", by_task)):
        for r in results:
            name = r.get(key) or "unknown"
            item = bucket.setdefault(name, {"total": 0, "success": 0, "success_rate": 0.0})
            item["total"] += 1
            item["success"] += int(bool(r.get("success")))
        for item in bucket.values():
            item["success_rate"] = item["success"] / item["total"] if item["total"] else 0.0
    return {
        "total": total,
        "success": success,
        "success_rate": success / total if total else 0.0,
        "by_category": by_category,
        "by_task": by_task,
        "errors": sum(1 for r in results if r.get("error")),
    }


def write_outputs(policy: str, output_dir: Path, results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)
    (output_dir / "results.jsonl").write_text(
        "".join(json.dumps(r, default=str) + "\n" for r in results),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "category",
                "task",
                "resolved_task",
                "seed",
                "policy",
                "no_human_applied",
                "success",
                "play_success",
                "elapsed_s",
                "error",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in writer.fieldnames})
    print(
        f"{policy}: {summary['success']}/{summary['total']} "
        f"= {summary['success_rate']:.3f}; errors={summary['errors']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=Path("eval_sets/eval_v0_smoke"),
        help="Eval-set directory or manifest.jsonl path.",
    )
    parser.add_argument(
        "--policy",
        choices=POLICIES,
        action="append",
        default=None,
        help="Policy to evaluate. Repeat to run more than one. Default: expert_full_state.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/expert_eval_set/eval_v0_smoke"),
        help="Directory for results.",
    )
    parser.add_argument("--save-video", action="store_true", help="Save debug videos.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N episodes.")
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based episode index to start from after loading the eval set.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Zero-based exclusive episode index to stop at after loading the eval set.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of episode worker processes per policy. Default: 1.",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default=None,
        help="Override the renderer for every episode (e.g. 'batch' for the "
             "Madrona batch renderer, 'rasterizer'). Default: use each "
             "episode's manifest value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global _RENDERER_OVERRIDE
    _RENDERER_OVERRIDE = args.renderer
    policies = args.policy or ["expert_full_state"]
    episodes = load_eval_set(args.eval_set)
    start_index = max(0, int(args.start_index))
    end_index = args.end_index if args.end_index is None else max(start_index, int(args.end_index))
    episodes = episodes[start_index:end_index]
    if args.limit is not None:
        episodes = episodes[: max(0, args.limit)]
    print(
        f"Loaded {len(episodes)} eval-set initializations from {args.eval_set} "
        f"(start_index={start_index}, end_index={end_index})"
    )

    for policy in policies:
        policy_dir = args.output_dir / policy
        results = []
        workers = max(1, int(args.workers))
        if args.save_video:
            workers = 1
        if workers == 1 or len(episodes) <= 1:
            for i, episode in enumerate(episodes, start=1):
                print(
                    f"[{policy}] {i}/{len(episodes)} "
                    f"{episode['task']} seed={episode['seed']}",
                    flush=True,
                )
                result = run_episode(episode, policy, policy_dir, save_video=args.save_video)
                results.append(result)
                status = "SUCCESS" if result["success"] else "FAIL"
                if result.get("error"):
                    status = "ERROR"
                print(f"  {status} elapsed={result['elapsed_s']:.1f}s", flush=True)
                if result.get("error"):
                    print(result["error"], flush=True)
                write_outputs(policy, policy_dir, results)
        else:
            print(f"[{policy}] running with workers={workers}", flush=True)
            indexed = list(enumerate(episodes, start=1))
            ctx = get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
                futures = {
                    executor.submit(run_episode, episode, policy, policy_dir, args.save_video): i
                    for i, episode in indexed
                }
                completed = {}
                for future in as_completed(futures):
                    i = futures[future]
                    result = future.result()
                    completed[i] = result
                    status = "SUCCESS" if result["success"] else "FAIL"
                    if result.get("error"):
                        status = "ERROR"
                    print(
                        f"[{policy}] {i}/{len(episodes)} "
                        f"{result['task']} seed={result['seed']} "
                        f"{status} elapsed={result['elapsed_s']:.1f}s",
                        flush=True,
                    )
                    if result.get("error"):
                        print(result["error"], flush=True)
                    results = [completed[j] for j in sorted(completed)]
                    write_outputs(policy, policy_dir, results)
        write_outputs(policy, policy_dir, results)


if __name__ == "__main__":
    main()
