"""Render a neutral-avatar motion to completion (robot stays idle).

The normal `collect.py` rollout ties the avatar's neutral motion to the robot
task's step count, so a long neutral clip (e.g. the 603-frame
put-/take-objects-in-bowl motion) gets truncated when the robot finishes first.
This script instead builds the neutral task, starts the avatar motion, and steps
the sim until the avatar clip ends — capturing the full motion + prop
attach/detach for review. Useful for any long neutral motion.

Usage:
  python scripts/render_neutral_motion_full.py \
      --task stack_bowls_three_neutral \
      --config config/neutral_motion_take_objects_from_bowl.yml \
      --out data/neutput/take_objects_from_bowl/video/full_motion.mp4 \
      --video-stride 20 --seed 0
"""
import argparse
import os
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml  # noqa: E402

from envs.tasks import TASK_MAP, resolve_task_class  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="neutral task name")
    ap.add_argument("--config", required=True, help="config YAML (forces the motion)")
    ap.add_argument("--out", required=True, help="output mp4 path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video-stride", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=30000,
                    help="hard cap on sim steps (safety net)")
    ap.add_argument("--settle-steps", type=int, default=60,
                    help="extra steps recorded after the motion ends")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f) or {}
    config["video_stride"] = max(1, args.video_stride)
    config.setdefault("track_avatar_collision", True)

    resolved_name, TaskClass = resolve_task_class(
        args.task, no_human=bool(config.get("no_human", False)),
    )
    print(f"[render] task={resolved_name} motion={config.get('neutral_avatar_motion')}")

    task = TaskClass(config)
    task.reset(seed=args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    task.start_video(args.out)

    # Start the avatar's neutral motion immediately (no robot rollout).
    task._start_neutral_avatar_table_work(add_start_delay=False)

    steps = 0
    # Wait a few steps for the ease-in to actually begin before polling spare().
    warmup = 5
    while steps < args.max_steps:
        task.step_sim()
        steps += 1
        if steps > warmup and task.avatar.spare():
            break
        if steps % 500 == 0:
            print(f"[render] stepped {steps} (avatar spare={task.avatar.spare()})", flush=True)

    print(f"[render] motion ended after {steps} steps; settling {args.settle_steps}")
    for _ in range(max(0, args.settle_steps)):
        task.step_sim()
        steps += 1

    task.save_video()

    coll = task.avatar_collision_summary() if hasattr(task, "avatar_collision_summary") else None
    safe = task.safe_distance_summary() if hasattr(task, "safe_distance_summary") else None
    print(f"[render] DONE steps={steps} -> {args.out}")
    print(f"[render] avatar-collision: {coll}")
    print(f"[render] safe-distance: {safe}")


if __name__ == "__main__":
    main()
