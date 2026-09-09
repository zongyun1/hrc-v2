"""Data collection: run task rollouts and save trajectories."""

import argparse
import os
import sys
import yaml
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.tasks import TASK_MAP, resolve_task_class
from envs.utils import load_config


RESET_RETRY_TASKS = {
    "place_burger_fries_neutral",
    "dump_bin_neutral",
    "stack_bowls_three_neutral",
}


def _should_retry_reset(task_name: str, config: dict) -> bool:
    if bool(config.get("retry_reset_until_success", False)):
        return True
    if task_name in RESET_RETRY_TASKS:
        return True
    return False


def collect(task_name: str, config: dict, num_episodes: int, save_dir: str, start_seed: int = 0):
    """Run data collection for a task."""
    resolved_task_name, TaskClass = resolve_task_class(
        task_name,
        no_human=bool(config.get("no_human", False)),
    )
    os.makedirs(save_dir, exist_ok=True)

    success_count = 0
    completed_episodes = 0
    seed = int(start_seed)
    reset_retry = _should_retry_reset(resolved_task_name, config)
    max_reset_attempts = int(config.get("max_reset_attempts", max(100, num_episodes * 50)))
    reset_attempts = 0
    while completed_episodes < num_episodes:
        print(f"\n{'='*60}")
        print(f"Episode {completed_episodes + 1}/{num_episodes} (seed={seed})")
        print(f"{'='*60}")

        task = TaskClass(config)
        try:
            obs = task.reset(seed=seed)
        except Exception as e:
            reset_attempts += 1
            if not reset_retry:
                raise
            print(f"  Reset failed for seed={seed}: {type(e).__name__}: {e}")
            if reset_attempts >= max_reset_attempts:
                raise RuntimeError(
                    f"Exceeded max reset attempts ({max_reset_attempts}) while "
                    f"collecting {resolved_task_name}; completed "
                    f"{completed_episodes}/{num_episodes}"
                ) from e
            seed += 1
            print(f"  Retrying reset with next seed={seed}")
            continue

        # Video is useful for normal demo collection but expensive during
        # project-wide smoke scans because each captured frame renders every
        # configured recording view. Keep physics/trajectory execution intact
        # when explicitly disabled.
        record_video = not bool(config.get("disable_video", False))
        if record_video:
            video_path = os.path.join(save_dir, "video", f"seed_{seed}.mp4")
            task.start_video(video_path)

        t0 = time.time()
        try:
            success = task.play_once()
        except Exception as e:
            print(f"  Episode failed with error: {e}")
            success = False
        elapsed = time.time() - t0

        metrics = task.evaluate() if success else {
            "plan_success": False, "success": False,
            "avatar_collision": task.avatar_collision_summary(),
            "safe_distance": task.safe_distance_summary(),
        }
        task_success = bool(metrics.get("success", False))

        # Always save an enabled video (success or fail).
        if record_video:
            task.save_video()

        if "target_dist_xy" in metrics:
            tilt = metrics.get("target_tilt_deg")
            tilt_s = f", tilt={tilt:.1f}°" if tilt is not None else ""
            print(f"  target[{metrics.get('target_label','?')}]: "
                  f"dist_xy={metrics['target_dist_xy']:.3f}m, "
                  f"dist_3d={metrics['target_dist_3d']:.3f}m, "
                  f"dz={metrics['target_dz']:+.3f}m{tilt_s}")
        # Per-episode avatar-collision report (if tracking enabled).
        coll = metrics.get("avatar_collision", {})
        if coll.get("enabled"):
            n = coll["n_checks"]; hits = coll["n_collisions"]
            dd = coll.get("deepest_depth_m")
            dd_str = f"{dd*100:+.1f} cm" if dd is not None else "—"
            print(f"  avatar-collision: {hits}/{n} checks hit, "
                  f"deepest={dd_str}, pair={coll.get('deepest_pair')}")
        safe_dist = metrics.get("safe_distance", {})
        if safe_dist.get("enabled"):
            sd = safe_dist.get("safe_distance_m")
            sd_str = f"{sd*100:+.1f} cm" if sd is not None else "—"
            print(f"  safe-distance: min={sd_str}, "
                  f"pair={safe_dist.get('closest_pair')}")

        if task_success:
            success_count += 1
            save_path = os.path.join(save_dir, f"seed_{seed}.pkl")
            task.save_trajectory(save_path)
            print(f"  SUCCESS ({elapsed:.1f}s) -> {save_path}")
        else:
            print(f"  FAIL ({elapsed:.1f}s)")

        completed_episodes += 1
        seed += 1

    print(f"\n{'='*60}")
    print(f"Done: {success_count}/{num_episodes} successful episodes")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Collect task demonstrations")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--save-dir", type=str, default=None, help="Save directory")
    parser.add_argument("--start-seed", type=int, default=0, help="Starting seed")
    parser.add_argument("--retry-reset-until-success", action="store_true",
                        help="On reset failures, skip the failed seed and keep trying "
                             "until the requested number of episodes have reset and "
                             "run. Enabled by default for selected neutral tasks.")
    parser.add_argument("--max-reset-attempts", type=int, default=None,
                        help="Maximum reset attempts when reset retry is enabled.")
    parser.add_argument("--renderer", type=str, default=None,
                        choices=["rasterizer", "raytracer", "nyx"],
                        help="Override renderer (nyx/raytracer require GPU)")
    parser.add_argument("--randomize-avatar", action="store_true",
                        help="Randomize avatar glb per episode (from task.AVATAR_POOL).")
    parser.add_argument("--no-human", action="store_true",
                        help="Collapse an assist/interrupt/neutral family task to "
                             "its canonical robot-only no-human task. Intent tasks "
                             "do not support this flag.")
    parser.add_argument("--randomize-inspect", action="store_true",
                        help="(interrupt tasks) Randomize avatar inspect motion per "
                             "episode from task.INSPECT_POOL (mixin default = "
                             "[Inspect, Inspect2]).")
    parser.add_argument("--randomize-table", action="store_true",
                        help="Debug-gated: randomize the table mesh from the SAPIEN table library. "
                             "Requires DEBUG_TABLE_RANDOMIZATION=1 unless "
                             "--debug-table-randomization is also set.")
    parser.add_argument("--table-random-objects",
                        "--task-irrelevant-objects",
                        "--with-irrelevant-objects",
                        dest="table_random_objects", action="store_true",
                        help="Enable random background tabletop objects for tasks "
                             "that define decorative object regions. Default: disabled.")
    parser.add_argument("--hard", action="store_true",
                        help="Use the legacy multi-object task variant where supported.")
    parser.add_argument("--no-table-random-objects",
                        "--no-task-irrelevant-objects",
                        dest="no_table_random_objects",
                        action="store_true",
                        help="Disable random background tabletop objects, overriding config.")
    parser.add_argument("--no-randomize-table", action="store_true",
                        help="Keep the original procedural table even when other "
                             "collection randomization flags are enabled.")
    parser.add_argument("--debug-table-randomization", action="store_true",
                        help="Enable the debug gate for --randomize-table in this run.")
    parser.add_argument("--table-variant-name", type=str, default=None,
                        help="Force a specific table variant, e.g. sapien-table-20279. "
                             "Implies --randomize-table for this run.")
    parser.add_argument("--inspect-motion-name", type=str, default=None,
                        help="(interrupt tasks) Force a specific inspect motion by "
                             "name (must exist in INSPECT_POOL). Overrides "
                             "--randomize-inspect.")
    parser.add_argument("--random-interrupt-target", action="store_true",
                        help="(categorize_interrupt only) Randomize which column "
                             "(0/1/2) the avatar inspects per episode. Default = middle.")
    parser.add_argument("--video-stride", type=int, default=None,
                        help="Capture 1 video frame every N sim steps (default: task VIDEO_STRIDE). "
                             "Higher = faster rollouts, choppier video.")
    parser.add_argument("--no-video", action="store_true",
                        help="Disable video rendering and encoding for smoke scans.")
    parser.add_argument("--track-avatar-collision", action="store_true",
                        help="Run the analytic robot↔avatar-capsule collision checker "
                             "each episode and print/log the per-episode summary. "
                             "Physics is untouched; this is pure detection.")
    parser.add_argument("--collision-stride", type=int, default=None,
                        help="Run the collision check every N sim steps "
                             "(default: 30 ≈ 60 ms @ 500 Hz). Lower = finer, slower.")
    parser.add_argument("--collision-margin", type=float, default=None,
                        help="Flag collisions within M metres of a capsule surface "
                             "(default 0 = strict overlap; positive = near-miss window).")
    parser.add_argument("--show-collider", action="store_true",
                        help="Render the 43 bone cylinders in the video — debug only.")
    parser.add_argument("--record-stride", type=int, default=None,
                        help="Record one obs frame (image + proprio) every N step_sim calls. "
                             "Default: off. 50 ~= 10 Hz, 100 ~= 5 Hz (Bridge-V2 style). "
                             "Required for OpenVLA / RLDS-format collection.")
    parser.add_argument("--setting", choices=["sim", "real"], default=None,
                        help="Collection setting. 'real' uses the real-world "
                             "interaction axis where the robot arm is at world x- "
                             "and the human is at world x+ for tasks that support it.")
    args = parser.parse_args()

    # config/default.yml is the base (renderer, nyx spp/denoise/env_texture,
    # camera resolutions); --config overlays it.
    config = load_config(args.config)

    if args.renderer:
        config["renderer"] = args.renderer
        if args.renderer in ("raytracer", "nyx"):
            os.environ.setdefault("GENESIS_BACKEND", "gpu")
        if args.renderer == "nyx":
            config.setdefault("nyx", {"spp": 1, "denoise": False, "open_window": False})
        if args.renderer == "raytracer":
            config.setdefault("raytracer", {
                "tracing_depth": 32, "env_radius": 1000.0,
                "lights": [{"pos": (0, 0, 10), "color": (1, 1, 1), "intensity": 10, "radius": 4}],
            })

    if args.randomize_avatar:
        config["randomize_avatar"] = True
    if args.no_human:
        config["no_human"] = True
        config["randomize_avatar"] = False
        config["track_avatar_collision"] = False
        config["avatar_retreat_enabled"] = False
    if args.randomize_inspect:
        config["randomize_inspect"] = True
    if args.table_random_objects:
        config["table_random_objects"] = True
        config["task_irrelevant_objects"] = True
    if args.hard:
        config["hard"] = True
        config["difficulty"] = "hard"
    if args.retry_reset_until_success:
        config["retry_reset_until_success"] = True
    if args.max_reset_attempts is not None:
        config["max_reset_attempts"] = max(1, args.max_reset_attempts)
    if args.no_table_random_objects:
        config["table_random_objects"] = False
        config["task_irrelevant_objects"] = False
    collection_randomization_on = (
        args.randomize_avatar
        or args.randomize_inspect
        or args.random_interrupt_target
        or bool(config.get("randomize_avatar", False))
        or bool(config.get("randomize_inspect", False))
        or bool(config.get("random_interrupt_target", False))
    )
    if args.no_randomize_table:
        config["randomize_table"] = False
    elif (
        args.randomize_table
        or args.table_variant_name
        or (collection_randomization_on and "randomize_table" not in config)
    ):
        config["randomize_table"] = True
    if args.debug_table_randomization:
        config["debug_table_randomization"] = True
    if args.table_variant_name:
        config["table_variant_name"] = args.table_variant_name
    if args.inspect_motion_name:
        config["inspect_motion_name"] = args.inspect_motion_name
    if args.random_interrupt_target:
        config["random_interrupt_target"] = True

    if args.video_stride is not None:
        config["video_stride"] = max(1, args.video_stride)
    if args.no_video:
        config["disable_video"] = True

    if args.track_avatar_collision:
        config["track_avatar_collision"] = True
    if args.collision_stride is not None:
        config["collision_check_stride"] = max(1, args.collision_stride)
    if args.collision_margin is not None:
        config["collision_margin"] = float(args.collision_margin)
    if args.show_collider:
        config["use_avatar_collider"] = True
        config["collider_visualization"] = True

    if args.record_stride is not None:
        config["record_stride"] = max(1, args.record_stride)
        # OpenVLA collect path only needs the primary camera; trim the others
        # to keep raw pickles small (5-camera default → ~5x larger). Override
        # via config["record_cameras"] in the YAML if needed.
        config.setdefault("record_cameras", ["head_camera", "left_wrist", "right_wrist"])
    if args.setting is not None:
        config["setting"] = args.setting

    resolved_task_name, _ = resolve_task_class(
        args.task,
        no_human=bool(config.get("no_human", False)),
    )

    if args.save_dir:
        save_dir = args.save_dir
    elif args.record_stride is not None:
        # OpenVLA-format collection: route raw pickles into the dataset's _raw dir.
        save_dir = os.path.join("openvla", "data", "genesis_hr_bench_raw", resolved_task_name)
    else:
        save_dir = os.path.join("data", resolved_task_name)
    collect(args.task, config, args.episodes, save_dir, args.start_seed)


if __name__ == "__main__":
    main()
