"""Task-native grasp/lift/hold probe for one categorize pool object.

The scene is ``CategorizeCooperative`` in no-human, one-object mode.  The
probe runs the same top-down radius grasp used by the task, but stops after
the vertical lift so grasp-pose quality is not confounded by basket transport.

This is intentionally a one-attempt diagnostic.  It writes one MP4, one final
PNG, and one JSON result under ``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.manipulation import PickSpec, PlaceSpec, TopDownPickPlaceMixin
from envs.object_catalog import get_entry, object_set_pairs
from envs.tasks.categorize_cooperative import CategorizeCooperative
from envs.utils import Pose, to_numpy


POOL = "tabletop_pick_pool"
POOL_TOKENS = tuple(
    f"{name}@{model_id}" if name != "cube" else "cube"
    for name, model_id in object_set_pairs(POOL)
)


class CategorizeGraspSweepTask(CategorizeCooperative):
    """One robot-side object in the real cooperative categorize scene."""

    use_avatar = False
    NUM_CATEGORIES = 1


def _actor_pose(actor) -> Pose:
    if hasattr(actor, "get_pose"):
        return actor.get_pose()
    entity = getattr(actor, "entity", actor)
    return Pose(
        to_numpy(entity.get_pos()).ravel()[:3],
        to_numpy(entity.get_quat()).ravel()[:4],
    )


def _current_close_value(task, label: str):
    if label in task.RADIUS_CLOSE_OVERRIDE:
        return float(task.RADIUS_CLOSE_OVERRIDE[label])
    value = get_entry(label).close_value
    return None if value is None else float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", required=True, dest="object_token")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-stem")
    parser.add_argument("--xy-offset", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--z-offset", type=float, default=0.0)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--close-value", type=float)
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--hold-steps", type=int, default=100)
    args = parser.parse_args()

    if args.object_token not in POOL_TOKENS:
        parser.error(
            f"--object must be one of {', '.join(POOL_TOKENS)}; "
            f"got {args.object_token!r}"
        )

    CategorizeGraspSweepTask.OBJECT_SET = [args.object_token]
    config = {
        "robot_type": "franka",
        "robot_single_arm": True,
        "no_human": True,
        "renderer": "rasterizer",
        "show_viewer": False,
        "table_random_objects": False,
        "video_stride": 8,
        "video_quality": 8,
        "pick_place_settle_above_place_steps": 0,
        "pick_place_hold_steps": 0,
    }
    task = CategorizeGraspSweepTask(config)
    task.reset(seed=args.seed)

    if len(task._sort_targets) != 1:
        raise RuntimeError(f"expected one robot target, got {len(task._sort_targets)}")
    actor, label, model_id, _basket_idx = task._sort_targets[0]
    entity = getattr(actor, "entity", actor)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    default_stem = args.object_token.replace("@", "_m")
    stem = args.output_stem or default_stem
    video_path = output_dir / f"{stem}.mp4"
    image_path = output_dir / f"{stem}.png"
    result_path = output_dir / f"{stem}.json"

    task.start_video(str(video_path))
    task._boost_arm_pd()
    task._open_basket_lids()
    task._move_to_safe("right")
    for _ in range(max(0, int(args.settle_steps))):
        task.step_sim()

    center_fn = lambda: task._get_object_world_center(actor, label, model_id)
    center_before = np.asarray(center_fn(), dtype=float)
    try:
        aabb = np.asarray(to_numpy(entity.get_AABB()), dtype=float)
        radius = float(np.min((aabb[1] - aabb[0])[:2]) / 2.0)
    except Exception:
        radius = float(task._get_object_cross_section_radius(label, model_id))

    close_value = (
        float(args.close_value)
        if args.close_value is not None
        else _current_close_value(task, label)
    )
    yaw_rad = float(np.deg2rad(args.yaw_deg))
    pick = PickSpec(
        get_center=center_fn,
        radius=radius,
        label=label,
        close_value=close_value,
        approach_xy_offset=tuple(float(x) for x in args.xy_offset),
        get_grasp_yaw=(lambda: yaw_rad),
        grasp_z_offset=float(args.z_offset),
        require_lift=True,
        min_lift_delta=0.05,
    )
    hold_z = float(task.TABLE_TOP_Z + 0.24)
    place = PlaceSpec(
        pos=np.array([center_before[0], center_before[1], hold_z]),
        label="elevated hold",
        transport_z=hold_z,
        release=False,
        stop_after_lift=True,
    )

    planned = bool(
        TopDownPickPlaceMixin.pick_and_place(task, pick, place, "right")
    )
    center_at_lift = np.asarray(center_fn(), dtype=float)

    # Capture every hold tick so the review video clearly shows progressive
    # slip instead of reducing the whole hold to a handful of frames.
    task._video_stride = 1
    task._video_cap_tick = 0
    hold_frame_start = len(task._video_frames)
    arm = task.robot.get_arm("right")
    ee_at_lift = Pose.from_pose7(arm.get_ee_pose())
    rel_at_lift = ee_at_lift.inv() * _actor_pose(actor)
    relative_drifts = []
    for _ in range(max(0, int(args.hold_steps))):
        task.step_sim()
        ee_now = Pose.from_pose7(arm.get_ee_pose())
        rel_now = ee_now.inv() * _actor_pose(actor)
        relative_drifts.append(float(np.linalg.norm(
            np.asarray(rel_now.p, dtype=float)
            - np.asarray(rel_at_lift.p, dtype=float)
        )))

    center_after = np.asarray(center_fn(), dtype=float)
    hold_frames = len(task._video_frames) - hold_frame_start
    lift_dz = float(center_at_lift[2] - center_before[2])
    end_drift = float(relative_drifts[-1] if relative_drifts else 0.0)
    max_drift = float(max(relative_drifts, default=0.0))
    hold_z_drop = float(center_at_lift[2] - center_after[2])
    retained = bool(end_drift <= 0.015 and max_drift <= 0.020)
    success = bool(
        planned
        and lift_dz >= 0.05
        and retained
        and hold_frames == max(0, int(args.hold_steps))
    )

    if task._video_frames:
        import imageio.v2 as imageio

        imageio.imwrite(image_path, task._video_frames[-1])
    task.save_video(fps=30)

    result = {
        "task": "categorize_cooperative",
        "pool": POOL,
        "object_token": args.object_token,
        "object_name": label,
        "model_id": int(model_id),
        "seed": int(args.seed),
        "pose": {
            "xy_offset_m": [float(x) for x in args.xy_offset],
            "z_offset_m": float(args.z_offset),
            "yaw_deg": float(args.yaw_deg),
        },
        "close_value": close_value,
        "planned": planned,
        "lift_dz_m": lift_dz,
        "hold_steps": int(args.hold_steps),
        "hold_frames": hold_frames,
        "hold_end_relative_drift_m": end_drift,
        "hold_max_relative_drift_m": max_drift,
        "hold_z_drop_m": hold_z_drop,
        "retained_through_hold": retained,
        "success": success,
        "video": video_path.name,
        "image": image_path.name,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"CATEGORIZE_GRASP_RESULT {json.dumps(result, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
