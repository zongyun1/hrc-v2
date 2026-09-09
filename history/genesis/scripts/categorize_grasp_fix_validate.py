"""Validate one corrected grasp through the full categorize pick/place path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import transforms3d as t3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.tasks.categorize_cooperative import CategorizeCooperative
from envs.grasp import load_grasp_poses


class OneObjectCategorize(CategorizeCooperative):
    use_avatar = False
    NUM_CATEGORIES = 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--object",
        required=True,
        choices=(
            "048_stapler@0",
            "038_milk-box@0",
            "035_apple@1",
            "073_rubikscube@1",
        ),
    )
    parser.add_argument("--grasp-name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-stem")
    parser.add_argument(
        "--spawn-xy",
        type=float,
        nargs=2,
        help="optional robot-object XY override for reproducing a failed layout",
    )
    args = parser.parse_args()

    object_name = args.object.split("@", 1)[0]
    if object_name in {"038_milk-box", "073_rubikscube"} and not args.grasp_name:
        parser.error(f"{object_name} validation requires --grasp-name")
    if object_name == "048_stapler" and args.grasp_name:
        parser.error(f"{object_name} uses the production radius grasp; omit --grasp-name")

    OneObjectCategorize.OBJECT_SET = [args.object]
    task = OneObjectCategorize({
        "robot_type": "franka",
        "robot_single_arm": True,
        "no_human": True,
        "renderer": "rasterizer",
        "show_viewer": False,
        "table_random_objects": False,
        "video_stride": 8,
        "video_quality": 8,
    })
    if args.grasp_name:
        # Force exactly one horizontal candidate so production keepers and
        # prospective replacements can be measured independently instead of
        # always taking the first reachable side.
        by_name = {
            grasp.name: grasp
            for grasp in load_grasp_poses(
                object_name, model_id=0, robot_type="franka",
            )
        }
        grasp = by_name.get(args.grasp_name)
        if grasp is None:
            parser.error(f"unknown milk-box grasp {args.grasp_name!r}")
        if object_name == "038_milk-box":
            approach = t3d.quaternions.quat2mat(grasp.pose.q)[:, 2]
            if abs(float(approach[1])) >= 0.20:
                parser.error(
                    f"{args.grasp_name!r} is not a side grasp: "
                    f"object-frame approach={approach.tolist()}"
                )
        task.EXCLUSIVE_GRASP_NAMES = {
            **task.EXCLUSIVE_GRASP_NAMES,
            object_name: (args.grasp_name,),
        }
        task.YAML_ONLY_GRASP_LABELS = frozenset({
            *task.YAML_ONLY_GRASP_LABELS,
            object_name,
        })

    task.reset(seed=args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem or (
        object_name if not args.grasp_name else f"{object_name}_{args.grasp_name}"
    )
    video_path = output_dir / f"{stem}.mp4"
    image_path = output_dir / f"{stem}.png"
    result_path = output_dir / f"{stem}.json"
    actor, spawned_name, model_id, basket_idx = task._sort_targets[0]
    if args.spawn_xy is not None:
        entity = getattr(actor, "entity", actor)
        pose = actor.get_pose()
        entity.set_pos(np.array([
            float(args.spawn_xy[0]),
            float(args.spawn_xy[1]),
            float(pose.p[2]),
        ]))
        for _ in range(120):
            task.step_sim()
    task.start_video(str(video_path))

    basket_xy = np.asarray(task.basket_poses[basket_idx].p[:2], dtype=float)
    success = bool(task.play_once())
    final_center = np.asarray(
        task._get_object_world_center(actor, spawned_name, model_id), dtype=float,
    )
    basket_dist_xy = float(np.linalg.norm(final_center[:2] - basket_xy))
    if task._video_frames:
        imageio.imwrite(image_path, task._video_frames[-1])
    task.save_video(fps=30)

    result = {
        "task": "categorize_cooperative",
        "object_token": args.object,
        "object_name": object_name,
        "model_id": int(model_id),
        "seed": int(args.seed),
        "grasp_name": args.grasp_name,
        "grasp_policy": (
            "short_axis_topdown_yaw90" if object_name == "048_stapler"
            else "apple_authored_topdown_candidate" if object_name == "035_apple" and args.grasp_name
            else "apple_radius_close_0.58" if object_name == "035_apple"
            else "exclusive_topdown_face_candidate" if object_name == "073_rubikscube"
            else "exclusive_side_candidate"
        ),
        "spawn_xy_override": args.spawn_xy,
        "basket_dist_xy_m": basket_dist_xy,
        "success": bool(success and basket_dist_xy <= 0.10),
        "video": video_path.name,
        "image": image_path.name,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"CATEGORIZE_GRASP_FIX_RESULT {json.dumps(result, sort_keys=True)}", flush=True)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
