"""Render one grasp-only demonstration for a pooled target object.

The driver reuses DumpBin's simple one-object tabletop scene and the shared
TopDownPickPlaceMixin.  It stops after a vertical lift and elevated hold; no
task-specific delivery or placement is attempted.

Usage:
    python scripts/object_pool_grasp_demo.py \
        --pool tabletop_pick_pool --object 035_apple --output-dir data/object/grasp_videos
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.manipulation import PickSpec, PlaceSpec
from envs.grasp import GraspPose, read_save_object_quat, tcp_to_link_pose
from envs.object_catalog import CATALOG, ObjectEntry
from envs.task_bases.dump_bin import DumpBin
from envs.utils import ASSETS_PATH, Pose, to_numpy


FACTORY_CLOSE_VALUES = {
    "113_coffee-box": 0.42,
    "112_tea-box": 0.45,
    "023_tissue-box": 0.46,
}

SIDE_GRASP_NAMES = {
    "038_milk-box": "grasp_065",
}

DEFAULT_GRASP_METHODS = {
    "038_milk-box": "side",
}


# A few task-specific targets predate the central object catalog.  Register
# them only in this diagnostic process so the simple DumpBin scene can render
# the same asset/model without changing benchmark pool membership.
DEMO_ONLY_ENTRIES = (
    ObjectEntry("005_french-fries", object_id="005_french-fries", model_id=0,
                display_name="french fries", friction=5.0),
    ObjectEntry("006_hamburg", object_id="006_hamburg", model_id=5,
                display_name="hamburger", friction=5.0),
    ObjectEntry("020_hammer", object_id="020_hammer", model_id=0,
                display_name="hammer", friction=4.0),
    ObjectEntry("032_screwdriver", object_id="032_screwdriver", model_id=0,
                display_name="screwdriver", friction=4.0),
    ObjectEntry("034_knife", object_id="034_knife", model_id=0,
                display_name="knife", friction=4.0),
    ObjectEntry("106_skillet", object_id="106_skillet", model_id=0,
                display_name="skillet", friction=4.0),
    ObjectEntry("block-small", kind="primitive", display_name="small block",
                friction=4.0, prim_half=0.018),
    ObjectEntry("block-medium", kind="primitive", display_name="medium block",
                friction=4.0, prim_half=0.024),
    ObjectEntry("block-large", kind="primitive", display_name="large block",
                friction=4.0, prim_half=0.030),
)


Q_UPRIGHT = (0.70710678, 0.70710678, 0.0, 0.0)
Q_MUG_HANDLE_TOWARD_ROBOT = (0.0, 0.0, 0.70710678, 0.70710678)
Q_FACTORY_TOOL_FLAT_X = (0.70710678, 0.0, 0.0, -0.70710678)

# Complete variant-aware inventory of objects physically grasped by the robot
# in registered benchmark tasks.  Receptacles, visual clutter, human-only
# props, pushed documents, and the articulated microwave handle are excluded.
# Every profile is rendered with the same grasp -> vertical lift -> 100-step
# stationary hold sequence.
DEMO_PROFILES = {
    # Shared random pools.
    "stapler_m0": {"object_name": "048_stapler@0"},
    "jam_jar_m0": {"object_name": "031_jam-jar@0"},
    "cube_40mm": {"object_name": "cube", "primitive_half": 0.020},
    "milk_box_m0": {"object_name": "038_milk-box@0", "grasp_method": "side",
                    "grasp_name": "grasp_065"},
    "apple_m1": {"object_name": "035_apple@1"},
    "woodenblock_m1": {"object_name": "086_woodenblock@1"},
    "seal_m1": {"object_name": "100_seal@1"},
    "rubikscube_m1": {"object_name": "073_rubikscube@1"},
    "coffee_box_m0": {"object_name": "113_coffee-box@0", "close_value": 0.42},
    "tea_box_m0": {"object_name": "112_tea-box@0", "close_value": 0.45},
    "tissue_box_m0": {"object_name": "023_tissue-box@0", "close_value": 0.46},
    "rubikscube_m0": {"object_name": "073_rubikscube@0"},
    # model 0 is a near-cube used by the factory tasks.  Do not inherit the
    # +3 cm high-body correction tuned for the taller model 1 block.
    "woodenblock_m0": {"object_name": "086_woodenblock@0", "close_value": 0.44,
                       "grasp_z_offset": 0.0},

    # Fixed and task-specific mesh targets.
    "shoe_m0": {"object_name": "041_shoe@0", "grasp_method": "annotated",
                "grasp_name": "manual_topdown_center_y000", "close_value": 0.20},
    "bowl_m1": {"object_name": "002_bowl@1", "grasp_method": "annotated",
                "grasp_name": "dish_rack_rim", "close_value": 0.025,
                "pin_during_close": True, "spawn_quat": Q_UPRIGHT},
    "bread_m0": {"object_name": "075_bread@0", "grasp_method": "annotated",
                 "grasp_name": "manual_topdown_center_yaw45", "close_value": 0.35,
                 "logical_center": True, "logical_z_offset": -0.011,
                 "spawn_quat": Q_UPRIGHT},
    "bread_m1": {"object_name": "075_bread@1", "close_value": 0.30},
    "olive_oil_m0": {"object_name": "029_olive-oil@0", "grasp_method": "annotated",
                     "grasp_categories": ["pour_side_primary"], "close_value": 0.04,
                     "spawn_quat": Q_UPRIGHT, "lift_height": 0.12},
    "mug_m0": {"object_name": "039_mug@0", "grasp_method": "annotated",
               "grasp_name": "manual_body_edge_topdown_posx_posz",
               "spawn_quat": Q_MUG_HANDLE_TOWARD_ROBOT},
    "knife_m0": {"object_name": "034_knife@0", "grasp_method": "annotated",
                 "grasp_name": "manual_000", "close_value": 0.10},
    "hamburger_m5": {"object_name": "006_hamburg@5", "close_value": 0.50},
    "french_fries_m0": {"object_name": "005_french-fries@0", "close_value": 0.25},
    "skillet_m0": {"object_name": "106_skillet@0", "grasp_method": "annotated",
                   # Handle points along +Y at this spawn; yaw 90 closes the
                   # fingers across its narrow width, matching the task path.
                   "grasp_name": "manual_topdown_handle_nearbody_yaw90",
                   "close_value": 0.22, "spawn_quat": Q_UPRIGHT},
    "apple_m0": {"object_name": "035_apple@0"},
    "screwdriver_m0": {"object_name": "032_screwdriver@0",
                       # The Franka YAML is a pruned seven-grasp validated set;
                       # lay the tool flat exactly as factory_line_feeding does.
                       "grasp_method": "annotated",
                       "grasp_name": "grasp_075",
                       "spawn_quat": Q_FACTORY_TOOL_FLAT_X,
                       "close_value": 0.22},
    "hammer_m0": {"object_name": "020_hammer@0", "grasp_method": "factory_tool",
                  "spawn_quat": Q_FACTORY_TOOL_FLAT_X,
                  "factory_grasp_offset": (0.0, -0.045, 0.0),
                  "factory_grasp_axis": (1.0, 0.0, 0.0),
                  "close_value": 0.18},

    # Size-ranking primitive variants (the 40 mm RGB/dump block is above).
    "block_36mm": {"object_name": "block-small", "primitive_half": 0.018},
    "block_48mm": {"object_name": "block-medium", "primitive_half": 0.024},
    "block_60mm": {"object_name": "block-large", "primitive_half": 0.030},
}


class GraspDemoTask(DumpBin):
    """One-object scene configured at runtime for any catalog pool."""

    EASY_NUM_OBJECTS = 1
    HARD_NUM_OBJECTS = 1
    recording_camera_pos = [0.90, -0.72, 1.42]
    recording_camera_lookat = [0.08, -0.20, 0.84]
    side_camera_pos = [-0.72, -0.62, 1.15]
    side_camera_lookat = [0.08, -0.20, 0.83]

    def _sample_dump_object_spec(self, idx: int) -> dict:
        spec = super()._sample_dump_object_spec(idx)
        if self.config.get("demo_grasp_method") == "annotated":
            spec["quat"] = read_save_object_quat(
                str(spec["object_id"]), robot_type="franka",
            )
        spawn_quat = self.config.get("demo_spawn_quat")
        if spawn_quat is not None:
            spec["quat"] = tuple(float(x) for x in spawn_quat)
        return spec


def _pick_center(task: GraspDemoTask, actor, label: str, model_id: int):
    if label == "cube":
        return task._object_pos(actor)
    return task._get_object_world_center(actor, label, model_id)


def _actor_pose(actor) -> Pose:
    """Return a Pose for both catalog Actor wrappers and raw primitives."""
    if hasattr(actor, "get_pose"):
        return actor.get_pose()
    entity = getattr(actor, "entity", actor)
    return Pose(
        to_numpy(entity.get_pos()).ravel()[:3],
        to_numpy(entity.get_quat()).ravel()[:4],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(DEMO_PROFILES))
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--pool")
    parser.add_argument("--object", dest="object_name")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-stem")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--grasp-method", choices=("topdown", "annotated", "side", "factory_tool"),
    )
    parser.add_argument("--close-value", type=float)
    parser.add_argument("--grasp-name")
    parser.add_argument("--grasp-category", action="append", dest="grasp_categories")
    parser.add_argument("--pre-dist", type=float)
    parser.add_argument("--spawn-quat", type=float, nargs=4, metavar=("W", "X", "Y", "Z"))
    parser.add_argument("--factory-grasp-offset", type=float, nargs=3)
    parser.add_argument("--factory-grasp-axis", type=float, nargs=3)
    parser.add_argument(
        "--logical-center", action="store_true",
        help="Use the live bbox centre as the annotation-frame origin.",
    )
    parser.add_argument("--logical-z-offset", type=float, default=0.0)
    parser.add_argument(
        "--pin-during-close", action="store_true",
        help="Hold the object at its live pose during the gripper close only.",
    )
    parser.add_argument("--primitive-half", type=float)
    parser.add_argument("--lift-height", type=float)
    parser.add_argument(
        "--grasp-z-offset", type=float,
        help="Task-local offset applied only to the top-down grasp target.",
    )
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--hold-steps", type=int, default=100)
    args = parser.parse_args()

    if args.list_profiles:
        print("\n".join(DEMO_PROFILES))
        return 0
    if args.profile:
        profile = DEMO_PROFILES[args.profile]
        for key, value in profile.items():
            current = getattr(args, key)
            if current is None or current is False:
                setattr(args, key, value)
        if args.output_stem is None:
            args.output_stem = args.profile
    if not args.object_name:
        parser.error("one of --profile or --object is required")

    for demo_entry in DEMO_ONLY_ENTRIES:
        CATALOG.setdefault(demo_entry.key, demo_entry)

    object_key = args.object_name.partition("@")[0]
    grasp_method = args.grasp_method or DEFAULT_GRASP_METHODS.get(
        object_key, "topdown"
    )

    GraspDemoTask.OBJECT_SET = args.pool or [args.object_name]
    if args.primitive_half is not None:
        GraspDemoTask.OBJECT_HALF = float(args.primitive_half)
    cfg = {
        "robot_type": "franka",
        "robot_single_arm": True,
        "object_name": args.object_name,
        "random_dump_objects": True,
        "renderer": "rasterizer",
        "show_viewer": False,
        "table_random_objects": False,
        "video_stride": 8,
        "video_quality": 8,
        "demo_grasp_method": grasp_method,
        "demo_spawn_quat": args.spawn_quat,
        # This is a grasp/lift clip, so do not add the task mixin's placement
        # settle after the object reaches its elevated hold pose.
        "pick_place_settle_above_place_steps": 0,
        "pick_place_hold_steps": 0,
    }
    task = GraspDemoTask(cfg)
    task.reset(seed=args.seed)

    entry = task._target_entry
    object_id = entry.object_id or entry.key
    model_id = int(entry.model_id)
    actor = task.objects[0]
    radius = float(task._object_pick_radii[id(actor)])
    label = str(task._object_labels[id(actor)])
    close_value = task._object_close_values.get(id(actor))
    if object_id in FACTORY_CLOSE_VALUES:
        close_value = FACTORY_CLOSE_VALUES[object_id]
    if args.close_value is not None:
        close_value = float(args.close_value)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = args.output_stem or object_id
    video_path = output_dir / f"{output_stem}.mp4"
    result_path = output_dir / f"{output_stem}.json"
    task.start_video(str(video_path))

    task._boost_finger_pd("right")
    for _ in range(max(0, int(args.settle_steps))):
        task.step_sim()

    center_before = np.asarray(
        _pick_center(task, actor, label, model_id), dtype=float,
    )
    hold_z = float(task.TABLE_TOP_Z + 0.24)
    grasp_name = None
    if grasp_method in ("annotated", "side", "factory_tool"):
        with open(
            ASSETS_PATH / "objects" / object_id / f"model_data{model_id}.json"
        ) as f:
            raw_scale = json.load(f).get("scale", 1.0)
        object_scale = float(
            raw_scale[0] if isinstance(raw_scale, (list, tuple)) else raw_scale
        )
        object_pose = actor.get_pose()
        if args.logical_center:
            logical_p = np.asarray(
                _pick_center(task, actor, label, model_id), dtype=float,
            )
            logical_p[2] += float(args.logical_z_offset)
            object_pose = Pose(logical_p, object_pose.q)
        task.open_gripper("right")
        grasp_tcp_override = None
        if grasp_method == "factory_tool":
            import transforms3d as t3d

            if args.factory_grasp_offset is None or args.factory_grasp_axis is None:
                raise ValueError("factory_tool requires offset and grasp axis")
            object_center = np.asarray(
                _pick_center(task, actor, label, model_id), dtype=float,
            )
            object_R = t3d.quaternions.quat2mat(
                np.asarray(actor.get_pose().q, dtype=float),
            )
            target = object_center + object_R @ np.asarray(
                args.factory_grasp_offset, dtype=float,
            )
            close_axis = object_R @ np.asarray(
                args.factory_grasp_axis, dtype=float,
            )
            yaw = float(np.arctan2(close_axis[1], close_axis[0]) - np.pi / 2.0)
            yaw = float(np.arctan2(np.sin(yaw), np.cos(yaw)))
            c, s = np.cos(yaw), np.sin(yaw)
            Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            grasp_tcp_override = Pose(
                target,
                t3d.quaternions.mat2quat(task._R_DOWN @ Rz),
            )
            world_grasp = GraspPose(
                name="factory_flat_handle_topdown",
                position=grasp_tcp_override.p,
                quaternion=grasp_tcp_override.q,
                pre_distance=float(args.pre_dist or 0.12),
                source="manual",
                scale_frame="runtime",
            )
            plans = task._plan_one_grasp(
                world_grasp, Pose(), "right", object_scale=1.0,
                log_prefix="[factory-tool] ",
            )
            if plans is None:
                grasp_result = None
            else:
                result_pre, result_grasp, grasp_link, pre_link = plans
                task.execute_plan(result_pre, "right")
                task.execute_plan(result_grasp, "right")
                grasp_result = (grasp_link, pre_link, world_grasp)
        elif grasp_method == "side":
            grasp_name = args.grasp_name or SIDE_GRASP_NAMES.get(object_id)
            if grasp_name is None:
                raise ValueError(f"No side grasp configured for {object_id}")
            grasp_result = task.try_grasp_by_name(
                object_name=object_id,
                object_pose=object_pose,
                arm_tag="right",
                grasp_name=grasp_name,
                model_id=model_id,
                robot_type="franka",
                object_scale=object_scale,
            )
        elif args.grasp_name:
            grasp_result = task.try_grasp_by_name(
                object_name=object_id,
                object_pose=object_pose,
                arm_tag="right",
                grasp_name=args.grasp_name,
                model_id=model_id,
                robot_type="franka",
                object_scale=object_scale,
                pre_dist=args.pre_dist,
            )
        else:
            grasp_result = task.select_and_execute_grasp(
                object_name=object_id,
                object_pose=object_pose,
                arm_tag="right",
                model_id=model_id,
                robot_type="franka",
                max_candidates=40,
                object_scale=object_scale,
                categories=args.grasp_categories,
                pre_dist=args.pre_dist,
            )
        planned = grasp_result is not None
        if planned:
            _grasp_link, _pre_link, grasp = grasp_result
            grasp_name = grasp.name
            if args.pin_during_close:
                pin_pose = actor.get_pose()
                target = 0.0 if close_value is None else float(close_value)
                arm = task.robot.get_arm("right")
                for value in np.linspace(float(arm.gripper_val), target, 200):
                    actor.entity.set_pos(np.asarray(pin_pose.p, dtype=float))
                    actor.entity.set_quat(np.asarray(pin_pose.q, dtype=float))
                    try:
                        actor.entity.set_dofs_velocity(np.zeros(6, dtype=float))
                    except Exception:
                        pass
                    task.robot.set_gripper(float(value), "right")
                    task.step_sim()
            elif close_value is None:
                task.close_gripper("right")
            else:
                task.set_gripper(float(close_value), "right", num_steps=200)
            grasp_tcp = (
                grasp_tcp_override
                if grasp_tcp_override is not None
                else grasp.to_world(object_pose, object_scale)
            )
            approach_axis = grasp_tcp.to_matrix()[:3, 2]
            print(
                f"[grasp_demo] grasp={grasp.name} tcp={grasp_tcp.p} "
                f"approach_axis={approach_axis}",
                flush=True,
            )
            lift_tcp = Pose(
                grasp_tcp.p + np.array([
                    0.0, 0.0,
                    float(args.lift_height if args.lift_height is not None else 0.20),
                ]),
                grasp_tcp.q,
            )
            lift_link = tcp_to_link_pose(
                lift_tcp, task.robot.get_arm("right").tcp_offset,
            )
            # A fresh global RRT may satisfy the same lift pose through a
            # distant redundant IK branch (the oil demo once wound joint 7
            # by 5.67 rad).  A lift is a local constant-orientation motion:
            # require the branch-continuous screw plan and reject large joint
            # excursions instead of accepting a visually absurd wrist spin.
            arm = task.robot.get_arm("right")
            start_full_q = np.asarray(arm.get_arm_qpos(), dtype=float)
            start_q = start_full_q.ravel()[:7]
            lift_result = None
            # Screw is preferred.  A seeded IK interpolation is the local
            # fallback, followed by a few RRT redraws.  Every candidate is
            # capped at 2 rad from the grasp configuration, so an equivalent
            # 2π wrist branch can never be accepted.
            lift_planners = (
                arm.planner.plan_screw_path,
                arm.planner.solve_ik,
                arm.planner.plan_path,
                arm.planner.plan_path,
                arm.planner.plan_path,
            )
            for plan_fn in lift_planners:
                candidate = plan_fn(start_full_q, lift_link.to_pose7())
                if candidate is None or not candidate.success or candidate.position.size == 0:
                    continue
                travel = float(np.max(np.abs(
                    np.asarray(candidate.position, dtype=float)[:, :7]
                    - start_q[None, :]
                )))
                if travel > 2.0:
                    print(
                        f"[grasp_demo] rejecting lift joint travel={travel:.3f}rad",
                        flush=True,
                    )
                    continue
                lift_result = candidate
                break
            planned = lift_result is not None
            if planned:
                task.execute_plan(lift_result, "right")
    else:
        pick = PickSpec(
            get_center=lambda: _pick_center(task, actor, label, model_id),
            radius=radius,
            label=label,
            close_value=close_value,
            grasp_z_offset=float(
                args.grasp_z_offset
                if args.grasp_z_offset is not None
                else entry.topdown_grasp_z_offset or 0.0
            ),
            approach_xy_offset=entry.topdown_approach_xy_offset or (0.0, 0.0),
            preopen_steps=int(entry.topdown_preopen_steps or 0),
        )
        place = PlaceSpec(
            pos=np.array([center_before[0], center_before[1], hold_z]),
            label="elevated hold",
            transport_z=hold_z,
            release=False,
            stop_after_lift=True,
        )
        planned = bool(task.pick_and_place(pick, place, "right"))

    center_at_lift = np.asarray(
        _pick_center(task, actor, label, model_id), dtype=float,
    )
    print(
        f"[grasp_demo] immediate lift dz={center_at_lift[2] - center_before[2]:.4f}m",
        flush=True,
    )
    # Render every hold step.  The approach remains downsampled for compact
    # videos, while the final 100 simulation steps become 100 visible frames
    # (~3.3 s at 30 fps) instead of the previous barely visible 12 frames.
    task._video_stride = 1
    task._video_cap_tick = 0
    hold_frame_start = len(task._video_frames)
    arm = task.robot.get_arm("right")
    ee_at_lift = Pose.from_pose7(arm.get_ee_pose())
    object_pose_at_lift = _actor_pose(actor)
    rel_at_lift = ee_at_lift.inv() * object_pose_at_lift
    hold_rel_drifts = []
    hold_zs = []
    for _ in range(max(0, int(args.hold_steps))):
        task.step_sim()
        ee_now = Pose.from_pose7(arm.get_ee_pose())
        object_pose_now = _actor_pose(actor)
        rel_now = ee_now.inv() * object_pose_now
        hold_rel_drifts.append(float(np.linalg.norm(
            np.asarray(rel_now.p, dtype=float)
            - np.asarray(rel_at_lift.p, dtype=float)
        )))
        hold_zs.append(float(_pick_center(task, actor, label, model_id)[2]))
    center_after = np.asarray(
        _pick_center(task, actor, label, model_id), dtype=float,
    )
    hold_frames = len(task._video_frames) - hold_frame_start
    lift_dz = float(center_after[2] - center_before[2])
    hold_end_relative_drift = float(hold_rel_drifts[-1] if hold_rel_drifts else 0.0)
    hold_max_relative_drift = float(max(hold_rel_drifts, default=0.0))
    hold_z_drop = float(
        center_at_lift[2] - center_after[2]
        if hold_zs else 0.0
    )
    # A lifted-but-falling object is not a successful hold.  The limits are
    # intentionally much tighter than one finger width: stable objects move
    # only a few millimetres, while the mug's brief compliant settling peaks
    # below 15 mm.  This rejects any visibly progressive slide.
    retained = bool(
        hold_end_relative_drift <= 0.015
        and hold_max_relative_drift <= 0.020
    )
    success = bool(
        planned
        and lift_dz >= 0.05
        and retained
        and hold_frames == max(0, int(args.hold_steps))
    )
    if task._video_frames:
        import imageio.v2 as imageio
        imageio.imwrite(output_dir / f"{output_stem}.png", task._video_frames[-1])
    task.save_video(fps=30)

    result = {
        "pool": args.pool,
        "object_token": args.object_name,
        "object_name": object_id,
        "model_id": model_id,
        "planned": planned,
        "grasp_method": grasp_method,
        "grasp_name": grasp_name,
        "sequence": "grasp_lift_hold",
        "hold_steps": int(args.hold_steps),
        "hold_frames": hold_frames,
        "hold_end_relative_drift_m": hold_end_relative_drift,
        "hold_max_relative_drift_m": hold_max_relative_drift,
        "hold_z_drop_m": hold_z_drop,
        "retained_through_hold": retained,
        "lift_dz": lift_dz,
        "success": success,
        "video": video_path.name,
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"GRASP_DEMO_RESULT {json.dumps(result, sort_keys=True)}", flush=True)
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
