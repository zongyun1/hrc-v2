"""Deliver to human (easy): pick up an object and deliver it to a static human's hand.

Per-object grasp method (selected by ``OBJECT_GRASP_METHOD``):
  - ``annotated``: read grasp pose from ``grasp_poses_franka.yml``.  Object
    is spawned with ``save_object_quat`` so the annotation is geometrically
    consistent.  Uses the shared annotated-grasp delivery pipeline.
  - ``primitive``: top-down approach via ``TopDownPickPlaceMixin``.  Object
    spawned with identity quat; gripper descends straight down with
    ``radius * 0.7`` close target.  Works well for objects whose smallest
    cross-section fits the Franka gripper.

Choice was tuned against coverage runs 56254065 (primitive) and 56262744
(annotated): each object goes with whichever method scored ≥ 50%.

Object selection is sampled by default from the shared
``human_transfer_pool``.  Config ``object_name`` / ``model_id`` remains an
explicit override, and ``random_object: false`` selects the pool's first item
for deterministic compatibility runs.

Both methods end with the gripper held at the deliver pose (no release) —
the arm extends toward the avatar's right hand and stays there.
"""

import json

import numpy as np
import transforms3d as t3d

from ..avatar.eval_mode_mixin import EvalModeAvatarMixin
from ..base_task import BaseTask, TargetSpec
from ..manipulation import TopDownPickPlaceMixin, PickSpec, PlaceSpec
from ..utils import (
    ASSETS_PATH,
    Pose,
    load_object,
    create_primitive,
)
from ..grasp import read_save_object_quat, tcp_to_link_pose


class DeliverToHumanEasy(EvalModeAvatarMixin, TopDownPickPlaceMixin, BaseTask):
    """Pick up an object and deliver to static avatar; per-object grasp method."""

    INSTRUCTION = "pick up the {object} from the table and deliver it to the human"
    OBJECT_SET = "human_transfer_pool"
    use_avatar = True
    avatar_init_pos = np.array([0.0, 1.5, -0.18])
    avatar_init_rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)

    # Preserve the avatar-side edge at y=+0.87, but extend the robot-side edge
    # to y=-0.50 so the Franka base at y=-0.35 sits 0.15 m inboard.
    TABLE_HALF_SIZE = (0.53, 0.685)
    table_offset = np.array([0.0, 0.185])

    recording_camera_pos = [1.3, 0.6, 2.5]
    recording_camera_lookat = [0.0, 0.4, 0.75]

    DEFAULT_OBJ_NAME = "035_apple"
    DEFAULT_OBJ_MODEL_ID = 0

    # Per-object method selection — empirical from coverage runs 56254065
    # (primitive), 56262744 (annotated), 56268109 (per-object), 56393175
    # (post first synthetic-grasp resample), 56395629 (max_candidates
    # 10 → 40 so RRT can fall back to negative-score arm-side grasps),
    # 56396866 (after a 10x denser synthetic resample for the still-
    # failing three).  Only objects with >= 50% success are kept.
    # Synthetic grasps came from ``preprocess_grasp_flying_gripper.py``
    # (jobs 56360674 then 56396642).  Still below threshold:
    # 046_alarm-clock (1/3), 018_microphone (0/3 — synth grasps below
    # the table-height filter).
    OBJECT_GRASP_METHOD = {
        "048_stapler": "annotated",       # 3/3
        "024_scanner": "annotated",       # 1/1 (small sample)
        "023_tissue-box": "annotated",    # 3/3 (synthetic-augmented)
        "065_soy-sauce": "annotated",     # 2/3 (synthetic + max_cand=40)
        "113_coffee-box": "annotated",    # 2/3 (10x synthetic resample)
        "005_french-fries": "primitive",  # 2/3
        "052_dumbbell": "annotated",      # 2/3 (manual_topdown_X/Y_center)
        "039_mug": "annotated",           # 3/3 (manual_topdown_X/Y_center)
        "047_mouse": "annotated",         # 3/3 (manual_topdown_yaw45_*)
        "086_woodenblock": "annotated",   # 3/3 (m1 + manual_topdown_yaw45_*)
        "020_hammer": "annotated",        # 2/3 (lay-flat save_quat + manual_topdown_X_center)
        "035_apple": "annotated",         # validated m1 top-down annotation
        "073_rubikscube": "primitive",    # compact box; radius-based top-down pick
        "029_olive-oil": "annotated",     # 2/3 (m1 + lay-flat + B_higher)
        "066_vinegar": "annotated",       # 2/3 (m1 + lay-flat + A_center)
        "095_glue": "annotated",          # 2/3 (m6 + lay-flat + A_center)
        "100_seal": "annotated",          # 3/3 (m2 + tight spawn + A_center, round-5)
        "055_small-speaker": "annotated", # 2/3 (m1 + A_center, round-6 re-run on new helper)
        "038_milk-box": "annotated",      # object-aug: grasp 3/3, deliver seed 512 pass
    }

    # Per-object model_id override (default 0).  Some objects have a
    # smaller variant that fits the franka 8 cm gripper better.
    OBJECT_MODEL_ID = {
        "035_apple": 1,        # 5.4 cm (vs 6.6 cm at model 0)
        "086_woodenblock": 1,  # 7.1 × 10.2 × 7.1 cm (vs 10.2 cube at model 0)
        "029_olive-oil": 1,    # body 7.7 cm (vs 13.3 cm at m0)
        "066_vinegar": 1,      # body 8.9 cm (vs 16.6 cm at m0)
        "030_drill": 1,        # 5.3 × 15.4 × 15.4
        "100_seal": 2,         # 3.6 × 7.8 × 3.2 — small handle stamp
        "095_glue": 6,         # 4.3 × 15.5 × 4.3 — slim glue stick
        "055_small-speaker": 1,
    }

    # Per-object gripper close target (None = full close).  Some objects
    # need a partial close so the gripper doesn't squeeze them out.
    OBJECT_CLOSE_TARGET = {
        "035_apple": 0.55,     # round apple — full close rolls it out
    }

    # Per-object spawn (x_lo, x_hi, y_lo, y_hi). Default applies when missing.
    # 100_seal: small stamp, IK marginal at edges of default range; this tight
    # corner is where round-5 A_center hit 3/3.
    OBJECT_SPAWN_RANGE = {
        "100_seal":     ( 0.00, 0.05, -0.10, -0.07),
    }

    TASK_OBJECTS = ("035_apple", "073_rubikscube")
    SUCCESS_DIST_3D_M = 0.65
    # The hand is above the tabletop; a table-resting object can be within
    # 0.65 m in 3-D just because the avatar stands nearby. Require the final
    # object to be no more than 40 cm below the palm, which still allows the
    # reachable deliver pose but rejects dropped/unpicked table objects.
    SUCCESS_DZ_MIN_M = -0.40
    SUCCESS_TABLE_CLEARANCE_M = 0.05
    FINAL_HOLD_SETTLE_STEPS = 300
    DELIVER_HAND_STANDOFF_M = 0.18
    DELIVER_HAND_Z_OFFSET_M = -0.05

    def __init__(self, config):
        super().__init__(config)
        # The actual entry is sampled in load_actors, after reset(seed) has
        # seeded numpy.  ``random_object`` is retained as a compatibility knob;
        # random selection is now the default.
        self.obj_name = None
        self.obj_model_id = self.DEFAULT_OBJ_MODEL_ID
        self.random_object = bool(config.get("random_object", True))
        self.grasp_method_override = config.get("grasp_method")

    def _grasp_method(self) -> str:
        if self.grasp_method_override:
            return self.grasp_method_override
        return self.OBJECT_GRASP_METHOD.get(self.obj_name, "annotated")

    def _load_robot(self):
        self.config.setdefault("robot_type", "franka")
        if self.config["robot_type"] == "franka":
            kwargs = self.config.setdefault("robot_kwargs", {})
            kwargs.setdefault("pos", [0.0, -0.35, 0.75])
            kwargs.setdefault("quat", [0.707, 0.0, 0.0, 0.707])
        super()._load_robot()

    def _create_table(self, table_height=0.74):
        dx, dy = self.table_offset
        return self._create_rectangular_table(
            table_height=table_height,
            half_size=self.TABLE_HALF_SIZE,
            center_xy=(dx, dy),
        )

    def _settle_scene(self, n_steps: int = None):
        if not getattr(self, "_avatar_head_hand_aligned", False):
            target = self._get_object_world_center(
                self.obj_actor, self.obj_name, self.obj_model_id
            )
            self._avatar_head_hand_yaw_deg = self._align_avatar_head_hand_to_xy(
                target[:2], hand_id=1, motion_name="take_from_human",
            )
            self._avatar_head_hand_aligned = True
        super()._settle_scene(n_steps)

    def load_actors(self):
        entry = self.resolve_target_object(randomize=self.random_object)
        self.obj_name = entry.object_id or entry.key
        self.obj_model_id = int(self.config.get("model_id", entry.model_id))

        table_top = self.TABLE_TOP_Z + 0.02
        # Annotated method needs the spawn quat that matches the grasp's
        # frame-of-reference; primitive top-down doesn't care, identity is fine.
        if self._grasp_method() == "annotated":
            obj_q = read_save_object_quat(self.obj_name, robot_type="franka")
        else:
            obj_q = np.array([1.0, 0.0, 0.0, 0.0])

        # Spawn region: per-object override falls back to default range that
        # stays inside the dexterous workspace at any annotated grasp orientation.
        spawn = self.OBJECT_SPAWN_RANGE.get(self.obj_name, (-0.10, 0.15, -0.15, -0.05))
        obj_x = np.random.uniform(spawn[0], spawn[1])
        obj_y = np.random.uniform(spawn[2], spawn[3])
        self.obj_actor = load_object(
            self.scene, Pose([obj_x, obj_y, table_top], obj_q), self.obj_name,
            model_id=self.obj_model_id, convex=True, is_static=False,
        )

        self.target = TargetSpec(
            object=self.obj_actor,
            position=lambda: self.avatar.robot.get_palm_center(hand_id=1),
            label="right_palm",
            object_pos_fn=lambda: self._get_object_world_center(
                self.obj_actor, self.obj_name, self.obj_model_id,
            ),
            success_dist_3d=self.SUCCESS_DIST_3D_M,
            success_dz_min=self.SUCCESS_DZ_MIN_M,
            success_require_no_avatar_collision=True,
        )

    def _object_bottom_z(self) -> float:
        md_path = (
            ASSETS_PATH / "objects" / self.obj_name
            / f"model_data{self.obj_model_id}.json"
        )
        with open(md_path) as f:
            md = json.load(f)
        extents = np.asarray(md["extents"], dtype=np.float64)
        scale = np.asarray(md.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
        if scale.size == 1:
            scale = np.repeat(scale, 3)
        center = np.asarray(md.get("center", [0.0, 0.0, 0.0]), dtype=np.float64)
        half = 0.5 * extents * scale
        center = center * scale
        corners = np.array([
            center + np.array([sx * half[0], sy * half[1], sz * half[2]])
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ])
        pose = self.obj_actor.get_pose()
        R = t3d.quaternions.quat2mat(np.asarray(pose.q, dtype=np.float64))
        world = np.asarray(pose.p, dtype=np.float64) + corners @ R.T
        return float(np.min(world[:, 2]))

    def _object_table_clearance(self) -> float:
        return self._object_bottom_z() - float(self.TABLE_TOP_Z)

    def _object_still_held(self) -> bool:
        return self._object_table_clearance() > float(self.SUCCESS_TABLE_CLEARANCE_M)

    def _clear_avatar_collision_state(self) -> None:
        """Start grading after the avatar's presentation pre-roll."""
        self.avatar_collided = False
        self.avatar_collision_log = []
        self._avatar_collision_tick = 0

    # ------------------------------------------------------------------
    # Deliver-pose helpers
    # ------------------------------------------------------------------
    def _deliver_target_and_direction(self, arm_tag: str) -> tuple[np.ndarray, np.ndarray]:
        """Return target xyz and horizontal gripper direction toward the palm."""
        arm = self.robot.get_arm(arm_tag)
        arm_base = np.asarray(arm.origin_pose.p, dtype=np.float64)
        palm = np.asarray(self.avatar.robot.get_palm_center(hand_id=1), dtype=np.float64)

        palm_to_robot = arm_base[:2] - palm[:2]
        n = np.linalg.norm(palm_to_robot)
        if n < 1e-6:
            palm_to_robot = np.array([0.0, -1.0], dtype=np.float64)
        else:
            palm_to_robot = palm_to_robot / n

        target = palm.copy()
        target[:2] = palm[:2] + palm_to_robot * float(self.DELIVER_HAND_STANDOFF_M)
        target[2] = palm[2] + float(self.DELIVER_HAND_Z_OFFSET_M)

        toward_palm = palm[:2] - target[:2]
        n = np.linalg.norm(toward_palm)
        toward_palm = toward_palm / n if n > 1e-6 else -palm_to_robot
        return target, toward_palm

    def _deliver_pos(self, arm_tag: str) -> np.ndarray:
        """Object-center target for the primitive top-down delivery path."""
        target, _ = self._deliver_target_and_direction(arm_tag)
        return target

    def _compute_deliver_pose(self, arm_tag: str) -> Pose:
        """Link pose with TCP z=down and TCP x pointing toward the palm.

        Used by the annotated path via ``move_and_execute``.
        """
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset

        deliver_pos, toward_palm = self._deliver_target_and_direction(arm_tag)

        tcp_z = np.array([0.0, 0.0, -1.0])
        tcp_x = np.array([toward_palm[0], toward_palm[1], 0.0])
        tcp_x /= np.linalg.norm(tcp_x) + 1e-8
        tcp_y = np.cross(tcp_z, tcp_x)
        tcp_y /= np.linalg.norm(tcp_y) + 1e-8

        R = np.column_stack([tcp_x, tcp_y, tcp_z])
        q = t3d.quaternions.mat2quat(R)
        deliver_tcp = Pose(deliver_pos, q)
        return tcp_to_link_pose(deliver_tcp, tcp_offset)

    def _move_screw_to_link(self, link_pose7, arm_tag: str) -> bool:
        """Straight-line Cartesian move to a link pose (mplib plan_screw).

        Falls back to ``move_and_execute`` (RRT plan_pose) if screw planning
        fails. Returns True on success.
        """
        arm = self.robot.get_arm(arm_tag)
        planner = arm.planner
        plan_screw_path = getattr(planner, "plan_screw_path", None)
        if plan_screw_path is None:
            return self.move_and_execute(link_pose7, arm_tag) is not None
        result = plan_screw_path(arm.get_arm_qpos(), np.asarray(link_pose7).ravel()[:7])
        if result is not None and getattr(result, "success", False):
            self.execute_plan(result, arm_tag)
            return True
        return self.move_and_execute(link_pose7, arm_tag) is not None

    # ------------------------------------------------------------------
    # Per-method play_once
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Eval-mode hook (single-shot at episode start): same as play_once's
    # opening, just queue the take_from_human animation and let the
    # avatar tick through it as the policy steps the simulator.
    # ------------------------------------------------------------------
    def _eval_at_reset(self):
        self.avatar.play_animation("take_from_human")

    def eval_pre_policy_warmup(self):
        if (
            self.avatar is None
            or not bool(self.config.get("eval_mode", False))
            or not bool(self.config.get("eval_avatar_pre_policy", True))
        ):
            return self.get_obs()
        while not self.avatar.spare():
            self.step_sim()
        for _ in range(int(self.config.get("eval_avatar_post_warmup_settle_steps", 50))):
            self.step_sim()
        self._clear_avatar_collision_state()
        return self.get_obs()

    def play_once(self) -> bool:
        if self.avatar is not None:
            self.avatar.play_animation("take_from_human")
            while not self.avatar.spare():
                self.step_sim()

        for _ in range(50):
            self.step_sim()
        # The opening human-only "take_from_human" presentation can move
        # randomized avatar hands/forearms through the parked robot before
        # the robot has started the delivery.  That pre-roll is scene setup,
        # not a robot-avatar safety violation for the scripted rollout.
        self._clear_avatar_collision_state()

        method = self._grasp_method()
        if method == "annotated":
            return self._play_once_annotated()
        else:
            return self._play_once_primitive()

    def _play_once_annotated(self) -> bool:
        arm_tag = "right"
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset

        obj_pose = self.obj_actor.get_pose()
        self._z0 = float(obj_pose.p[2])
        self._stage_failed = None

        with open(
            ASSETS_PATH / "objects" / self.obj_name
            / f"model_data{self.obj_model_id}.json"
        ) as f:
            raw_scale = json.load(f).get("scale", 1.0)
        object_scale = float(
            raw_scale[0] if isinstance(raw_scale, (list, tuple)) else raw_scale
        )

        self.open_gripper(arm_tag)
        result = self.select_and_execute_grasp(
            self.obj_name, obj_pose, arm_tag,
            max_candidates=40, model_id=self.obj_model_id,
            robot_type="franka",
            object_scale=object_scale,
        )
        if result is None:
            self._stage_failed = "select_grasp"
            return False

        grasp_link, pre_link, grasp = result
        self._grasp_used = grasp.name
        grasp_tcp = grasp.to_world(obj_pose, object_scale)

        close_target = self.OBJECT_CLOSE_TARGET.get(self.obj_name)
        if close_target is None:
            self.close_gripper(arm_tag)
        else:
            self.robot.set_gripper(close_target, arm_tag)
            for _ in range(60):
                self.step_sim()

        lift_tcp = Pose(grasp_tcp.p + np.array([0.0, 0.0, 0.12]), grasp_tcp.q)
        lift_link = tcp_to_link_pose(lift_tcp, tcp_offset)
        if self.move_and_execute(lift_link.to_pose7(), arm_tag) is None:
            self._stage_failed = "move_lift"
            return False

        deliver_link = self._compute_deliver_pose(arm_tag)
        if self.move_and_execute(deliver_link.to_pose7(), arm_tag) is None:
            self._stage_failed = "move_deliver"
            return False

        for _ in range(self.FINAL_HOLD_SETTLE_STEPS):
            self.step_sim()

        final_pose = self.obj_actor.get_pose()
        self._z1 = float(final_pose.p[2])
        self._held = self._object_still_held()
        return self._held

    def _play_once_primitive(self) -> bool:
        arm_tag = "right"
        obj_pose = self.obj_actor.get_pose()
        self._z0 = float(obj_pose.p[2])

        pick = PickSpec(
            get_center=lambda: self._get_object_world_center(
                self.obj_actor, self.obj_name, self.obj_model_id
            ),
            radius=self._get_object_cross_section_radius(self.obj_name, self.obj_model_id),
            label=self.obj_name,
        )
        deliver_pos = self._deliver_pos(arm_tag)
        transport_z = max(float(deliver_pos[2]), self.TABLE_TOP_Z + 0.20) + 0.05
        place = PlaceSpec(
            pos=deliver_pos,
            label="hand",
            transport_z=transport_z,
            release=False,
        )
        ok = self.pick_and_place(pick, place, arm_tag)
        if not ok:
            return False
        for _ in range(self.FINAL_HOLD_SETTLE_STEPS):
            self.step_sim()

        final_pose = self.obj_actor.get_pose()
        self._z1 = float(final_pose.p[2])
        self._held = self._object_still_held()
        return self._held

    def check_success(self) -> bool:
        # `self._held` is set only by the scripted `play_once` path. Under
        # policy eval `play_once` never runs, so the attribute is absent and
        # the old `getattr(..., False)` gate made `check_success` return
        # False for *every* eval episode regardless of the policy. Gate on
        # the live hold check (`_object_still_held`) instead — it recomputes
        # the same thing from the current scene and works in both flows.
        if not self.plan_success or not self._object_still_held():
            return False
        if self.avatar_collision_summary().get("any_collision"):
            return False
        metrics = self._compute_target_metrics()
        if metrics is None:
            return False
        if metrics["dist_3d"] > float(self.SUCCESS_DIST_3D_M):
            return False
        if metrics["dz"] < float(self.SUCCESS_DZ_MIN_M):
            return False
        return True

    def evaluate(self) -> dict:
        out = super().evaluate()
        out["object_name"] = self.obj_name
        out["object_model_id"] = int(self.obj_model_id)
        out["grasp_method"] = self._grasp_method()
        if hasattr(self, "_stage_failed"):
            out["stage_failed"] = self._stage_failed
        if hasattr(self, "_grasp_used"):
            out["grasp_used"] = self._grasp_used
        try:
            out["object_bottom_z"] = self._object_bottom_z()
            out["object_table_clearance"] = self._object_table_clearance()
            out["object_table_clearance_threshold"] = self.SUCCESS_TABLE_CLEARANCE_M
            out["object_still_held"] = self._object_still_held()
        except Exception as e:
            out["object_hold_metric_error"] = str(e)
        return out
