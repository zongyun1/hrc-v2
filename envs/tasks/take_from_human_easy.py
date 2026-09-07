"""Take from human (easy): grasp an object from the static human's hand and place it."""

import os
import json
import numpy as np
import transforms3d as t3d
import genesis as gs
from ..avatar.eval_mode_mixin import EvalModeAvatarMixin
from ..base_task import BaseTask, TargetSpec
from ..genesis_compat import (
    set_dofs_kp_kv_compat,
    update_dofs_force_range_compat,
)
from ..utils import Pose, load_object, create_primitive, to_numpy, ASSETS_PATH, place_decorative_objects
from ..grasp import tcp_to_link_pose


class TakeFromHumanEasy(EvalModeAvatarMixin, BaseTask):
    """Grasp a compact object from the human's hand and place it on the table."""

    INSTRUCTION = "take the {object} from the person and place it on the table"
    OBJECT_SET = "human_transfer_pool"

    use_avatar = True

    # Avatar base pose (no randomization). The avatar sits at +Y edge of the
    # table, facing -Y. Per-episode offsets are layered on in _sample_avatar_pose.
    AVATAR_BASE_POS = np.array([0.218, 0.768, -0.053])
    AVATAR_BASE_ROT = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)
    avatar_init_pos = AVATAR_BASE_POS
    avatar_init_rot = AVATAR_BASE_ROT

    table_offset = np.array([0.0, -0.1])

    # VLA primary view: near top-down over the full 0.99 x 1.31 m table.
    static_camera_list = [{
        "name": "head_camera",
        "position": [0.0, -0.095, 2.42],
        "forward": [0.0, -0.005, -1.36],
    }]

    # Recording cam: side/back view with avatar visible.
    recording_camera_pos = [0.9, 1.8, 1.90]
    recording_camera_lookat = [0.0, 0.15, 1.40]

    # Side cam: pulled back along -Y past the table edge so the full table
    # width fits in the frame, looking up-and-across toward the avatar-arm
    # interaction center. Captures avatar's upper body + arm + the whole
    # table surface.
    side_camera_pos = [0.55, -1.10, 1.60]
    side_camera_lookat = [0.0, 0.35, 1.35]

    # Per-episode layout randomization. Avatar slides along the near table edge
    # (X axis only — Y stays close to table) with a modest yaw perturbation;
    # plate lives on the opposite (-Y) side so the arm has clear placing room.
    AVATAR_X_RANGE = (-0.12, 0.12)
    AVATAR_YAW_DEG_RANGE = (-30.0, 30.0)
    PLATE_X_RANGE = (0.15, 0.40)
    PLATE_Y_RANGE = (-0.50, -0.30)
    TABLE_OBJECT_Z_CLEARANCE = 0.02
    PLACE_Z_ABOVE_PLATE = 0.06
    # A full descent to PLACE_Z_ABOVE_PLATE crosses an unstable Franka IK
    # region.  Stop above it, but substantially below the transport height,
    # to reduce impact energy without losing the grasp during descent.
    RELEASE_DESCENT_ABOVE_PLACE = 0.14
    DECOR_REGION = (-0.40, 0.40, -0.65, 0.50)
    DECOR_PLATE_EXCLUSION_RADIUS = 0.18
    DECOR_OBJECT_COUNT = 3
    DECOR_TARGET_SIZE = 0.10
    ARM_MAX_REACH = 0.82
    PRE_GRASP_HEIGHT = 0.10
    PRE_GRASP_EXTRA_CLEARANCE = 0.05
    MID_WAYPOINT_Y_FROM_BASE = 0.35
    MID_WAYPOINT_Z_FROM_BASE = 0.10
    HELD_APPLE_GRIPPER_VALUE = 0.55
    OBJECT_GRIPPER_VALUES = {
        "035_apple": 0.55,
        "073_rubikscube": 0.35,
    }
    FINGER_KP = 5000.0
    FINGER_KV = 500.0
    FINGER_FORCE_LIMIT = 200.0
    SET_GRIPPER_STEPS = 100
    AVATAR_HAND_SETTLE_STEPS = 50
    GRASP_SETTLE_STEPS = 100
    POST_DETACH_SETTLE_STEPS = 50
    LIFT_HEIGHT_FROM_APPLE = 0.10
    LIFT_CARTESIAN_STEPS = 40
    LONG_CARTESIAN_STEPS = 60
    RETRACT_CARTESIAN_STEPS = 20
    CARTESIAN_SIM_PER_STEP = 10
    RELEASE_SETTLE_STEPS = 50
    FINAL_SETTLE_STEPS = 100
    HAND_APPLE_OFFSET_LOCAL = np.array([0.0, 0.07, 0.03], dtype=np.float64)
    EVAL_APPLE_SUPPORT_HALF_SIZE = (0.04, 0.04, 0.003)

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("collision_ignore_capsule_prefixes", (
            "r_palm",
            "r_thumb",
            "r_index",
            "r_middle",
            "r_ring",
            "r_pinky",
        ))
        super().__init__(cfg)
        # Sampled in load_actors after reset(seed) seeds numpy.
        self.obj_name = None
        self.obj_model_id = 0

    def _setup_scene(self):
        """Use smaller dt for better collision response during grasping.

        Mirrors BaseTask._setup_scene but with dt=0.001 (vs 0.002) — and
        delegates renderer construction to `_build_renderer` so the
        `renderer: raytracer` config path works here too.
        """
        try:
            backend = gs.gpu if os.environ.get("GENESIS_BACKEND") == "gpu" else gs.cpu
            gs.init(backend=backend, logging_level="error")
        except Exception:
            pass
        renderer = self._build_renderer()
        scene_kwargs = dict(
            show_viewer=self.config.get("show_viewer", False),
            sim_options=gs.options.SimOptions(dt=0.001),
        )
        if renderer is not None:
            scene_kwargs["renderer"] = renderer
        self.scene = gs.Scene(**scene_kwargs)
        renderer_type = self.config.get("renderer", "rasterizer").lower()
        self.scene.add_entity(gs.morphs.Plane(visualization=not self._hide_ground_visuals()))
        self._create_floor()

    def _load_robot(self):
        """Place Franka on the standard-height workstation."""
        self.config.setdefault("robot_type", "franka")
        if self.config["robot_type"] == "franka":
            kwargs = self.config.setdefault("robot_kwargs", {})
            kwargs.setdefault("pos", [0.0, -0.3, 0.75])
            kwargs.setdefault("quat", [0.707, 0.0, 0.0, 0.707])
        super()._load_robot()

    def _create_table(self, table_height=0.74):
        dx, dy = self.table_offset

        # Table sized at 2/3 × 1.4 of original (half_x: 0.353×1.4=0.494, half_y: 0.467×1.4=0.654)
        half_x = 0.494
        half_y = 0.654
        center_y = dy
        return self._create_rectangular_table(
            table_height=table_height,
            half_size=(half_x, half_y),
            center_xy=(dx, center_y),
            leg_inset=(half_x - 0.45, half_y - 0.60),
        )

    def _sample_avatar_pose(self):
        """Sample per-episode avatar (pos, rot). Uses the seeded np.random."""
        dx = float(np.random.uniform(*self.AVATAR_X_RANGE))
        yaw_deg = float(np.random.uniform(*self.AVATAR_YAW_DEG_RANGE))
        yaw = np.deg2rad(yaw_deg)
        cz, sz = np.cos(yaw), np.sin(yaw)
        rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        pos = self.AVATAR_BASE_POS + np.array([dx, 0.0, 0.0])
        rot = rot_z @ self.AVATAR_BASE_ROT
        return pos, rot, dx, yaw_deg

    def _sample_plate_xy(self):
        """Sample per-episode plate XY on the side opposite the avatar."""
        return np.array([
            float(np.random.uniform(*self.PLATE_X_RANGE)),
            float(np.random.uniform(*self.PLATE_Y_RANGE)),
        ])

    def _settle_scene(self, n_steps: int = None):
        if not getattr(self, "_avatar_head_hand_aligned", False):
            self._avatar_head_hand_yaw_deg = self._align_avatar_head_hand_to_xy(
                self.place_pos[:2], hand_id=1, motion_name="deliver_to_human",
            )
            self._avatar_head_hand_aligned = True
        super()._settle_scene(n_steps)

    def load_actors(self):
        entry = self.resolve_target_object(randomize=True)
        self.obj_name = entry.object_id or entry.key
        self.obj_model_id = int(self.config.get("model_id", entry.model_id))
        self._plate_xy = self._sample_plate_xy()
        self.avatar_init_pos, self.avatar_init_rot, _dx, _yaw_deg = self._sample_avatar_pose()

        table_top = self.TABLE_TOP_Z + self.TABLE_OBJECT_Z_CLEARANCE
        obj_q = np.array([1.0, 0.0, 0.0, 0.0])
        upright_q = np.array([0.707, 0.707, 0.0, 0.0])

        # Apple starts near the avatar's idle palm; attachment binds it to the
        # hand at play_once, so this pose is only an initial placement.
        self.apple_pose = Pose([0.0, 0.768, 0.69], obj_q)
        self.apple_actor = load_object(
            self.scene, self.apple_pose, self.obj_name,
            model_id=self.obj_model_id, convex=True, is_static=False,
            friction=4.0,
        )
        obj_radius = self._get_object_radius(self.obj_name, self.obj_model_id)
        support_half = (max(0.025, 0.8 * obj_radius),) * 2 + (
            self.EVAL_APPLE_SUPPORT_HALF_SIZE[2],
        )
        self._eval_object_support_half_size = support_half
        self._eval_apple_support = create_primitive(
            self.scene, "box",
            Pose(p=[0.0, 0.0, -1.0]),
            size={"half_size": support_half},
            color=(0.0, 0.0, 0.0),
            is_static=True,
            collision=True,
            visual=False,
        )
        self._eval_apple_support_active = False

        plate_x, plate_y = float(self._plate_xy[0]), float(self._plate_xy[1])
        self.plate_actor = load_object(
            self.scene, Pose([plate_x, plate_y, table_top], upright_q), "003_plate",
            model_id=0, convex=True, is_static=False,
        )
        self.place_pos = np.array([
            plate_x, plate_y, table_top + self.PLACE_Z_ABOVE_PLATE,
        ])

        # Evaluation metric: apple→plate lateral distance.  The apple mesh
        # origin is offset ~27 mm from its bbox centre, so route the object
        # position through `_get_object_center_world` instead of the default
        # `entity.get_pos()`.
        self.target = TargetSpec(
            object=self.apple_actor,
            position=self.place_pos,
            label="plate",
            object_pos_fn=lambda: self._get_object_center_world(self.apple_actor),
            success_dist_xy=0.12,
        )

        self._clutter_actors = []

    # For top-down grasps on round objects, grip slightly below the equator so
    # contact normals push the object upward into the gripper palm.
    GRASP_DEPTH_FRACTION = 0.3  # 30% of radius below center

    @staticmethod
    def _get_object_radius(object_name: str, model_id: int = 0) -> float:
        """Get approximate object radius from mesh bounding-box metadata."""
        md_path = ASSETS_PATH / "objects" / object_name / f"model_data{model_id}.json"
        with open(md_path) as f:
            md = json.load(f)
        extents = np.array(md["extents"])
        scale = np.array(md.get("scale", [1, 1, 1]))
        return float(np.max(extents * scale) / 2)

    @staticmethod
    def _get_object_center_world(actor) -> np.ndarray:
        """Compute world-frame bounding-box center.

        `actor.entity.get_pos()` returns the mesh origin, which for many assets
        is offset from the visual/collision bounding-box center (see the
        `center` field in model_data.json). Applying the actor's current
        rotation to that offset gives the true center in world coordinates.
        """
        pos = to_numpy(actor.entity.get_pos()).astype(float).ravel()[:3]
        quat_wxyz = to_numpy(actor.entity.get_quat()).astype(float).ravel()[:4]
        center_local = np.asarray(actor.config.get("center", [0.0, 0.0, 0.0]), dtype=float)
        scale_raw = actor.config.get("scale", [1.0, 1.0, 1.0])
        scale = (np.array([scale_raw, scale_raw, scale_raw], dtype=float)
                 if np.ndim(scale_raw) == 0
                 else np.asarray(scale_raw, dtype=float))
        R = t3d.quaternions.quat2mat(quat_wxyz)
        return pos + R @ (center_local * scale)

    def _compute_grasp_poses(self, arm_tag: str):
        """Compute pre-grasp and grasp TCP poses for the apple.

        Uses a top-down approach with the grasp depth derived from the object's
        mesh bounding box — no per-object hardcoding needed.
        """
        arm = self.robot.get_arm(arm_tag)
        arm_base = np.array(arm.origin_pose.p)

        # Use the bounding-box center (not mesh origin) — crucial for assets
        # like 035_apple whose mesh origin is offset ~27mm from true center.
        apple_pos = self._get_object_center_world(self.apple_actor)

        # Clamp to arm reach (Franka reach ~0.855m)
        to_apple = apple_pos - arm_base
        dist_3d = np.linalg.norm(to_apple)
        if dist_3d > self.ARM_MAX_REACH:
            grasp_pos = arm_base + to_apple * (self.ARM_MAX_REACH / dist_3d)
        else:
            grasp_pos = apple_pos.copy()

        # Gripper straight down
        R_down = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        q_down = t3d.quaternions.mat2quat(R_down)

        # Lower TCP below object center by a fraction of the object's radius,
        # so finger pads grip the lower hemisphere (upward contact forces).
        obj_radius = self._get_object_radius(self.obj_name, self.obj_model_id)
        grasp_offset = obj_radius * self.GRASP_DEPTH_FRACTION
        grasp_pos[2] -= grasp_offset

        # Pre-grasp: well above the grasp point for clearance
        pre_grasp_pos = grasp_pos.copy()
        pre_grasp_pos[2] += self.PRE_GRASP_HEIGHT

        return Pose(pre_grasp_pos, q_down), Pose(grasp_pos, q_down)

    # ------------------------------------------------------------------
    # IK / teleport / cartesian helpers (adopted from categorize_cooperative)
    # ------------------------------------------------------------------

    def _top_down_tcp(self, pos):
        R_down = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        q = t3d.quaternions.mat2quat(R_down)
        return Pose(np.asarray(pos, dtype=float), q)

    def _solve_ik(self, tcp_pose, arm_tag):
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset
        link_pose = tcp_to_link_pose(tcp_pose, tcp_offset)
        pose7 = link_pose.to_pose7()
        ik_fn = getattr(arm.entity, "inverse_kinematics", None)
        if ik_fn is None:
            return None
        try:
            seed_qpos = getattr(arm, "_cached_target", None)
            if seed_qpos is None:
                seed_qpos = to_numpy(arm.entity.get_qpos())
            seed_qpos = np.asarray(seed_qpos, dtype=np.float64).ravel()
            qpos_sol = ik_fn(
                link=arm.ee_link,
                pos=np.array(pose7[:3]),
                quat=np.array(pose7[3:7]),
                init_qpos=seed_qpos,
            )
            if qpos_sol is None:
                return None
            return to_numpy(qpos_sol).ravel()[:arm.n_arm]
        except Exception:
            return None

    def _boost_arm_pd(self, arm_tag: str = "right"):
        """Raise arm/finger gains so tracking and contact grasp are stable.

        With default kp=4000 the arm lags the planned trajectory by ~30cm at
        the end-effector near the edge of Franka's reach. kp=20000 brings that
        to millimetre accuracy. Latest Genesis can expose MJCF actuators as
        non-PD-reducible, so fall back to the direct actuator gain/bias API.
        """
        from ..robot.franka_robot import _get_dof_idx
        arm = self.robot.get_arm(arm_tag)
        finger_indices = []
        for j in arm.finger_joints:
            if j is not None:
                idx = _get_dof_idx(j)
                if idx is not None:
                    finger_indices.append(idx)
        if finger_indices:
            arm._finger_dof_indices = finger_indices

        try:
            kp = to_numpy(arm.entity.get_dofs_kp()).copy()
            kv = to_numpy(arm.entity.get_dofs_kv()).copy()
        except Exception:
            gain = to_numpy(arm.entity.get_dofs_act_gain()).copy()
            bias0_raw, bias1_raw, bias2_raw = arm.entity.get_dofs_act_bias()
            bias0 = to_numpy(bias0_raw).copy()
            bias1 = to_numpy(bias1_raw).copy()
            bias2 = to_numpy(bias2_raw).copy()
            for j in arm.arm_joints:
                if j is not None:
                    idx = _get_dof_idx(j)
                    if idx is not None:
                        gain[idx] = 20000.0
                        bias0[idx] = 0.0
                        bias1[idx] = -20000.0
                        bias2[idx] = -1000.0
            for idx in finger_indices:
                gain[idx] = self.FINGER_KP
                bias0[idx] = 0.0
                bias1[idx] = -self.FINGER_KP
                bias2[idx] = -self.FINGER_KV
            arm.entity.set_dofs_act_gain(gain)
            arm.entity.set_dofs_act_bias(bias0, bias1, bias2)
            self._set_finger_force_limit(arm, finger_indices)
            return

        for j in arm.arm_joints:
            if j is not None:
                idx = _get_dof_idx(j)
                if idx is not None:
                    kp[idx] = 20000.0
                    kv[idx] = 1000.0
        for idx in finger_indices:
            kp[idx] = self.FINGER_KP
            kv[idx] = self.FINGER_KV
        set_dofs_kp_kv_compat(arm.entity, kp=kp, kv=kv)
        self._set_finger_force_limit(arm, finger_indices)

    def _set_finger_force_limit(self, arm, finger_indices):
        if not finger_indices:
            return
        update_dofs_force_range_compat(
            arm.entity,
            finger_indices,
            lower_value=-self.FINGER_FORCE_LIMIT,
            upper_value=+self.FINGER_FORCE_LIMIT,
        )

    def _move_cartesian(self, start_pos, end_pos, arm_tag, n_steps=50, sim_per_step=20):
        from ..robot.franka_robot import _get_dof_idx
        arm = self.robot.get_arm(arm_tag)
        base_target = getattr(arm, "_cached_target", None)
        if base_target is None:
            base_target = to_numpy(arm.entity.get_qpos())
        base_target = np.array(base_target, dtype=float).copy()
        measured_qpos = np.asarray(to_numpy(arm.entity.get_qpos()), dtype=float).ravel()
        seed_target = base_target.copy()
        for j in arm.arm_joints:
            if j is not None:
                dof_idx = _get_dof_idx(j)
                if dof_idx is not None:
                    seed_target[dof_idx] = float(measured_qpos[dof_idx])
        arm._cached_target = seed_target.copy()
        qpos_full = base_target.copy()
        for i in range(1, n_steps + 1):
            t = i / n_steps
            pos = np.asarray(start_pos) * (1 - t) + np.asarray(end_pos) * t
            arm_qpos = self._solve_ik(self._top_down_tcp(pos), arm_tag)
            if arm_qpos is None:
                continue
            qpos_full = base_target.copy()
            for j_idx, j in enumerate(arm.arm_joints):
                if j is not None:
                    dof_idx = _get_dof_idx(j)
                    if dof_idx is not None:
                        qpos_full[dof_idx] = float(arm_qpos[j_idx])
            arm.entity.control_dofs_position(qpos_full)
            arm._cached_target = qpos_full.copy()
            for _ in range(sim_per_step):
                self.step_sim()
        arm._cached_target = qpos_full.copy()

    # ------------------------------------------------------------------
    # Eval-mode hook (single-shot): replicate the avatar setup from
    # play_once — calibrate apple position relative to the avatar's right
    # hand, attach it, and queue the deliver_to_human animation.  After
    # _eval_at_reset returns, the policy starts stepping and the avatar
    # progresses through the animation in the background.
    # ------------------------------------------------------------------
    def _eval_at_reset(self):
        if self.config.get("debug_eval_handoff", False):
            print("[take_eval] _eval_at_reset begin", flush=True)
        self._boost_arm_pd("right")
        skin = self.avatar.robot.skin
        wrist = to_numpy(skin.get_global_translation("RightHand")[0]).ravel()[:3]
        mid1 = to_numpy(skin.get_global_translation("RightHandMiddle1")[0]).ravel()[:3]
        palm_center = (wrist + mid1) / 2.0
        _, R_hand = self.avatar.robot._get_hand_frame(1)
        apple_pos = palm_center + R_hand @ self.HAND_APPLE_OFFSET_LOCAL
        self.apple_actor.entity.set_pos(apple_pos.astype(float))
        self.avatar.attach_object_to_hand(self.apple_actor.entity, hand_id=1)
        self._eval_apple_attached_to_avatar = True
        self._eval_handoff_closed_steps = 0
        self._eval_handoff_debug = None
        if self.config.get("debug_eval_handoff", False):
            print("[take_eval] play deliver_to_human", flush=True)
        self.avatar.play_animation("deliver_to_human")
        if self.config.get("debug_eval_handoff", False):
            print("[take_eval] _eval_at_reset done", flush=True)

    def _place_eval_apple_support_and_detach(self):
        """In eval, replace the human-hand hold with a hidden support shelf.

        Policy control makes gripper-based detach timing ambiguous.  Once the
        handoff animation has finished, hold the apple from below with a small
        invisible static box, then release the avatar attachment so the policy
        sees a dynamic apple at the same offered pose.
        """
        support = getattr(self, "_eval_apple_support", None)
        if support is None or self.avatar is None:
            return
        if bool(getattr(self, "_eval_apple_support_active", False)):
            return

        apple_center = self._get_object_center_world(self.apple_actor)
        radius = self._get_object_radius(self.obj_name, self.obj_model_id)
        support_half_z = float(self._eval_object_support_half_size[2])
        support_pos = np.array([
            apple_center[0],
            apple_center[1],
            apple_center[2] - radius - support_half_z,
        ], dtype=np.float64)
        support.set_pos(support_pos)
        self.avatar.robot.detach_object(1)
        self._eval_apple_attached_to_avatar = False
        self._eval_apple_support_active = True
        if self.config.get("debug_eval_handoff", False):
            print(
                "[take_eval] apple detached onto hidden support at "
                f"{support_pos.tolist()}",
                flush=True,
            )

    def _eval_at_step(self, step_idx: int):
        if not getattr(self, "_eval_apple_attached_to_avatar", False):
            return
        arm = self.robot.get_arm("right")
        ee_pos = np.asarray(arm.get_ee_pose(), dtype=np.float64).ravel()[:3]
        apple_pos = self._get_object_center_world(self.apple_actor)
        ee_apple_dist = float(np.linalg.norm(ee_pos - apple_pos))
        near_threshold = float(self.config.get("eval_handoff_near_threshold", 0.08))
        close_value = self.OBJECT_GRIPPER_VALUES[self.obj_name]
        closed = arm.gripper_val <= close_value + 1e-3
        if closed and ee_apple_dist <= near_threshold:
            self._eval_handoff_closed_steps = (
                getattr(self, "_eval_handoff_closed_steps", 0) + 1
            )
            action_substeps = int(self.config.get("action_substeps", 25))
            required_closed_steps = int(np.ceil(self.SET_GRIPPER_STEPS / max(1, action_substeps)))
            required_closed_steps = max(
                1,
                int(self.config.get("eval_handoff_closed_steps", required_closed_steps)),
            )
            if self._eval_handoff_closed_steps >= required_closed_steps:
                self._eval_handoff_debug = {
                    "step_idx": int(step_idx),
                    "gripper": float(arm.gripper_val),
                    "ee_pos": ee_pos.tolist(),
                    "apple_pos": np.asarray(apple_pos, dtype=np.float64).ravel()[:3].tolist(),
                    "ee_apple_dist": ee_apple_dist,
                    "closed_steps": int(self._eval_handoff_closed_steps),
                    "required_closed_steps": int(required_closed_steps),
                }
                self.avatar.robot.detach_object(1)
                self._eval_apple_attached_to_avatar = False
        else:
            self._eval_handoff_closed_steps = 0
            if self.config.get("debug_eval_handoff", False) and closed:
                print(
                    "[take_eval] closed but not near: "
                    f"step={step_idx} gripper={arm.gripper_val:.3f} "
                    f"ee_apple_dist={ee_apple_dist:.3f}",
                    flush=True,
                )

    def eval_pre_policy_warmup(self):
        if (
            self.avatar is None
            or not bool(self.config.get("eval_mode", False))
            or not bool(self.config.get("eval_avatar_pre_policy", True))
        ):
            return self.get_obs()
        while not self.avatar.spare():
            self.step_sim()
        for _ in range(int(self.config.get(
            "eval_avatar_post_warmup_settle_steps",
            self.AVATAR_HAND_SETTLE_STEPS,
        ))):
            self.step_sim()
        if bool(self.config.get("eval_detach_apple_to_support", True)):
            self._place_eval_apple_support_and_detach()
            for _ in range(int(self.config.get("eval_apple_support_settle_steps", 20))):
                self.step_sim()
        return self.get_obs()

    def evaluate(self) -> dict:
        out = super().evaluate()
        out["object_name"] = self.obj_name
        out["object_model_id"] = int(self.obj_model_id)
        debug = getattr(self, "_eval_handoff_debug", None)
        if debug is not None:
            out["eval_handoff_debug"] = debug
        out["eval_apple_attached_to_avatar"] = bool(
            getattr(self, "_eval_apple_attached_to_avatar", False)
        )
        out["eval_apple_support_active"] = bool(
            getattr(self, "_eval_apple_support_active", False)
        )
        return out

    def play_once(self) -> bool:
        arm_tag = "right"
        self._boost_arm_pd(arm_tag)
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset

        if self.avatar is not None:
            # True palm center: midpoint of wrist (RightHand) and middle-finger knuckle.
            # The built-in get_palm_center averages 4 fingers (excluding thumb) which
            # biases the position toward the pinky side.
            skin = self.avatar.robot.skin
            wrist = to_numpy(skin.get_global_translation("RightHand")[0]).ravel()[:3]
            mid1 = to_numpy(skin.get_global_translation("RightHandMiddle1")[0]).ravel()[:3]
            palm_center = (wrist + mid1) / 2.0
            _, R_hand = self.avatar.robot._get_hand_frame(1)
            # Hand frame basis: e1=Thumb−Pinky, e2=along fingers (toward tips),
            # e3=e1×e2 (points into back of hand). Offset chosen empirically from
            # the debug sweep: 7cm toward fingertips (+e2) and 3cm into the palm
            # (+e3) places the apple nestled against the palm near the fingers.
            apple_pos = palm_center + R_hand @ self.HAND_APPLE_OFFSET_LOCAL
            self.apple_actor.entity.set_pos(apple_pos.astype(float))

            self.avatar.attach_object_to_hand(self.apple_actor.entity, hand_id=1)

            self.avatar.play_animation("deliver_to_human")
            while not self.avatar.spare():
                self.step_sim()
            # Leave the apple kinematically attached to the human hand while
            # the robot plans its approach. Detaching early would need a
            # support plane under the apple (an obstacle MPLib has to plan
            # around, which destroys grasp accuracy). Instead we detach right
            # before closing the gripper, so the apple only enters physics
            # simulation when the fingers are already in place to catch it.
            for _ in range(self.AVATAR_HAND_SETTLE_STEPS):
                self.step_sim()
        else:
            for _ in range(self.AVATAR_HAND_SETTLE_STEPS):
                self.step_sim()

        # Phase 2: Robot approaches apple in human's hand
        self.open_gripper(arm_tag)

        # Gripper-down orientation for waypoints and placing
        R_down = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        q_down = t3d.quaternions.mat2quat(R_down)

        # Mid-height waypoint to extend arm before reaching toward human
        arm_base = np.array(arm.origin_pose.p)
        mid_tcp = Pose(
            arm_base + np.array([
                0.0,
                self.MID_WAYPOINT_Y_FROM_BASE,
                self.MID_WAYPOINT_Z_FROM_BASE,
            ]),
            q_down,
        )
        mid_link = tcp_to_link_pose(mid_tcp, tcp_offset)

        result = self.move_and_execute(mid_link.to_pose7(), arm_tag)
        if result is None:
            return False

        # Move to pre-grasp (15cm above apple) via the motion planner.
        pre_grasp_tcp, _ = self._compute_grasp_poses(arm_tag)
        pre_pos_hi = np.array(pre_grasp_tcp.p) + np.array([
            0.0, 0.0, self.PRE_GRASP_EXTRA_CLEARANCE,
        ])
        pre_link = tcp_to_link_pose(self._top_down_tcp(pre_pos_hi), tcp_offset)
        if self.move_and_execute(pre_link.to_pose7(), arm_tag) is None:
            return False

        # Final descent via MPLib. With the support plane removed (apple is
        # still kinematically attached to the human hand), MPLib can plan
        # directly to the grasp pose without detouring around an obstacle.
        _, grasp_tcp = self._compute_grasp_poses(arm_tag)
        grasp_link = tcp_to_link_pose(grasp_tcp, tcp_offset)
        if self.move_and_execute(grasp_link.to_pose7(), arm_tag) is None:
            return False
        for _ in range(self.GRASP_SETTLE_STEPS):
            self.step_sim()
        apple_pre = to_numpy(self.apple_actor.entity.get_pos())

        # Close the gripper while the apple is still kinematically held by the
        # human hand. The fingers encounter the apple as a fixed obstacle and
        # stop at its convex-hull surface (~joint=37mm), building up ~150N per
        # finger of PD contact force. Then detach the apple — it transitions
        # to physics already pinned between the fingers, so the grip holds
        # without the apple free-falling out from between the fingers.
        self.set_gripper(
            self.OBJECT_GRIPPER_VALUES[self.obj_name],
            arm_tag,
            num_steps=self.SET_GRIPPER_STEPS,
        )
        if self.avatar is not None:
            self.avatar.robot.detach_object(1)
        for _ in range(self.POST_DETACH_SETTLE_STEPS):
            self.step_sim()

        # Phase 4: Cartesian lift — slow motion preserves finger grip during
        # the transition from static close to moving grip. _move_cartesian
        # does per-step IK + control_dofs_position (no set_qpos).
        grasp_pos = np.array(grasp_tcp.p)
        lift_pos = grasp_pos.copy()
        lift_pos[2] = apple_pre[2] + self.LIFT_HEIGHT_FROM_APPLE
        self._move_cartesian(
            grasp_pos, lift_pos, arm_tag,
            n_steps=self.LIFT_CARTESIAN_STEPS,
            sim_per_step=self.CARTESIAN_SIM_PER_STEP,
        )
        object_lifted = to_numpy(self.apple_actor.entity.get_pos())
        lift_dz = float(object_lifted[2] - apple_pre[2])
        print(
            f"[take] {self.obj_name} post-lift dz={lift_dz:.4f}m "
            f"pos={np.asarray(object_lifted).round(4).tolist()}",
            flush=True,
        )
        if lift_dz < 0.03:
            return False

        # Lateral transport over a mid-waypoint (y = arm_base[1] + 0.35) to
        # keep the shoulder from twisting through its joint limit. Larger
        # n_steps here = slower velocity = less inertial slip at the fingers.
        mid_pos = np.array([
            0.0, arm_base[1] + self.MID_WAYPOINT_Y_FROM_BASE, lift_pos[2],
        ])
        self._move_cartesian(
            lift_pos, mid_pos, arm_tag,
            n_steps=self.LONG_CARTESIAN_STEPS,
            sim_per_step=self.CARTESIAN_SIM_PER_STEP,
        )

        # Phase 5: Human retracts hand (reverse animation)
        if self.avatar is not None:
            self.avatar.play_animation_reverse("deliver_to_human")
            while not self.avatar.spare():
                self.step_sim()

        # Phase 6: Lateral traverse to a hover above the plate, followed by a
        # slow vertical last mile.  Releasing directly from the transport
        # height gives the apple enough energy to bounce and roll beyond the
        # plate footprint (seed 0 consistently settled about 14 cm away).
        # `_move_cartesian` seeds every IK solve from the previous command, so
        # keeping the top-down orientation fixed avoids the old wrist-branch
        # jump while allowing a low-energy release.
        above_plate = np.array([self.place_pos[0], self.place_pos[1], lift_pos[2]])
        self._move_cartesian(
            mid_pos, above_plate, arm_tag,
            n_steps=self.LONG_CARTESIAN_STEPS,
            sim_per_step=self.CARTESIAN_SIM_PER_STEP,
        )
        release_pos = np.asarray(self.place_pos, dtype=float).copy()
        release_pos[2] += self.RELEASE_DESCENT_ABOVE_PLACE
        self._move_cartesian(
            above_plate, release_pos, arm_tag,
            n_steps=self.LONG_CARTESIAN_STEPS,
            sim_per_step=self.CARTESIAN_SIM_PER_STEP,
        )

        # Release just above the plate so the object settles instead of
        # bouncing out of the success footprint.
        self.open_gripper(arm_tag)
        for _ in range(self.RELEASE_SETTLE_STEPS):
            self.robot.set_gripper(arm.gripper_val, arm_tag)
            self.step_sim()

        if self.vla_recorder is not None:
            self.vla_recorder.mark_episode_end("take_from_human_easy_post_release")

        hold_qpos = np.asarray(to_numpy(arm.entity.get_qpos()), dtype=float).ravel()
        for _ in range(self.FINAL_SETTLE_STEPS):
            arm.entity.control_dofs_position(hold_qpos)
            self.robot.set_gripper(arm.gripper_val, arm_tag)
            self.step_sim()

        return True
