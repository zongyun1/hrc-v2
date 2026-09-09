"""Pour water: avatar drinks from cup, then Franka arm grasps mug, lifts, and pours toward cup."""

import numpy as np
import torch
import transforms3d as t3d

from ..avatar.eval_mode_mixin import EvalModeAvatarMixin
from ..base_task import BaseTask, TargetSpec
from ..genesis_compat import update_dofs_force_range_compat, update_dofs_kp_kv_compat
from ..utils import Pose, load_mesh, load_object, to_numpy, ASSETS_PATH
from ..grasp import load_object_config, tcp_to_link_pose
from ..robot.base import _get_dof_idx

# Object IDs.  The grasped/manipulated target is the mug (sourced from the
# central catalog); the cup is the pour receptacle.
from ..object_catalog import get_entry
_MUG_ID = get_entry("039_mug").object_id
_CUP_ID = "021_cup"
_Q_UPRIGHT = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64)
_Q_MUG_HANDLE_TOWARD_ROBOT = np.array(
    [0.0, 0.0, 0.70710678, 0.70710678], dtype=np.float64,
)
_MUG_BODY_GRASP_ORDER = (
    "manual_body_edge_topdown_posx_posz",
    "manual_body_edge_topdown",
    "manual_body_edge_topdown_negx_posz",
    "manual_body_edge_topdown_posx",
    "manual_body_edge_topdown_negx",
)


class PourWater(EvalModeAvatarMixin, BaseTask):
    """Avatar plays drink animation; then Franka arm grasps mug, lifts, and pours into cup."""

    INSTRUCTION = "pour water from the mug into the cup"
    OBJECT_SET = ["039_mug"]

    use_avatar = True
    avatar_init_pos = np.array([-0.02, 0.91, -0.10])
    avatar_init_rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)

    recording_camera_pos = [1.1, 0.0, 2.05]
    recording_camera_lookat = [0.1, 0.2, 0.95]

    # Mug pour parameters.  Tuned for handle grip: handle EE sits ~10 cm
    # to the SIDE of the mug (not above it like a body topdown grip), so
    # the EE z budget after grasp is tight (~0.93 m max with same wrist
    # orientation).  Heights are kept low enough that the EE stays in
    # reach; the mug rim is itself ~9 cm above the mug origin so the
    # spout still clears the cup rim at PRE_POUR_HEIGHT=0.
    POUR_TILT_DEG = 85          # lower wrist angle still yields >60° physical mug tilt
    LIFT_HEIGHT = 0.00          # no separate lift — pre-pour move subsumes it
    PRE_POUR_HEIGHT = 0.00      # mug origin level with cup rim (rim ~9 cm above)
    POUR_TILT_HEIGHT = 0.18     # natural pour height with >10 cm measured rim clearance

    GRIPPER_KP = 50000.0
    GRIPPER_KV = 1000.0
    GRIPPER_FORCE_RANGE = (-200.0, 200.0)

    # Wider table: double Y dimension, keep human-side edge at Y=0.35
    table_offset = np.array([0.0, -0.35])

    def _create_table(self, table_height=0.74):
        from ..utils import create_primitive
        self.TABLE_HEIGHT = table_height
        self.TABLE_THICKNESS = 0.05
        self.TABLE_TOP_Z = table_height + self.TABLE_THICKNESS / 2
        self._table_top_z_fixed = False
        dx, dy = self.table_offset
        if self._try_create_table_variant(table_height, half_size=(0.6, 0.70), center_xy=(dx, dy)):
            return

        self.table = create_primitive(
            self.scene, "box",
            Pose(p=[dx, dy, table_height]),
            size={"half_size": (0.6, 0.70, self.TABLE_THICKNESS / 2)},
            color=(0.8, 0.75, 0.65),
            is_static=True,
        )
        leg_h = table_height - self.TABLE_THICKNESS / 2
        for x, y in [(-0.55, -0.65), (0.55, -0.65), (-0.55, 0.65), (0.55, 0.65)]:
            create_primitive(
                self.scene, "cylinder",
                Pose(p=[x + dx, y + dy, leg_h / 2]),
                size={"radius": 0.025, "half_length": leg_h / 2},
                color=(0.5, 0.5, 0.5),
                is_static=True,
            )

    def _load_robot(self):
        self.config.setdefault("robot_type", "franka")
        if str(self.config.get("robot_type", "franka")).lower() == "franka":
            robot_kwargs = dict(self.config.get("robot_kwargs", {}) or {})
            robot_kwargs.setdefault(
                "mjcf_file",
                "xml/franka_emika_panda/panda_no_tendon.xml",
            )
            self.config["robot_kwargs"] = robot_kwargs
        super()._load_robot()

    def load_actors(self):
        self.resolve_target_object()   # registry/override hook (size-1 set)
        table_top = self.TABLE_TOP_Z + 0.02
        # Mesh loading preserves the authored frame. These assets use local
        # +Y as up, so rotate +Y onto world +Z explicitly.
        upright_q = _Q_UPRIGHT
        old_mug_q = _Q_MUG_HANDLE_TOWARD_ROBOT

        # Avatar+cup shift along the table edge (X axis): ±0.10 m.
        # Cup and the calibration target shift together so the
        # cup-in-hand geometry at frame 33 of drink_water_1 is preserved.
        # Range tightened from ±0.15 to ±0.10 to keep the pour-tilt target
        # inside Franka's reliable reach/wrist envelope (±0.15 hit the
        # plan_screw + Cartesian-IK failure mode in 2/3 of seeds).
        self._avatar_shift_x = float(np.random.uniform(-0.10, 0.10))

        # Cup — upright; X shifted with avatar
        self.cup_scale = 0.088
        self.cup_rim_height = 1.017 * self.cup_scale
        self.cup_rim_radius = 0.988 / 2 * self.cup_scale
        cup_x = -0.08 + self._avatar_shift_x
        cup_y = 0.0
        self.cup_pose = Pose([cup_x, cup_y, table_top + 0.1], upright_q)
        self.cup_actor = load_object(
            self.scene, self.cup_pose, _CUP_ID, model_id=0, convex=True,
        )

        # Mug — random position within Franka reach (base at y=-0.65).
        self.mug_scale = 0.08
        mug_cfg = load_object_config(_MUG_ID)
        self.mug_rim_radius = mug_cfg.get("rim_radius", 1.0) * self.mug_scale
        mug_q = old_mug_q
        self._mug_logical_q = old_mug_q
        # Tightened from (0.18-0.32) × (-0.18 to -0.02) to (0.20-0.30) × (-0.15 to -0.05).
        # The wider box tripped the planner on grasp+lift transitions for some seeds.
        mug_x = float(np.random.uniform(0.20, 0.30))
        mug_y = float(np.random.uniform(-0.15, -0.05))
        self._mug_physics_pose = Pose([mug_x, mug_y, table_top + 0.1], mug_q)
        self.mug_pose = Pose([mug_x, mug_y, table_top + 0.1], self._mug_logical_q)
        mug_mesh = self._find_mesh(ASSETS_PATH / "objects" / _MUG_ID / "visual", model_id=0)
        self.mug_entity = load_mesh(
            self.scene, mug_mesh, self._mug_physics_pose,
            scale=(self.mug_scale,) * 3, convex=True,
            friction=2.0, density=100.0,
        ) if mug_mesh else None

        # Evaluation: mug origin vs cup rim + mug tilt + no arm-avatar contact.
        # Mug's world "up" axis is its initial body-frame +Z transported by
        # the current orientation — when poured, this axis tilts away from
        # world +Z.  Thresholds set after a measurement sweep.
        if self.mug_entity is not None:
            R_mug_init = t3d.quaternions.quat2mat(self._mug_physics_pose.q)
            body_up = R_mug_init.T @ np.array([0.0, 0.0, 1.0])

            def _cup_rim_pos():
                cup_pose = self._get_object_pose(
                    self.cup_actor.entity if self.cup_actor else None,
                    self.cup_pose,
                )
                return np.array([cup_pose.p[0], cup_pose.p[1],
                                 cup_pose.p[2] + self.cup_rim_height])

            def _mug_up_axis():
                q = to_numpy(self.mug_entity.get_quat()).ravel()[:4]
                return t3d.quaternions.quat2mat(q) @ body_up

            # Thresholds chosen to label "is the mug positioned to pour water
            # into the cup?" rather than "did planner hit the exact target":
            #   dist_xy ≤ 0.20 m  — mug center close enough that water falls in
            #   dz ≥ 0.10 m       — mug clearly above cup rim (any positive dz
            #                       would physically pour, 0.10 leaves margin)
            #   tilt ≥ 60°        — mug tipped enough for water to flow
            #   no arm↔avatar contact during the rollout — only gated when
            #     collision tracking is actually enabled.  The collision
            #     checker is built only when config["track_avatar_collision"]
            #     is set; default collect/eval runs leave it off, so
            #     `avatar_collision_summary()["enabled"]` is False.  Requiring
            #     no-collision against a disabled checker made check_success
            #     return False unconditionally (the summary reports
            #     enabled=False → treated as "couldn't confirm no-collision"),
            #     so every default rollout FAILed despite a correct pour.
            #     Only require it when we are actually measuring collisions.
            require_no_collision = (
                not self.no_human_enabled()
                and bool(self.config.get("track_avatar_collision", False))
            )
            self.target = TargetSpec(
                object=self.mug_entity,
                position=_cup_rim_pos,
                label="cup_rim",
                object_axis_fn=_mug_up_axis,
                success_dist_xy=0.20,
                success_dz_min=0.10,
                success_tilt_min_deg=60.0,
                success_require_no_avatar_collision=require_no_collision,
            )

    def _find_mesh(self, directory, model_id=None):
        from pathlib import Path
        d = Path(directory)
        if not d.exists():
            return None
        if model_id is not None:
            for ext in (".glb", ".obj"):
                f = d / f"base{model_id}{ext}"
                if f.exists():
                    return f
        for ext in (".glb", ".obj"):
            files = sorted(d.glob(f"*{ext}"))
            if files:
                return files[0]
        return None

    def _get_object_pose(self, entity, fallback_pose: Pose) -> Pose:
        if entity is None:
            return fallback_pose
        pos = to_numpy(entity.get_pos()).ravel()[:3]
        quat = to_numpy(entity.get_quat()).ravel()[:4]
        return Pose(pos, quat)

    def _get_mug_logical_pose(self) -> Pose:
        """Current mug position in the raw benchmark frame used by grasp YAML."""
        physics_pose = self._get_object_pose(
            self.mug_entity, getattr(self, "_mug_physics_pose", self.mug_pose)
        )
        return Pose(physics_pose.p, getattr(self, "_mug_logical_q", physics_pose.q))

    def _get_gripper_dof_indices(self, arm):
        gripper_dofs = list(getattr(arm, "_finger_dof_indices", None) or [])
        if not gripper_dofs:
            gripper_dofs = list(getattr(arm, "_driver_dof_indices", None) or [])
        if not gripper_dofs:
            joints = list(getattr(arm, "finger_joints", None) or [])
            joints += list(getattr(arm, "driver_joints", None) or [])
            seen = set()
            for joint in joints:
                if joint is None:
                    continue
                idx = _get_dof_idx(joint)
                if idx is not None and idx not in seen:
                    seen.add(idx)
                    gripper_dofs.append(idx)
        return gripper_dofs

    def _boost_gripper_force(self, arm):
        gripper_dofs = self._get_gripper_dof_indices(arm)
        if not gripper_dofs:
            return

        update_dofs_kp_kv_compat(
            arm.entity,
            gripper_dofs,
            kp_value=self.GRIPPER_KP,
            kv_value=self.GRIPPER_KV,
        )
        low, high = self.GRIPPER_FORCE_RANGE
        update_dofs_force_range_compat(
            arm.entity,
            gripper_dofs,
            lower_value=low,
            upper_value=high,
        )

    # ------------------------------------------------------------------
    # Motion planning helpers
    # ------------------------------------------------------------------

    def _move_cartesian(self, pose7, arm_tag: str, label: str = ""):
        """Move via screw planning, falling back to RRT transit.
        Verifies actual EE pose after execution against the requested target;
        silent IK relax (planner reports Success but lands far off) is
        flagged and aborts the rollout instead of silently corrupting the
        downstream pour math."""
        if not self.plan_success:
            return None
        arm = self.robot.get_arm(arm_tag)
        pose7 = np.asarray(pose7, dtype=np.float64).ravel()[:7]

        def _try_and_verify(plan_fn):
            res = plan_fn(arm.get_arm_qpos(), pose7)
            if not res.success:
                return None, None, None
            self.execute_plan(res, arm_tag)
            ee_actual = np.asarray(arm.get_ee_pose(), dtype=np.float64).ravel()[:7]
            xyz_err = float(np.linalg.norm(ee_actual[:3] - pose7[:3]))
            dot = float(abs(np.dot(ee_actual[3:], pose7[3:])))
            q_err_deg = float(np.degrees(2 * np.arccos(min(1.0, dot))))
            return res, xyz_err, q_err_deg

        # Try plan_screw first (preserves orientation along approach).
        result, xyz_err, q_err = _try_and_verify(arm.planner.plan_screw_path)
        if result is not None and xyz_err <= 0.05 and q_err <= 15.0:
            return result

        # RRT fallback: for transit legs (lift / pre-pour), a curved path
        # is acceptable when the straight-line passes through a joint
        # limit and silently relaxes.
        result, xyz_err, q_err = _try_and_verify(arm.planner.plan_path)
        if result is not None and xyz_err <= 0.05 and q_err <= 15.0:
            return result

        self.plan_success = False
        return None

    def _mug_to_link(self, T_world_mug: Pose, arm_tag: str) -> Pose:
        """Convert desired mug world pose → EE link pose for IK."""
        arm = self.robot.get_arm(arm_tag)
        T_world_ee = T_world_mug * self._T_ee_mug.inv()
        return T_world_ee * arm.tcp_offset.inv()

    # ------------------------------------------------------------------
    # Pour pose computation
    # ------------------------------------------------------------------

    def _cup_mug_geometry(self):
        """Compute shared geometry: cup_rim_z, cup_xy, toward_mug unit vector, tilt_axis."""
        cup_pose = self._get_object_pose(
            self.cup_actor.entity if self.cup_actor else None, self.cup_pose,
        )
        cup_xy = cup_pose.p[:2]
        cup_rim_z = cup_pose.p[2] + self.cup_rim_height

        mug_pose = self._get_object_pose(
            self.mug_entity if self.mug_entity is not None else None,
            getattr(self, "_mug_physics_pose", self.mug_pose),
        )
        mug_xy = mug_pose.p[:2]
        delta = np.array([cup_xy[0] - mug_xy[0], cup_xy[1] - mug_xy[1], 0.0])
        toward_cup = delta / np.linalg.norm(delta)
        toward_mug = -toward_cup[:2]

        tilt_axis = np.cross([0, 0, 1], toward_cup)
        tilt_axis /= np.linalg.norm(tilt_axis)

        return cup_rim_z, cup_xy, toward_mug, tilt_axis

    def _compute_pour_origin(self):
        """Compute shared mug_origin XY and Z base for pre-pour and pour-tilt."""
        cup_rim_z, cup_xy, toward_mug, tilt_axis = self._cup_mug_geometry()
        # Body grasping is pure contact, so the mug lags the planned EE pose
        # slightly during tilt. Aim just closer than the geometric rim-to-rim
        # clearance so the final mug origin stays inside the 20 cm eval band.
        origin_xy = cup_xy + toward_mug * (self.cup_rim_radius + self.mug_rim_radius + 0.05)
        return cup_rim_z, origin_xy, tilt_axis

    # Yaw rotation about the mug's own +z axis is invariant for the
    # handle grip (the gripper rotates in lockstep), so we have one
    # free dof at the pour pose: the mug yaw.  Sweeping it lets the
    # planner pick a wrist-roll that frees joint-4 enough to reach
    # the commanded EE z.  Initial 0° is tried first so the original
    # spawn yaw wins when feasible.
    _POUR_YAW_SEARCH_DEG = (0, -45, 45, -90, 90, -135, 135, 180)

    # Avatar obstacle cloud used only for the near-cup pre-pour / tilt legs.
    # The live collision failure is link7 against the thin oriented r_palm
    # box, so sampling only the avatar's bone capsules is insufficient: the
    # palm box itself must be present in mplib's world.
    _AVATAR_OBSTACLE_RES = 0.025
    _AVATAR_OBSTACLE_PAD_M = 0.01

    # Eval-mode avatar schedule.  Policy eval bypasses play_once and calls
    # take_action(action)->obs repeatedly.  For pour-water the avatar should
    # finish drinking before the robot starts, matching the scripted rollout.
    EVAL_TRIGGER_STEP_MIN = 100
    EVAL_TRIGGER_STEP_MAX = 200

    def _yaw_rotated_quat(self, base_q, yaw_deg: float):
        R_yaw = t3d.axangles.axangle2mat([0, 0, 1], np.radians(yaw_deg))
        R_base = t3d.quaternions.quat2mat(base_q)
        return t3d.quaternions.mat2quat(R_yaw @ R_base)

    def _avatar_obstacle_points(self) -> np.ndarray | None:
        """Sample the live avatar capsules *and palm boxes* for mplib."""
        collider = self.avatar_collider
        if collider is None:
            return None

        spacing = self._AVATAR_OBSTACLE_RES
        pad = self._AVATAR_OBSTACLE_PAD_M
        parts = []

        for _name, pa, pb, radius in collider.current_capsules():
            pa = np.asarray(pa, dtype=np.float64)
            pb = np.asarray(pb, dtype=np.float64)
            segment = pb - pa
            length = float(np.linalg.norm(segment))
            if length < 1e-8:
                continue
            axis = segment / length
            tmp = (
                np.array([0.0, 0.0, 1.0])
                if abs(axis[2]) < 0.9
                else np.array([1.0, 0.0, 0.0])
            )
            u = np.cross(axis, tmp)
            u /= np.linalg.norm(u) + 1e-12
            v = np.cross(axis, u)
            v /= np.linalg.norm(v) + 1e-12
            n_axis = max(2, int(np.ceil(length / spacing)) + 1)
            centers = pa[None, :] + np.linspace(0.0, 1.0, n_axis)[:, None] * segment
            shell_radius = float(radius) + pad
            shell = []
            for theta in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
                shell.append(
                    centers
                    + shell_radius * (np.cos(theta) * u + np.sin(theta) * v)[None, :]
                )
            parts.extend([centers, *shell])

        # Palms are oriented boxes, not capsules.  Sample all six padded box
        # faces densely enough that mplib's voxelization cannot route link7
        # through the broad face between sparse corner points.
        for _name, center, axes, half_extents in collider.current_boxes():
            center = np.asarray(center, dtype=np.float64)
            axes = np.asarray(axes, dtype=np.float64)
            half_extents = np.asarray(half_extents, dtype=np.float64) + pad
            grids = [
                np.linspace(-extent, extent, max(2, int(np.ceil(2 * extent / spacing)) + 1))
                for extent in half_extents
            ]
            face_points = []
            for fixed_axis in range(3):
                free_axes = [axis for axis in range(3) if axis != fixed_axis]
                aa, bb = np.meshgrid(
                    grids[free_axes[0]], grids[free_axes[1]], indexing="ij",
                )
                for sign in (-1.0, 1.0):
                    local = np.zeros((aa.size, 3), dtype=np.float64)
                    local[:, fixed_axis] = sign * half_extents[fixed_axis]
                    local[:, free_axes[0]] = aa.ravel()
                    local[:, free_axes[1]] = bb.ravel()
                    face_points.append(center[None, :] + local @ axes.T)
            parts.extend(face_points)

        return np.concatenate(parts, axis=0) if parts else None

    def _refresh_avatar_planner_obstacles(self, arm_tag: str) -> None:
        points = self._avatar_obstacle_points()
        if points is None or points.size == 0:
            return
        self.robot.get_arm(arm_tag).planner.update_obstacles(
            points, resolution=self._AVATAR_OBSTACLE_RES,
        )

    def _compute_pre_pour_pose(self, arm_tag: str, yaw_deg: float = 0.0) -> Pose:
        """Mug upright (with optional yaw offset about world +z), origin near cup rim."""
        cup_rim_z, origin_xy, _ = self._compute_pour_origin()
        mug_origin = np.array([origin_xy[0], origin_xy[1],
                               cup_rim_z + self.PRE_POUR_HEIGHT])
        q = self._yaw_rotated_quat(self._mug_physics_pose.q, yaw_deg)
        return self._mug_to_link(Pose(mug_origin, q), arm_tag)

    def _compute_pour_tilt_pose(self, arm_tag: str, yaw_deg: float = 0.0) -> Pose:
        """Mug tilted to pour — same origin XY, gripper just rotates in place."""
        cup_rim_z, origin_xy, tilt_axis = self._compute_pour_origin()

        # Apply yaw first (about mug's local +z), then the pour tilt.
        R_yawed = t3d.quaternions.quat2mat(
            self._yaw_rotated_quat(self._mug_physics_pose.q, yaw_deg),
        )
        R_tilt = t3d.axangles.axangle2mat(tilt_axis, np.radians(self.POUR_TILT_DEG))
        R_tilted = R_tilt @ R_yawed

        mug_origin = np.array([origin_xy[0], origin_xy[1],
                               cup_rim_z + self.POUR_TILT_HEIGHT])
        return self._mug_to_link(
            Pose(mug_origin, t3d.quaternions.mat2quat(R_tilted)), arm_tag,
        )

    # ------------------------------------------------------------------
    # Main task sequence
    # ------------------------------------------------------------------

    # Reference wrist XY at attachment frame (F33 of drink_water_1) for the
    # original tuned avatar (custom_Adrian_Keller @ avatar_init_pos). All other
    # avatars are shifted so their wrist lands at this same world point — this
    # makes the captured cup-attachment offset (computed in attach_object_to_hand
    # from wrist position and orientation) identical to the tuned baseline.
    REFERENCE_WRIST_XY = np.array([-0.16662797, 0.08912287], dtype=np.float64)

    def _calibrate_avatar_pos(self):
        """Adjust avatar position so wrist matches the reference attach-frame pose.

        Plays drink_water_1 fully (no attachment), reads the wrist position at
        F33 (the attachment frame), and shifts the avatar so wrist XY equals
        REFERENCE_WRIST_XY. The wrist is exactly the point used as the origin
        for the cup attachment offset, so this preserves the tuned wrist→cup
        relative pose for every avatar.
        """
        if self.avatar is None:
            return

        # Play drink_water_1 fully (no cup attachment) to get wrist position
        self.avatar.play_animation("drink_water_1")
        while not self.avatar.spare():
            self.scene.step()
            self.avatar.step()

        wrist_pos = self.avatar.get_hand_pos(hand_id=1)
        target_xy = self.REFERENCE_WRIST_XY + np.array([self._avatar_shift_x, 0.0])
        delta = np.zeros(3, dtype=np.float64)
        delta[0] = target_xy[0] - wrist_pos[0]
        delta[1] = target_xy[1] - wrist_pos[1]
        self.avatar.reset(self.avatar_init_pos.copy() + delta, self.avatar_init_rot.copy())
        for _ in range(50):
            self.scene.step()

    # ------------------------------------------------------------------
    # Eval-mode hooks: policy evaluation bypasses play_once, so the avatar
    # drinking sequence is driven from take_action via EvalModeAvatarMixin.
    # ------------------------------------------------------------------
    def _eval_at_reset(self):
        # Keep eval/testbed reset cheap.  Scripted play_once performs the
        # full dry-run wrist calibration; eval mode only needs a plausible
        # avatar drinking event while the policy drives the robot.  Shift the
        # default avatar with the randomized cup X so the hand/cup relation
        # stays close without burning minutes before the policy loop starts.
        shift = np.array([getattr(self, "_avatar_shift_x", 0.0), 0.0, 0.0])
        self.avatar.reset(self.avatar_init_pos.copy() + shift, self.avatar_init_rot.copy())
        if bool(self.config.get("eval_avatar_pre_policy", True)):
            self._eval_trigger_step = None
        else:
            lo = int(self.config.get("eval_trigger_step_min", self.EVAL_TRIGGER_STEP_MIN))
            hi = int(self.config.get("eval_trigger_step_max", self.EVAL_TRIGGER_STEP_MAX))
            if hi < lo:
                hi = lo
            # np.random is seeded by BaseTask.reset, so this is reproducible per seed.
            self._eval_trigger_step = int(np.random.randint(lo, hi + 1))
        self._eval_drink1_started = False
        self._eval_drink2_queued = False

    def _start_eval_drink1(self):
        cup_entity = self.cup_actor.entity if self.cup_actor else None
        self.avatar.play_animation("drink_water_1", attach_obj=cup_entity, hand_id=1)
        self._eval_drink1_started = True

    def _start_eval_drink2(self):
        self.avatar.play_animation_hold_reverse(
            "drink_water_2", hold_at_frame=39, hold_duration=70,
            reverse_to_frame=1, hold_end=30,
        )
        self._eval_drink2_queued = True

    def eval_pre_policy_warmup(self):
        if (
            self.avatar is None
            or not bool(self.config.get("eval_mode", False))
            or not bool(self.config.get("eval_avatar_pre_policy", True))
        ):
            return self.get_obs()
        if not self._eval_drink1_started:
            self._start_eval_drink1()
        while not self.avatar.spare():
            self.step_sim()
        if not self._eval_drink2_queued:
            self._start_eval_drink2()
        while not self.avatar.spare():
            self.step_sim()
        for _ in range(int(self.config.get("eval_avatar_post_warmup_settle_steps", 0))):
            self.step_sim()
        return self.get_obs()

    def _eval_at_step(self, step_idx: int):
        if self._eval_trigger_step is None:
            return
        if not self._eval_drink1_started and step_idx >= self._eval_trigger_step:
            self._start_eval_drink1()
            return
        # Chain drink_water_2 the moment drink_water_1 finishes.
        if (self._eval_drink1_started
                and (not self._eval_drink2_queued)
                and self.avatar.spare()):
            self._start_eval_drink2()

    def play_once(self) -> bool:
        # Calibrate avatar position for this avatar model
        self._calibrate_avatar_pos()

        # Phase 1: Avatar drinks from cup
        if self.avatar is not None:
            cup_entity = self.cup_actor.entity if self.cup_actor else None
            # drink_water_1: attach cup at last frame
            self.avatar.play_animation("drink_water_1", attach_obj=cup_entity, hand_id=1)
            while not self.avatar.spare():
                self.step_sim()
            # drink_water_2: hold at frame 39, hold 70 frames, reverse to frame 1, hold
            self.avatar.play_animation_hold_reverse(
                "drink_water_2", hold_at_frame=39, hold_duration=70,
                reverse_to_frame=1, hold_end=30,
            )
            while not self.avatar.spare():
                self.step_sim()

        # Phase 2: Arm grasps mug, lifts, pours into cup
        if self.mug_entity is None:
            return False

        arm_tag = "right"
        arm = self.robot.get_arm(arm_tag)
        # Pre-flight + execute the first feasible body grasp candidate via
        # the shared helper: pre→grasp leg uses plan_screw (no RRT fallback)
        # so the gripper approaches the mug along the grasp z-axis.  The
        # latest mug poses intentionally grasp the mug body/rim edge; handle
        # grasps are not used for this task.
        robot_type = self.config.get("robot_type", "franka")
        self.open_gripper(arm_tag)
        mug_pose = self._get_mug_logical_pose()
        result = None
        for grasp_name in _MUG_BODY_GRASP_ORDER:
            result = self.try_grasp_by_name(
                _MUG_ID, mug_pose, arm_tag,
                grasp_name=grasp_name,
                robot_type=robot_type,
                object_scale=self.mug_scale,
            )
            if result is not None:
                break
        if result is None:
            result = self.select_and_execute_grasp(
                _MUG_ID, mug_pose, arm_tag,
                robot_type=robot_type, max_candidates=10,
                object_scale=self.mug_scale,
                categories=["body"],
            )
        if result is None:
            return False

        grasp_link, pre_link, grasp = result

        self._boost_gripper_force(arm)

        for _ in range(100):
            self.step_sim()
        self.close_gripper(arm_tag)
        for _ in range(50):
            self.step_sim()

        self._boost_gripper_force(arm)

        # Record grasp-time EE↔mug transform (for pose computation, no kinematic attach)
        ee_pose = Pose.from_pose7(arm.get_ee_pose())
        mug_now = self._get_object_pose(self.mug_entity, self._mug_physics_pose)
        self._T_ee_mug = ee_pose.inv() * mug_now

        # Lift straight up (skipped when LIFT_HEIGHT == 0 — handle grip's
        # post-grasp wrist config can't translate +z while preserving
        # orientation; the pre-pour move handles vertical clearance via
        # a curved RRT path with orientation freedom along the way).
        grasp_tcp = grasp.to_world(mug_pose, self.mug_scale)
        if self.LIFT_HEIGHT > 1e-6:
            lift_tcp = Pose(grasp_tcp.p + [0, 0, self.LIFT_HEIGHT], grasp_tcp.q)
            lift_link = tcp_to_link_pose(lift_tcp, arm.tcp_offset)
            if self._move_cartesian(lift_link.to_pose7(), arm_tag, label="lift") is None:
                return False

        # Move to cup rim → tilt to pour.  Sweep mug yaw to find a
        # wrist roll that lets the EE actually reach the commanded
        # pose; the handle grip is yaw-invariant so this is free.
        # Lock the chosen yaw on pre-pour so pour-tilt continues
        # from a feasible config.
        chosen_yaw = self._sweep_yaw_and_execute(
            self._compute_pre_pour_pose, arm_tag, label="pre-pour",
        )
        if chosen_yaw is None:
            return False
        if str(self.config.get("robot_type", "franka")).lower() == "xarm7":
            tilt_yaws = None
        else:
            # Prefer to preserve the pre-pour yaw, but do not lock the wrist
            # into it: an upright wrist pose can clear the palm while the same
            # yaw's tilted link7 pose intersects it.  Mug yaw about its own
            # axis is task-invariant, so search the remaining equivalents.
            tilt_yaws = (chosen_yaw,) + tuple(
                yaw for yaw in self._POUR_YAW_SEARCH_DEG if yaw != chosen_yaw
            )
        if self._sweep_yaw_and_execute(
            self._compute_pour_tilt_pose, arm_tag, label="pour-tilt",
            yaws=tilt_yaws,
        ) is None:
            return False
        return True

    def evaluate(self) -> dict:
        metrics = super().evaluate()
        if bool(self.config.get("eval_mode", False)):
            metrics.update({
                "eval_mode": True,
                "eval_trigger_step": getattr(self, "_eval_trigger_step", None),
                "eval_drink1_started": bool(getattr(self, "_eval_drink1_started", False)),
                "eval_drink2_queued": bool(getattr(self, "_eval_drink2_queued", False)),
            })
        return metrics

    def _sweep_yaw_and_execute(self, pose_fn, arm_tag: str, label: str,
                                yaws=None):
        """Try `pose_fn(arm_tag, yaw_deg=...)` over a list of yaws; FK-verify
        each plan BEFORE executing; execute the first one that lands within
        tolerance.  Returns the chosen yaw_deg, or None if nothing works."""
        arm = self.robot.get_arm(arm_tag)
        candidate_yaws = yaws if yaws is not None else self._POUR_YAW_SEARCH_DEG

        # Snapshot the avatar immediately before planning.  The drinking
        # animation is already at its final hold here, so this static cloud is
        # valid for the short pre-pour / tilt execution that follows.
        self._refresh_avatar_planner_obstacles(arm_tag)

        for yaw_deg in candidate_yaws:
            link_pose = pose_fn(arm_tag, yaw_deg=yaw_deg)
            target = np.asarray(link_pose.to_pose7(), dtype=np.float64).ravel()[:7]

            for plan_fn in (
                arm.planner.plan_screw_path,
                arm.planner.plan_path,
            ):
                res = plan_fn(arm.get_arm_qpos(), target)
                if not res.success:
                    continue
                # Pre-execution FK check: set qpos, read link, restore.
                final_qpos = (
                    res.position[-1] if res.position.size > 0
                    else arm.get_arm_qpos()
                )
                saved = arm.entity.get_qpos()
                full_qpos = arm._build_qpos_with_arm(final_qpos)
                arm.entity.set_qpos(
                    torch.tensor(full_qpos, dtype=torch.float32)
                    if not hasattr(full_qpos, "cpu") else full_qpos
                )
                arm.entity.get_links_pos()
                ee_link_pos = to_numpy(arm.ee_link.get_pos()).ravel()[:3]
                fk_err = float(np.linalg.norm(ee_link_pos - target[:3]))
                arm.entity.set_qpos(saved)
                arm.entity.get_links_pos()

                if fk_err <= 0.05:
                    self.execute_plan(res, arm_tag)
                    print(f"[pour] {label} selected yaw={yaw_deg:g}°")
                    return yaw_deg

        self.plan_success = False
        print(f"[pour] {label} failed for yaws={tuple(candidate_yaws)}")
        return None
