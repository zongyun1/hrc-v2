"""Stack three bowls: Franka picks two bowls and stacks them on a
third (base) bowl on the table. Single-arm, no avatar — RoboTwin's
`stack_bowls_three` task scaled down to one Franka.

Three identical bowls (002_bowl, model_id=1, 11.4 × 4.9 × 11.4 cm)
spawn at random xy in a 0.40 × 0.40 m region. Bowl 0 is the base;
the robot picks bowls 1 and 2 in turn and drops them on the stack.

Pure physics throughout (memory `feedback_no_kinematic_snap` forbids
held-object snap during rollout). The grasp helper picks the
`dish_rack_rim` grasp (fingers straddle the 3 mm rim wall — one
inside cavity, one outside). The task forces `kinematic_pin=True`
on the chosen grasp so the bowl is held still during close (the
shared 002_bowl YAML is used by other tasks and can't be edited).
A light close (set_gripper ≈ 0.025 → ~1 mm finger gap, ~10-30 N per
finger) avoids tunnelling through the thin wall.
"""

import json
import numpy as np
import transforms3d as t3d

from ..base_task import BaseTask
from ..utils import (
    Pose, load_object, create_primitive, to_numpy, ASSETS_PATH, TABLE_HEIGHT,
)
from ..grasp import tcp_to_link_pose
from ..object_catalog import get_entry


# Target object identity from the central catalog (model_id=1 = the smaller
# 11.4 × 4.9 cm bowl variant).
_BOWL_ENTRY = get_entry("002_bowl")
_BOWL_ID = _BOWL_ENTRY.object_id
_BOWL_MODEL_ID = _BOWL_ENTRY.model_id
_NUM_BOWLS = 3

# Randomization region — 0.40 × 0.40 m. Bowl outer dia ~11.4 cm so
# 0.16 m min separation gives 4-5 cm clearance for the gripper.
_REGION_X = (-0.18, +0.22)
_REGION_Y = (-0.18, +0.18)
_MIN_SEP_M = 0.16
_SPAWN_SAMPLE_ATTEMPTS = 256
_SPAWN_FALLBACK_MARGIN = 0.05

# Use the deep dish_rack_rim grasp (fingers straddle the 3 mm rim
# wall vertically — one finger inside cavity, one outside).  The
# shallower `rim_top` only contacts the top edge and slips immediately.
_GRASP_CATEGORIES = ["dish_rack_rim"]


def _read_model_data(name, model_id=0):
    p = ASSETS_PATH / "objects" / name / f"model_data{model_id}.json"
    with open(p) as f:
        return json.load(f)


def _read_scale(name, model_id=0):
    raw = _read_model_data(name, model_id).get("scale", 1.0)
    return float(raw[0] if isinstance(raw, (list, tuple)) else raw)


def _read_extents(name, model_id=0):
    return np.array(_read_model_data(name, model_id)["extents"], dtype=np.float64)


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def _sample_bowl_xy(rng, n_bowls):
    if n_bowls == 2:
        return [(-0.08, -0.06), (0.14, -0.16)]
    for _ in range(_SPAWN_SAMPLE_ATTEMPTS):
        pts = []
        ok = True
        for _ in range(n_bowls):
            x = float(rng.uniform(*_REGION_X))
            y = float(rng.uniform(*_REGION_Y))
            for px, py in pts:
                if (x - px) ** 2 + (y - py) ** 2 < _MIN_SEP_M ** 2:
                    ok = False
                    break
            if not ok:
                break
            pts.append((x, y))
        if ok and len(pts) == n_bowls:
            return pts
    fallback = [
        (
            _REGION_X[0] + _SPAWN_FALLBACK_MARGIN,
            _REGION_Y[0] + _SPAWN_FALLBACK_MARGIN,
        ),
        (
            _REGION_X[1] - _SPAWN_FALLBACK_MARGIN,
            _REGION_Y[1] - _SPAWN_FALLBACK_MARGIN,
        ),
        (0.0, 0.0),
    ]
    return fallback[:n_bowls]


class StackBowlsThree(BaseTask):
    """Robot picks two bowls and stacks them on a third (base) bowl."""

    INSTRUCTION = "stack the three bowls on top of each other"
    OBJECT_SET = ["002_bowl"]
    use_avatar = False
    EASY_NUM_OBJECTS = 2
    HARD_NUM_OBJECTS = _NUM_BOWLS

    TABLE_THICKNESS_VALUE = 0.05
    TABLE_HALF_SIZE = (0.6, 0.55)
    TABLE_CENTER_XY = (0.0, -0.35)
    TABLE_LEG_RADIUS = 0.025
    TABLE_LEG_XY = (
        (-0.55, -0.85),
        (0.55, -0.85),
        (-0.55, 0.15),
        (0.55, 0.15),
    )

    BOWL_UPRIGHT_QUAT = np.array(
        [0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64,
    )
    BOWL_YAW_LIMIT = np.pi / 4
    BOWL_SPAWN_Z_CLEARANCE = 0.02
    BOWL_FRICTION = 5.0
    BOWL_DENSITY = 10.0

    LIFT_HEIGHT = 0.20
    # Plate-bottom clearance above the stack top at release. Releasing
    # right at contact (DROP_BOWL_DZ=0) jerks the plate when the
    # gripper opens; 3 cm gives the plate time to fall and settle on
    # plate i's top surface (28 mm thick disk catches it cleanly).
    DROP_BOWL_DZ = 0.030

    # Success: loose stacking metric. A stacked bowl's origin only needs
    # to land inside bowl 0's horizontal footprint and be higher than
    # bowl 0. This intentionally ignores uprightness and exact centering.
    SUCCESS_XY_TOL = 0.10
    SUCCESS_DZ_MIN = 0.0
    # Kept for old configs/diagnostics; check_success no longer uses it.
    SUCCESS_UP_DOT_MIN = 0.5
    SUCCESS_ELEVATED_DZ_MIN = 0.0
    SUCCESS_BASE_BOUNDS_MARGIN = 0.0

    GRASP_MAX_CANDIDATES = 10
    GRASP_TABLE_Z_MARGIN = 0.02
    PIN_SETTLE_STEPS = 40
    GRIPPER_CLOSE_VALUE = 0.025
    GRIPPER_CLOSE_SETTLE_STEPS = 120
    LIFT_SUCCESS_Z_MARGIN = 0.02
    ABOVE_STACK_SETTLE_STEPS = 80
    RELEASE_PARTIAL_OPEN_VALUE = 0.40
    RELEASE_PARTIAL_SETTLE_STEPS = 60
    RELEASE_OPEN_SETTLE_STEPS = 80
    RETREAT_Z_ABOVE_TABLE = 0.30
    INITIAL_SETTLE_STEPS = 60
    FINAL_SETTLE_STEPS = 180

    recording_camera_pos = [1.1, 0.0, 1.95]
    recording_camera_lookat = [0.0, 0.0, 0.85]

    def __init__(self, config: dict = None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        # Keep mplib for the grasp phase (the helper needs plan_screw
        # which genesis_ik doesn't implement). After grasp we hand off
        # to genesis_ik in `_pick_and_stack_one` so the lift + place
        # use damped least squares from current qpos and don't suffer
        # the silent-relax mplib bug.
        super().__init__(cfg)
        self.resolve_target_object()   # registry/override hook (size-1 set)
        self._num_bowls = self.difficulty_object_count()

    def _measure_held_offset(self, bowl_idx, arm_tag):
        """TCP→bowl offset right after close, used to aim the place pose.

        The bowl is held purely by friction — Genesis updates its world
        pose every step. We only sample once (post-close) so we know how
        the bowl sits in the gripper for placement targeting.
        """
        arm = self.robot.get_arm(arm_tag)
        tcp_now = np.asarray(arm.get_ee_pose()[:3], dtype=np.float64)
        bowl_p = np.asarray(self.bowls[bowl_idx].get_pose().p, dtype=np.float64)
        return bowl_p - tcp_now

    def _create_table(self, table_height=TABLE_HEIGHT):
        # Wider table than default (matches place_bread_in_basket) so the
        # 0.40 × 0.40 m spawn region is fully inside the top.
        self.TABLE_HEIGHT = table_height
        self.TABLE_THICKNESS = self.TABLE_THICKNESS_VALUE
        self.TABLE_TOP_Z = table_height + self.TABLE_THICKNESS / 2
        self._table_top_z_fixed = False
        if self._try_create_table_variant(
            table_height,
            half_size=self.TABLE_HALF_SIZE,
            center_xy=self.TABLE_CENTER_XY,
        ):
            return
        self.table = create_primitive(
            self.scene, "box", Pose(p=[*self.TABLE_CENTER_XY, table_height]),
            size={"half_size": (*self.TABLE_HALF_SIZE, self.TABLE_THICKNESS / 2)},
            color=(0.8, 0.75, 0.65), is_static=True,
        )
        leg_h = table_height - self.TABLE_THICKNESS / 2
        for x, y in self.TABLE_LEG_XY:
            create_primitive(
                self.scene, "cylinder",
                Pose(p=[x, y, leg_h / 2]),
                size={"radius": self.TABLE_LEG_RADIUS, "half_length": leg_h / 2},
                color=(0.5, 0.5, 0.5), is_static=True,
            )

    def _swap_planner(self, arm_tag, planner_type):
        """Swap the arm's planner.  genesis_ik for post-grasp moves
        (mplib's silent relax breaks long Cartesian plans — see memory
        `mplib_silent_ik_relax`), mplib back for the next grasp helper
        invocation (which needs plan_screw, not in genesis_ik).

        Reads the URDF/SRDF stashed by the active Robot during
        ``init_joints`` so xArm7 / Franka / future embodiments swap
        against their own planner config, not a hardcoded Franka path.
        """
        arm = self.robot.get_arm(arm_tag)
        arm.init_planner(
            urdf_path=self.robot.planner_urdf_path,
            srdf_path=self.robot.planner_srdf_path,
            scene=self.scene,
            planner_type=planner_type,
        )

    def load_actors(self):
        # mesh-y → world-z (002_bowl is y-up).
        upright_q = self.BOWL_UPRIGHT_QUAT
        table_top = self.TABLE_TOP_Z

        rng = np.random.RandomState(np.random.randint(0, 2 ** 31 - 1))
        positions = _sample_bowl_xy(rng, self._num_bowls)

        self._bowl_scale = _read_scale(_BOWL_ID, _BOWL_MODEL_ID)
        bowl_extents = _read_extents(_BOWL_ID, _BOWL_MODEL_ID)
        self._bowl_extents = bowl_extents
        # mesh-y is the bowl's vertical axis; after upright_q it is world z.
        self._bowl_height = float(bowl_extents[1]) * self._bowl_scale

        # Yaw clamped to ±π/4: the rim grasp candidates are written for
        # the four cardinal mesh directions (`cont_bowl_top_down_v1` set
        # in the shared YAML); the helper's approach-toward-robot score
        # picks whichever cardinal is most reachable, so a moderate yaw
        # spread doesn't hurt.
        yaws = [
            float(rng.uniform(-self.BOWL_YAW_LIMIT, self.BOWL_YAW_LIMIT))
            for _ in range(self._num_bowls)
        ]

        # Lift mesh slightly above the table so the rim wall doesn't
        # collide on spawn.
        spawn_z = table_top + self.BOWL_SPAWN_Z_CLEARANCE
        self.bowls = []
        self._spawn_xy = []
        for i, ((cx, cy), yaw) in enumerate(zip(positions, yaws)):
            yaw_q = np.asarray(
                t3d.quaternions.mat2quat(t3d.euler.euler2mat(0, 0, yaw, "sxyz")),
                dtype=np.float64,
            )
            full_q = _quat_mul(yaw_q, upright_q)
            # density 10 kg/m³ is unrealistic but keeps plate mass
            # ~12 g; gripper torque during transit is the dominant
            # slip-cause on a rim grip and lower mass cuts it linearly.
            actor = load_object(
                self.scene,
                Pose([cx, cy, spawn_z], full_q),
                _BOWL_ID, model_id=_BOWL_MODEL_ID,
                convex=True, is_static=False,
                friction=self.BOWL_FRICTION,
                density=self.BOWL_DENSITY,
            )
            self.bowls.append(actor)
            self._spawn_xy.append((cx, cy))

        # Bowl 0 = base (stays put); the rest get stacked on top.
        self._stacked_indices = list(range(1, self._num_bowls))

    # ------------------------------------------------------------------
    # Main task sequence
    # ------------------------------------------------------------------

    def _bowl_xyz(self, idx):
        return to_numpy(self.bowls[idx].entity.get_pos()).ravel()[:3]

    def _bowl_xyz_str(self, idx):
        p = self._bowl_xyz(idx)
        return f"({p[0]:+.3f},{p[1]:+.3f},{p[2]:.3f})"

    def _bowl_up_dot(self, idx):
        quat = to_numpy(self.bowls[idx].entity.get_quat()).ravel()[:4]
        rot = t3d.quaternions.quat2mat(quat)
        # Mesh local +Y is the bowl opening/up direction.
        up_world = rot[:, 1]
        return float(np.dot(up_world, np.array([0.0, 0.0, 1.0])))

    def _bowl_xy_half_bounds(self):
        scale = float(getattr(self, "_bowl_scale", _read_scale(_BOWL_ID, _BOWL_MODEL_ID)))
        extents = getattr(self, "_bowl_extents", None)
        if extents is None:
            extents = _read_extents(_BOWL_ID, _BOWL_MODEL_ID)
        extents = np.asarray(extents, dtype=np.float64)
        # 002_bowl is mesh-Y-up, so mesh X/Z are the table-plane axes.
        margin = float(getattr(self, "SUCCESS_BASE_BOUNDS_MARGIN", 0.0))
        return np.array([
            0.5 * float(extents[0]) * scale + margin,
            0.5 * float(extents[2]) * scale + margin,
        ], dtype=np.float64)

    def _pick_and_stack_one(self, src_idx, stack_position):
        arm_tag = "right"
        arm = self.robot.get_arm(arm_tag)
        tcp_offset = arm.tcp_offset

        src_pose_geom = self.bowls[src_idx].get_pose()
        live_p = np.asarray(src_pose_geom.p, dtype=np.float64)
        live_q = np.asarray(src_pose_geom.q, dtype=np.float64)
        # The entity pose already contains BOWL_UPRIGHT_QUAT because mesh
        # loading preserves the authored Y-up frame. Feed that physical pose
        # directly to the Y-up grasp annotations; composing upright again
        # would rotate the rim approach onto its side.
        src_pose = Pose(live_p, live_q)

        self.open_gripper(arm_tag)

        # Standard rim grasp via the shared helper. table_z lowered by
        # 2 cm so the rim TCP (≈4 mm above bowl bottom) doesn't get
        # filtered by the helper's table-clearance gate.
        result = self.select_and_execute_grasp(
            object_name=_BOWL_ID,
            object_pose=src_pose,
            arm_tag=arm_tag,
            model_id=_BOWL_MODEL_ID,
            object_scale=self._bowl_scale,
            max_candidates=self.GRASP_MAX_CANDIDATES,
            categories=_GRASP_CATEGORIES,
            table_z=self.TABLE_TOP_Z - self.GRASP_TABLE_Z_MARGIN,
        )
        if result is None:
            return False
        grasp_link, _pre_link, grasp = result
        _ = grasp_link

        # Force pin during close. The 002_bowl YAML is shared with
        # other tasks and can't carry
        # `kinematic_pin: true`; we override locally so the bowl
        # stays still while the gripper closes on its rim.
        # Initialization-phase pin is permitted by memory
        # `feedback_no_kinematic_snap`; the held-object snap during
        # transit/lift is what's forbidden.
        grasp.kinematic_pin = True
        do_pin = bool(grasp.kinematic_pin)
        if do_pin:
            pin_pose = self.bowls[src_idx].get_pose()
            pin_pos = np.asarray(pin_pose.p, dtype=np.float64).copy()
            pin_quat = np.asarray(pin_pose.q, dtype=np.float64).copy()

        def _step_pinned(n):
            ent = self.bowls[src_idx].entity
            for _ in range(n):
                if do_pin:
                    ent.set_pos(pin_pos.astype(float))
                    ent.set_quat(pin_quat.astype(float))
                    try:
                        ent.set_dofs_velocity(np.zeros(6, dtype=np.float64))
                    except Exception:
                        pass
                self.step_sim()

        # Settle, then close on the bowl's 3 mm rim wall.  The wall
        # is much thinner than the plate (28 mm), so the close target
        # has to be tighter — set_gripper(0.025) → 1 mm per finger
        # = 2 mm gap, squeezing the wall by 1 mm overshoot → ~10 N
        # per finger.  Light enough to not tunnel through the thin
        # wall; firm enough to engage the rim.
        _step_pinned(self.PIN_SETTLE_STEPS)
        self.set_gripper(self.GRIPPER_CLOSE_VALUE, arm_tag)
        _step_pinned(self.GRIPPER_CLOSE_SETTLE_STEPS)
        # Pin lifts here — pure physics from this point on.

        # Capture TCP→bowl offset now (one-shot). Pure physics owns
        # the bowl from here; we only use this offset to aim the
        # place pose so the BOWL (not the TCP) lands at the stack.
        held_offset_p = self._measure_held_offset(src_idx, arm_tag)
        pre_lift_z = float(self._bowl_xyz(src_idx)[2])

        # Switch the arm planner to genesis_ik for the lift + place +
        # retreat. mplib's silent relax on the lift broke the place
        # IK in v19/v20; genesis_ik tracks current qpos faithfully.
        self._swap_planner(arm_tag, "genesis_ik")

        grasp_tcp = grasp.to_world(src_pose, self._bowl_scale)
        place_q = grasp_tcp.q

        # Lift along world +z to a safe transit altitude.
        lift_tcp = Pose(
            grasp_tcp.p + np.array([0.0, 0.0, self.LIFT_HEIGHT]),
            grasp_tcp.q,
        )
        lift_link = tcp_to_link_pose(lift_tcp, tcp_offset)
        if self.move_and_execute(lift_link.to_pose7(), arm_tag) is None:
            return False
        # Slip guard: if the bowl barely rose (<2 cm) after the lift
        # trajectory, the rim grip lost contact and we should abort.
        # Compare against the pose immediately before the lift rather
        # than an absolute table height: the bowl mesh origin is already
        # about 2.4 cm above TABLE_TOP_Z while resting, so the old check
        # could accept a bowl that never left the table.
        post_lift_z = float(self._bowl_xyz(src_idx)[2])
        if post_lift_z - pre_lift_z < self.LIFT_SUCCESS_Z_MARGIN:
            return False

        # Re-measure held offset post-lift — with the soft close
        # the plate slides a few mm in the gripper during the lift,
        # so the offset captured at close-time is stale.
        held_offset_p = self._measure_held_offset(src_idx, arm_tag)

        # Move above the stack target with the rim TCP. The held bowl
        # is offset from the TCP by held_offset_p (because the rim
        # grasp puts TCP at the rim edge, ~5 cm from the bowl center,
        # ~3-4 cm below the TCP). We solve for the TCP target so the
        # BOWL (mesh origin = bottom) sits `DROP_BOWL_DZ` above the
        # stack top:
        #   bowl_world ≈ TCP_world + held_offset_p
        #   bowl_xyz   = (target_x, target_y, stack_top_z + DROP_BOWL_DZ)
        # → TCP_world = bowl_xyz - held_offset_p
        target_x, target_y, stack_top_z = stack_position
        place_p = np.array([
            target_x - float(held_offset_p[0]),
            target_y - float(held_offset_p[1]),
            stack_top_z + self.DROP_BOWL_DZ - float(held_offset_p[2]),
        ])
        place_link = tcp_to_link_pose(Pose(place_p, place_q), tcp_offset)
        if self.move_and_execute(place_link.to_pose7(), arm_tag) is None:
            return False
        # Settle so PD converges before release.
        for _ in range(self.ABOVE_STACK_SETTLE_STEPS):
            self.step_sim()

        # Release. Open in two stages — first to gap=plate-thickness
        # (so fingers no longer clamp), then full open. One-shot
        # full-open from a soft-close target flings the plate
        # because PD ramps fingers from 8 mm to 40 mm in one step.
        self.set_gripper(self.RELEASE_PARTIAL_OPEN_VALUE, arm_tag)
        for _ in range(self.RELEASE_PARTIAL_SETTLE_STEPS):
            self.step_sim()
        self.open_gripper(arm_tag)
        for _ in range(self.RELEASE_OPEN_SETTLE_STEPS):
            self.step_sim()

        # Retreat upward so the gripper is clear before the next pick.
        retreat_p = np.array([
            target_x, target_y,
            self.TABLE_TOP_Z + self.RETREAT_Z_ABOVE_TABLE,
        ])
        retreat_link = tcp_to_link_pose(Pose(retreat_p, place_q), tcp_offset)
        self.move_and_execute(retreat_link.to_pose7(), arm_tag)

        # Restore mplib so the next pick's grasp helper can plan_screw.
        self._swap_planner(arm_tag, "mplib")
        return True

    def play_once(self) -> bool:
        # Pre-task object settle; excluded from recordings.
        with self.suppress_recording():
            for _ in range(self.INITIAL_SETTLE_STEPS):
                self.step_sim()
        if any(b is None for b in self.bowls):
            return False

        # Pick each non-base bowl in turn and place above the base.
        # The stack-top reference is the highest already-placed bowl's
        # mesh-origin (= bowl bottom) + bowl_height. For the first
        # stacked bowl this is base_top; for the second it includes
        # whatever bowl 1 actually settled at.
        for stack_idx, src_idx in enumerate(self._stacked_indices):
            base_xyz = self._bowl_xyz(0)
            placed_so_far = [0] + self._stacked_indices[:stack_idx]
            top_z = max(self._bowl_xyz(i)[2] for i in placed_so_far)
            stack_top_z = top_z + self._bowl_height
            stack_position = (base_xyz[0], base_xyz[1], stack_top_z)
            if not self._pick_and_stack_one(src_idx, stack_position):
                return False

        for _ in range(self.FINAL_SETTLE_STEPS):
            self.step_sim()
        return True

    def check_success(self) -> bool:
        if any(b is None for b in self.bowls):
            return False
        positions = [self._bowl_xyz(i) for i in range(self._num_bowls)]
        base_xy = positions[0][:2]
        base_z = float(positions[0][2])
        half_bounds = self._bowl_xy_half_bounds()
        # Stacked bowl centers must be inside bowl 0's footprint and above it.
        for i in self._stacked_indices:
            delta_xy = np.abs(positions[i][:2] - base_xy)
            dz = float(positions[i][2]) - base_z
            if bool(np.any(delta_xy > half_bounds)):
                return False
            if dz <= self.SUCCESS_DZ_MIN:
                return False
        return True
