"""Stretch SE3 mobile manipulator loaded from URDF."""

from typing import Callable, Optional

import numpy as np
import genesis as gs

from ..utils import Pose, to_numpy, ASSETS_PATH
from ..genesis_compat import set_dofs_kp_kv_compat, update_dofs_force_range_compat
from .base import Robot, _find_joint_by_name, _find_link_by_name, _get_dof_idx


# Joints whose PD targets should be re-asserted every tick while the base
# drives, so base acceleration + wheel reactions can't push the telescoping
# stages and lift out of their retracted pose.
_HOLD_JOINT_NAMES = (
    "joint_lift",
    "joint_arm_l0", "joint_arm_l1", "joint_arm_l2", "joint_arm_l3",
    "joint_wrist_yaw", "joint_wrist_pitch", "joint_wrist_roll",
    "joint_gripper_finger_left", "joint_gripper_finger_right",
)


# Stretch SE3 actuated joints (mobile base wheels + head excluded from "arm").
# Order: vertical lift → 4 telescoping arm stages → 3-DOF wrist.
STRETCH_ARM_JOINTS = [
    "joint_lift",
    "joint_arm_l3",
    "joint_arm_l2",
    "joint_arm_l1",
    "joint_arm_l0",
    "joint_wrist_yaw",
    "joint_wrist_pitch",
    "joint_wrist_roll",
]
STRETCH_FINGER_JOINTS = ["joint_gripper_finger_left", "joint_gripper_finger_right"]
STRETCH_WHEEL_JOINTS = ["joint_left_wheel", "joint_right_wheel"]
# link_grasp_center is a URDF-defined frame at the fingertip center — its pose
# already is the TCP, so no additional tcp_offset is needed.
STRETCH_EE_LINK = "link_grasp_center"

# Finger joint angle: 0.35 rad ≈ open (fingertip gap ~0.16 m), 0.0 ≈ closed.
STRETCH_GRIPPER_OPEN = 0.35
STRETCH_GRIPPER_CLOSED = 0.0

# Mobile-base geometry (from stretch_sg3.urdf wheel joint origins).
STRETCH_WHEEL_RADIUS = 0.0508    # wheel z-offset from base_link — wheels rest on ground at z=0
STRETCH_TRACK_WIDTH = 0.3407     # distance between wheel centers (2 * 0.17035)

STRETCH_URDF_REL = "example-robot-data/robots/stretch_description/stretch_sg3.urdf"


def _qpos_start(joint):
    """Entity-local start index of a joint's qpos slice.

    ``joint.q_start`` is a scene-global index — if another dynamic body is
    loaded before the robot, it includes that body's qpos and goes off the
    end of ``entity.get_qpos()``. ``q_idx_local`` / ``qs_idx_local`` are the
    entity-local equivalents.
    """
    local = getattr(joint, "q_idx_local", None)
    if local is not None:
        if isinstance(local, (list, tuple, np.ndarray)):
            return int(local[0]) if len(local) > 0 else None
        return int(local)
    qs_local = getattr(joint, "qs_idx_local", None)
    if qs_local is not None and len(qs_local) > 0:
        return int(qs_local[0])
    return None


class StretchArm:
    """Stretch SE3 'arm': lift + telescoping stages + 3-DOF wrist + 2-finger gripper.

    When the parent ``StretchRobot`` is created with ``mobile_base=True``,
    the two wheel joints are also actuated via ``drive(v, omega)``.
    """

    def __init__(self, entity, origin_pose: Pose, mobile_base: bool = False):
        self.entity = entity
        self.origin_pose = origin_pose
        self.mobile_base = mobile_base
        self.n_arm = len(STRETCH_ARM_JOINTS)
        self.gripper_val = 0.0
        self.planner = None

        self.gripper_open = STRETCH_GRIPPER_OPEN
        self.gripper_closed = STRETCH_GRIPPER_CLOSED

        # Home: lift mid, arm fully retracted, wrist zeroed.
        self.homestate = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # link_grasp_center is already at fingertip center → TCP offset is identity.
        self.tcp_offset = Pose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])

        self.arm_joint_names = list(STRETCH_ARM_JOINTS)

        # Resolved after scene.build()
        self.ee_link = None
        self.base_link = None
        self.arm_joints = []
        self.finger_joints = []
        self.wheel_joints = []  # [left, right]
        self.arm_dof_indices = []
        self.arm_qpos_indices = []
        self.finger_dof_indices = []
        self.finger_qpos_indices = []
        self.wheel_dof_indices = []

        # Mobile-base state
        self._locked_base_pose: Optional[np.ndarray] = None  # qpos[0:7] snapshot
        self._joint_by_name: dict = {}

    def _noop_step(self):
        pass

    def init_joints(self):
        """Resolve joint/link references and configure PD gains. Runs after scene.build()."""
        self.ee_link = (
            _find_link_by_name(self.entity, STRETCH_EE_LINK)
            # merge_fixed_links=True would collapse link_grasp_center into its
            # fixed parent; fall back to that parent so things still work.
            or _find_link_by_name(self.entity, "link_gripper_s3_body")
            or (self.entity.links[-1] if self.entity.links else None)
        )
        self.base_link = _find_link_by_name(self.entity, "base_link")

        self.arm_joints = [_find_joint_by_name(self.entity, n) for n in STRETCH_ARM_JOINTS]
        self.finger_joints = [_find_joint_by_name(self.entity, n) for n in STRETCH_FINGER_JOINTS]
        self.wheel_joints = [_find_joint_by_name(self.entity, n) for n in STRETCH_WHEEL_JOINTS]

        self._joint_by_name = {j.name: j for j in self.entity.joints}

        self.arm_dof_indices = [_get_dof_idx(j) for j in self.arm_joints if j is not None]
        self.arm_qpos_indices = [_qpos_start(j) for j in self.arm_joints if j is not None]
        self.finger_dof_indices = [_get_dof_idx(j) for j in self.finger_joints if j is not None]
        self.finger_qpos_indices = [_qpos_start(j) for j in self.finger_joints if j is not None]
        self.wheel_dof_indices = [_get_dof_idx(j) for j in self.wheel_joints if j is not None]

        try:
            n_dofs = self.entity.n_dofs if hasattr(self.entity, "n_dofs") else len(to_numpy(self.entity.get_dofs_kp()))
            kp = np.zeros(n_dofs, dtype=np.float64)
            kv = np.full(n_dofs, 50.0, dtype=np.float64)
            # Lift needs high stiffness — holds the arm + payload against gravity.
            for j in self.arm_joints:
                if j is None:
                    continue
                idx = _get_dof_idx(j)
                if idx is None or idx >= n_dofs:
                    continue
                if j.name == "joint_lift":
                    kp[idx] = 8000.0
                    kv[idx] = 500.0
                else:
                    kp[idx] = 3000.0
                    kv[idx] = 300.0
            # Fingers need very strong PD to squeeze objects (see test_stretch_grasp.py).
            for j in self.finger_joints:
                if j is None:
                    continue
                idx = _get_dof_idx(j)
                if idx is None or idx >= n_dofs:
                    continue
                kp[idx] = 30000.0
                kv[idx] = 500.0
            # Wheels use velocity control — kp=0, kv provides the "motor damping"
            # that makes the PD velocity loop converge to the target rate.
            for idx in self.wheel_dof_indices:
                if idx is None or idx >= n_dofs:
                    continue
                kp[idx] = 0.0
                kv[idx] = 200.0
            set_dofs_kp_kv_compat(self.entity, kp=kp, kv=kv)
        except Exception as e:
            print(f"[StretchArm] Warning: could not set PD gains: {e}")

    def init_planner(self, *args, **kwargs):
        """Stretch has a telescoping arm that mplib/cuRobo don't model well —
        no planner by default. Consumers that need motion planning should
        use Genesis IK + simple straight-line trajectories."""
        self.planner = None

    # ---- joint state readback -----------------------------------------

    def _qpos_value(self, qpos_start, fallback_idx=None):
        """Read a scalar qpos value at a joint's q_start. Falls back to
        DOF index when q_start isn't populated by the installed Genesis build."""
        qpos = to_numpy(self.entity.get_qpos())
        if qpos_start is not None and 0 <= qpos_start < len(qpos):
            return float(qpos[qpos_start])
        if fallback_idx is not None and 0 <= fallback_idx < len(qpos):
            return float(qpos[fallback_idx])
        return 0.0

    def get_joint_state(self) -> list:
        state = []
        for j, qi, di in zip(self.arm_joints, self.arm_qpos_indices, self.arm_dof_indices):
            if j is None:
                continue
            state.append(self._qpos_value(qi, di))
        state.append(self.gripper_val)
        return state

    def get_arm_qpos(self) -> np.ndarray:
        state = []
        for j, qi, di in zip(self.arm_joints, self.arm_qpos_indices, self.arm_dof_indices):
            if j is None:
                continue
            state.append(self._qpos_value(qi, di))
        return np.array(state, dtype=np.float64)

    # ---- control ------------------------------------------------------

    def set_arm_joints(self, position: np.ndarray):
        position = np.asarray(position, dtype=np.float64).ravel()
        n = min(len(position), len(self.arm_dof_indices))
        if n == 0:
            return
        self.entity.control_dofs_position(
            position[:n], dofs_idx_local=self.arm_dof_indices[:n]
        )

    def teleport_arm_joints(self, position: np.ndarray):
        """Hard-set arm qpos AND the PD target (so the arm stays put)."""
        position = np.asarray(position, dtype=np.float64).ravel()
        qpos = to_numpy(self.entity.get_qpos()).copy()
        for i, (qi, di) in enumerate(zip(self.arm_qpos_indices, self.arm_dof_indices)):
            if i >= len(position):
                break
            slot = qi if (qi is not None and qi < len(qpos)) else di
            if slot is None or slot >= len(qpos):
                continue
            qpos[slot] = float(position[i])
        self.entity.set_qpos(qpos)
        self.set_arm_joints(position)

    def set_gripper(self, val: float):
        """1=open, 0=closed. Both finger joints mirror the same angle."""
        val = float(np.clip(val, 0.0, 1.0))
        self.gripper_val = val
        finger_pos = self.gripper_closed + val * (self.gripper_open - self.gripper_closed)

        if not self.finger_dof_indices:
            return
        self.entity.control_dofs_position(
            np.full(len(self.finger_dof_indices), finger_pos, dtype=np.float64),
            dofs_idx_local=self.finger_dof_indices,
        )

    def get_ee_pose(self) -> list:
        if self.ee_link is None:
            return list(self.origin_pose.p) + list(self.origin_pose.q)
        pos = to_numpy(self.ee_link.get_pos()).ravel()[:3]
        quat = to_numpy(self.ee_link.get_quat()).ravel()[:4]
        link_pose = Pose(pos, quat)
        tcp_pose = link_pose * self.tcp_offset
        return tcp_pose.p.tolist() + tcp_pose.q.tolist()

    def move_to_homestate(self):
        """Teleport arm to home + open gripper."""
        self.teleport_arm_joints(np.asarray(self.homestate, dtype=np.float64))
        self.set_gripper(1.0)
        # Stop wheels in case mobile_base is active.
        if self.mobile_base:
            self.stop_base()

    # ---- mobile base --------------------------------------------------

    def drive(self, v_forward: float, omega_yaw: float):
        """Differential-drive twist command.

        ``v_forward`` (m/s) is linear speed along the base +X axis (Stretch's
        forward). ``omega_yaw`` (rad/s) is rotational speed about +Z: positive
        turns left (counter-clockwise when viewed from above).

        Silently no-ops if ``mobile_base=False`` because the base is welded
        and wheel torques would just deflect against the world constraint.
        """
        if not self.mobile_base:
            return
        if len(self.wheel_dof_indices) != 2:
            return

        half = STRETCH_TRACK_WIDTH / 2.0
        v_left_contact = v_forward - omega_yaw * half
        v_right_contact = v_forward + omega_yaw * half
        omega_left = v_left_contact / STRETCH_WHEEL_RADIUS
        omega_right = v_right_contact / STRETCH_WHEEL_RADIUS

        # STRETCH_WHEEL_JOINTS = [left, right] → wheel_dof_indices = [left_dof, right_dof]
        self.entity.control_dofs_velocity(
            np.array([omega_left, omega_right], dtype=np.float64),
            dofs_idx_local=self.wheel_dof_indices,
        )

    def stop_base(self):
        """Zero wheel velocity targets. No-op if base is fixed."""
        if not self.mobile_base or len(self.wheel_dof_indices) != 2:
            return
        self.entity.control_dofs_velocity(
            np.zeros(2, dtype=np.float64), dofs_idx_local=self.wheel_dof_indices
        )

    def get_base_pose(self) -> tuple:
        """Return (x, y, yaw) of the base_link in world coordinates.

        yaw is the rotation about +Z extracted from the base_link quaternion.
        """
        if self.base_link is None:
            return (float(self.origin_pose.p[0]), float(self.origin_pose.p[1]), 0.0)
        pos = to_numpy(self.base_link.get_pos()).ravel()
        quat = to_numpy(self.base_link.get_quat()).ravel()  # (w, x, y, z)
        w, x, y, z = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        return (float(pos[0]), float(pos[1]), yaw)

    # ---- arm-hold during base motion ---------------------------------

    def snapshot_hold_state(self):
        """Capture current PD targets for the arm+wrist+fingers so the base
        can drive without the telescoping stages or lift drifting.

        Base acceleration + wheel reaction can push those joints out of
        their current pose even with PD active; re-issuing the targets
        every tick pins the arm "inside" the base footprint during drive.
        """
        qpos = to_numpy(self.entity.get_qpos())
        vals, dof_idx = [], []
        for name in _HOLD_JOINT_NAMES:
            j = self._joint_by_name.get(name)
            if j is None:
                continue
            q = _qpos_start(j)
            d = _get_dof_idx(j)
            if q is None or q >= len(qpos) or d is None:
                continue
            vals.append(float(qpos[q]))
            dof_idx.append(d)
        return (np.array(vals, dtype=np.float64), dof_idx)

    def apply_hold(self, hold_state):
        """Re-issue the PD targets snapshotted by ``snapshot_hold_state``."""
        if hold_state is None:
            return
        vals, dof_idx = hold_state
        if not dof_idx:
            return
        self.entity.control_dofs_position(vals, dofs_idx_local=dof_idx)

    # ---- base lock / snap ---------------------------------------------

    def lock_base(self):
        """Pin the base qpos so gripper-squeeze reaction can't roll it.

        After this call, the parent robot's ``on_post_step`` re-snaps the
        base qpos[0:7] and zeros its 6 velocity DOFs every physics tick.
        Only meaningful when ``mobile_base=True``.
        """
        if not self.mobile_base:
            return
        self._locked_base_pose = to_numpy(self.entity.get_qpos())[0:7].copy()

    def unlock_base(self):
        """Release the base-lock. Subsequent ticks leave the base free."""
        self._locked_base_pose = None

    def is_base_locked(self) -> bool:
        return self._locked_base_pose is not None

    def snap_base_to_locked(self):
        """Internal: re-snap qpos[0:7] + zero base velocity. Called per-tick."""
        if self._locked_base_pose is None:
            return
        qpos = to_numpy(self.entity.get_qpos()).copy()
        qpos[0:7] = self._locked_base_pose
        self.entity.set_qpos(qpos)
        try:
            self.entity.set_dofs_velocity(
                np.zeros(6, dtype=np.float64), dofs_idx_local=[0, 1, 2, 3, 4, 5]
            )
        except Exception:
            pass

    # ---- lift force-range ---------------------------------------------

    def widen_lift_force_range(self, limit: float = 500.0) -> bool:
        """Widen ``joint_lift``'s force range from the URDF cap of ±100 N.

        Genesis uses the URDF ``effort`` attribute as the PD output clamp.
        100 N is consumed by gravity on the extended arm + stiction in the
        telescoping stages, leaving nothing to accelerate the mast — the
        lift won't actually raise under load without this call.
        """
        j = self._joint_by_name.get("joint_lift")
        if j is None:
            return False
        d_lift = _get_dof_idx(j)
        if d_lift is None:
            return False
        ok = update_dofs_force_range_compat(
            self.entity,
            [d_lift],
            lower_value=-float(limit),
            upper_value=float(limit),
        )
        if not ok:
            print("[StretchArm] widen_lift_force_range failed")
        return bool(ok)

    # ---- base-level motion --------------------------------------------

    def teleport_base(self, x: float, y: float, yaw: float = 0.0):
        """Snap the free-base root joint to (x, y, yaw) and zero its velocity.

        Base qpos layout: qpos[0:3] = (x, y, z), qpos[3:7] = (qw, qx, qy, qz).
        The free-base DOFs 0..5 (linear + angular velocity) must be zeroed
        or the body keeps drifting from its pre-teleport velocity. z is
        kept at its current value so the wheels stay on the ground.
        """
        if not self.mobile_base:
            return
        qpos = to_numpy(self.entity.get_qpos()).copy()
        qpos[0] = float(x)
        qpos[1] = float(y)
        half = 0.5 * float(yaw)
        qpos[3] = float(np.cos(half))
        qpos[4] = 0.0
        qpos[5] = 0.0
        qpos[6] = float(np.sin(half))
        self.entity.set_qpos(qpos)
        try:
            self.entity.set_dofs_velocity(
                np.zeros(6, dtype=np.float64), dofs_idx_local=[0, 1, 2, 3, 4, 5]
            )
        except Exception:
            try:
                self.entity.zero_all_dofs_velocity()
            except Exception:
                pass
        self.stop_base()

    def move_base_to(
        self,
        x: float,
        y: float,
        yaw: float = 0.0,
        step_fn: Optional[Callable[[], None]] = None,
        hold_arm: bool = True,
        physics_drive_v: float = 0.18,
        settle_steps: int = 200,
        dt: float = 0.002,
    ):
        """Drive the base to ``(x, y, yaw)`` via open-loop physics drive +
        precise teleport snap.

        The Stretch base's mast-mass asymmetry plus the sliding-sphere
        caster produce a systematic Y-drift of ~10–16 cm per meter that
        closed-loop diff-drive can't fully remove in the ~2 cm precision
        window the arm reach demands. We do a visible physics drive over
        most of the distance so wheels spin on camera, then teleport to
        the exact waypoint.

        Args:
            x, y, yaw: target base pose in world frame.
            step_fn: callable invoked once per physics tick. Tasks pass
                ``self.step_sim`` so video recording / attached-object
                sync / avatar update still happen. Defaults to a no-op
                stepper that just calls ``scene.step`` is not usable here
                because the arm has no scene reference — callers should
                always pass ``step_fn``.
            hold_arm: snapshot the arm's current PD targets and re-assert
                them every tick so the telescoping stages + lift stay put
                under base acceleration.
        """
        if not self.mobile_base:
            return
        if step_fn is None:
            step_fn = self._noop_step

        hold_state = self.snapshot_hold_state() if hold_arm else None

        x0, y0, yaw0 = self.get_base_pose()
        dist = float(np.hypot(x - x0, y - y0))

        # Visible forward drive over most of the X-distance. Slightly
        # undershoot so we don't bump past the target before the teleport.
        # Hold the yaw the base already has — callers that want the drive
        # aimed at the target should pivot first.
        drive_dist = max(dist - 0.1, 0.0)
        drive_time = drive_dist / physics_drive_v if physics_drive_v > 0 else 0.0
        if drive_time > 0.5:
            self._physics_drive_forward(
                physics_drive_v, drive_time, step_fn, hold_state,
                dt=dt, target_yaw=yaw0,
            )

        self.teleport_base(x, y, yaw)
        self.apply_hold(hold_state)

        for _ in range(settle_steps):
            self.apply_hold(hold_state)
            step_fn()

    def _physics_drive_forward(
        self,
        v_forward: float,
        duration_s: float,
        step_fn: Callable[[], None],
        hold_state,
        ramp_n: int = 150,
        k_yaw: float = 1.5,
        dt: float = 0.002,
        target_yaw: float = 0.0,
    ):
        """Open-loop forward drive with ramp in/out + weak yaw correction.

        Yaw feedback holds the robot at ``target_yaw`` but precision is
        owned by the subsequent teleport — this exists for visible wheel
        rolling in the video, not for pose accuracy.
        """
        def _wrap(a):
            return float(np.arctan2(np.sin(a), np.cos(a)))

        max_n = int(duration_s / dt)
        hold_n = max(max_n - 2 * ramp_n, 0)
        for k in range(ramp_n):
            a = (k + 1) / ramp_n
            yaw_err = _wrap(self.get_base_pose()[2] - target_yaw)
            self.drive(a * v_forward, -k_yaw * yaw_err)
            self.apply_hold(hold_state)
            step_fn()
        for _ in range(hold_n):
            yaw_err = _wrap(self.get_base_pose()[2] - target_yaw)
            self.drive(v_forward, -k_yaw * yaw_err)
            self.apply_hold(hold_state)
            step_fn()
        for k in range(ramp_n):
            a = 1.0 - (k + 1) / ramp_n
            yaw_err = _wrap(self.get_base_pose()[2] - target_yaw)
            self.drive(a * v_forward, -k_yaw * yaw_err)
            self.apply_hold(hold_state)
            step_fn()
        self.stop_base()

    def pivot_to(
        self,
        target_yaw: float,
        step_fn: Optional[Callable[[], None]] = None,
        hold_arm: bool = True,
        tol: float = 0.008,
        k: float = 2.5,
        omega_max: float = 0.4,
        max_duration_s: float = 6.0,
        dt: float = 0.002,
    ):
        """Pivot in place to ``target_yaw`` (world frame)."""
        if not self.mobile_base:
            return
        if step_fn is None:
            step_fn = self._noop_step
        hold_state = self.snapshot_hold_state() if hold_arm else None

        def _wrap(a):
            return float(np.arctan2(np.sin(a), np.cos(a)))

        max_n = int(max_duration_s / dt)
        for _ in range(max_n):
            yaw = self.get_base_pose()[2]
            err = _wrap(target_yaw - yaw)
            if abs(err) < tol:
                break
            self.drive(0.0, float(np.clip(k * err, -omega_max, omega_max)))
            self.apply_hold(hold_state)
            step_fn()
        self.stop_base()
        for _ in range(100):
            self.apply_hold(hold_state)
            step_fn()


class StretchRobot(Robot):
    """Stretch SE3 mobile manipulator (single arm, differential-drive base)."""

    def __init__(
        self,
        pos=None,
        quat=None,
        merge_fixed_links: bool = False,
        mobile_base: bool = False,
    ):
        """
        Args:
            pos: base position [x, y, z]. Default places the mobile base on the
                 floor (z=0) roughly 0.9 m in +Y from the world origin so the
                 telescoping arm (which extends along -Y from the base) reaches
                 over the typical manual-grasp table scene.
            quat: base orientation [w, x, y, z]. Default is identity (arm -Y).
            merge_fixed_links: if True, Genesis merges fixed-joint children into
                 their parents. Default False so link_grasp_center and finger
                 collision geometry remain queryable.
            mobile_base: if True, the URDF is loaded with ``fixed=False`` so the
                 base can roll. Differential-drive control is then available
                 through ``drive(v_forward, omega_yaw)``. If False (default),
                 the base is welded to the world and drive commands are no-ops.
        """
        if pos is None:
            pos = [0.0, 0.9, 0.0]
        if quat is None:
            quat = [1.0, 0.0, 0.0, 0.0]
        self.base_pose = Pose(pos, quat)
        self.merge_fixed_links = merge_fixed_links
        self.mobile_base = mobile_base
        self.left_arm = None  # single-arm robot; left_arm = right_arm = the one arm
        self.right_arm = None

    @property
    def urdf_path(self) -> str:
        return str(ASSETS_PATH / STRETCH_URDF_REL)

    def add_to_scene(self, scene: gs.Scene, **kwargs):
        entity = scene.add_entity(
            gs.morphs.URDF(
                file=self.urdf_path,
                pos=tuple(self.base_pose.p),
                quat=tuple(self.base_pose.q),
                fixed=not self.mobile_base,
                visualization=True,
                collision=True,
                merge_fixed_links=self.merge_fixed_links,
            ),
            material=gs.materials.Rigid(friction=1.5),
        )
        self.left_arm = StretchArm(entity, self.base_pose, mobile_base=self.mobile_base)
        self.right_arm = self.left_arm

    def init_joints(self, scene: gs.Scene):
        self.left_arm.init_joints()

    def move_to_homestate(self):
        self.left_arm.move_to_homestate()

    def get_arm(self, arm_tag: str) -> StretchArm:
        return self.left_arm

    def get_left_joint_state(self) -> list:
        return self.left_arm.get_joint_state()

    def get_right_joint_state(self) -> list:
        return self.left_arm.get_joint_state()

    def set_arm_joints(self, position, arm_tag: str):
        self.left_arm.set_arm_joints(position)

    def teleport_arm_joints(self, position, arm_tag: str):
        self.left_arm.teleport_arm_joints(position)

    def set_gripper(self, val: float, arm_tag: str):
        self.left_arm.set_gripper(val)

    def get_ee_pose(self, arm_tag: str) -> list:
        return self.left_arm.get_ee_pose()

    def get_left_gripper_val(self) -> float:
        return self.left_arm.gripper_val

    def get_right_gripper_val(self) -> float:
        return self.left_arm.gripper_val

    def open_gripper(self, arm_tag: str):
        self.set_gripper(1.0, arm_tag)

    def close_gripper(self, arm_tag: str):
        self.set_gripper(0.0, arm_tag)

    def set_origin_endpose(self):
        self.left_original_pose = self.left_arm.get_ee_pose()
        self.right_original_pose = self.left_arm.get_ee_pose()

    # ---- mobile base passthroughs -------------------------------------

    def has_mobile_base(self) -> bool:
        return bool(self.mobile_base)

    def drive(self, v_forward: float, omega_yaw: float):
        """Differential-drive twist. See ``StretchArm.drive`` for details."""
        self.left_arm.drive(v_forward, omega_yaw)

    def stop_base(self):
        self.left_arm.stop_base()

    def get_base_pose(self) -> tuple:
        return self.left_arm.get_base_pose()

    def move_base_to(
        self,
        x: float,
        y: float,
        yaw: float = 0.0,
        step_fn: Optional[Callable[[], None]] = None,
        hold_arm: bool = True,
        **kwargs,
    ):
        """Drive the base to ``(x, y, yaw)``. See ``StretchArm.move_base_to``."""
        self.left_arm.move_base_to(
            x, y, yaw, step_fn=step_fn, hold_arm=hold_arm, **kwargs
        )

    def pivot_to(
        self,
        target_yaw: float,
        step_fn: Optional[Callable[[], None]] = None,
        hold_arm: bool = True,
        **kwargs,
    ):
        """Pivot in place. See ``StretchArm.pivot_to``."""
        self.left_arm.pivot_to(
            target_yaw, step_fn=step_fn, hold_arm=hold_arm, **kwargs
        )

    def teleport_base(self, x: float, y: float, yaw: float = 0.0):
        self.left_arm.teleport_base(x, y, yaw)

    def lock_base(self):
        self.left_arm.lock_base()

    def unlock_base(self):
        self.left_arm.unlock_base()

    def is_base_locked(self) -> bool:
        return self.left_arm.is_base_locked()

    def widen_lift_force_range(self, limit: float = 500.0) -> bool:
        """Widen the lift joint force range. Must be called after scene.build()."""
        return self.left_arm.widen_lift_force_range(limit=limit)

    def snapshot_hold_state(self):
        return self.left_arm.snapshot_hold_state()

    def apply_hold(self, hold_state):
        self.left_arm.apply_hold(hold_state)

    def on_post_step(self, scene):
        """Re-snap the base each tick if it's locked. Called from the task's
        ``step_sim`` after ``scene.step``."""
        self.left_arm.snap_base_to_locked()
