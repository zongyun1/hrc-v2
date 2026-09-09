"""Franka Emika Panda robot loaded via Genesis MJCF API."""

import numpy as np
import genesis as gs

from ..utils import Pose, to_numpy, ASSETS_PATH
from ..genesis_compat import set_dofs_kp_kv_compat
from .base import Robot, Arm, _find_joint_by_name, _find_link_by_name, _get_dof_idx
from ..planning import MplibPlanner, GenesisIKPlanner


# Joint name mapping: MJCF uses short names, URDF uses panda_ prefix.
# The planner uses the URDF, so we need to map between them.
MJCF_ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
MJCF_FINGER_JOINTS = ["finger_joint1", "finger_joint2"]
MJCF_EE_LINK = "hand"

URDF_ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
URDF_FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
URDF_EE_LINK = "panda_hand"


class FrankaArm:
    """Single Franka Panda arm loaded from MJCF."""

    def __init__(self, entity, origin_pose: Pose):
        self.entity = entity
        self.origin_pose = origin_pose
        self.n_arm = 7
        self.gripper_val = 0.0
        self.planner = None

        # Franka gripper range: 0.0 (closed) to 0.04 (open) per finger
        self.gripper_open = 0.04
        self.gripper_closed = 0.0

        # Home state from MJCF keyframe
        self.homestate = [0, 0.19634954, 0.0, -2.61799388, 0.0, 2.94159265, 0.78539816]

        # TCP offset: from hand link to fingertip center
        # Franka hand to fingertip ~0.1034m along Z
        self.tcp_offset = Pose(
            [0.0, 0.0, 0.1034],
            [1.0, 0.0, 0.0, 0.0],
        )

        # Resolved after scene.build()
        self.ee_link = None
        self.arm_joints = []
        self.finger_joints = []
        self._finger_dof_indices = []
        self._cached_target = None  # cached PD target to avoid overwrite

    def init_joints(self):
        """Resolve joint/link references after scene.build()."""
        self.ee_link = _find_link_by_name(self.entity, MJCF_EE_LINK)
        self.arm_joints = [_find_joint_by_name(self.entity, n) for n in MJCF_ARM_JOINTS]
        self.finger_joints = [_find_joint_by_name(self.entity, n) for n in MJCF_FINGER_JOINTS]

        # Boost PD gains for accurate trajectory tracking
        try:
            kp = to_numpy(self.entity.get_dofs_kp()).copy()
            kv = to_numpy(self.entity.get_dofs_kv()).copy()
            # Arm joints: high stiffness for position tracking
            for j in self.arm_joints:
                if j is not None:
                    idx = _get_dof_idx(j)
                    if idx is not None:
                        kp[idx] = max(kp[idx], 4000.0)
                        kv[idx] = max(kv[idx], 400.0)
            # Finger joints: extra strong for grip
            for j in self.finger_joints:
                if j is not None:
                    idx = _get_dof_idx(j)
                    if idx is not None:
                        self._finger_dof_indices.append(idx)
                        kp[idx] = max(kp[idx], 10000.0)
                        kv[idx] = max(kv[idx], 500.0)
            set_dofs_kp_kv_compat(self.entity, kp=kp, kv=kv)
        except Exception:
            pass  # Non-critical

    def init_planner(self, urdf_path: str, srdf_path: str, scene, planner_type: str = "mplib"):
        """Create motion planner. `planner_type` = "mplib" (default) or "genesis_ik"."""
        joint_index = {}
        for j in self.entity.joints:
            idx = _get_dof_idx(j)
            if idx is not None:
                joint_index[j.name] = idx

        if planner_type == "genesis_ik":
            self.planner = GenesisIKPlanner(
                entity=self.entity,
                ee_link_name=MJCF_EE_LINK,
                n_arm=self.n_arm,
                arm_joint_names=MJCF_ARM_JOINTS,
                arm_joint_index=joint_index,
            )
            return

        # Default: mplib (uses URDF joint names internally; MJCF names map via joint_index).
        self.planner = MplibPlanner(
            urdf_path=urdf_path,
            srdf_path=srdf_path,
            move_group=URDF_EE_LINK,
            base_pose=self.origin_pose.to_pose7(),
            entity=self.entity,
            n_arm=self.n_arm,
            arm_joint_names=MJCF_ARM_JOINTS,
            arm_joint_index=joint_index,
            entity_ee_link_name=MJCF_EE_LINK,
        )

    def get_joint_state(self) -> list:
        """Get arm joint positions + gripper value."""
        qpos = to_numpy(self.entity.get_qpos())
        state = []
        for j in self.arm_joints:
            if j is not None:
                dof_idx = _get_dof_idx(j)
                if dof_idx is not None and 0 <= dof_idx < len(qpos):
                    state.append(float(qpos[dof_idx]))
        state.append(self.gripper_val)
        return state

    def get_arm_qpos(self) -> np.ndarray:
        """Get arm joint positions only (no gripper)."""
        qpos = to_numpy(self.entity.get_qpos())
        state = []
        for j in self.arm_joints:
            if j is not None:
                dof_idx = _get_dof_idx(j)
                if dof_idx is not None and 0 <= dof_idx < len(qpos):
                    state.append(float(qpos[dof_idx]))
        return np.array(state, dtype=np.float64)

    def set_arm_joints(self, position: np.ndarray):
        """Set arm joint PD targets."""
        qpos = self._build_qpos_with_arm(position)
        self._cached_target = qpos.copy()
        self.entity.control_dofs_position(qpos)

    def teleport_arm_joints(self, position: np.ndarray):
        """Directly set arm joint positions and sync PD target."""
        qpos = self._build_qpos_with_arm(position)
        self._cached_target = qpos.copy()
        self.entity.set_qpos(qpos)
        self.entity.control_dofs_position(qpos)

    def _get_target(self) -> np.ndarray:
        """Get cached PD target, initializing from current qpos if needed."""
        if self._cached_target is None:
            self._cached_target = to_numpy(self.entity.get_qpos()).copy()
        return self._cached_target

    def _build_qpos_with_arm(self, position: np.ndarray) -> np.ndarray:
        """Build full qpos array with arm joint values replaced."""
        qpos = self._get_target().copy()
        for i, j in enumerate(self.arm_joints):
            if i >= len(position) or j is None:
                break
            dof_idx = _get_dof_idx(j)
            if dof_idx is not None and 0 <= dof_idx < len(qpos):
                qpos[dof_idx] = float(position[i])
        return qpos

    def set_gripper(self, val: float):
        """Set gripper: 1=open, 0=closed."""
        val = float(np.clip(val, 0.0, 1.0))
        self.gripper_val = val
        finger_pos = self.gripper_closed + val * (self.gripper_open - self.gripper_closed)

        qpos = self._get_target().copy()
        for j in self.finger_joints:
            if j is None:
                continue
            dof_idx = _get_dof_idx(j)
            if dof_idx is not None and 0 <= dof_idx < len(qpos):
                qpos[dof_idx] = float(finger_pos)
        self._cached_target = qpos.copy()
        self.entity.control_dofs_position(qpos)

    def get_ee_pose(self) -> list:
        """Get end-effector (gripper TCP) pose as [x, y, z, qw, qx, qy, qz]."""
        if self.ee_link is None:
            return list(self.origin_pose.p) + list(self.origin_pose.q)
        pos = to_numpy(self.ee_link.get_pos()).ravel()[:3]
        quat = to_numpy(self.ee_link.get_quat()).ravel()[:4]
        # Apply TCP offset to get actual gripper TCP pose (consistent with base Arm class)
        link_pose = Pose(pos, quat)
        tcp_pose = link_pose * self.tcp_offset
        return tcp_pose.p.tolist() + tcp_pose.q.tolist()

    def move_to_homestate(self):
        """Teleport arm to home configuration via set_qpos."""
        qpos = to_numpy(self.entity.get_qpos()).copy()
        for i, j in enumerate(self.arm_joints):
            if j is None or i >= len(self.homestate):
                continue
            dof_idx = _get_dof_idx(j)
            if dof_idx is not None and 0 <= dof_idx < len(qpos):
                qpos[dof_idx] = self.homestate[i]
        # Also open gripper in homestate
        for j in self.finger_joints:
            if j is not None:
                dof_idx = _get_dof_idx(j)
                if dof_idx is not None and 0 <= dof_idx < len(qpos):
                    qpos[dof_idx] = self.gripper_open
        self.entity.set_qpos(qpos)
        # Also set PD target to homestate so it holds position
        self.entity.control_dofs_position(qpos)


class FrankaRobot(Robot):
    """Franka Emika Panda robot loaded from Genesis built-in MJCF."""

    def __init__(
        self,
        pos=None,
        quat=None,
        planner_type: str = "mplib",
        mjcf_file: str = None,
    ):
        """
        Args:
            pos: robot base position [x, y, z]. Default places it behind the table.
            quat: robot base orientation [w, x, y, z].
            planner_type: "mplib" (default) or "genesis_ik".
            mjcf_file: Genesis MJCF asset to load. ``None`` auto-selects: under
                Genesis 1.0.0+ the tendon-coupled gripper in the stock
                ``panda.xml`` does NOT open (``set_gripper(0.04)`` leaves the
                fingers at 0.0), so top-down grasps descend with a closed
                gripper and jam the wrist on the object. The ``panda_no_tendon``
                variant drives both finger joints directly and opens to 0.08 m,
                which is what every grasp pipeline expects. The old ``vico``
                fork keeps the tested tendon ``panda.xml``.
        """
        if pos is None:
            pos = [0.0, -0.65, 0.75]
        if quat is None:
            quat = [0.707, 0.0, 0.0, 0.707]
        if mjcf_file is None:
            from ..genesis_compat import is_latest_genesis
            mjcf_file = (
                "xml/franka_emika_panda/panda_no_tendon.xml"
                if is_latest_genesis()
                else "xml/franka_emika_panda/panda.xml"
            )
        self.base_pose = Pose(pos, quat)
        self.planner_type = planner_type
        self.mjcf_file = mjcf_file
        # Single-arm robot: follows Piper single-arm convention (left_arm=None,
        # right_arm is the primary). Keeps BaseTask.take_action dual-arm branching sane.
        self.left_arm = None
        self.right_arm = None

    def add_to_scene(self, scene: gs.Scene, **kwargs):
        """Add Franka entity to scene using MJCF."""
        entity = scene.add_entity(
            gs.morphs.MJCF(
                file=self.mjcf_file,
                pos=tuple(self.base_pose.p),
                quat=tuple(self.base_pose.q),
            ),
        )
        self.right_arm = FrankaArm(entity, self.base_pose)

    def init_joints(self, scene: gs.Scene):
        """Initialize joints and planner after scene.build()."""
        self.right_arm.init_joints()

        # Use URDF from assets for mplib planner
        franka_dir = ASSETS_PATH / "embodiments" / "franka-panda"
        self.planner_urdf_path = str(franka_dir / "panda.urdf")
        self.planner_srdf_path = str(franka_dir / "panda.srdf")
        self.right_arm.init_planner(
            self.planner_urdf_path, self.planner_srdf_path, scene,
            planner_type=self.planner_type,
        )

    def move_to_homestate(self):
        self.right_arm.move_to_homestate()

    def get_arm(self, arm_tag: str) -> FrankaArm:
        return self.right_arm

    def get_left_joint_state(self) -> list:
        return self.right_arm.get_joint_state()

    def get_right_joint_state(self) -> list:
        return self.right_arm.get_joint_state()

    def set_arm_joints(self, position, arm_tag: str):
        self.right_arm.set_arm_joints(position)

    def teleport_arm_joints(self, position, arm_tag: str):
        self.right_arm.teleport_arm_joints(position)

    def set_gripper(self, val: float, arm_tag: str):
        self.right_arm.set_gripper(val)

    def get_ee_pose(self, arm_tag: str) -> list:
        return self.right_arm.get_ee_pose()

    def get_left_gripper_val(self) -> float:
        return self.right_arm.gripper_val

    def get_right_gripper_val(self) -> float:
        return self.right_arm.gripper_val

    def open_gripper(self, arm_tag: str):
        self.set_gripper(1.0, arm_tag)

    def close_gripper(self, arm_tag: str):
        self.set_gripper(0.0, arm_tag)

    def set_origin_endpose(self):
        self.left_original_pose = self.right_arm.get_ee_pose()
        self.right_original_pose = self.right_arm.get_ee_pose()
