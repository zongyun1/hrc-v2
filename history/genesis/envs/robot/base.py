"""Robot abstraction: base interface + Arm implementation."""

import os
import numpy as np
import genesis as gs
from abc import ABC, abstractmethod
from copy import deepcopy

from ..utils import Pose, to_numpy, ASSETS_PATH
from ..genesis_compat import set_dofs_kp_kv_compat
from ..planning import MplibPlanner, GenesisIKPlanner, CUROBO_AVAILABLE

if CUROBO_AVAILABLE:
    from ..planning import CuroboPlanner
else:
    CuroboPlanner = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_joint_by_name(entity, name):
    for joint in entity.joints:
        if joint.name == name:
            return joint
    return None


def _find_link_by_name(entity, name):
    for link in entity.links:
        if link.name == name:
            return link
    return None


def _get_dof_idx(joint):
    """Get DOF index for a joint."""
    idx = getattr(joint, "dof_idx_local", None)
    if idx is None and hasattr(joint, "get_dof_indices_local"):
        indices = joint.get_dof_indices_local()
        idx = indices[0] if indices else None
    return int(idx) if idx is not None else None


# ---------------------------------------------------------------------------
# Arm: one arm with its entity, joints, gripper, and planner
# ---------------------------------------------------------------------------

class Arm:
    """Represents one robot arm (entity + joints + gripper + planner)."""

    def __init__(
        self,
        entity,
        config: dict,
        arm_index: int,
        origin_pose: Pose,
    ):
        self.entity = entity
        self.origin_pose = origin_pose
        self.arm_index = arm_index

        # Parse config for this arm index
        self.move_group = config["move_group"][arm_index]
        self.ee_joint_name = config["ee_joints"][arm_index]
        self.arm_joint_names = config["arm_joints_name"][arm_index]
        self.gripper_config = config["gripper_name"][arm_index]
        self.gripper_bias = config["gripper_bias"]
        self.gripper_scale = config["gripper_scale"]
        self.homestate = config.get("homestate", [[0] * len(self.arm_joint_names)])[arm_index]

        self.n_arm = len(self.arm_joint_names)
        self.gripper_val = 0.0
        self.planner = None

        _pl = config.get("planner", "mplib")
        if isinstance(_pl, str):
            self.planner_type = _pl.split("#")[0].strip().strip('"').strip("'").lower()
        else:
            self.planner_type = "mplib"

        # PD gains from config
        self.joint_stiffness = config.get("joint_stiffness", 1000)
        self.joint_damping = config.get("joint_damping", 200)
        self.gripper_stiffness = config.get("gripper_stiffness", 1000)
        self.gripper_damping = config.get("gripper_damping", 200)

        # TCP offset (from ee_link to fingertip center)
        tcp_cfg = config.get("tcp_offset", {})
        self.tcp_offset = Pose(
            tcp_cfg.get("position", [0, 0, 0]),
            tcp_cfg.get("quaternion", [1, 0, 0, 0]),
        )

        # These are resolved after scene.build()
        self.ee_joint = None
        self.ee_link = None
        self.arm_joints = []
        self.gripper_joints = []  # list of (joint, multiplier, offset)
        self._cached_target = None  # cached PD target to avoid overwrite

    def init_joints(self):
        """Resolve joint/link references after scene.build()."""
        self.ee_joint = _find_joint_by_name(self.entity, self.ee_joint_name)
        self.ee_link = (
            _find_link_by_name(self.entity, self.move_group)
            or _find_link_by_name(self.entity, self.ee_joint_name)
            or (self.entity.links[-1] if self.entity.links else None)
        )
        self.arm_joints = [_find_joint_by_name(self.entity, n) for n in self.arm_joint_names]

        # Parse gripper joints
        gc = self.gripper_config
        base_joint = _find_joint_by_name(self.entity, gc["base"])
        self.gripper_joints = [(base_joint, 1.0, 0.0)]
        for mimic in gc.get("mimic", []):
            j = _find_joint_by_name(self.entity, mimic[0])
            if j:
                self.gripper_joints.append((j, mimic[1], mimic[2]))

        # Apply PD gains to all DOFs
        n_dofs = len(self.entity.joints)
        kp = np.zeros(n_dofs, dtype=np.float64)
        kv = np.zeros(n_dofs, dtype=np.float64)
        for j in self.entity.joints:
            idx = _get_dof_idx(j)
            if idx is None or idx >= n_dofs:
                continue
            if j.name in self.arm_joint_names:
                kp[idx] = self.joint_stiffness
                kv[idx] = self.joint_damping
            else:
                kp[idx] = self.gripper_stiffness
                kv[idx] = self.gripper_damping
        try:
            set_dofs_kp_kv_compat(self.entity, kp=kp, kv=kv)
        except Exception as e:
            print(f"[Arm] Warning: could not set PD gains: {e}")

    def init_planner(self, urdf_path: str, srdf_path: str, scene, package_keyword: str = ""):
        """Create motion planner for this arm (``planner`` in robot ``config.yml``: ``mplib`` or ``curobo``)."""
        # Build joint name -> dof index mapping
        joint_index = {}
        for j in self.entity.joints:
            idx = _get_dof_idx(j)
            if idx is not None:
                joint_index[j.name] = idx

        if self.planner_type == "genesis_ik":
            self.planner = GenesisIKPlanner(
                entity=self.entity,
                ee_link_name=self.move_group,
                n_arm=self.n_arm,
                arm_joint_names=self.arm_joint_names,
                arm_joint_index=joint_index,
            )
            return

        use_curobo = (
            self.planner_type == "curobo"
            and CUROBO_AVAILABLE
            and CuroboPlanner is not None
        )
        if use_curobo:
            try:
                self.planner = CuroboPlanner(
                    urdf_path=urdf_path,
                    ee_link=self.move_group,
                    base_pose=self.origin_pose.to_pose7(),
                    joint_names=list(self.arm_joint_names),
                    entity=self.entity,
                    arm_joint_index=joint_index,
                )
                return
            except Exception as e:
                print(f"[Arm] CuroboPlanner failed ({e}); falling back to MplibPlanner.")

        if self.planner_type == "curobo" and not use_curobo:
            print("[Arm] planner=curobo requested but cuRobo is unavailable; using MplibPlanner.")

        self.planner = MplibPlanner(
            urdf_path=urdf_path,
            srdf_path=srdf_path,
            move_group=self.move_group,
            base_pose=self.origin_pose.to_pose7(),
            entity=self.entity,
            n_arm=self.n_arm,
            arm_joint_names=self.arm_joint_names,
            arm_joint_index=joint_index,
            package_keyword=package_keyword,
        )

    def get_joint_state(self) -> list:
        """Get arm joint positions + gripper value."""
        qpos = to_numpy(self.entity.get_qpos())
        state = []
        for name in self.arm_joint_names:
            joint = _find_joint_by_name(self.entity, name)
            if joint is not None:
                dof_idx = _get_dof_idx(joint)
                if dof_idx is not None and 0 <= dof_idx < len(qpos):
                    state.append(float(qpos[dof_idx]))
        state.append(self.gripper_val)
        return state

    def get_arm_qpos(self) -> np.ndarray:
        """Get arm joint positions only (no gripper)."""
        qpos = to_numpy(self.entity.get_qpos())
        state = []
        for name in self.arm_joint_names:
            joint = _find_joint_by_name(self.entity, name)
            if joint is not None:
                dof_idx = _get_dof_idx(joint)
                if dof_idx is not None and 0 <= dof_idx < len(qpos):
                    state.append(float(qpos[dof_idx]))
        return np.array(state, dtype=np.float64)

    def set_arm_joints(self, position: np.ndarray):
        """Set arm joint PD targets (for normal trajectory tracking)."""
        qpos = self._build_qpos_with_arm(position)
        self._cached_target = qpos.copy()
        self.entity.control_dofs_position(qpos)

    def teleport_arm_joints(self, position: np.ndarray):
        """Directly set arm joint positions (for DEBUG_TELEPORT)."""
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
        joint_by_name = {j.name: j for j in self.entity.joints}

        for i, name in enumerate(self.arm_joint_names):
            if i >= len(position):
                break
            joint = joint_by_name.get(name)
            if joint is None:
                continue
            dof_idx = _get_dof_idx(joint)
            if dof_idx is not None and 0 <= dof_idx < len(qpos):
                qpos[dof_idx] = float(position[i])
        return qpos

    def set_gripper(self, val: float):
        """Set gripper: 1=open, 0=closed."""
        val = float(np.clip(val, 0.0, 1.0))
        self.gripper_val = val

        scale = self.gripper_scale
        if isinstance(scale, (list, tuple)) and len(scale) >= 2:
            base_actual = scale[0] + val * (scale[1] - scale[0])
        else:
            base_actual = val * float(scale)

        qpos = self._get_target().copy()
        for joint, mult, offset in self.gripper_joints:
            if joint is None:
                continue
            dof_idx = _get_dof_idx(joint)
            if dof_idx is not None and 0 <= dof_idx < len(qpos):
                qpos[dof_idx] = float(base_actual * mult + offset)
        self._cached_target = qpos.copy()
        self.entity.control_dofs_position(qpos)

    def get_ee_pose(self) -> list:
        """Get end-effector (gripper TCP) pose as [x, y, z, qw, qx, qy, qz]."""
        if self.ee_link is None:
            return list(self.origin_pose.p) + list(self.origin_pose.q)
        # Get link6 pose from Genesis (quaternion is wxyz format)
        pos = to_numpy(self.ee_link.get_pos()).ravel()[:3]
        quat = to_numpy(self.ee_link.get_quat()).ravel()[:4]

        link_pose = Pose(pos, quat)

        # Apply TCP offset to get actual gripper TCP pose
        tcp_pose = link_pose * self.tcp_offset
        return tcp_pose.p.tolist() + tcp_pose.q.tolist()

    def move_to_homestate(self):
        """Snap arm to the home configuration.

        Hard-sets qpos AND the PD target so the arm is immediately at the
        home pose when this returns. Using ``control_dofs_position`` alone
        is not enough — it only sets a PD *target*, and callers typically
        run only a handful of settle steps before starting their own
        control loop, which would overwrite the target before PD
        converges.
        """
        qpos = to_numpy(self.entity.get_qpos()).copy()
        joint_names_all = [j.name for j in self.entity.joints]
        for i, name in enumerate(self.arm_joint_names):
            if name in joint_names_all:
                idx = joint_names_all.index(name)
                if idx < len(qpos):
                    qpos[idx] = self.homestate[i]
        self._cached_target = qpos.copy()
        self.entity.set_qpos(qpos)
        self.entity.control_dofs_position(qpos)


# ---------------------------------------------------------------------------
# Robot base interface
# ---------------------------------------------------------------------------

class Robot(ABC):
    """Abstract robot interface. Subclasses configure specific embodiments."""

    @abstractmethod
    def add_to_scene(self, scene: gs.Scene, **kwargs):
        """Add robot entities to the Genesis scene (before scene.build())."""
        ...

    @abstractmethod
    def init_joints(self, scene: gs.Scene):
        """Initialize joints and planners (after scene.build())."""
        ...

    @abstractmethod
    def move_to_homestate(self):
        ...

    @abstractmethod
    def get_arm(self, arm_tag: str) -> Arm:
        ...

    @abstractmethod
    def get_left_joint_state(self) -> list:
        ...

    @abstractmethod
    def get_right_joint_state(self) -> list:
        ...

    @abstractmethod
    def set_arm_joints(self, position, arm_tag: str):
        ...

    @abstractmethod
    def teleport_arm_joints(self, position, arm_tag: str):
        ...

    @abstractmethod
    def set_gripper(self, val: float, arm_tag: str):
        ...

    @abstractmethod
    def get_ee_pose(self, arm_tag: str) -> list:
        ...

    # ------------------------------------------------------------------
    # Mobile-base hooks (concrete no-op defaults; override in mobile robots)
    # ------------------------------------------------------------------

    def has_mobile_base(self) -> bool:
        """Whether this robot's base can translate/rotate in the world."""
        return False

    def drive(self, v_forward: float, omega_yaw: float):
        """Differential-drive twist command. No-op for fixed-base robots."""
        return

    def stop_base(self):
        """Zero any base velocity command. No-op for fixed-base robots."""
        return

    def get_base_pose(self) -> tuple:
        """Return (x, y, yaw) of the mobile base in world coordinates.

        Fixed-base robots return their ``base_pose`` projection, which callers
        can still use as the robot's world origin.
        """
        base_pose = getattr(self, "base_pose", None)
        if base_pose is None:
            return (0.0, 0.0, 0.0)
        w, x, y, z = (float(base_pose.q[0]), float(base_pose.q[1]),
                      float(base_pose.q[2]), float(base_pose.q[3]))
        yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        return (float(base_pose.p[0]), float(base_pose.p[1]), yaw)

    def move_base_to(self, x: float, y: float, yaw: float = 0.0,
                     step_fn=None, hold_arm: bool = True):
        """Drive the mobile base to (x, y, yaw). Raises for fixed-base robots."""
        raise NotImplementedError(
            f"{type(self).__name__} has no mobile base. Override "
            "move_base_to() in a mobile subclass."
        )

    def lock_base(self):
        """Pin the base so reaction forces (e.g. gripper squeeze) can't drift it."""
        return

    def unlock_base(self):
        """Release the base lock."""
        return

    def on_post_step(self, scene):
        """Hook called by the task's ``step_sim`` after each physics step.

        Mobile robots use this to re-snap a locked base. Fixed-base robots
        ignore it.
        """
        return
