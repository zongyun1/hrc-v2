"""UFactory xArm7 (7-DOF arm + xArm parallel gripper) loaded via Genesis MJCF API.

Asset source: google-deepmind/mujoco_menagerie/ufactory_xarm7 (BSD-3-Clause),
mirrored at ``assets/embodiments/xarm7/xarm7.xml``. The MJCF bundles the arm
plus the xArm parallel gripper; the original URDF in
``assets/example-robot-data/robots/xarm_description/urdf/xarm7.urdf`` is
arm-only (no gripper) so we keep the MJCF as the source of truth.
Motion planning defaults to ``mplib`` (OMPL against the arm-only
URDF); ``genesis_ik`` is also available.
"""

import os

import numpy as np
import genesis as gs

from ..utils import Pose, to_numpy, ASSETS_PATH
from ..genesis_compat import set_dofs_kp_kv_compat
from .base import Robot, _find_joint_by_name, _find_link_by_name, _get_dof_idx
from ..planning import MplibPlanner, GenesisIKPlanner


XARM7_MJCF_REL = "embodiments/xarm7/xarm7.xml"

# 7-DOF arm joints (MJCF joint names; URDF uses the same names).
XARM7_ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]

# Driver joints for the parallel gripper. The four other gripper joints
# (inner_knuckle, follower fingers) are kinematically coupled via the MJCF
# tendon "split" + equality joint constraint, so commanding both drivers
# to the same value is sufficient. Genesis approximates the tendon as a
# joint actuator on each driver — see warning at scene.build().
XARM7_DRIVER_JOINTS = ["left_driver_joint", "right_driver_joint"]

# End-effector link in the MJCF. The site `link_tcp` at pos=(0, 0, 0.172)
# in this link's frame is the tool-center-point.
XARM7_EE_LINK = "xarm_gripper_base_link"

# URDF-side names (for mplib path: arm-only URDF, EE = link7).
XARM7_URDF_ARM_JOINTS = list(XARM7_ARM_JOINTS)
XARM7_URDF_EE_LINK = "link7"

# Driver joint range: 0 = fully open, 0.85 rad ≈ fully closed.
XARM7_GRIPPER_OPEN = 0.0
XARM7_GRIPPER_CLOSED = 0.85


class XArm7Arm:
    """Single xArm7 arm + parallel gripper loaded from MJCF."""

    def __init__(self, entity, origin_pose: Pose):
        self.entity = entity
        self.origin_pose = origin_pose
        self.n_arm = 7
        self.gripper_val = 0.0
        self.planner = None

        self.gripper_open = XARM7_GRIPPER_OPEN
        self.gripper_closed = XARM7_GRIPPER_CLOSED

        # Home from MJCF keyframe `home` qpos[:7].
        self.homestate = [0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0]

        # TCP offset: from xarm_gripper_base_link origin to the link_tcp site.
        # Site is at local pos (0, 0, 0.172) with identity orientation.
        self.tcp_offset = Pose(
            [0.0, 0.0, 0.172],
            [1.0, 0.0, 0.0, 0.0],
        )

        # Resolved after scene.build()
        self.ee_link = None
        self.arm_joints = []
        self.driver_joints = []
        self._driver_dof_indices = []
        # Franka-compatible finger aliases — populated in init_joints().
        self.finger_joints = []
        self._finger_dof_indices = []
        self._cached_target = None

    def init_joints(self):
        """Resolve joint/link references after scene.build()."""
        self.ee_link = _find_link_by_name(self.entity, XARM7_EE_LINK)
        self.arm_joints = [_find_joint_by_name(self.entity, n) for n in XARM7_ARM_JOINTS]
        self.driver_joints = [_find_joint_by_name(self.entity, n) for n in XARM7_DRIVER_JOINTS]

        # Boost PD gains for accurate trajectory tracking. Match Franka-style
        # stiffness — xArm7 joint torques are similar order of magnitude.
        try:
            kp = to_numpy(self.entity.get_dofs_kp()).copy()
            kv = to_numpy(self.entity.get_dofs_kv()).copy()
            for j in self.arm_joints:
                if j is None:
                    continue
                idx = _get_dof_idx(j)
                if idx is None:
                    continue
                kp[idx] = max(kp[idx], 4000.0)
                kv[idx] = max(kv[idx], 400.0)
            self._driver_dof_indices = []
            for j in self.driver_joints:
                if j is None:
                    continue
                idx = _get_dof_idx(j)
                if idx is None:
                    continue
                self._driver_dof_indices.append(idx)
                # Driver joints actuate against soft springs in the inner-knuckle
                # joints (stiffness 0.05) — modest gains are enough.
                kp[idx] = max(kp[idx], 1000.0)
                kv[idx] = max(kv[idx], 100.0)
            set_dofs_kp_kv_compat(self.entity, kp=kp, kv=kv)
        except Exception:
            pass

        # FrankaArm exposes the gripper-grip interface as `finger_joints` /
        # `_finger_dof_indices`, which many tasks iterate directly. The xArm7
        # parallel-jaw gripper's actuated grip joints are the drivers (the
        # inner-knuckle / follower joints track via MJCF equality
        # constraints), so alias the finger interface onto the drivers.
        self.finger_joints = self.driver_joints
        self._finger_dof_indices = self._driver_dof_indices

    def init_planner(self, urdf_path: str, srdf_path: str, scene, planner_type: str = "mplib"):
        """Create motion planner. Default ``mplib`` — OMPL-based planning
        against the arm-only URDF. ``genesis_ik`` is also supported: it
        uses Genesis's damped-LS IK against the live entity, so the
        gripper geometry is accounted for and no URDF is needed.

        The ``mplib`` path uses the arm-only URDF where
        ``link7`` is the EE (no gripper). The MJCF places
        ``xarm_gripper_base_link`` as a child of ``link7`` with
        ``quat="0 0 0 1"`` (180° around Z, same origin), so when the mplib
        path is selected we rebake ``tcp_offset`` to be relative to
        ``link7`` instead of the gripper-base. This way callers (e.g.
        ``pour_water._mug_to_link``) can compose ``T_world_TCP *
        arm.tcp_offset.inv()`` with either planner and get the right
        link-frame target.
        """
        if planner_type == "mplib":
            # link7 → TCP = (link7 → gripper_base) ⋅ (gripper_base → TCP)
            # = Pose(p=0, q=Z180°) * Pose(p=[0,0,0.172], q=identity)
            # Z-rotation preserves the [0,0,0.172] translation; quat
            # remains the 180°-Z rotation.
            self.tcp_offset = Pose(
                [0.0, 0.0, 0.172],
                [0.0, 0.0, 0.0, 1.0],
            )
            # Rebind ee_link so get_ee_pose() composes from the same frame
            # the planner targets (link7), keeping the TCP frame consistent.
            link7 = _find_link_by_name(self.entity, XARM7_URDF_EE_LINK)
            if link7 is not None:
                self.ee_link = link7
        joint_index = {}
        for j in self.entity.joints:
            idx = _get_dof_idx(j)
            if idx is not None:
                joint_index[j.name] = idx

        if planner_type == "genesis_ik":
            self.planner = GenesisIKPlanner(
                entity=self.entity,
                ee_link_name=XARM7_EE_LINK,
                n_arm=self.n_arm,
                arm_joint_names=XARM7_ARM_JOINTS,
                arm_joint_index=joint_index,
            )
            return

        # The xarm URDF references meshes via
        # ``package://example-robot-data/robots/xarm_description/<rest>``.
        # Sapien's URDF loader (under mplib) strips leading `/` from absolute
        # paths, so we precompute a URDF with mesh filenames written as
        # relative paths from the URDF directory.
        urdf_path = self._materialize_relative_path_urdf(urdf_path)

        # Frame mismatch: the menagerie MJCF puts ``link_base`` at
        # ``pos="0 0 0.12"`` w.r.t. the robot anchor, while the URDF treats
        # ``link_base`` as the root with zero offset from ``world``. Shift
        # the base pose given to mplib up by 0.12m so its URDF chain aligns
        # with Genesis's world-frame link poses.
        base7 = self.origin_pose.to_pose7().copy()
        R = self.origin_pose.to_matrix()[:3, :3]
        base7[:3] = base7[:3] + R @ np.array([0.0, 0.0, 0.12])

        self.planner = MplibPlanner(
            urdf_path=urdf_path,
            srdf_path=srdf_path,
            move_group=XARM7_URDF_EE_LINK,
            base_pose=base7,
            entity=self.entity,
            n_arm=self.n_arm,
            arm_joint_names=XARM7_ARM_JOINTS,
            arm_joint_index=joint_index,
            entity_ee_link_name=XARM7_EE_LINK,
        )

    @staticmethod
    def _materialize_relative_path_urdf(urdf_path: str) -> str:
        """Produce a sibling URDF whose ``filename=...`` mesh refs are
        relative to the URDF directory. Idempotent; cached on disk."""
        import re

        urdf_path = str(urdf_path)
        out_path = urdf_path.replace(".urdf", "_relpaths.urdf")
        urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
        with open(urdf_path, "r") as f:
            text = f.read()

        def _sub(match):
            full = match.group(1)  # e.g. package://example-robot-data/robots/xarm_description/meshes/...
            if full.startswith("package://"):
                # Map `package://X/<rest>` → assets/<X-without-package-prefix>/<rest>
                # by trying common roots: ASSETS_PATH and the URDF's
                # great-grand-parent (the description root).
                rest = full[len("package://"):]
                pkg_name, _, tail = rest.partition("/")
                # Heuristic: search ASSETS_PATH for the package directory.
                candidates = [
                    os.path.join(str(ASSETS_PATH), pkg_name, tail),
                    # Fallback: scan ASSETS_PATH/example-robot-data/robots/* for `<pkg>`.
                    os.path.join(str(ASSETS_PATH), "example-robot-data", "robots", pkg_name, tail),
                ]
                resolved = next((c for c in candidates if os.path.isfile(c)), None)
                if resolved is None:
                    return match.group(0)  # leave untouched if we can't resolve
                rel = os.path.relpath(resolved, urdf_dir)
                return f'filename="{rel}"'
            return match.group(0)

        new_text = re.sub(r'filename="([^"]+)"', _sub, text)
        with open(out_path, "w") as f:
            f.write(new_text)
        return out_path

    def get_joint_state(self) -> list:
        qpos = to_numpy(self.entity.get_qpos())
        state = []
        for j in self.arm_joints:
            if j is None:
                continue
            dof_idx = _get_dof_idx(j)
            if dof_idx is not None and 0 <= dof_idx < len(qpos):
                state.append(float(qpos[dof_idx]))
        state.append(self.gripper_val)
        return state

    def get_arm_qpos(self) -> np.ndarray:
        qpos = to_numpy(self.entity.get_qpos())
        state = []
        for j in self.arm_joints:
            if j is None:
                continue
            dof_idx = _get_dof_idx(j)
            if dof_idx is not None and 0 <= dof_idx < len(qpos):
                state.append(float(qpos[dof_idx]))
        return np.array(state, dtype=np.float64)

    def set_arm_joints(self, position: np.ndarray):
        qpos = self._build_qpos_with_arm(position)
        self._cached_target = qpos.copy()
        self.entity.control_dofs_position(qpos)

    def teleport_arm_joints(self, position: np.ndarray):
        qpos = self._build_qpos_with_arm(position)
        self._cached_target = qpos.copy()
        self.entity.set_qpos(qpos)
        self.entity.control_dofs_position(qpos)

    def _get_target(self) -> np.ndarray:
        if self._cached_target is None:
            self._cached_target = to_numpy(self.entity.get_qpos()).copy()
        return self._cached_target

    def _build_qpos_with_arm(self, position: np.ndarray) -> np.ndarray:
        qpos = self._get_target().copy()
        for i, j in enumerate(self.arm_joints):
            if i >= len(position) or j is None:
                break
            dof_idx = _get_dof_idx(j)
            if dof_idx is not None and 0 <= dof_idx < len(qpos):
                qpos[dof_idx] = float(position[i])
        return qpos

    def set_gripper(self, val: float):
        """Set gripper: 1=open, 0=closed.

        Driver joint maps: open=0.0 rad, closed=0.85 rad. Both driver
        joints get the same target — the equality+tendon in the MJCF
        couples them, but commanding both directly is more robust to
        Genesis's tendon-as-actuator approximation.
        """
        val = float(np.clip(val, 0.0, 1.0))
        self.gripper_val = val
        driver_pos = self.gripper_closed + val * (self.gripper_open - self.gripper_closed)

        qpos = self._get_target().copy()
        for j in self.driver_joints:
            if j is None:
                continue
            dof_idx = _get_dof_idx(j)
            if dof_idx is not None and 0 <= dof_idx < len(qpos):
                qpos[dof_idx] = float(driver_pos)
        self._cached_target = qpos.copy()
        self.entity.control_dofs_position(qpos)

    def get_ee_pose(self) -> list:
        if self.ee_link is None:
            return list(self.origin_pose.p) + list(self.origin_pose.q)
        pos = to_numpy(self.ee_link.get_pos()).ravel()[:3]
        quat = to_numpy(self.ee_link.get_quat()).ravel()[:4]
        link_pose = Pose(pos, quat)
        tcp_pose = link_pose * self.tcp_offset
        return tcp_pose.p.tolist() + tcp_pose.q.tolist()

    def move_to_homestate(self):
        qpos = to_numpy(self.entity.get_qpos()).copy()
        for i, j in enumerate(self.arm_joints):
            if j is None or i >= len(self.homestate):
                continue
            dof_idx = _get_dof_idx(j)
            if dof_idx is not None and 0 <= dof_idx < len(qpos):
                qpos[dof_idx] = self.homestate[i]
        # Open gripper at home.
        for j in self.driver_joints:
            if j is None:
                continue
            dof_idx = _get_dof_idx(j)
            if dof_idx is not None and 0 <= dof_idx < len(qpos):
                qpos[dof_idx] = self.gripper_open
        self._cached_target = qpos.copy()
        self.entity.set_qpos(qpos)
        self.entity.control_dofs_position(qpos)
        self.gripper_val = 1.0


class XArm7Robot(Robot):
    """UFactory xArm7 single-arm robot loaded from the menagerie MJCF."""

    def __init__(self, pos=None, quat=None, planner_type: str = "mplib"):
        """
        Args:
            pos: robot base position [x, y, z]. Default places it behind the
                table at table-height, mirroring the Franka placement.
            quat: robot base orientation [w, x, y, z]. Default rotates the
                arm to face +Y (toward the workspace).
            planner_type: ``"mplib"`` (default) or ``"genesis_ik"``.
        """
        if pos is None:
            pos = [0.0, -0.65, 0.75]
        if quat is None:
            quat = [0.707, 0.0, 0.0, 0.707]
        self.base_pose = Pose(pos, quat)
        self.planner_type = planner_type
        # Single-arm convention (left_arm=None, right_arm primary) — matches
        # FrankaRobot so BaseTask.take_action's dual-arm branching works.
        self.left_arm = None
        self.right_arm = None

    def add_to_scene(self, scene: gs.Scene, **kwargs):
        mjcf_path = str(ASSETS_PATH / XARM7_MJCF_REL)
        entity = scene.add_entity(
            gs.morphs.MJCF(
                file=mjcf_path,
                pos=tuple(self.base_pose.p),
                quat=tuple(self.base_pose.q),
            ),
        )
        self.right_arm = XArm7Arm(entity, self.base_pose)

    def init_joints(self, scene: gs.Scene):
        self.right_arm.init_joints()

        # mplib path uses the arm-only URDF/SRDF shipped with the repo.
        xarm_dir = ASSETS_PATH / "example-robot-data" / "robots" / "xarm_description"
        self.planner_urdf_path = str(xarm_dir / "urdf" / "xarm7.urdf")
        self.planner_srdf_path = str(xarm_dir / "srdf" / "xarm7.srdf")
        self.right_arm.init_planner(
            self.planner_urdf_path, self.planner_srdf_path, scene,
            planner_type=self.planner_type,
        )

    def move_to_homestate(self):
        self.right_arm.move_to_homestate()

    def get_arm(self, arm_tag: str) -> XArm7Arm:
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
