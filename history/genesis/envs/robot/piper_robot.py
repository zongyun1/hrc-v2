"""Piper dual-arm robot loaded via Genesis URDF API."""

import os
import yaml
import genesis as gs

from ..utils import Pose, ASSETS_PATH
from .base import Robot, Arm


class PiperRobot(Robot):
    """Piper robot: one or two URDF instances (see ``single_arm`` / ``dual_arm`` in config)."""

    def __init__(self, config_path: str = None, arms_distance: float = 0.3):
        if config_path is None:
            config_path = str(ASSETS_PATH / "embodiments" / "piper" / "config.yml")
        self.config_dir = os.path.dirname(os.path.abspath(config_path))

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.arms_distance = arms_distance
        self.left_arm = None
        self.right_arm = None

    def _resolve_path(self, rel_path: str) -> str:
        if rel_path.startswith("./"):
            rel_path = rel_path[2:]
        return os.path.abspath(os.path.join(self.config_dir, rel_path))

    @property
    def urdf_path(self) -> str:
        return self._resolve_path(self.config["urdf_path"])

    @property
    def srdf_path(self):
        rel = self.config.get("srdf_path")
        if not rel:
            return None
        return self._resolve_path(rel)

    def add_to_scene(self, scene: gs.Scene, **kwargs):
        """Add arm entities to scene.

        By default only the right arm is loaded (``single_arm: true`` in config).
        Set ``single_arm: false`` to load both arms.
        """
        cfg = self.config
        robot_poses = cfg.get("robot_pose", [[0, -0.3, 0.65, 1, 0, 0, 0], [0, -0.3, 0.75, 1, 0, 0, 0]])

        left_pose_raw = robot_poses[0]
        right_pose_raw = robot_poses[1] if len(robot_poses) > 1 else robot_poses[0]
        left_pose = Pose(left_pose_raw[:3], left_pose_raw[3:])
        right_pose = Pose(right_pose_raw[:3], right_pose_raw[3:])

        is_dual_arm = cfg.get("dual_arm", False)
        self._single_arm = cfg.get("single_arm", True)

        if not is_dual_arm and not self._single_arm:
            left_pose.p[0] -= self.arms_distance / 2
            right_pose.p[0] += self.arms_distance / 2

        urdf = self.urdf_path

        if not self._single_arm:
            left_entity = scene.add_entity(
                gs.morphs.URDF(
                    file=urdf,
                    pos=tuple(left_pose.p), quat=tuple(left_pose.q),
                    fixed=True, visualization=True, collision=True,
                    merge_fixed_links=True,
                ),
                material=gs.materials.Rigid(),
            )

        if is_dual_arm and not self._single_arm:
            right_entity = left_entity
        else:
            right_entity = scene.add_entity(
                gs.morphs.URDF(
                    file=urdf,
                    pos=tuple(right_pose.p), quat=tuple(right_pose.q),
                    fixed=True, visualization=True, collision=True,
                    merge_fixed_links=True,
                ),
                material=gs.materials.Rigid(),
            )

        if not self._single_arm:
            self.left_arm = Arm(left_entity, cfg, arm_index=0, origin_pose=left_pose)
        else:
            self.left_arm = None
        self.right_arm = Arm(right_entity, cfg, arm_index=1, origin_pose=right_pose)

    def init_joints(self, scene: gs.Scene):
        if self.left_arm is not None:
            self.left_arm.init_joints()
            self.left_arm.init_planner(self.urdf_path, self.srdf_path, scene)
        self.right_arm.init_joints()
        self.right_arm.init_planner(self.urdf_path, self.srdf_path, scene)

    def move_to_homestate(self):
        if self.left_arm is not None:
            self.left_arm.move_to_homestate()
        self.right_arm.move_to_homestate()

    def get_arm(self, arm_tag: str) -> Arm:
        if arm_tag == "left" and self.left_arm is None:
            raise RuntimeError("Left arm not available in single_arm mode")
        return self.left_arm if arm_tag == "left" else self.right_arm

    def get_left_joint_state(self) -> list:
        if self.left_arm is None:
            return []
        return self.left_arm.get_joint_state()

    def get_right_joint_state(self) -> list:
        return self.right_arm.get_joint_state()

    def set_arm_joints(self, position, arm_tag: str):
        self.get_arm(arm_tag).set_arm_joints(position)

    def teleport_arm_joints(self, position, arm_tag: str):
        self.get_arm(arm_tag).teleport_arm_joints(position)

    def set_gripper(self, val: float, arm_tag: str):
        self.get_arm(arm_tag).set_gripper(val)

    def get_ee_pose(self, arm_tag: str) -> list:
        return self.get_arm(arm_tag).get_ee_pose()

    def get_left_gripper_val(self) -> float:
        if self.left_arm is None:
            return 0.0
        return self.left_arm.gripper_val

    def get_right_gripper_val(self) -> float:
        return self.right_arm.gripper_val

    def open_gripper(self, arm_tag: str):
        self.set_gripper(1.0, arm_tag)

    def close_gripper(self, arm_tag: str):
        self.set_gripper(0.0, arm_tag)

    def set_origin_endpose(self):
        if self.left_arm is not None:
            self.left_original_pose = self.left_arm.get_ee_pose()
        else:
            self.left_original_pose = None
        self.right_original_pose = self.right_arm.get_ee_pose()
