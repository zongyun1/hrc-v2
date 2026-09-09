"""UR5 + WSG-50 gripper single-arm robot (URDF) loaded via PiperRobot machinery.

``ur5_wsg_gripper.urdf`` + ``embodiments/ur5-wsg/config.yml`` follow the same
schema as Piper, so we delegate to PiperRobot. Defaults to single-arm; the
gripper is the WSG-50 (prismatic ``base_joint_gripper_left`` mirrored onto
``base_joint_gripper_right`` via the config's mimic entry).
"""

from ..utils import ASSETS_PATH
from .piper_robot import PiperRobot


class UR5WSGRobot(PiperRobot):
    def __init__(self, config_path: str = None, arms_distance: float = 0.3):
        if config_path is None:
            config_path = str(ASSETS_PATH / "embodiments" / "ur5-wsg" / "config.yml")
        super().__init__(config_path=config_path, arms_distance=arms_distance)
