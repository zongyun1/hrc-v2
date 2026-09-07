"""ARX X5 single-arm robot (URDF) loaded via the generic PiperRobot machinery.

The X5A.urdf + ``embodiments/ARX-X5/config.yml`` ship in the same schema the
Arm class understands (joint names, gripper mimic, gripper_bias/scale, home
state). Defaults to single-arm; flip ``dual_arm: True`` in config.yml to load
two instances side-by-side.
"""

from ..utils import ASSETS_PATH
from .piper_robot import PiperRobot


class ARXX5Robot(PiperRobot):
    def __init__(self, config_path: str = None, arms_distance: float = 0.3):
        if config_path is None:
            config_path = str(ASSETS_PATH / "embodiments" / "ARX-X5" / "config.yml")
        super().__init__(config_path=config_path, arms_distance=arms_distance)
