"""Handle on the PandaOmron that rides inside the RoboCasa MJCF.

Genesis imports the RoboCasa export as *one* entity: the kitchen, its doors and drawers,
and the robot all share a single 59-dof kinematic chain. So there is no robot object to
hold — only a set of dof indices to pick out of the scene entity. This class does the
picking, by joint name, and carries the two corrections the export needs before the arm
can be driven at all:

* **The neutral qpos is out of bounds.** robosuite's authored home pose exceeds several
  of the arm's own joint limits on import, and Genesis then reports neutral-pose
  self-collisions. :meth:`reset_to_safe_pose` seats the arm at the middle of its declared
  range instead.
* **The gains have to be set per dof group.** The kitchen's own articulation (doors,
  drawers, knobs) shares the entity, so gains must be applied to the robot's dofs
  specifically, or stepping the scene would start actuating the furniture.

Dof groups, by joint-name prefix in the export:

======================  ==================================================
``robot0_joint*``       the 7 Panda arm joints
``mobilebase0_joint*``  forward / side slides, yaw hinge, torso column
``gripper0*``           the two finger slides
everything else         kitchen articulation — not ours to actuate
======================  ==================================================
"""

from __future__ import annotations

import numpy as np

ARM_PREFIX = "robot0_joint"
BASE_PREFIX = "mobilebase0_joint"
GRIPPER_PREFIX = "gripper0"
EEF_LINK = "gripper0_right_right_gripper"


class PandaOmron:
    """Dof bookkeeping and basic control for the robot embedded in a RoboCasa scene."""

    def __init__(self, entity):
        self.entity = entity
        self.arm, self.base, self.gripper, self.scene_dofs = [], [], [], []
        self.base_dof_by_name: dict[str, list[int]] = {}
        for joint in entity.joints:
            dofs = list(joint.dofs_idx_local)
            if not dofs:
                continue
            name = joint.name
            if name.startswith(ARM_PREFIX):
                self.arm += dofs
            elif name.startswith(BASE_PREFIX):
                self.base += dofs
                self.base_dof_by_name[name] = dofs
            elif name.startswith(GRIPPER_PREFIX):
                self.gripper += dofs
            else:
                self.scene_dofs += dofs
        self.arm = np.array(sorted(self.arm), dtype=int)
        self.base = np.array(sorted(self.base), dtype=int)
        self.gripper = np.array(sorted(self.gripper), dtype=int)
        self.robot_dofs = np.array(sorted({*self.arm, *self.base, *self.gripper}), dtype=int)

    def __repr__(self) -> str:
        return (f"PandaOmron(arm={len(self.arm)}, base={len(self.base)}, "
                f"gripper={len(self.gripper)}, scene={len(self.scene_dofs)})")

    # -- setup -----------------------------------------------------------------

    def safe_qpos(self) -> np.ndarray:
        """Current dof positions, with the arm moved to the middle of its limits."""
        q = np.asarray(self.entity.get_dofs_position().cpu()).copy()
        lo, hi = (np.asarray(x.cpu()) for x in self.entity.get_dofs_limit())
        bounded = np.isfinite(lo) & np.isfinite(hi)
        mid = np.where(bounded, 0.5 * (lo + hi), q)
        q[self.arm] = mid[self.arm]
        return q

    def reset_to_safe_pose(self) -> np.ndarray:
        """Seat the arm inside its joint limits. Call once after ``scene.build()``."""
        q = self.safe_qpos()
        self.entity.set_dofs_position(q, zero_velocity=True)
        return q

    def set_gains(self, kp: float = 300.0, kv: float = 30.0) -> None:
        """Apply PD gains to the robot's dofs only, leaving kitchen articulation alone."""
        n = len(self.robot_dofs)
        self.entity.set_dofs_kp(np.full(n, kp), self.robot_dofs)
        self.entity.set_dofs_kv(np.full(n, kv), self.robot_dofs)

    # -- state -----------------------------------------------------------------

    @property
    def eef(self):
        return self.entity.get_link(EEF_LINK)

    def eef_pos(self) -> np.ndarray:
        return np.asarray(self.eef.get_pos().cpu()).reshape(-1)

    def base_pos(self) -> np.ndarray:
        return np.asarray(self.entity.get_link("robot0_link0").get_pos().cpu()).reshape(-1)

    def arm_qpos(self) -> np.ndarray:
        return np.asarray(self.entity.get_dofs_position(self.arm).cpu()).reshape(-1)

    # -- control ---------------------------------------------------------------

    def hold(self, q_target: np.ndarray) -> None:
        """Command every robot dof to a target configuration (full-length vector)."""
        self.entity.control_dofs_position(q_target[self.robot_dofs], self.robot_dofs)


__all__ = ["PandaOmron", "EEF_LINK", "ARM_PREFIX", "BASE_PREFIX", "GRIPPER_PREFIX"]
