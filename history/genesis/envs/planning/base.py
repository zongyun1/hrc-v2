"""Abstract planner interface."""

from abc import ABC, abstractmethod
import numpy as np


class PlanResult:
    """Standardized planning result."""

    def __init__(self, success: bool, position: np.ndarray = None, velocity: np.ndarray = None):
        self.success = success
        self.position = position if position is not None else np.array([])
        self.velocity = velocity if velocity is not None else np.zeros_like(self.position)

    @property
    def status(self) -> str:
        return "Success" if self.success else "Fail"

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "position": self.position,
            "velocity": self.velocity,
        }


class Planner(ABC):
    """Base interface for motion planners.

    All planners must implement:
        - plan_path: plan from current joint config to target EE pose
        - solve_ik: solve inverse kinematics for a target pose

    Optional:
        - update_obstacles: add collision obstacles
        - plan_batch: plan through a sequence of waypoints
    """

    @abstractmethod
    def plan_path(
        self,
        current_qpos: np.ndarray,
        target_pose: np.ndarray,
        **kwargs,
    ) -> PlanResult:
        """Plan a collision-free path to target_pose.

        Args:
            current_qpos: current joint positions (n_arm,)
            target_pose: target EE pose [x, y, z, qw, qx, qy, qz]

        Returns:
            PlanResult with waypoint positions and velocities.
        """
        ...

    @abstractmethod
    def solve_ik(
        self,
        current_qpos: np.ndarray,
        target_pose: np.ndarray,
        **kwargs,
    ) -> PlanResult:
        """Solve IK and return interpolated path to goal config.

        Args:
            current_qpos: current joint positions (n_arm,)
            target_pose: target EE pose [x, y, z, qw, qx, qy, qz]

        Returns:
            PlanResult with interpolated joint trajectory.
        """
        ...

    def update_obstacles(self, point_cloud: np.ndarray, resolution: float = 0.02):
        """Update obstacle point cloud for collision avoidance."""
        pass

    def plan_batch(
        self,
        current_qpos: np.ndarray,
        target_poses: list[np.ndarray],
        **kwargs,
    ) -> list[PlanResult]:
        """Plan through a sequence of waypoints."""
        results = []
        qpos = np.asarray(current_qpos, dtype=np.float64)
        for target in target_poses:
            result = self.plan_path(qpos, target, **kwargs)
            results.append(result)
            if result.success and result.position.size > 0:
                qpos = result.position[-1]
        return results

    def plan_gripper(self, current_val: float, target_val: float, num_steps: int = 200) -> np.ndarray:
        """Interpolate gripper value from current to target."""
        return np.linspace(current_val, target_val, num_steps)

    def _pad_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Pad an arm-only qpos to the width a planner backend expects.

        Default: identity. Planners that act on the live Genesis entity
        (``GenesisIKPlanner``, ``GenesisPlanner``) take arm-only qpos as-is;
        ``MplibPlanner`` overrides this to pad up to its URDF's user-joint
        count.

        Defined on the base class so task code that calls
        ``planner._pad_qpos(...)`` ahead of an mplib-only fast path (e.g.
        ``manipulation.TopDownPickPlaceMixin._move_seeded``) does not raise
        ``AttributeError`` on a non-mplib planner. That code calls
        ``_pad_qpos`` outside its try/except but the following
        ``planner._mplib`` access inside it — so a no-op here lets the
        mplib branch fail gracefully and fall back to the planner-agnostic
        ``move_and_execute`` path.
        """
        return np.asarray(qpos, dtype=np.float64)

    def _to_mplib_pose(self, target_pose):
        """Normalize a target pose to a 7-element [x,y,z,qw,qx,qy,qz] array.

        Default: returns a plain numpy pose7. ``MplibPlanner`` overrides
        this to return a backend ``pymp.Pose``. Defined on the base class
        for the same reason as ``_pad_qpos``: task code (e.g.
        ``manipulation.TopDownPickPlaceMixin._move_screw``) calls
        ``planner._to_mplib_pose(...)`` outside the try/except that guards
        the mplib-only fast path, so a non-crashing default lets the
        following ``planner._mplib`` access fail gracefully into
        ``move_and_execute``.
        """
        if hasattr(target_pose, "p"):
            return np.concatenate([
                np.asarray(target_pose.p, dtype=np.float64).ravel(),
                np.asarray(target_pose.q, dtype=np.float64).ravel(),
            ])
        return np.asarray(target_pose, dtype=np.float64).ravel()
