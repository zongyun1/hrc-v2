"""Backend-neutral interface for controllable human/avatar motion.

The current avatar implementation can continue to replay authored clips, while
future backends (motion generators, physics refinement, or learned policies)
can expose the same state contract to robot planners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np


@dataclass
class HumanMotionState:
    """Snapshot consumed by collision checkers and robot planners."""

    root_pose: np.ndarray | None = None       # position(3) + quaternion(4)
    root_velocity: np.ndarray | None = None  # linear(3) + angular(3)
    joint_positions: np.ndarray | None = None
    joint_velocities: np.ndarray | None = None
    future_trajectory: np.ndarray | None = None
    collision_capsules: list[Mapping[str, Any]] = field(default_factory=list)
    action_phase: str = "idle"
    confidence: float = 1.0
    timestamp: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly snapshot for logging/planning."""
        return {
            "root_pose": self.root_pose,
            "root_velocity": self.root_velocity,
            "joint_positions": self.joint_positions,
            "joint_velocities": self.joint_velocities,
            "future_trajectory": self.future_trajectory,
            "collision_capsules": list(self.collision_capsules),
            "action_phase": self.action_phase,
            "confidence": float(self.confidence),
            "timestamp": float(self.timestamp),
        }


class HumanMotionController(Protocol):
    """Minimal controller contract independent of Genesis or a motion source."""

    def reset(self, *, seed: int | None = None) -> None: ...

    def step(
        self,
        dt: float,
        *,
        robot_state: Mapping[str, Any] | None = None,
        scene_state: Mapping[str, Any] | None = None,
    ) -> HumanMotionState: ...

    def set_goal(self, goal: Mapping[str, Any] | None) -> None: ...

    def get_state(self) -> HumanMotionState: ...


class StaticHumanMotionController:
    """Safe default adapter for existing static/authored avatar scenes."""

    def __init__(self, state: HumanMotionState | None = None):
        self._state = state or HumanMotionState()
        self._goal: Mapping[str, Any] | None = None

    def reset(self, *, seed: int | None = None) -> None:
        self._state.timestamp = 0.0

    def step(self, dt: float, *, robot_state=None, scene_state=None) -> HumanMotionState:
        self._state.timestamp += float(dt)
        return self._state

    def set_goal(self, goal: Mapping[str, Any] | None) -> None:
        self._goal = goal

    def get_state(self) -> HumanMotionState:
        return self._state

