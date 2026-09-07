"""SE2 planner for a mobile base.

Plans a collision-free sequence of ``(x, y)`` waypoints through a set of
axis-aligned rectangular obstacles, plus a goal yaw. The path is an RRT
extension on the 2D ground plane with post-hoc shortcut smoothing; yaw is
not part of the search space (the base is omnidirectional enough via
pivot-then-drive that heading can be chosen per-segment at execution time).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..utils import to_numpy


@dataclass
class AABB2D:
    """Axis-aligned rectangle in the world xy-plane. ``z`` is ignored."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def inflate(self, r: float) -> "AABB2D":
        return AABB2D(self.xmin - r, self.ymin - r, self.xmax + r, self.ymax + r)

    def contains(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def segment_intersects(
        self, p0: Tuple[float, float], p1: Tuple[float, float]
    ) -> bool:
        """Liang–Barsky clip: true iff segment p0→p1 enters this rectangle."""
        x0, y0 = p0
        x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0
        t0, t1 = 0.0, 1.0
        for p, q in (
            (-dx, x0 - self.xmin),
            ( dx, self.xmax - x0),
            (-dy, y0 - self.ymin),
            ( dy, self.ymax - y0),
        ):
            if abs(p) < 1e-12:
                if q < 0:
                    return False
                continue
            t = q / p
            if p < 0:
                if t > t1:
                    return False
                if t > t0:
                    t0 = t
            else:
                if t < t0:
                    return False
                if t < t1:
                    t1 = t
        return t0 <= t1

    @classmethod
    def from_entity(cls, entity, z_min: float = -np.inf, z_max: float = np.inf) -> "AABB2D":
        """Build a 2D footprint AABB from a Genesis entity's 3D AABB.

        ``z_min`` / ``z_max`` filter entities that don't actually obstruct the
        base — e.g., a ceiling. By default every entity is treated as an
        obstacle regardless of height.
        """
        aabb = to_numpy(entity.get_AABB()).reshape(-1, 3)
        lo, hi = aabb[0], aabb[-1]
        if hi[2] < z_min or lo[2] > z_max:
            return cls(np.inf, np.inf, -np.inf, -np.inf)  # empty — ignored
        return cls(float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))


class SE2BasePlanner:
    """RRT on (x, y) with axis-aligned rectangular obstacles.

    Obstacles are inflated by ``robot_radius`` so the planner can treat the
    base as a point. The output is a list of ``(x, y)`` waypoints plus the
    caller-supplied goal yaw; heading between waypoints is chosen at
    execution time so the robot pivots-then-drives forward.
    """

    def __init__(
        self,
        obstacles: Iterable[AABB2D],
        robot_radius: float,
        bounds: Tuple[float, float, float, float] = (-5.0, -5.0, 5.0, 5.0),
        step_size: float = 0.25,
        goal_bias: float = 0.15,
        max_iter: int = 4000,
        goal_tol: float = 0.1,
        shortcut_iter: int = 200,
        seed: int = 0,
    ):
        self.obstacles = [o.inflate(robot_radius) for o in obstacles]
        self.bounds = bounds
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.max_iter = max_iter
        self.goal_tol = goal_tol
        self.shortcut_iter = shortcut_iter
        self.rng = np.random.default_rng(seed)

    # ---- collision queries -------------------------------------------

    def point_free(self, p: Tuple[float, float]) -> bool:
        return not any(o.contains(p[0], p[1]) for o in self.obstacles)

    def segment_free(
        self, p0: Tuple[float, float], p1: Tuple[float, float]
    ) -> bool:
        return not any(o.segment_intersects(p0, p1) for o in self.obstacles)

    # ---- RRT ---------------------------------------------------------

    def plan(
        self,
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float],
    ) -> Optional["BasePath"]:
        """Plan from ``(sx, sy, _)`` to ``(gx, gy, gyaw)``.

        Returns ``None`` if no path is found within ``max_iter`` samples.
        The start yaw is read from the current base pose at execution time,
        so the third component of ``start`` is ignored here.
        """
        sxy = (float(start[0]), float(start[1]))
        gxy = (float(goal[0]), float(goal[1]))
        goal_yaw = float(goal[2])

        if not self.point_free(sxy):
            # Start buried inside an obstacle — can happen if the robot
            # began too close to a wall. The caller's best recovery is to
            # nudge out, so surface this rather than silently fail.
            return None
        if not self.point_free(gxy):
            return None

        if self.segment_free(sxy, gxy):
            return BasePath(waypoints=[sxy, gxy], goal_yaw=goal_yaw)

        nodes: List[Tuple[float, float]] = [sxy]
        parents: List[int] = [-1]
        xmin, ymin, xmax, ymax = self.bounds

        for _ in range(self.max_iter):
            if self.rng.random() < self.goal_bias:
                sample = gxy
            else:
                sample = (
                    float(self.rng.uniform(xmin, xmax)),
                    float(self.rng.uniform(ymin, ymax)),
                )

            near_idx = self._nearest(nodes, sample)
            near = nodes[near_idx]
            dx, dy = sample[0] - near[0], sample[1] - near[1]
            d = float(np.hypot(dx, dy))
            if d < 1e-6:
                continue
            step = min(self.step_size, d)
            new_pt = (near[0] + dx / d * step, near[1] + dy / d * step)
            if not self.segment_free(near, new_pt):
                continue
            nodes.append(new_pt)
            parents.append(near_idx)

            if (
                float(np.hypot(new_pt[0] - gxy[0], new_pt[1] - gxy[1])) < self.goal_tol
                and self.segment_free(new_pt, gxy)
            ):
                nodes.append(gxy)
                parents.append(len(nodes) - 2)
                path = self._reconstruct(nodes, parents)
                path = self._shortcut(path)
                return BasePath(waypoints=path, goal_yaw=goal_yaw)

        return None

    @staticmethod
    def _nearest(nodes: Sequence[Tuple[float, float]], q: Tuple[float, float]) -> int:
        arr = np.asarray(nodes)
        d2 = (arr[:, 0] - q[0]) ** 2 + (arr[:, 1] - q[1]) ** 2
        return int(np.argmin(d2))

    @staticmethod
    def _reconstruct(
        nodes: List[Tuple[float, float]], parents: List[int]
    ) -> List[Tuple[float, float]]:
        path, i = [], len(nodes) - 1
        while i != -1:
            path.append(nodes[i])
            i = parents[i]
        path.reverse()
        return path

    def _shortcut(
        self, path: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Randomly pick index pairs and splice in a straight segment if free."""
        if len(path) <= 2:
            return path
        path = list(path)
        for _ in range(self.shortcut_iter):
            n = len(path)
            if n <= 2:
                break
            i = int(self.rng.integers(0, n - 1))
            j = int(self.rng.integers(i + 1, n))
            if j - i <= 1:
                continue
            if self.segment_free(path[i], path[j]):
                path = path[: i + 1] + path[j:]
        return path


@dataclass
class BasePath:
    """Planner output: waypoints on the xy-plane plus final yaw."""

    waypoints: List[Tuple[float, float]]
    goal_yaw: float

    def __len__(self) -> int:
        return len(self.waypoints)

    def __iter__(self):
        return iter(self.waypoints)
