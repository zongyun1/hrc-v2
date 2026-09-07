"""CuroboPlanner — persistent subprocess bridge to cuRobo motion planning.

Architecture
------------
cuRobo requires a CUDA-capable A100/V100 GPU and lives in a separate conda env:
  /scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/curobo

This class launches ``curobo_worker.py`` as a **long-lived** subprocess via
``Popen``.  The worker builds cuRobo MotionGen once (warmup ~15-25 s on V100)
and reuses it for every subsequent plan request — no cold-start overhead.

Protocol: newline-delimited JSON on stdin/stdout (one object per line).

    result = planner.plan_path(qpos_start, target_pose_7)
    # PlanResult with .success, .position (N, n_arm), .velocity (N, n_arm)
"""

from __future__ import annotations

import json
import select
import subprocess
import time
from pathlib import Path

import numpy as np

from .base import Planner, PlanResult

# ── paths ─────────────────────────────────────────────────────────────────────
CUROBO_PYTHON: str = (
    "/scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/curobo/bin/python"
)
_WORKER: str = str(Path(__file__).parent / "curobo_worker.py")

# Per-request wall-clock timeout (seconds).
# First call includes MotionGen warmup (~20 s); subsequent calls are ~2-15 s.
REQUEST_TIMEOUT_FIRST: float = 120.0
REQUEST_TIMEOUT: float = 90.0


class CuroboPlanner(Planner):
    """Persistent subprocess bridge to cuRobo MotionGen.

    Parameters
    ----------
    urdf_path    : absolute path to the robot URDF
    ee_link      : end-effector link name (e.g. ``"link6"`` or ``"panda_hand"``)
    base_pose    : robot base pose in world frame ``[x, y, z, qw, qx, qy, qz]``
    joint_names  : arm joint names in order (length == n_arm)
    num_waypoints: number of interpolation steps returned by MotionGen
    entity       : Genesis entity (used for IK fallback via solve_ik)
    """

    def __init__(
        self,
        urdf_path: str,
        ee_link: str,
        base_pose,
        joint_names: list[str],
        num_waypoints: int = 200,
        entity=None,
        arm_joint_index: dict[str, int] = None,
        **kwargs,
    ) -> None:
        self.urdf_path     = str(urdf_path)
        self.ee_link       = ee_link
        self._base_pose    = np.asarray(base_pose, dtype=np.float64).ravel()
        self.joint_names   = list(joint_names)
        self.n_arm         = len(joint_names)
        self.num_waypoints = num_waypoints
        self._is_first     = True
        self._entity       = entity
        self._ee_link_name = ee_link
        self._arm_joint_names = list(joint_names)
        self._arm_joint_index = dict(arm_joint_index) if arm_joint_index else None

        if not Path(CUROBO_PYTHON).exists():
            raise FileNotFoundError(
                f"cuRobo Python not found: {CUROBO_PYTHON}\n"
                "Check that the cuRobo conda env is installed."
            )
        if not Path(_WORKER).exists():
            raise FileNotFoundError(f"cuRobo worker not found: {_WORKER}")

        self._proc: subprocess.Popen | None = None
        self._launch_worker()

    # ── subprocess lifecycle ─────────────────────────────────────────────────

    def _launch_worker(self):
        """Start (or restart) the persistent worker subprocess."""
        self._kill_worker()
        self._proc = subprocess.Popen(
            [CUROBO_PYTHON, _WORKER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,         # inherit parent stderr (logs visible)
            text=True,
            bufsize=1,           # line-buffered
        )
        self._is_first = True

    def _kill_worker(self):
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception:
                pass
            self._proc = None

    def _worker_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def close(self):
        """Cleanly shut down the worker."""
        self._kill_worker()

    def __del__(self):
        self._kill_worker()

    # ── read with timeout ────────────────────────────────────────────────────

    def _read_line(self, timeout: float) -> str | None:
        """Read one line from worker stdout, with wall-clock timeout."""
        if self._proc is None:
            return None
        ready = select.select([self._proc.stdout], [], [], timeout)
        if ready[0]:
            line = self._proc.stdout.readline()
            return line if line else None
        return None

    # ── Planner interface ────────────────────────────────────────────────────

    def plan_path(
        self,
        current_qpos: np.ndarray,
        target_pose: np.ndarray,
        log: bool = True,
        **kwargs,
    ) -> PlanResult:
        """Plan a collision-free trajectory using cuRobo subprocess.

        Args:
            current_qpos: current joint positions (n_arm,)
            target_pose: target EE pose [x, y, z, qw, qx, qy, qz]

        Returns:
            PlanResult with waypoint positions and velocities.
        """
        result = self._plan_raw(current_qpos, target_pose, log=log)
        status = result.get("status", "Fail")
        if status == "Success":
            pos = result["position"]
            vel = result.get("velocity", np.zeros_like(pos))
            return PlanResult(True, pos, vel)
        return PlanResult(False)

    def solve_ik(
        self,
        current_qpos: np.ndarray,
        target_pose: np.ndarray,
        num_waypoints: int = None,
        log: bool = True,
        **kwargs,
    ) -> PlanResult:
        """Solve IK using Genesis entity and return interpolated trajectory.

        Falls back to Genesis IK since cuRobo IK requires the worker.
        """
        if self._entity is None:
            return PlanResult(False)

        from .mplib_planner import MplibPlanner
        # Delegate to a lightweight IK solver
        target7 = np.asarray(target_pose, dtype=np.float64).ravel()[:7]
        pos_ik = target7[:3]
        quat_ik = target7[3:7]

        raw_qpos = self._entity.get_qpos()
        init_full = raw_qpos.detach().cpu().numpy().ravel() if hasattr(raw_qpos, "cpu") else np.asarray(raw_qpos).ravel()

        if self._arm_joint_names and self._arm_joint_index:
            for i, jname in enumerate(self._arm_joint_names[:self.n_arm]):
                if jname in self._arm_joint_index and i < len(current_qpos):
                    init_full[self._arm_joint_index[jname]] = float(current_qpos[i])

        ee_link = None
        for link in self._entity.links:
            if link.name == self._ee_link_name:
                ee_link = link
                break
        if ee_link is None:
            if log:
                print(f"[curobo] IK: ee_link '{self._ee_link_name}' not found")
            return PlanResult(False)

        ik_fn = getattr(self._entity, "inverse_kinematics", None)
        if ik_fn is None:
            if log:
                print("[curobo] IK: entity has no inverse_kinematics")
            return PlanResult(False)

        try:
            qpos_goal = ik_fn(link=ee_link, pos=pos_ik, quat=quat_ik, init_qpos=init_full)
        except TypeError:
            qpos_goal = ik_fn(link=ee_link, pos=pos_ik, quat=quat_ik)
        except Exception as e:
            if log:
                print(f"[curobo] IK exception: {e}")
            return PlanResult(False)

        if qpos_goal is None:
            return PlanResult(False)

        qpos_goal = qpos_goal.detach().cpu().numpy().ravel() if hasattr(qpos_goal, "cpu") else np.asarray(qpos_goal).ravel()

        if self._arm_joint_names and self._arm_joint_index:
            arm_cols = [self._arm_joint_index[jn] for jn in self._arm_joint_names if jn in self._arm_joint_index]
            q_goal = np.array([qpos_goal[idx] for idx in arm_cols[:self.n_arm]], dtype=np.float64)
            q_start = np.array([current_qpos[i] if i < len(current_qpos) else 0.0 for i in range(self.n_arm)], dtype=np.float64)
        else:
            q_goal = qpos_goal[:self.n_arm]
            q_start = np.asarray(current_qpos[:self.n_arm], dtype=np.float64)

        n_wp = max(2, num_waypoints or self.num_waypoints)
        t = np.linspace(0, 1, n_wp)
        position = (1 - t[:, None]) * q_start + t[:, None] * q_goal
        dt = 1.0 / 250.0
        velocity = np.zeros_like(position)
        if position.shape[0] > 1:
            velocity[:-1] = np.diff(position, axis=0) / dt
        velocity = np.clip(velocity, -1.0, 1.0)

        if log:
            print(f"[curobo] IK Success, waypoints={position.shape[0]}")
        return PlanResult(True, position, velocity)

    # ── raw plan via subprocess ──────────────────────────────────────────────

    def _plan_raw(self, qpos_start, target_pose, log=True) -> dict:
        """Send plan request to worker and parse response."""
        qpos     = np.asarray(qpos_start,  dtype=np.float64).ravel()[:self.n_arm]
        pose_arr = np.asarray(target_pose, dtype=np.float64).ravel()[:7]

        if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(pose_arr)):
            return self._fail("non-finite input")

        if not self._worker_alive():
            if log:
                print("[curobo] worker not alive, restarting…")
            self._launch_worker()

        payload = {
            "qpos":          qpos.tolist(),
            "target_pose":   pose_arr.tolist(),
            "urdf_path":     self.urdf_path,
            "ee_link":       self.ee_link,
            "base_pose":     self._base_pose.tolist(),
            "joint_names":   self.joint_names,
            "num_waypoints": self.num_waypoints,
        }

        timeout = REQUEST_TIMEOUT_FIRST if self._is_first else REQUEST_TIMEOUT
        t0 = time.perf_counter()

        try:
            self._proc.stdin.write(json.dumps(payload) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            if log:
                print(f"[curobo] write failed ({e}), restarting worker")
            self._launch_worker()
            return self._fail(f"worker write error: {e}")

        response_line = self._read_line(timeout)
        elapsed = time.perf_counter() - t0

        if response_line is None:
            if not self._worker_alive():
                if log:
                    print(f"[curobo] worker died ({elapsed:.1f}s)")
                self._launch_worker()
                return self._fail("worker process died")
            else:
                if log:
                    print(f"[curobo] timeout after {elapsed:.1f}s (limit={timeout:.0f}s)")
                self._launch_worker()
                return self._fail("timeout")

        self._is_first = False

        try:
            result = json.loads(response_line)
        except json.JSONDecodeError as e:
            if log:
                print(f"[curobo] invalid JSON ({elapsed:.1f}s): {e}")
            return self._fail("invalid JSON from worker")

        status = result.get("status", "Fail")
        if status == "Success":
            pos = np.array(result["position"], dtype=np.float64)
            vel_raw = result.get("velocity", [])
            vel = (np.array(vel_raw, dtype=np.float64)
                   if vel_raw else np.zeros_like(pos))
            if vel.shape != pos.shape:
                vel = np.zeros_like(pos)
            if log:
                print(f"[curobo] Success — {pos.shape[0]} waypoints in {elapsed:.2f}s")
            return {"status": "Success", "position": pos, "velocity": vel}
        else:
            err = result.get("error", "")
            if log:
                print(f"[curobo] Fail ({elapsed:.2f}s)")
                if err:
                    print(f"[curobo] error: {err}")
            return self._fail(err)

    @staticmethod
    def _fail(reason: str = "") -> dict:
        out: dict = {
            "status":   "Fail",
            "position": np.array([]),
            "velocity": np.array([]),
        }
        if reason:
            out["error"] = reason
        return out
