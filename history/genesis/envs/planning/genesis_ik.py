"""Genesis IK-based planner — uses Genesis built-in inverse_kinematics."""

import numpy as np

try:  # torch ships with Genesis; entity.set_qpos wants a tensor
    import torch
except Exception:
    torch = None

from .base import Planner, PlanResult


class GenesisIKPlanner(Planner):
    """Planner using Genesis entity's built-in damped least-squares IK solver.

    This planner does NOT do collision-free path planning — it solves IK for
    the target pose and linearly interpolates joint values from start to goal.
    Use this when speed matters more than collision avoidance.

    Parameters
    ----------
    entity           : Genesis RigidEntity (the robot)
    ee_link_name     : end-effector link name
    n_arm            : number of arm DOFs
    arm_joint_names  : arm joint names in order
    arm_joint_index  : mapping from joint name to DOF index in entity qpos
    num_waypoints    : interpolation steps for the output trajectory
    screw_max_joint_step : per-waypoint joint-jump ceiling for
                       ``plan_screw_path`` — a larger delta means the IK
                       solver flipped branches and the screw plan is aborted
    """

    def __init__(
        self,
        entity,
        ee_link_name: str,
        n_arm: int,
        arm_joint_names: list[str],
        arm_joint_index: dict[str, int],
        num_waypoints: int = 200,
        screw_max_joint_step: float = 0.5,
        **kwargs,
    ):
        self._entity = entity
        self._ee_link_name = ee_link_name
        self.n_arm = n_arm
        self._arm_joint_names = list(arm_joint_names)
        self._arm_joint_index = dict(arm_joint_index)
        self.num_waypoints = num_waypoints
        # Per-waypoint joint-jump ceiling for plan_screw_path: a delta larger
        # than this between consecutive Cartesian waypoints means the IK
        # solver flipped branches, so the screw plan is aborted.
        self._screw_max_joint_step = float(screw_max_joint_step)

        # Resolve EE link
        self._ee_link = None
        for link in entity.links:
            if link.name == ee_link_name:
                self._ee_link = link
                break
        if self._ee_link is None:
            raise ValueError(f"EE link '{ee_link_name}' not found in entity")

    def _to_numpy(self, arr):
        if hasattr(arr, "cpu"):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    def _solve(self, current_qpos, target_pose, log=True):
        """Run Genesis IK and return (q_start, q_goal) arm-only arrays, or None on failure."""
        target7 = np.asarray(target_pose, dtype=np.float64).ravel()[:7]
        pos_ik = target7[:3]
        quat_ik = target7[3:7]

        # Build init_qpos with current arm values
        raw_qpos = self._entity.get_qpos()
        init_full = self._to_numpy(raw_qpos).ravel().copy()

        for i, jname in enumerate(self._arm_joint_names[:self.n_arm]):
            if jname in self._arm_joint_index and i < len(current_qpos):
                init_full[self._arm_joint_index[jname]] = float(current_qpos[i])

        ik_fn = getattr(self._entity, "inverse_kinematics", None)
        if ik_fn is None:
            if log:
                print("[genesis_ik] entity has no inverse_kinematics method")
            return None

        try:
            # max_samples=1 → a single seeded damped-LS solve, no random
            # multi-start. Genesis IK defaults to max_samples=50 random
            # restarts, which makes the regular plan_path / solve_ik path
            # nondeterministic across runs even with task seed. Mirrors
            # what plan_screw_path:308 already does.
            qpos_goal = ik_fn(
                link=self._ee_link,
                pos=pos_ik,
                quat=quat_ik,
                init_qpos=init_full,
                max_samples=1,
            )
        except TypeError:
            # Older Genesis IK signature without max_samples kwarg.
            try:
                qpos_goal = ik_fn(
                    link=self._ee_link, pos=pos_ik, quat=quat_ik,
                    init_qpos=init_full,
                )
            except TypeError:
                qpos_goal = ik_fn(link=self._ee_link, pos=pos_ik, quat=quat_ik)
        except Exception as e:
            if log:
                print(f"[genesis_ik] IK exception: {e}")
            return None

        if qpos_goal is None:
            if log:
                print("[genesis_ik] IK returned None")
            return None

        qpos_goal = self._to_numpy(qpos_goal).ravel()

        # Extract arm joint values
        arm_cols = [self._arm_joint_index[jn]
                    for jn in self._arm_joint_names
                    if jn in self._arm_joint_index]
        q_goal = np.array([qpos_goal[idx] for idx in arm_cols[:self.n_arm]],
                          dtype=np.float64)
        q_start = np.array([current_qpos[i] if i < len(current_qpos) else 0.0
                            for i in range(self.n_arm)], dtype=np.float64)
        return q_start, q_goal

    def _interpolate(self, q_start, q_goal, num_waypoints=None):
        """Linear interpolation from q_start to q_goal."""
        n_wp = max(2, num_waypoints or self.num_waypoints)
        t = np.linspace(0, 1, n_wp)
        position = (1 - t[:, None]) * q_start + t[:, None] * q_goal
        dt = 1.0 / 250.0
        velocity = np.zeros_like(position)
        if position.shape[0] > 1:
            velocity[:-1] = np.diff(position, axis=0) / dt
        velocity = np.clip(velocity, -1.0, 1.0)
        return position, velocity

    def plan_path(
        self,
        current_qpos: np.ndarray,
        target_pose: np.ndarray,
        log: bool = True,
        **kwargs,
    ) -> PlanResult:
        """Solve IK and interpolate to target pose (no collision avoidance)."""
        result = self._solve(current_qpos, target_pose, log=log)
        if result is None:
            return PlanResult(False)

        q_start, q_goal = result
        position, velocity = self._interpolate(q_start, q_goal)

        if log:
            print(f"[genesis_ik] Success, waypoints={position.shape[0]}")
        return PlanResult(True, position, velocity)

    def solve_ik(
        self,
        current_qpos: np.ndarray,
        target_pose: np.ndarray,
        num_waypoints: int = None,
        log: bool = True,
        **kwargs,
    ) -> PlanResult:
        """Solve IK and return interpolated trajectory."""
        result = self._solve(current_qpos, target_pose, log=log)
        if result is None:
            return PlanResult(False)

        q_start, q_goal = result
        position, velocity = self._interpolate(q_start, q_goal, num_waypoints)

        if log:
            print(f"[genesis_ik] IK Success, waypoints={position.shape[0]}")
        return PlanResult(True, position, velocity)

    @staticmethod
    def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between two (w,x,y,z) quaternions."""
        q0 = np.asarray(q0, dtype=np.float64)
        q1 = np.asarray(q1, dtype=np.float64)
        n0, n1 = np.linalg.norm(q0), np.linalg.norm(q1)
        if n0 > 0:
            q0 = q0 / n0
        if n1 > 0:
            q1 = q1 / n1
        dot = float(np.dot(q0, q1))
        if dot < 0.0:  # take the shorter great-circle arc
            q1 = -q1
            dot = -dot
        if dot > 0.9995:  # nearly parallel — lerp + renormalize
            q = q0 + t * (q1 - q0)
            nq = np.linalg.norm(q)
            return q / nq if nq > 0 else q0
        theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
        theta = theta0 * t
        s0 = np.sin(theta0 - theta) / np.sin(theta0)
        s1 = np.sin(theta) / np.sin(theta0)
        return s0 * q0 + s1 * q1

    def _set_entity_qpos(self, qpos):
        """``entity.set_qpos`` accepting a numpy array (wraps to torch)."""
        q = qpos
        if not hasattr(q, "cpu"):
            if torch is not None:
                q = torch.tensor(np.asarray(qpos, dtype=np.float32))
            else:
                q = np.asarray(qpos, dtype=np.float32)
        self._entity.set_qpos(q)

    def _ee_pose_at(self, full_qpos, restore_qpos):
        """Forward kinematics: (pos, quat) of the EE link at ``full_qpos``.

        Genesis exposes no side-effect-free FK on a RigidEntity, so this
        briefly sets the entity to ``full_qpos``, reads the link, then restores
        ``restore_qpos`` — the same set/restore pattern base_task.grasp_pick
        uses for its post-screw FK check.
        """
        self._set_entity_qpos(full_qpos)
        self._entity.get_links_pos()
        pos = self._to_numpy(self._ee_link.get_pos()).ravel()[:3]
        quat = self._to_numpy(self._ee_link.get_quat()).ravel()[:4]
        self._set_entity_qpos(restore_qpos)
        self._entity.get_links_pos()
        return pos, quat

    def plan_screw_path(
        self,
        current_qpos: np.ndarray,
        target_pose: np.ndarray,
        num_waypoints: int = None,
        log: bool = True,
        **kwargs,
    ) -> PlanResult:
        """Straight-line Cartesian motion — the genesis_ik counterpart of
        ``MplibPlanner.plan_screw_path``.

        ``plan_path`` / ``solve_ik`` interpolate in *joint* space, so the EE
        bows off a straight line — it can swing laterally and knock objects on
        descents/inserts. This instead interpolates the EE *pose* (lerp
        position, slerp orientation) from the current pose to ``target_pose``
        and solves IK at each step, seeding every solve from the previous
        waypoint's solution so the joint trajectory stays on one IK branch.

        The start pose is read from the live entity link, which assumes the
        entity is at ``current_qpos`` — true for the manipulation-mixin callers
        that pass ``arm.get_arm_qpos()``. Returns ``PlanResult(False)`` if any
        waypoint IK fails or the solver flips branches, so the caller can fall
        back to a seeded joint-space plan.
        """
        target7 = np.asarray(target_pose, dtype=np.float64).ravel()[:7]
        goal_pos = target7[:3]
        goal_quat = target7[3:7]

        ik_fn = getattr(self._entity, "inverse_kinematics", None)
        if ik_fn is None:
            if log:
                print("[genesis_ik] entity has no inverse_kinematics method")
            return PlanResult(False)

        n_wp = max(2, num_waypoints or self.num_waypoints)
        ts = np.linspace(0.0, 1.0, n_wp)

        arm_cols = [self._arm_joint_index[jn]
                    for jn in self._arm_joint_names
                    if jn in self._arm_joint_index][:self.n_arm]
        q_prev = np.array([current_qpos[i] if i < len(current_qpos) else 0.0
                           for i in range(self.n_arm)], dtype=np.float64)

        # Full entity qpos baseline; arm columns are overwritten per waypoint
        # with the running solution so each IK solve is seeded from the last.
        saved_qpos = self._to_numpy(self._entity.get_qpos()).ravel().copy()
        init_full = saved_qpos.copy()
        for col, val in zip(arm_cols, q_prev):
            init_full[col] = val

        # Start pose = FK of `current_qpos`, NOT the live entity pose. Callers
        # such as base_task.grasp_pick pass a *planned* config (`end_pre_qpos`)
        # the entity has not been moved to yet, so reading the live link would
        # interpolate from the wrong pose and the continuity guard would trip
        # on the very first waypoint.
        start_pos, start_quat = self._ee_pose_at(init_full, saved_qpos)

        positions = [q_prev.copy()]

        # Adaptive Cartesian sub-stepping. A "branch flip" — the per-waypoint
        # joint jump exceeding `_screw_max_joint_step` — means the seeded IK
        # snapped to a different solution between two waypoints. Rather than
        # aborting the whole screw (which forces callers onto a weak joint-
        # space fallback that bows the EE off the straight line), bisect the
        # offending [t_prev, t] segment and re-solve at finer resolution — the
        # genesis_ik analogue of MplibPlanner._move_screw's qpos_step
        # 0.1→0.05→0.02 retry. Give up only when even a near-zero Cartesian
        # step still flips (a genuine kinematic singularity on the path).
        nominal_seg = 1.0 / max(1, n_wp - 1)
        min_seg = nominal_seg / 16.0          # finest sub-step before abort
        max_wp = 8 * n_wp                     # guard against runaway bisection
        t_prev = 0.0
        targets = [float(t) for t in ts[1:]]  # ascending list of waypoint t's
        n_subdiv = 0

        while targets:
            t = targets[0]
            pos_t = (1.0 - t) * start_pos + t * goal_pos
            quat_t = self._slerp(start_quat, goal_quat, float(t))
            for col, val in zip(arm_cols, q_prev):
                init_full[col] = val
            # max_samples=1 → a single seeded damped-LS solve, no random
            # multi-start. Genesis IK defaults to max_samples=50 restarts,
            # which jump IK branches even for a target near the seed and so
            # break screw continuity. Per-waypoint Cartesian steps are tiny,
            # so one seeded solve converges locally and stays on-branch.
            try:
                sol = ik_fn(link=self._ee_link, pos=pos_t, quat=quat_t,
                            init_qpos=init_full, max_samples=1)
            except TypeError:
                sol = ik_fn(link=self._ee_link, pos=pos_t, quat=quat_t,
                            init_qpos=init_full)
            except Exception as e:
                if log:
                    print(f"[genesis_ik] screw IK exception at t={t:.2f}: {e}")
                return PlanResult(False)
            if sol is None:
                if log:
                    print(f"[genesis_ik] screw IK returned None at t={t:.2f}")
                return PlanResult(False)
            sol = self._to_numpy(sol).ravel()
            q_t = np.array([sol[col] for col in arm_cols], dtype=np.float64)
            jump = float(np.max(np.abs(q_t - q_prev)))

            if jump <= self._screw_max_joint_step:
                positions.append(q_t)
                q_prev = q_t
                t_prev = t
                targets.pop(0)
                continue

            # Branch flip — bisect [t_prev, t] and retry the midpoint first.
            if (t - t_prev) <= min_seg or len(positions) >= max_wp:
                if log:
                    print(f"[genesis_ik] screw IK branch flip at t={t:.3f} "
                          f"(delta={jump:.2f} rad > {self._screw_max_joint_step}); "
                          f"segment {t - t_prev:.4f} <= {min_seg:.4f}, abort")
                return PlanResult(False)
            targets.insert(0, 0.5 * (t_prev + t))
            n_subdiv += 1

        if log and n_subdiv:
            print(f"[genesis_ik] screw sub-stepped {n_subdiv}x to hold continuity")

        position = np.asarray(positions, dtype=np.float64)
        dt = 1.0 / 250.0
        velocity = np.zeros_like(position)
        if position.shape[0] > 1:
            velocity[:-1] = np.diff(position, axis=0) / dt
        velocity = np.clip(velocity, -1.0, 1.0)

        if log:
            print(f"[genesis_ik] screw Success, waypoints={position.shape[0]}")
        return PlanResult(True, position, velocity)
