"""MPLib-based motion planner with TOPP-RA trajectory smoothing."""

import os
import numpy as np
import mplib
import mplib.pymp as pymp
import toppra as ta
import time
from typing import Optional

from .base import Planner, PlanResult

ta.setup_logging("CRITICAL")


class MplibPlanner(Planner):
    """Motion planner using MPLib (RRT-Connect) with TOPP-RA smoothing."""

    def __init__(
        self,
        urdf_path: str,
        srdf_path: str,
        move_group: str,
        base_pose: np.ndarray,
        entity=None,
        n_arm: int = 6,
        num_waypoints: int = 200,
        timeout: float = None,
        max_retry: int = 5,
        arm_joint_names: list[str] = None,
        arm_joint_index: dict[str, int] = None,
        entity_ee_link_name: str = None,
        package_keyword: str = "",
    ):
        self.n_arm = n_arm
        self.num_waypoints = num_waypoints
        self.max_retry = max_retry
        self._timeout = timeout
        self._entity = entity
        self._ee_link_name = move_group
        self._entity_ee_link_name = entity_ee_link_name or move_group
        self._arm_joint_names = list(arm_joint_names) if arm_joint_names else None
        self._arm_joint_index = dict(arm_joint_index) if arm_joint_index else None

        # Build mplib planner
        links = [link.name for link in entity.links] if entity else []
        joints = [joint.name for joint in entity.joints] if entity else []

        if "base_link" not in links:
            self._mplib = mplib.Planner(
                urdf=urdf_path, srdf=srdf_path,
                move_group=move_group, use_convex=False,
                new_package_keyword=package_keyword,
            )
        else:
            self._mplib = mplib.Planner(
                urdf=urdf_path, srdf=srdf_path,
                move_group=move_group,
                user_link_names=links, user_joint_names=joints,
                use_convex=False,
                new_package_keyword=package_keyword,
            )

        # Set base pose
        pose7 = np.asarray(base_pose, dtype=np.float64).ravel()
        p = pose7[:3].astype(np.float64)
        q = pose7[3:7].astype(np.float64)
        self._mplib.set_base_pose(pymp.Pose(p=p, q=q))
        self._base_pose = pose7.copy()

    def _to_mplib_pose(self, target_pose) -> pymp.Pose:
        """Convert target pose to mplib Pose."""
        arr = np.asarray(target_pose, dtype=np.float64).ravel()
        if hasattr(target_pose, "p"):
            arr = np.concatenate([
                np.asarray(target_pose.p).ravel(),
                np.asarray(target_pose.q).ravel(),
            ])
        p = arr[:3].astype(np.float64)
        q = arr[3:7].astype(np.float64)
        return pymp.Pose(p=p, q=q)

    def _pad_qpos(self, qpos: np.ndarray) -> np.ndarray:
        """Pad qpos to match mplib expected size."""
        qpos = np.asarray(qpos, dtype=np.float64)
        try:
            n_user = len(self._mplib.robot.get_user_joint_names())
            if qpos.shape[0] < n_user:
                qpos = np.concatenate([qpos, np.zeros(n_user - qpos.shape[0], dtype=np.float64)])
        except Exception:
            pass
        return qpos

    @staticmethod
    def _bounded_paths_debug_enabled() -> bool:
        """Whether to use the experimental bounded-joint planning path.

        This is deliberately an environment flag rather than a default config
        option while the implementation is being validated.  With the flag
        absent, planner behavior is byte-for-byte the existing MPLib path.
        """
        return os.environ.get("DEBUG_BOUNDED_JOINT_PATHS", "").strip() == "1"

    @staticmethod
    def _joint_path_metrics(
        current_qpos: np.ndarray,
        position: np.ndarray,
        n_arm: int,
    ) -> dict[str, np.ndarray | float]:
        """Return joint-space path-quality metrics for one arm trajectory.

        Per-step continuity is not enough for a redundant arm: a path can use
        hundreds of tiny steps to wind a wrist through its complete range.
        The cumulative travel and start-relative excursion expose that failure
        without assuming which joint is redundant.
        """
        q0 = np.asarray(current_qpos, dtype=np.float64).ravel()[:n_arm]
        pos = np.asarray(position, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[0] == 0 or pos.shape[1] < n_arm:
            return {
                "travel": np.full(n_arm, np.inf),
                "excursion": np.full(n_arm, np.inf),
                "max_step": np.inf,
                "weighted_length": np.inf,
            }
        arm = pos[:, :n_arm]
        if not np.allclose(arm[0], q0, atol=1e-6, rtol=0.0):
            arm = np.vstack([q0, arm])
        delta = np.diff(arm, axis=0)
        return {
            "travel": np.sum(np.abs(delta), axis=0),
            "excursion": np.max(np.abs(arm - q0[None, :]), axis=0),
            "max_step": float(np.max(np.abs(delta))) if delta.size else 0.0,
            "weighted_length": float(np.sum(np.linalg.norm(delta, axis=1))),
        }

    def _debug_path_quality_ok(
        self,
        current_qpos: np.ndarray,
        position: np.ndarray,
        *,
        motion_kind: str,
        log: bool,
    ) -> bool:
        """Experimental hard gate against globally unreasonable joint paths."""
        metrics = self._joint_path_metrics(current_qpos, position, self.n_arm)
        travel = np.asarray(metrics["travel"], dtype=np.float64)
        excursion = np.asarray(metrics["excursion"], dtype=np.float64)

        # Local Cartesian motions should remain on the starting IK branch.
        # Global RRT transits get a wider allowance for obstacle detours, but
        # still may not sweep a bounded Franka joint through most of its range.
        if motion_kind == "screw":
            max_excursion = 1.75  # 100 deg on any one joint
            max_travel = 2.25     # permits modest backtracking, not winding
        else:
            max_excursion = 2.80  # 160 deg
            max_travel = 3.50     # 200 deg cumulative

        bad = np.flatnonzero(
            (excursion > max_excursion) | (travel > max_travel)
        )
        if bad.size:
            if log:
                details = ", ".join(
                    f"j{i + 1}: excursion={excursion[i]:.3f}, "
                    f"travel={travel[i]:.3f}rad"
                    for i in bad
                )
                print(
                    f"[bounded_path] reject {motion_kind} trajectory ({details})"
                )
            return False
        return True

    def _plan_bounded_screw(
        self,
        current_qpos: np.ndarray,
        target_pose: np.ndarray,
        *,
        time_step: float = 1 / 250,
        qpos_step: float = 0.05,
        log: bool = True,
    ) -> PlanResult:
        """Plan a branch-preserving Cartesian path with MPLib kinematics.

        MPLib's stock ``plan_screw`` integrates ``pinv(J) @ twist``.  That is
        minimum-norm for each individual step, but it has no memory of total
        joint travel and no cost for approaching a joint limit.  On a 7-DOF
        arm it can therefore drift through the null space for several full
        radians while the TCP orientation barely changes.

        This experimental version keeps all kinematics and collision checking
        inside MPLib, but uses a joint-limit-weighted damped pseudoinverse and
        rejects any path that leaves a bounded neighborhood of its starting IK
        branch.  It is reachable only through ``DEBUG_BOUNDED_JOINT_PATHS=1``.
        """
        backend = self._mplib
        q_full = backend.pad_move_group_qpos(
            np.asarray(current_qpos, dtype=np.float64).copy()
        )
        backend.robot.set_qpos(q_full, True)

        goal = self._to_mplib_pose(target_pose)
        goal = backend._transform_goal_to_wrt_base(goal)
        pin = backend.pinocchio_model
        ee_index = backend.link_name_2_idx[backend.move_group]
        move_idx = np.asarray(backend.move_group_joint_indices, dtype=int)
        q_start = q_full.copy()

        def _skew(vec):
            return np.array([
                [0.0, -vec[2], vec[1]],
                [vec[2], 0.0, -vec[0]],
                [-vec[1], vec[0], 0.0],
            ])

        def _pose_log(pose):
            matrix = pose.to_transformation_matrix()
            rotation = matrix[:3, :3]
            trace_arg = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
            theta = float(np.arccos(trace_arg))
            if theta < 1e-9:
                return np.concatenate([matrix[:3, 3], np.zeros(3)])
            if np.pi - theta < 1e-6:
                raise ValueError("180-degree screw rotation is ill-conditioned")
            omega_axis = np.array([
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]) / (2.0 * np.sin(theta))
            omega_hat = _skew(omega_axis)
            inv_left_jacobian = (
                np.eye(3) / theta
                - 0.5 * omega_hat
                + (1.0 / theta - 0.5 / np.tan(theta / 2.0))
                * omega_hat @ omega_hat
            )
            linear = inv_left_jacobian @ matrix[:3, 3]
            return np.concatenate([linear, omega_axis]) * theta

        pin.compute_forward_kinematics(q_full)
        relative = goal * pin.get_link_pose(ee_index).inv()
        try:
            remaining = _pose_log(relative).reshape(-1)
        except ValueError as exc:
            if log:
                print(f"[bounded_path] screw rejected: {exc}")
            return PlanResult(False)

        limits = np.asarray(backend.joint_limits, dtype=np.float64)
        lo = limits[move_idx, 0]
        hi = limits[move_idx, 1]
        width = np.maximum(hi - lo, 1e-6)
        start_arm = q_start[move_idx]
        q_arm = start_arm.copy()
        path = [q_arm.copy()]
        max_excursion = 1.75
        damping = 0.03

        for _ in range(2000):
            pin.compute_full_jacobian(q_full)
            jac_full = np.asarray(pin.get_link_jacobian(ee_index, local=False))
            jac = jac_full[:, move_idx]

            # Penalize joints close to either hard limit and joints that have
            # already consumed much of the local motion budget.  Weighted DLS
            # then satisfies the Cartesian twist using less constrained joints.
            limit_fraction = np.minimum(q_arm - lo, hi - q_arm) / width
            limit_weight = 1.0 + 0.02 / np.square(np.maximum(limit_fraction, 0.02))
            used_fraction = np.abs(q_arm - start_arm) / max_excursion
            excursion_weight = 1.0 + 8.0 * np.power(used_fraction, 4)
            weight_inv = np.diag(1.0 / (limit_weight * excursion_weight))
            task_matrix = jac @ weight_inv @ jac.T + damping ** 2 * np.eye(6)
            try:
                delta = weight_inv @ jac.T @ np.linalg.solve(task_matrix, remaining)
            except np.linalg.LinAlgError:
                if log:
                    print("[bounded_path] screw rejected: singular weighted Jacobian")
                return PlanResult(False)

            delta_norm = float(np.linalg.norm(delta))
            if not np.isfinite(delta_norm) or delta_norm < 1e-10:
                if log:
                    print("[bounded_path] screw rejected: no Cartesian progress")
                return PlanResult(False)
            if delta_norm > qpos_step:
                delta *= qpos_step / delta_norm

            achieved = jac @ delta
            achieved_norm = float(np.linalg.norm(achieved))
            remaining_norm = float(np.linalg.norm(remaining))
            if achieved_norm > remaining_norm and achieved_norm > 0.0:
                ratio = remaining_norm / achieved_norm
                delta *= ratio
                achieved *= ratio

            candidate = q_arm + delta
            if (
                np.any(candidate < lo - 1e-3)
                or np.any(candidate > hi + 1e-3)
                or np.any(np.abs(candidate - start_arm) > max_excursion)
            ):
                if log:
                    print("[bounded_path] screw rejected: joint branch budget exceeded")
                return PlanResult(False)

            q_full[move_idx] = candidate
            backend.planning_world.set_qpos_all(candidate)
            if backend.planning_world.is_state_colliding():
                if log:
                    print("[bounded_path] screw rejected: collision")
                return PlanResult(False)

            q_arm = candidate
            path.append(q_arm.copy())
            remaining -= achieved

            if float(np.linalg.norm(remaining)) < 1e-4:
                break
        else:
            if log:
                print("[bounded_path] screw rejected: iteration limit")
            return PlanResult(False)

        # Verify the integrated endpoint in the same MPLib FK model before
        # time-parameterizing it.  Pose.distance combines translation and
        # quaternion error and is the same metric MPLib's IK uses.
        pin.compute_forward_kinematics(q_full)
        endpoint_error = float(goal.distance(pin.get_link_pose(ee_index)))
        # The weighted DLS integration converges to about 11 mm on long
        # workspace-spanning translations in the real Franka model.  Keep the
        # debug gate well inside the shared grasp helper's 30 mm FK tolerance,
        # but do not discard a bounded Cartesian path for a 1 mm numerical
        # miss and replace it with a less constrained global transit.
        endpoint_tolerance = 0.015
        if endpoint_error > endpoint_tolerance:
            if log:
                print(
                    f"[bounded_path] screw rejected: endpoint error "
                    f"{endpoint_error:.4f} > {endpoint_tolerance:.3f}"
                )
            return PlanResult(False)

        try:
            _times, pos, vel, _acc, _duration = backend.TOPP(
                np.vstack(path), time_step
            )
        except Exception as exc:
            if log:
                print(f"[bounded_path] screw TOPP failed: {exc}")
            return PlanResult(False)

        if not self._debug_path_quality_ok(
            current_qpos, pos, motion_kind="screw", log=log
        ):
            return PlanResult(False)
        if log:
            metrics = self._joint_path_metrics(current_qpos, pos, self.n_arm)
            print(
                "[bounded_path] screw accepted: max excursion="
                f"{np.max(metrics['excursion']):.3f}rad, max travel="
                f"{np.max(metrics['travel']):.3f}rad"
            )
        return PlanResult(True, pos, vel)

    def _plan_bounded_transit(
        self,
        current_qpos: np.ndarray,
        target_pose: np.ndarray,
        *,
        log: bool = True,
    ) -> PlanResult:
        """Plan to the closest reasonable MPLib IK branch in debug mode.

        Stock ``plan_pose`` passes every sampled IK goal to RRT-Connect at
        once.  RRT-Connect stops at the first branch it connects, even when a
        much closer goal configuration was sampled in the same call.  Rank
        those collision-checked MPLib IK solutions in joint space and plan to
        them one at a time so feasibility does not override path quality.
        """
        backend = self._mplib
        q_full = backend.pad_move_group_qpos(
            np.asarray(current_qpos, dtype=np.float64).copy()
        )
        goal = self._to_mplib_pose(target_pose)
        goal = backend._transform_goal_to_wrt_base(goal)

        status, goals = backend.IK(
            goal_pose=goal,
            start_qpos=q_full,
            n_init_qpos=max(20, 5 * self.max_retry),
            threshold=0.001,
            return_closest=False,
            verbose=log,
        )
        if status != "Success" or goals is None:
            if log:
                print(f"[bounded_path] transit IK failed: {status}")
            return PlanResult(False)

        goals = np.asarray(goals, dtype=np.float64)
        if goals.ndim == 1:
            goals = goals[None, :]
        move_idx = np.asarray(backend.move_group_joint_indices, dtype=int)
        q_arm = q_full[move_idx]
        limits = np.asarray(backend.joint_limits, dtype=np.float64)[move_idx]
        width = np.maximum(limits[:, 1] - limits[:, 0], 1e-6)

        candidate_scores = []
        for index, candidate in enumerate(goals):
            candidate_arm = candidate[move_idx]
            direct = float(np.linalg.norm(candidate_arm - q_arm))
            margin = np.minimum(
                candidate_arm - limits[:, 0], limits[:, 1] - candidate_arm
            ) / width
            # Prefer nearby branches, then break close ties in favor of goals
            # with more room before a hard limit.  The limit term is small so
            # it cannot make a distant IK branch look artificially attractive.
            limit_penalty = float(np.sum(1.0 / np.maximum(margin, 0.02)))
            candidate_scores.append((direct + 0.002 * limit_penalty, index))

        for score, index in sorted(candidate_scores):
            candidate = goals[index]
            try:
                result = backend.plan_qpos(
                    goal_qposes=[candidate],
                    current_qpos=q_full,
                    time_step=1 / 250,
                    planning_time=self._timeout or 10,
                    rrt_range=0.1,
                    verbose=log,
                )
            except Exception as exc:
                if log:
                    print(f"[bounded_path] transit plan_qpos exception: {exc}")
                continue
            if not result or result.get("status") != "Success":
                continue
            pos = np.asarray(result["position"])
            if not self._debug_path_quality_ok(
                current_qpos, pos, motion_kind="transit", log=log
            ):
                continue
            vel = result.get("velocity", np.zeros_like(pos))
            if log:
                metrics = self._joint_path_metrics(current_qpos, pos, self.n_arm)
                print(
                    f"[bounded_path] transit accepted IK score={score:.3f}: "
                    f"max excursion={np.max(metrics['excursion']):.3f}rad, "
                    f"max travel={np.max(metrics['travel']):.3f}rad"
                )
            return PlanResult(True, pos, vel)

        if log:
            print("[bounded_path] no bounded transit candidate")
        return PlanResult(False)

    def plan_path(
        self,
        current_qpos: np.ndarray,
        target_pose: np.ndarray,
        log: bool = True,
        **kwargs,
    ) -> PlanResult:
        """Plan collision-free path using RRT-Connect.

        When DEBUG_TELEPORT=1, skips RRT and uses IK with 1 waypoint for fast debugging.
        """
        # DEBUG_TELEPORT: skip full planning, just IK teleport
        if os.environ.get("DEBUG_TELEPORT", "").strip() == "1":
            if log:
                print("[planner] DEBUG_TELEPORT=1, using IK teleport")
            return self.solve_ik(current_qpos, target_pose, num_waypoints=1, log=log)

        if self._bounded_paths_debug_enabled():
            return self._plan_bounded_transit(current_qpos, target_pose, log=log)

        goal = self._to_mplib_pose(target_pose)
        qpos = self._pad_qpos(current_qpos)

        result = None
        for attempt in range(self.max_retry):
            t0 = time.perf_counter()
            try:
                result = self._mplib.plan_pose(
                    goal_pose=goal,
                    current_qpos=qpos,
                    time_step=1 / 250,
                    planning_time=self._timeout or 10,
                    wrt_world=True,
                    verbose=log,
                )
            except Exception as e:
                if log:
                    print(f"[planner] MPLib exception: {e}")
                result = {"status": "Fail"}
            elapsed = time.perf_counter() - t0

            if result and result.get("status") == "Success":
                pos = result["position"]
                if self._bounded_paths_debug_enabled() and not self._debug_path_quality_ok(
                    current_qpos, pos, motion_kind="transit", log=log
                ):
                    # A feasibility planner is allowed to return a needlessly
                    # circuitous RRT/IK branch.  In debug mode, treat that as a
                    # rejected candidate and let the normal retry loop sample
                    # another solution instead of executing it.
                    continue
                if log:
                    print(f"[planner] MPLib Success ({elapsed:.2f}s, attempt {attempt + 1})")
                vel = result.get("velocity", np.zeros_like(pos))
                return PlanResult(True, pos, vel)
            else:
                if log:
                    status = result.get("status", "Unknown") if result else "None"
                    print(f"[planner] MPLib attempt {attempt + 1} failed: {status} ({elapsed:.2f}s)")

        if log:
            print("[planner] MPLib failed all attempts — skipping (no Genesis IK fallback)")
        return PlanResult(False)

    # NOTE: _plan_with_genesis_ik is intentionally disabled — confirmed unused
    # 2026-05-02. plan_path() is RRT-only (mplib.plan_pose); on failure it returns
    # PlanResult(False) without any silent fallback. Kept here as a reference
    # implementation in case a future planner needs Genesis-IK seeding, but DO
    # NOT call from plan_path or any production path — the whole point of
    # select_and_execute_grasp is that leg 1 is pure RRT with no surprises.
    def _plan_with_genesis_ik_DISABLED(self, current_qpos, goal_pose, log=True):
        """[DISABLED] Fallback: Genesis IK + mplib plan_qpos. Not wired to plan_path."""
        import torch as _torch

        # Find EE link
        ee_link = None
        for link in self._entity.links:
            if link.name == self._entity_ee_link_name:
                ee_link = link
                break
        if ee_link is None:
            return None

        target_pos = np.array(goal_pose.p).ravel()[:3]
        target_quat = np.array(goal_pose.q).ravel()[:4]

        ik_fn = getattr(self._entity, "inverse_kinematics", None)
        if ik_fn is None:
            return None

        raw = self._entity.get_qpos()
        init_full = raw.detach().cpu().numpy().ravel() if hasattr(raw, "cpu") else np.asarray(raw).ravel()
        # Write current arm joints into init_full
        if self._arm_joint_names and self._arm_joint_index:
            for i, jname in enumerate(self._arm_joint_names[:self.n_arm]):
                if jname in self._arm_joint_index and i < len(current_qpos):
                    init_full[self._arm_joint_index[jname]] = float(current_qpos[i])

        # Get joint limits for random seed generation and clamping
        joint_limits = None
        try:
            pm = self._mplib.robot.get_pinocchio_model()
            joint_limits = pm.get_joint_limits()
        except Exception:
            pass

        # Try multiple IK seeds, keep best (lowest FK error)
        best_qpos = None
        best_fk_err = float("inf")
        n_ik_attempts = 10

        for ik_attempt in range(n_ik_attempts):
            try:
                if ik_attempt == 0:
                    seed_qpos = init_full.copy()
                else:
                    # Random perturbation of arm joints
                    seed_qpos = init_full.copy()
                    if self._arm_joint_names and self._arm_joint_index:
                        for i, jname in enumerate(self._arm_joint_names[:self.n_arm]):
                            if jname in self._arm_joint_index:
                                idx = self._arm_joint_index[jname]
                                if joint_limits is not None and i < len(joint_limits):
                                    lo, hi = joint_limits[i][0][0], joint_limits[i][0][1]
                                    seed_qpos[idx] = np.random.uniform(lo, hi)
                                else:
                                    seed_qpos[idx] += np.random.uniform(-1.0, 1.0)

                try:
                    qpos_sol = ik_fn(link=ee_link, pos=target_pos, quat=target_quat,
                                     init_qpos=seed_qpos)
                except TypeError:
                    qpos_sol = ik_fn(link=ee_link, pos=target_pos, quat=target_quat)

                if qpos_sol is None:
                    continue
                qpos_sol = qpos_sol.detach().cpu().numpy().ravel() if hasattr(qpos_sol, "cpu") else np.asarray(qpos_sol).ravel()

                # Clamp to joint limits
                if joint_limits is not None:
                    for i, lim in enumerate(joint_limits):
                        lo, hi = lim[0][0], lim[0][1]
                        margin = 0.01
                        if i < len(qpos_sol):
                            qpos_sol[i] = np.clip(qpos_sol[i], lo + margin, hi - margin)

                # Extract arm joints
                if self._arm_joint_names and self._arm_joint_index:
                    arm_cols = [self._arm_joint_index[jn] for jn in self._arm_joint_names if jn in self._arm_joint_index]
                    qpos_arm = np.array([qpos_sol[idx] for idx in arm_cols[:self.n_arm]], dtype=np.float64)
                else:
                    qpos_arm = qpos_sol[:self.n_arm].copy()

                # FK verification
                saved_qpos = self._entity.get_qpos()
                verify_qpos = saved_qpos.detach().cpu().numpy().ravel().copy() if hasattr(saved_qpos, "cpu") else np.asarray(saved_qpos).ravel().copy()
                if self._arm_joint_names and self._arm_joint_index:
                    for i, jname in enumerate(self._arm_joint_names[:self.n_arm]):
                        if jname in self._arm_joint_index and i < len(qpos_arm):
                            verify_qpos[self._arm_joint_index[jname]] = float(qpos_arm[i])
                qpos_tensor = _torch.tensor(verify_qpos, dtype=_torch.float32)
                self._entity.set_qpos(qpos_tensor)
                self._entity.get_links_pos()
                fk_pos = ee_link.get_pos().detach().cpu().numpy().ravel()[:3]
                fk_err = float(np.linalg.norm(fk_pos - target_pos))
                self._entity.set_qpos(saved_qpos)

                if fk_err < best_fk_err:
                    best_fk_err = fk_err
                    best_qpos = qpos_arm.copy()
                    if fk_err < 0.01:  # < 1cm is excellent, stop early
                        break
            except Exception:
                continue

        if best_qpos is None:
            if log:
                print("[planner] Genesis IK returned None for all seeds")
            return None

        qpos_goal = best_qpos
        if log:
            print(f"[planner] Genesis IK FK error: {best_fk_err:.4f}m (best of {n_ik_attempts} attempts)")
        if best_fk_err > 0.10:  # reject if > 10cm
            if log:
                print(f"[planner] Genesis IK rejected: FK error {best_fk_err:.4f}m > 0.10m")
            return None

        # Pad goal and current to match mplib user joints
        goal_padded = self._pad_qpos(qpos_goal)
        current_padded = self._pad_qpos(current_qpos)

        # mplib plan_qpos (collision-free RRT)
        for attempt in range(self.max_retry):
            t0 = time.perf_counter()
            try:
                result = self._mplib.plan_qpos(
                    goal_qposes=[goal_padded],
                    current_qpos=current_padded,
                    time_step=1 / 250,
                    planning_time=self._timeout or 10,
                    rrt_range=0.3,
                    verbose=log,
                )
            except Exception as e:
                if log:
                    print(f"[planner] plan_qpos exception: {e}")
                result = {"status": "Fail"}
            elapsed = time.perf_counter() - t0

            if result and result.get("status") == "Success":
                if log:
                    print(f"[planner] Genesis IK + plan_qpos Success ({elapsed:.2f}s, attempt {attempt + 1})")
                pos = result["position"]
                vel = result.get("velocity", np.zeros_like(pos))
                return PlanResult(True, pos, vel)

        if log:
            print("[planner] Genesis IK + plan_qpos failed")
        return None

    def solve_ik(
        self,
        current_qpos: np.ndarray,
        target_pose: np.ndarray,
        num_waypoints: int = None,
        log: bool = True,
        **kwargs,
    ) -> PlanResult:
        """Solve IK using mplib (Pinocchio) and return interpolated trajectory."""
        goal = self._to_mplib_pose(target_pose)
        qpos = self._pad_qpos(current_qpos)

        # mplib.Planner.IK expects the goal in BASE frame (unlike plan_pose,
        # which exposes a wrt_world flag). Transform world→base ourselves
        # so pinocchio's CLIK seeds at the right pose; otherwise IK fails
        # whenever the robot base is offset from the world origin.
        goal = self._mplib.robot.get_base_pose().inv() * goal

        status, q_goals = self._mplib.IK(
            goal_pose=goal,
            start_qpos=qpos,
            n_init_qpos=20,
            threshold=0.001,
            return_closest=True,
        )

        if status != "Success" or q_goals is None:
            if log:
                print(f"[planner] mplib IK failed: {status}")
            return PlanResult(False)

        q_goal = np.asarray(q_goals, dtype=np.float64).ravel()[:self.n_arm]
        q_start = np.asarray(current_qpos[:self.n_arm], dtype=np.float64)

        # Wrap joint angles to shortest path (avoid 360°+ rotations)
        diff = q_goal - q_start
        diff = (diff + np.pi) % (2 * np.pi) - np.pi
        q_goal = q_start + diff

        # Interpolate (min 2 waypoints to always include goal)
        n_wp = max(2, num_waypoints or self.num_waypoints)
        t = np.linspace(0, 1, n_wp)
        position = (1 - t[:, None]) * q_start + t[:, None] * q_goal
        dt = 1.0 / 250.0
        velocity = np.zeros_like(position)
        if position.shape[0] > 1:
            velocity[:-1] = np.diff(position, axis=0) / dt
        velocity = np.clip(velocity, -1.0, 1.0)

        if log:
            print(f"[planner] IK Success, waypoints={position.shape[0]}")
        return PlanResult(True, position, velocity)

    def plan_screw_path(
        self,
        current_qpos: np.ndarray,
        target_pose: np.ndarray,
        log: bool = True,
        **kwargs,
    ) -> PlanResult:
        """Plan a straight-line Cartesian path using mplib's screw motion planner."""
        if self._bounded_paths_debug_enabled():
            return self._plan_bounded_screw(
                current_qpos,
                target_pose,
                time_step=1 / 250,
                qpos_step=0.05,
                # The opt-in flag is a diagnostic mode: always expose why a
                # bounded Cartesian candidate was accepted or rejected, even
                # when the production caller normally suppresses planner logs.
                log=True,
            )

        goal = self._to_mplib_pose(target_pose)
        qpos = self._pad_qpos(current_qpos)

        for attempt in range(self.max_retry):
            t0 = time.perf_counter()
            try:
                result = self._mplib.plan_screw(
                    goal_pose=goal,
                    current_qpos=qpos,
                    time_step=1 / 250,
                    qpos_step=0.1,
                    wrt_world=True,
                    verbose=log,
                )
            except Exception as e:
                if log:
                    print(f"[planner] plan_screw exception: {e}")
                result = {"status": "Fail"}
            elapsed = time.perf_counter() - t0

            if result and result.get("status") == "Success":
                if log:
                    print(f"[planner] plan_screw Success ({elapsed:.2f}s, attempt {attempt + 1})")
                pos = result["position"]
                vel = result.get("velocity", np.zeros_like(pos))
                return PlanResult(True, pos, vel)
            else:
                if log:
                    status = result.get("status", "Unknown") if result else "None"
                    print(f"[planner] plan_screw attempt {attempt + 1} failed: {status} ({elapsed:.2f}s)")

        if log:
            print("[planner] plan_screw failed all attempts")
        return PlanResult(False)

    def solve_ik_cartesian(
        self,
        current_qpos: np.ndarray,
        target_pose: np.ndarray,
        num_waypoints: int = None,
        log: bool = True,
        **kwargs,
    ) -> PlanResult:
        """Cartesian-space interpolation: interpolate pos/quat in task space,
        solve IK per waypoint. No collision checking — use for attached-object transport."""
        if self._entity is None:
            return PlanResult(False)

        import torch as _torch

        ee_link = None
        for link in self._entity.links:
            if link.name == self._entity_ee_link_name:
                ee_link = link
                break
        if ee_link is None:
            return PlanResult(False)

        ik_fn = getattr(self._entity, "inverse_kinematics", None)
        if ik_fn is None:
            return PlanResult(False)

        # Get current full qpos with arm joints written in
        raw_qpos = self._entity.get_qpos()
        init_full = raw_qpos.detach().cpu().numpy().ravel() if hasattr(raw_qpos, "cpu") else np.asarray(raw_qpos).ravel()
        if self._arm_joint_names and self._arm_joint_index:
            for i, jname in enumerate(self._arm_joint_names[:self.n_arm]):
                if jname in self._arm_joint_index and i < len(current_qpos):
                    init_full[self._arm_joint_index[jname]] = float(current_qpos[i])

        # Read current EE pose via FK
        saved_qpos = self._entity.get_qpos()
        self._entity.set_qpos(_torch.tensor(init_full, dtype=_torch.float32))
        self._entity.get_links_pos()
        start_pos = ee_link.get_pos().detach().cpu().numpy().ravel()[:3]
        start_quat = ee_link.get_quat().detach().cpu().numpy().ravel()[:4]
        self._entity.set_qpos(saved_qpos)

        target7 = np.asarray(target_pose, dtype=np.float64).ravel()[:7]
        end_pos, end_quat = target7[:3], target7[3:7]

        n_wp = max(2, num_waypoints or self.num_waypoints)
        n_arm = self.n_arm
        position = np.zeros((n_wp, n_arm), dtype=np.float64)

        # Normalize quaternions and ensure shortest-path slerp
        q0 = start_quat / (np.linalg.norm(start_quat) + 1e-12)
        q1 = end_quat / (np.linalg.norm(end_quat) + 1e-12)
        if np.dot(q0, q1) < 0:
            q1 = -q1

        prev_qpos = init_full.copy()
        for i, alpha in enumerate(np.linspace(0, 1, n_wp)):
            pos_i = (1 - alpha) * start_pos + alpha * end_pos
            # Slerp
            dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
            if abs(dot) > 0.9995:
                quat_i = q0 + alpha * (q1 - q0)
                quat_i /= np.linalg.norm(quat_i)
            else:
                theta = np.arccos(abs(dot))
                quat_i = (np.sin((1 - alpha) * theta) * q0 + np.sin(alpha * theta) * q1) / np.sin(theta)

            try:
                q_sol = ik_fn(
                    link=ee_link, pos=pos_i, quat=quat_i,
                    init_qpos=prev_qpos,
                    max_samples=1, max_solver_iters=100,
                )
            except TypeError:
                q_sol = ik_fn(link=ee_link, pos=pos_i, quat=quat_i)
            if q_sol is None:
                if log:
                    print(f"[planner] Cartesian IK failed at waypoint {i}/{n_wp}")
                return PlanResult(False)

            q_sol = q_sol.detach().cpu().numpy().ravel() if hasattr(q_sol, "cpu") else np.asarray(q_sol).ravel()
            if self._arm_joint_names and self._arm_joint_index:
                arm_cols = [self._arm_joint_index[jn] for jn in self._arm_joint_names if jn in self._arm_joint_index]
                position[i] = np.array([q_sol[idx] for idx in arm_cols[:n_arm]], dtype=np.float64)
            else:
                position[i] = q_sol[:n_arm]
            prev_qpos = q_sol.copy()

        dt = 1.0 / 250.0
        velocity = np.zeros_like(position)
        if position.shape[0] > 1:
            velocity[:-1] = np.diff(position, axis=0) / dt
        velocity = np.clip(velocity, -1.0, 1.0)

        if log:
            print(f"[planner] Cartesian IK Success, waypoints={n_wp}")
        return PlanResult(True, position, velocity)

    def update_obstacles(self, point_cloud: np.ndarray, resolution: float = 0.02):
        self._mplib.update_point_cloud(point_cloud, resolution=resolution)
