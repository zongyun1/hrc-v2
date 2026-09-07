"""Genesis motion planner — uses Genesis built-in RRT/RRTConnect path planning."""

import numpy as np

from .base import Planner, PlanResult


class GenesisMotionPlanner(Planner):
    """Planner using Genesis entity's built-in RRTConnect motion planning.

    This planner does collision-free path planning using Genesis's integrated
    RRT/RRTConnect planner with automatic collision checking against the scene.

    Parameters
    ----------
    entity           : Genesis RigidEntity (the robot)
    ee_link_name     : end-effector link name
    n_arm            : number of arm DOFs
    arm_joint_names  : arm joint names in order
    arm_joint_index  : mapping from joint name to DOF index in entity qpos
    num_waypoints    : interpolation steps for the output trajectory
    planner_type     : "RRTConnect" (default) or "RRT"
    max_nodes        : max tree nodes for RRT
    resolution       : joint-space step size
    """

    def __init__(
        self,
        entity,
        ee_link_name: str,
        n_arm: int,
        arm_joint_names: list[str],
        arm_joint_index: dict[str, int],
        num_waypoints: int = 200,
        planner_type: str = "RRTConnect",
        max_nodes: int = 2000,
        resolution: float = 0.05,
        **kwargs,
    ):
        self._entity = entity
        self._ee_link_name = ee_link_name
        self.n_arm = n_arm
        self._arm_joint_names = list(arm_joint_names)
        self._arm_joint_index = dict(arm_joint_index)
        self.num_waypoints = num_waypoints
        self._planner_type = planner_type
        self._max_nodes = max_nodes
        self._resolution = resolution

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

    def _solve_ik(self, current_qpos, target_pose, log=True):
        """Run Genesis IK to get goal joint positions."""
        target7 = np.asarray(target_pose, dtype=np.float64).ravel()[:7]
        pos_ik = target7[:3]
        quat_ik = target7[3:7]

        raw_qpos = self._entity.get_qpos()
        init_full = self._to_numpy(raw_qpos).ravel().copy()

        for i, jname in enumerate(self._arm_joint_names[:self.n_arm]):
            if jname in self._arm_joint_index and i < len(current_qpos):
                init_full[self._arm_joint_index[jname]] = float(current_qpos[i])

        ik_fn = getattr(self._entity, "inverse_kinematics", None)
        if ik_fn is None:
            if log:
                print("[genesis_mp] entity has no inverse_kinematics method")
            return None

        try:
            qpos_goal = ik_fn(
                link=self._ee_link,
                pos=pos_ik,
                quat=quat_ik,
                init_qpos=init_full,
            )
        except TypeError:
            qpos_goal = ik_fn(link=self._ee_link, pos=pos_ik, quat=quat_ik)
        except Exception as e:
            if log:
                print(f"[genesis_mp] IK exception: {e}")
            return None

        if qpos_goal is None:
            if log:
                print("[genesis_mp] IK returned None")
            return None

        return self._to_numpy(qpos_goal).ravel()

    def plan_path(
        self,
        current_qpos: np.ndarray,
        target_pose: np.ndarray,
        log: bool = True,
        **kwargs,
    ) -> PlanResult:
        """Plan collision-free path using Genesis RRTConnect.

        Steps:
            1. Solve IK for target_pose to get goal joint positions
            2. Use entity.plan_path() for collision-free RRTConnect planning
            3. Extract arm-only joint trajectories from the full path
        """
        # Step 1: Solve IK to get full qpos goal
        qpos_goal_full = self._solve_ik(current_qpos, target_pose, log=log)
        if qpos_goal_full is None:
            return PlanResult(False)

        # Step 2: Use Genesis plan_path for collision-free planning
        plan_fn = getattr(self._entity, "plan_path", None)
        if plan_fn is None:
            if log:
                print("[genesis_mp] entity has no plan_path method")
            return PlanResult(False)

        try:
            import torch
            qpos_goal_t = torch.tensor(
                qpos_goal_full, dtype=torch.float32
            )

            path = plan_fn(
                qpos_goal=qpos_goal_t,
                num_waypoints=self.num_waypoints,
                max_nodes=self._max_nodes,
                resolution=self._resolution,
                planner=self._planner_type,
                smooth_path=True,
            )
        except Exception as e:
            if log:
                print(f"[genesis_mp] plan_path exception: {e}")
            return PlanResult(False)

        if path is None:
            if log:
                print("[genesis_mp] plan_path returned None")
            return PlanResult(False)

        # Step 3: Extract arm-only joint values from path
        path_np = self._to_numpy(path)

        # path shape could be (num_waypoints, n_envs, n_dofs) or (num_waypoints, n_dofs)
        if path_np.ndim == 3:
            path_np = path_np[:, 0, :]  # take first env
        elif path_np.ndim == 1:
            path_np = path_np.reshape(1, -1)

        # Extract arm DOF columns
        arm_cols = [self._arm_joint_index[jn]
                    for jn in self._arm_joint_names
                    if jn in self._arm_joint_index]
        position = path_np[:, arm_cols[:self.n_arm]].astype(np.float64)

        # Compute velocity
        dt = 1.0 / 250.0
        velocity = np.zeros_like(position)
        if position.shape[0] > 1:
            velocity[:-1] = np.diff(position, axis=0) / dt
        velocity = np.clip(velocity, -1.0, 1.0)

        if log:
            print(f"[genesis_mp] Success, waypoints={position.shape[0]}")
        return PlanResult(True, position, velocity)

    def solve_ik(
        self,
        current_qpos: np.ndarray,
        target_pose: np.ndarray,
        num_waypoints: int = None,
        log: bool = True,
        **kwargs,
    ) -> PlanResult:
        """Solve IK and return linearly interpolated trajectory (no collision avoidance)."""
        qpos_goal_full = self._solve_ik(current_qpos, target_pose, log=log)
        if qpos_goal_full is None:
            return PlanResult(False)

        arm_cols = [self._arm_joint_index[jn]
                    for jn in self._arm_joint_names
                    if jn in self._arm_joint_index]
        q_goal = np.array([qpos_goal_full[idx] for idx in arm_cols[:self.n_arm]],
                          dtype=np.float64)
        q_start = np.array([current_qpos[i] if i < len(current_qpos) else 0.0
                            for i in range(self.n_arm)], dtype=np.float64)

        n_wp = max(2, num_waypoints or self.num_waypoints)
        t = np.linspace(0, 1, n_wp)
        position = (1 - t[:, None]) * q_start + t[:, None] * q_goal
        dt = 1.0 / 250.0
        velocity = np.zeros_like(position)
        if position.shape[0] > 1:
            velocity[:-1] = np.diff(position, axis=0) / dt
        velocity = np.clip(velocity, -1.0, 1.0)

        if log:
            print(f"[genesis_mp] IK Success, waypoints={position.shape[0]}")
        return PlanResult(True, position, velocity)
