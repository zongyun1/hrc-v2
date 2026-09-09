#!/usr/bin/env python3
"""
Persistent cuRobo motion planning worker.

Runs INSIDE the cuRobo conda environment (requires A100 / V100 GPU).
Invoked as a long-lived subprocess by CuroboPlanner in the main env.

Protocol  (newline-delimited JSON, one object per line)
--------
  stdin  : one JSON object per line  (plan requests)
  stdout : one JSON object per line  (plan results)
  stderr : log / diagnostic text (inherited by parent)

The worker builds cuRobo MotionGen once on the first request (warmup ~15-25 s)
and reuses it for all subsequent requests — eliminating cold-start overhead.
"""

import sys
import json
import os
import time
import traceback

import numpy as np


# ─── GPU guard ────────────────────────────────────────────────────────────────

def _gpu_guard():
    """Exit early with a JSON Fail payload if no suitable GPU is present."""
    try:
        import torch
    except ImportError:
        return {"status": "Fail", "position": [], "velocity": [],
                "error": "torch not importable in cuRobo env"}

    if not torch.cuda.is_available():
        return {"status": "Fail", "position": [], "velocity": [],
                "error": "No CUDA GPU available"}

    gpu_name = torch.cuda.get_device_name(0).lower()
    _supported = ("a100", "v100", "a40", "l40", "h100", "gh200", "a16",
                  "2080", "3090", "4090")
    if not any(tag in gpu_name for tag in _supported):
        return {"status": "Fail", "position": [], "velocity": [],
                "error": f"cuRobo needs A100/V100 class GPU, got: {gpu_name}"}
    return None


_guard_result = _gpu_guard()
if _guard_result is not None:
    print(json.dumps(_guard_result), flush=True)
    sys.exit(0)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _world_to_base(t_world, q_world_wxyz, base_p, base_q_wxyz):
    """Transform a world-frame pose into the robot base frame (wxyz quaternions)."""
    from scipy.spatial.transform import Rotation as R

    base_R = R.from_quat([base_q_wxyz[1], base_q_wxyz[2],
                           base_q_wxyz[3], base_q_wxyz[0]])       # scipy xyzw
    t_base = base_R.inv().apply(np.array(t_world) - np.array(base_p))

    goal_R = R.from_quat([q_world_wxyz[1], q_world_wxyz[2],
                           q_world_wxyz[3], q_world_wxyz[0]])     # scipy xyzw
    q_base_xyzw = (base_R.inv() * goal_R).as_quat()               # xyzw
    q_base_wxyz = np.array([q_base_xyzw[3],
                             q_base_xyzw[0], q_base_xyzw[1], q_base_xyzw[2]])
    return t_base.astype(np.float32), q_base_wxyz.astype(np.float32)


def _fail(msg=""):
    out = {"status": "Fail", "position": [], "velocity": []}
    if msg:
        out["error"] = msg
    return out


def _load_collision_spheres(urdf_path):
    """Load collision spheres YAML from the same directory as the URDF.

    Tries collision_franka.yml, collision_piper.yml, then any collision_*.yml.
    """
    import yaml
    d = os.path.dirname(os.path.abspath(urdf_path))
    for name in ("collision_franka.yml", "collision_piper.yml"):
        path = os.path.join(d, name)
        if os.path.exists(path):
            with open(path) as f:
                data = yaml.safe_load(f)
            return data.get("collision_spheres", data)
    # Fallback: any collision_*.yml
    import glob
    for path in glob.glob(os.path.join(d, "collision_*.yml")):
        with open(path) as f:
            data = yaml.safe_load(f)
        return data.get("collision_spheres", data)
    return None


# Self-collision pairs to ignore — Piper links
_SELF_COLLISION_IGNORE_PIPER = {
    "base_link": ["link1", "link2", "link3"],
    "link1": ["link2", "link3"],
    "link2": ["link3", "link4"],
    "link3": ["link4", "link5", "link6", "link7", "link8"],
    "link4": ["link5", "link6", "link7", "link8"],
    "link5": ["link6", "link7", "link8"],
    "link6": ["link7", "link8"],
    "link7": ["link8"],
}

# Self-collision pairs to ignore — Franka Panda links
_SELF_COLLISION_IGNORE_FRANKA = {
    "panda_link0": ["panda_link1", "panda_link2"],
    "panda_link1": ["panda_link2", "panda_link3"],
    "panda_link2": ["panda_link3", "panda_link4"],
    "panda_link3": ["panda_link4", "panda_link5"],
    "panda_link4": ["panda_link5", "panda_link6"],
    "panda_link5": ["panda_link6", "panda_link7", "panda_hand"],
    "panda_link6": ["panda_link7", "panda_hand"],
    "panda_link7": ["panda_hand"],
}


def _pick_self_collision_config(urdf_path):
    """Pick self-collision ignore dict based on URDF link names."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(urdf_path)
    link_names = {l.get("name") for l in tree.findall(".//link")}
    if "panda_link0" in link_names:
        ignore = _SELF_COLLISION_IGNORE_FRANKA
        buffer = {k: 0.0 for k in ignore}
    elif "base_link" in link_names:
        ignore = _SELF_COLLISION_IGNORE_PIPER
        buffer = {k: 0.0 for k in ignore}
        buffer.update({"link8": 0.0})
    else:
        ignore = {}
        buffer = {}
    return ignore, buffer


# ─── persistent MotionGen wrapper ────────────────────────────────────────────

class _PersistentMotionGen:
    """Build cuRobo MotionGen once, reuse for every plan request."""

    def __init__(self):
        self.mg = None
        self._config_key = None
        self._tensor_args = None
        self._n_dof = None
        self._active_joint_names = None

    # ── build / rebuild MotionGen if config changed ──────────────────────────

    def _ensure_built(self, payload):
        urdf_path   = payload["urdf_path"]
        ee_link     = payload.get("ee_link", "link6")
        joint_names = payload.get("joint_names", [f"joint{i+1}" for i in range(6)])
        num_wp      = int(payload.get("num_waypoints", 200))

        key = (urdf_path, ee_link, tuple(joint_names), num_wp)
        if self.mg is not None and key == self._config_key:
            return
        t0 = time.perf_counter()

        import torch
        from curobo.types.base import TensorDeviceType
        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.geom.types import WorldConfig
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

        tensor_args = TensorDeviceType()
        self._tensor_args = tensor_args

        # Discover ALL movable joints in the URDF and their limits
        import xml.etree.ElementTree as ET
        tree = ET.parse(urdf_path)
        all_movable = []
        _joint_lower = {}
        _joint_upper = {}
        for j in tree.findall(".//joint"):
            if j.get("type") in ("revolute", "prismatic", "continuous"):
                jname = j.get("name")
                all_movable.append(jname)
                lim = j.find("limit")
                if lim is not None:
                    _joint_lower[jname] = float(lim.get("lower", -6.28))
                    _joint_upper[jname] = float(lim.get("upper",  6.28))

        planned_set = set(joint_names)
        lock = {name: 0.0 for name in all_movable if name not in planned_set}

        # Determine base_link name from URDF
        root_link = tree.find(".//link")
        base_link_name = root_link.get("name") if root_link is not None else "base_link"

        # Homestate for retract
        _home_defaults = {
            # Piper
            "joint2": 1.57, "joint3": -1.57,
            # Franka
            "panda_joint2": 0.1963, "panda_joint4": -2.618,
            "panda_joint6": 2.9416, "panda_joint7": 0.7854,
        }
        retract = [_home_defaults.get(n, 0.0) for n in all_movable]

        # Collision spheres
        spheres = _load_collision_spheres(urdf_path)
        self_collision_ignore, self_collision_buffer = _pick_self_collision_config(urdf_path)

        robot_cfg = {
            "kinematics": {
                "urdf_path":   urdf_path,
                "base_link":   base_link_name,
                "ee_link":     ee_link,
                "lock_joints": lock,
                "cspace": {
                    "joint_names":            all_movable,
                    "retract_config":         retract,
                    "null_space_weight":      [1.0] * len(all_movable),
                    "cspace_distance_weight": [1.0] * len(all_movable),
                    "max_acceleration":       15.0,
                    "max_jerk":               500.0,
                },
            }
        }

        if spheres is not None:
            robot_cfg["kinematics"]["collision_spheres"]       = spheres
            robot_cfg["kinematics"]["collision_link_names"]    = list(spheres.keys())
            robot_cfg["kinematics"]["collision_sphere_buffer"] = 0.004
            robot_cfg["kinematics"]["self_collision_ignore"]   = self_collision_ignore
            robot_cfg["kinematics"]["self_collision_buffer"]   = self_collision_buffer
            robot_cfg["kinematics"]["use_global_cumul"]        = True

        # cuRobo PRIMITIVE collision checker requires ≥1 world obstacle.
        # Place a tiny cuboid 10m underground so it never interferes.
        from curobo.geom.types import Cuboid as CuroboCuboid
        _dummy = CuroboCuboid(
            name="__dummy_far",
            pose=[0.0, 0.0, -10.0, 1, 0, 0, 0],
            dims=[0.01, 0.01, 0.01],
        )
        world_cfg = WorldConfig(cuboid=[_dummy])

        cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            tensor_args,
            trajopt_tsteps         = min(num_wp, 32),
            collision_checker_type = CollisionCheckerType.PRIMITIVE,
            num_trajopt_seeds      = 24,
            num_ik_seeds           = 300,
            interpolation_steps    = num_wp,
        )
        self.mg = MotionGen(cfg)
        self.mg.warmup()

        self._n_dof = self.mg.kinematics.get_dof()
        self._active_joint_names = [n for n in all_movable if n not in lock]
        self._jlim_lower = np.array(
            [_joint_lower.get(n, -6.28) for n in self._active_joint_names],
            dtype=np.float32,
        )
        self._jlim_upper = np.array(
            [_joint_upper.get(n,  6.28) for n in self._active_joint_names],
            dtype=np.float32,
        )
        self._config_key = key

        dt = time.perf_counter() - t0
        sys.stderr.write(
            f"[curobo_worker] MotionGen built: {self._n_dof} active DOFs, "
            f"warmup {dt:.1f}s, GPU={torch.cuda.get_device_name(0)}\n"
        )
        sys.stderr.flush()

    # ── plan one request ─────────────────────────────────────────────────────

    def plan(self, payload):
        self._ensure_built(payload)

        import torch
        from curobo.types.math  import Pose as CuroboPose
        from curobo.types.robot import JointState
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

        qpos_raw  = np.array(payload["qpos"],        dtype=np.float32)
        target    = np.array(payload["target_pose"],  dtype=np.float32)
        base_pose = np.array(payload.get("base_pose", [0, 0, 0, 1, 0, 0, 0]),
                             dtype=np.float32)

        # Clamp start qpos to URDF joint limits
        n_dof_pre = self._n_dof
        qpos = np.clip(qpos_raw[:n_dof_pre], self._jlim_lower, self._jlim_upper)
        if not np.allclose(qpos, qpos_raw[:n_dof_pre], atol=1e-6):
            sys.stderr.write(
                f"[curobo_worker] clamped qpos: {np.round(qpos_raw[:n_dof_pre], 5).tolist()}"
                f" -> {np.round(qpos, 5).tolist()}\n"
            )
            sys.stderr.flush()

        # World → base frame
        t_base, q_base = _world_to_base(
            target[:3], target[3:7], base_pose[:3], base_pose[3:7],
        )

        ta    = self._tensor_args
        n_dof = self._n_dof

        # ── diagnostic logging ───────────────────────────────────────────
        sys.stderr.write(
            f"[curobo_worker] plan request:\n"
            f"  qpos({n_dof})    = {np.round(qpos[:n_dof], 4).tolist()}\n"
            f"  target_world  = pos {np.round(target[:3], 4).tolist()}  "
            f"quat {np.round(target[3:7], 4).tolist()}\n"
            f"  target_base   = pos {np.round(t_base, 4).tolist()}  "
            f"quat {np.round(q_base, 4).tolist()}\n"
            f"  base_pose     = {np.round(base_pose, 4).tolist()}\n"
        )
        sys.stderr.flush()

        # FK at start state
        q_start_t = torch.tensor(qpos[:n_dof], dtype=ta.dtype,
                                  device=ta.device).unsqueeze(0)
        fk = self.mg.kinematics.get_state(q_start_t)
        fk_pos = fk.ee_position.cpu().numpy().ravel()
        fk_quat = fk.ee_quaternion.cpu().numpy().ravel()
        dist = float(np.linalg.norm(t_base - fk_pos))
        sys.stderr.write(
            f"  FK@start base = pos {np.round(fk_pos, 4).tolist()}  "
            f"quat {np.round(fk_quat, 4).tolist()}\n"
            f"  dist(target, FK) = {dist:.4f} m\n"
        )
        sys.stderr.flush()

        goal = CuroboPose(
            position   = torch.tensor(t_base,  dtype=ta.dtype, device=ta.device).unsqueeze(0),
            quaternion = torch.tensor(q_base,  dtype=ta.dtype, device=ta.device).unsqueeze(0),
        )

        start = JointState.from_position(
            torch.tensor(qpos[:n_dof], dtype=ta.dtype, device=ta.device).unsqueeze(0),
            joint_names=self._active_joint_names,
        )

        # Homestate start (for retry)
        _home_defaults = {
            "joint2": 1.57, "joint3": -1.57,
            "panda_joint2": 0.1963, "panda_joint4": -2.618,
            "panda_joint6": 2.9416, "panda_joint7": 0.7854,
        }
        home_q = np.array(
            [_home_defaults.get(n, 0.0) for n in self._active_joint_names],
            dtype=np.float32,
        )
        home_start = JointState.from_position(
            torch.tensor(home_q, dtype=ta.dtype, device=ta.device).unsqueeze(0),
            joint_names=self._active_joint_names,
        )

        # ── planning with progressive retry strategy ───────────────────
        _plan_configs = [
            ("standard",       start,      MotionGenPlanConfig(enable_graph=True, enable_opt=True, max_attempts=12, timeout=15.0)),
            ("retry_partial",  start,      MotionGenPlanConfig(enable_graph=True, enable_opt=True, max_attempts=20, timeout=20.0, partial_ik_opt=True)),
            ("from_home",      home_start, MotionGenPlanConfig(enable_graph=True, enable_opt=True, max_attempts=12, timeout=15.0)),
            ("home_partial",   home_start, MotionGenPlanConfig(enable_graph=True, enable_opt=True, max_attempts=24, timeout=25.0, partial_ik_opt=True)),
        ]

        result = None
        status_attr = None
        dt = 0.0
        for attempt_label, attempt_start, attempt_cfg in _plan_configs:
            t0 = time.perf_counter()
            result = self.mg.plan_single(attempt_start, goal, attempt_cfg)
            dt_i = time.perf_counter() - t0
            dt += dt_i
            status_attr = getattr(result, "status", None)
            sys.stderr.write(
                f"[curobo_worker] {attempt_label}: success={result.success.item()} "
                f"status={status_attr} dt={dt_i:.2f}s\n"
            )
            sys.stderr.flush()
            if result.success.item():
                break
            if attempt_label == "from_home" and not result.success.item():
                s1 = str(status_attr)
                if "IK_FAIL" in s1:
                    sys.stderr.write("[curobo_worker] IK_FAIL from home too — target likely unreachable\n")
                    sys.stderr.flush()
                    break

        if result.success.item():
            traj   = result.get_interpolated_plan()
            pos_np = traj.position.cpu().numpy()[:, :n_dof]
            vel_attr = getattr(traj, "velocity", None)
            vel_np = (vel_attr.cpu().numpy()[:, :n_dof]
                      if vel_attr is not None else np.zeros_like(pos_np))
            sys.stderr.write(f"[curobo_worker] OK {pos_np.shape[0]} wps in {dt:.2f}s\n")
            sys.stderr.flush()
            return {
                "status":   "Success",
                "position": pos_np.tolist(),
                "velocity": vel_np.tolist(),
            }
        else:
            sys.stderr.write(f"[curobo_worker] FAIL in {dt:.2f}s (all attempts exhausted)\n")
            sys.stderr.flush()
            return _fail(f"MotionGen failed: {status_attr}")


# ─── main loop ────────────────────────────────────────────────────────────────

def main():
    pmg = _PersistentMotionGen()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            payload = json.loads(line)
        except Exception as e:
            print(json.dumps(_fail(f"JSON parse error: {e}")), flush=True)
            continue

        try:
            output = pmg.plan(payload)
        except Exception:
            output = _fail(traceback.format_exc())

        print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
