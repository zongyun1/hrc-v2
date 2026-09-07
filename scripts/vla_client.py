"""Small runtime helpers for VLA evaluation.

This keeps the benchmark-facing HTTP client shims in one place without the
larger historical ``policies/`` package.
"""
from __future__ import annotations

import base64
import io
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


def get_task_instruction(task_name, override=None):
    if override:
        return override
    try:
        from envs.tasks import TASK_MAP
        task_cls = TASK_MAP.get(task_name)
        # default_instruction() fills any {object} template slot with a
        # representative object so static lookups never surface a raw
        # placeholder (per-episode the real object is set via set_instruction).
        if task_cls is not None and hasattr(task_cls, "default_instruction"):
            instruction = task_cls.default_instruction()
        else:
            instruction = getattr(task_cls, "INSTRUCTION", None) if task_cls else None
        if instruction:
            return instruction
    except Exception:
        pass
    return f"Complete the {task_name.replace('_', ' ')} task."


def encode_png(img_uint8: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_uint8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def resize_uint8(img_uint8: np.ndarray, hw) -> np.ndarray:
    return np.asarray(
        Image.fromarray(img_uint8).resize((hw[1], hw[0]), Image.LANCZOS),
        dtype=np.uint8,
    )


def extract_proprio(obs, action_type, dual_arm):
    parts = []
    arms = ["left", "right"] if dual_arm else ["right"]
    for arm in arms:
        if action_type in {"qpos", "qpos_abs"}:
            joint_state = list(obs["joint_state"][arm])
            gripper = obs.get("gripper_val", {}).get(arm)
            if gripper is not None and len(joint_state) in {6, 7}:
                joint_state.append(float(gripper))
            parts.extend(joint_state)
        else:
            parts.extend(obs["ee_pose"][arm])
    return np.array(parts, dtype=np.float32)


class BaseVLAPolicy(ABC):
    def __init__(self, checkpoint_path, task_instruction, config, action_type, dual_arm):
        self.checkpoint_path = checkpoint_path
        self.task_instruction = task_instruction
        self.config = config
        self.action_type = action_type
        self.dual_arm = dual_arm
        self.action_dim = 8 * (2 if dual_arm else 1)
        self._action_queue = []

    def set_instruction(self, instruction):
        """Update the language instruction (called per episode by the eval
        loop so tasks that sample their target object each reset feed the
        correct per-episode instruction to the model).  No-op on None/empty."""
        if instruction:
            self.task_instruction = instruction

    @abstractmethod
    def load_model(self):
        ...

    @abstractmethod
    def predict_raw(self, processed_obs: dict) -> np.ndarray:
        ...

    def preprocess_obs(self, obs: dict) -> dict:
        cameras = self.config.get("cameras", ["head_camera"])
        image_size = tuple(self.config.get("image_size", [224, 224]))
        images = {}
        for name in cameras:
            images[name] = np.asarray(
                Image.fromarray(obs["rgb"][name]).resize(
                    (image_size[1], image_size[0]), Image.LANCZOS
                ),
                dtype=np.float32,
            ) / 255.0
        proprio = extract_proprio(obs, self.action_type, self.dual_arm)
        return {"images": images, "proprio": proprio, "instruction": self.task_instruction}

    def postprocess_action(self, raw_action: np.ndarray) -> np.ndarray:
        return raw_action.astype(np.float64)

    def reset(self):
        self._action_queue = []

    def __call__(self, obs: dict) -> np.ndarray:
        if not self._action_queue:
            raw = np.atleast_2d(self.predict_raw(self.preprocess_obs(obs)))
            for action in raw:
                self._action_queue.append(self.postprocess_action(action))
        action = self._action_queue.pop(0)
        assert action.shape == (self.action_dim,), (
            f"Expected ({self.action_dim},), got {action.shape}"
        )
        return action


def _quat_wxyz_to_euler(q):
    import transforms3d as t3d

    return np.asarray(t3d.euler.quat2euler(q, axes="sxyz"), dtype=np.float32)


def _lerobot_delta_to_absolute_ee(delta_action, current_pose) -> np.ndarray:
    import transforms3d as t3d
    from scipy.spatial.transform import Rotation as R

    delta_action = np.asarray(delta_action, dtype=np.float64).ravel()
    current_pose = np.asarray(current_pose, dtype=np.float64).ravel()
    dxyz = delta_action[:3]
    drpy = delta_action[3:6]
    gripper_open = 1.0 - float(delta_action[6])

    cur_xyz = current_pose[:3]
    cur_quat_wxyz = current_pose[3:7]
    cur_rot = R.from_quat([
        cur_quat_wxyz[1],
        cur_quat_wxyz[2],
        cur_quat_wxyz[3],
        cur_quat_wxyz[0],
    ])
    dq_wxyz = t3d.euler.euler2quat(
        float(drpy[0]), float(drpy[1]), float(drpy[2]), axes="sxyz"
    )
    delta_rot = R.from_quat([dq_wxyz[1], dq_wxyz[2], dq_wxyz[3], dq_wxyz[0]])
    qx, qy, qz, qw = (cur_rot * delta_rot).as_quat()
    new_xyz = cur_xyz + dxyz
    return np.array(
        [new_xyz[0], new_xyz[1], new_xyz[2], qw, qx, qy, qz, gripper_open],
        dtype=np.float64,
    )


class RemoteVLAPolicy(BaseVLAPolicy):
    IMG_PRIMARY = (224, 224)
    IMG_WRIST = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ee_command_pose = None

    def reset(self):
        super().reset()
        self._ee_command_pose = None

    def load_model(self):
        import requests

        self.server_url = self.config.get("server_url", "http://127.0.0.1:8765").rstrip("/")
        self.request_timeout = float(self.config.get("server_timeout", 60))
        r = requests.get(f"{self.server_url}/health", timeout=5)
        r.raise_for_status()
        print(f"[{type(self).__name__}] connected to {self.server_url}: {r.json()}")

    def _ee_delta_base_pose(self, current_pose) -> np.ndarray:
        current_pose = np.asarray(current_pose, dtype=np.float64).ravel()[:7]
        mode = str(self.config.get("ee_delta_integration", "target")).lower()
        if mode in {"measured", "current", "feedback"}:
            return current_pose
        if self._ee_command_pose is None:
            self._ee_command_pose = current_pose.copy()
        return self._ee_command_pose

    def _remember_ee_command_pose(self, absolute_action) -> np.ndarray:
        absolute_action = np.asarray(absolute_action, dtype=np.float64).ravel()
        if absolute_action.size >= 7:
            self._ee_command_pose = absolute_action[:7].copy()
        return absolute_action

    def preprocess_obs(self, obs):
        primary_cam = self.config.get("primary_camera", "head_camera")
        images = {
            "image_primary": encode_png(
                resize_uint8(obs["rgb"][primary_cam], self.IMG_PRIMARY)
            )
        }
        wrist_cam = self.config.get("wrist_camera")
        if wrist_cam and self.IMG_WRIST and wrist_cam in obs["rgb"]:
            images["image_wrist"] = encode_png(
                resize_uint8(obs["rgb"][wrist_cam], self.IMG_WRIST)
            )
        secondary_cam = self.config.get("secondary_camera")
        if secondary_cam and secondary_cam in obs["rgb"]:
            images["image_secondary"] = encode_png(
                resize_uint8(obs["rgb"][secondary_cam], self.IMG_PRIMARY)
            )
        return {
            "images": images,
            "proprio": extract_proprio(obs, self.action_type, self.dual_arm),
            "instruction": self.task_instruction,
        }

    def predict_raw(self, processed_obs):
        import requests

        payload = {
            "instruction": self.task_instruction,
            "images": processed_obs["images"],
            "unnorm_key": self.config.get("unnorm_key"),
        }
        r = requests.post(
            f"{self.server_url}/predict",
            json=payload,
            timeout=self.request_timeout,
        )
        r.raise_for_status()
        return np.asarray(r.json()["action"], dtype=np.float64)


class LeRobotRemotePolicy(RemoteVLAPolicy):
    IMG_PRIMARY = (256, 256)
    IMG_WRIST = (256, 256)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._first_call = True
        self._gripper_open_cmd = 1.0

    def reset(self):
        super().reset()
        self._first_call = True
        self._gripper_open_cmd = 1.0

    def preprocess_obs(self, obs):
        processed = super().preprocess_obs(obs)
        if self.action_type == "ee":
            if self.dual_arm:
                raise ValueError(f"{type(self).__name__} is single-arm only for EE eval.")
            current_pose = np.asarray(obs["ee_pose"]["right"], dtype=np.float64).ravel()[:7]
            gripper_closed = 1.0 - float(obs["gripper_val"]["right"])
            processed["current_ee_pose"] = current_pose
            processed["proprio"] = np.concatenate(
                [current_pose[:3], _quat_wxyz_to_euler(current_pose[3:7]), [gripper_closed]]
            ).astype(np.float32)
        return processed

    def predict_raw(self, processed_obs):
        import requests

        if self.dual_arm:
            raise ValueError(f"{type(self).__name__} is single-arm only.")
        payload = {
            "instruction": self.task_instruction,
            "images": processed_obs["images"],
            "proprio": [float(x) for x in np.asarray(processed_obs["proprio"]).tolist()],
            "reset": self._first_call,
        }
        self._first_call = False
        r = requests.post(
            f"{self.server_url}/predict",
            json=payload,
            timeout=self.request_timeout,
        )
        r.raise_for_status()
        delta = np.asarray(r.json()["action"], dtype=np.float64)
        if self.action_type == "ee":
            action = _lerobot_delta_to_absolute_ee(
                delta, self._ee_delta_base_pose(processed_obs["current_ee_pose"])
            )
            return self._remember_ee_command_pose(action)
        return delta

    def postprocess_action(self, raw_action: np.ndarray) -> np.ndarray:
        action = super().postprocess_action(raw_action)
        if self.action_type not in {"qpos", "qpos_abs"} or "act_gripper_closed_cmd" not in self.config:
            return action
        close_threshold = float(self.config.get("act_gripper_close_threshold", 0.75))
        open_threshold = float(self.config.get("act_gripper_open_threshold", 0.90))
        closed_cmd = float(self.config["act_gripper_closed_cmd"])
        if action[-1] <= close_threshold:
            self._gripper_open_cmd = closed_cmd
        elif action[-1] >= open_threshold:
            self._gripper_open_cmd = 1.0
        action[-1] = self._gripper_open_cmd
        return action


class OpenVLAOFTPolicy(LeRobotRemotePolicy):
    IMG_PRIMARY = (224, 224)
    IMG_WRIST = (224, 224)

    def predict_raw(self, processed_obs):
        import requests

        if self.dual_arm:
            raise ValueError("OpenVLA-OFT single-arm only in this wiring.")
        payload = {
            "instruction": self.task_instruction,
            "images": processed_obs["images"],
            "proprio": [float(x) for x in np.asarray(processed_obs["proprio"]).tolist()],
            "unnorm_key": self.config.get("unnorm_key"),
            "reset": self._first_call,
        }
        self._first_call = False
        r = requests.post(
            f"{self.server_url}/predict",
            json=payload,
            timeout=self.request_timeout,
        )
        r.raise_for_status()
        delta = np.asarray(r.json()["action"], dtype=np.float64)
        if self.action_type == "ee":
            action = _lerobot_delta_to_absolute_ee(
                delta, self._ee_delta_base_pose(processed_obs["current_ee_pose"])
            )
            return self._remember_ee_command_pose(action)
        return delta


class RDTPolicy(RemoteVLAPolicy):
    IMG_PRIMARY = (384, 384)
    IMG_WRIST = (384, 384)

    def predict_raw(self, processed_obs):
        import requests

        proprio = np.asarray(processed_obs["proprio"], dtype=np.float32)
        if proprio.size < 8:
            proprio = np.concatenate([proprio, np.zeros(8 - proprio.size, dtype=np.float32)])
        payload = {
            "instruction": self.task_instruction,
            "images": processed_obs["images"],
            "proprio": [float(x) for x in proprio[:8].tolist()],
        }
        r = requests.post(
            f"{self.server_url}/predict",
            json=payload,
            timeout=self.request_timeout,
        )
        r.raise_for_status()
        return np.asarray(r.json()["action"], dtype=np.float64)


class DiffusionPolicyPolicy(RemoteVLAPolicy):
    IMG_PRIMARY = (96, 96)
    IMG_WRIST = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._first_call = True

    def reset(self):
        super().reset()
        self._first_call = True

    def predict_raw(self, processed_obs):
        import requests

        payload = {
            "instruction": self.task_instruction,
            "images": processed_obs["images"],
            "proprio": [float(x) for x in np.asarray(processed_obs["proprio"]).tolist()],
            "reset": self._first_call,
        }
        is_first = self._first_call
        self._first_call = False
        timeout = max(self.request_timeout, 300.0) if is_first else self.request_timeout
        r = requests.post(f"{self.server_url}/predict", json=payload, timeout=timeout)
        r.raise_for_status()
        return np.asarray(r.json()["action"], dtype=np.float64)


class ACTPolicy(LeRobotRemotePolicy):
    pass


class Pi0Policy(LeRobotRemotePolicy):
    pass


class Pi05Policy(Pi0Policy):
    pass


class Pi0FastPolicy(Pi0Policy):
    pass


class VQBeTPolicy(LeRobotRemotePolicy):
    IMG_WRIST = None


class LeRobotDiffusionPolicy(LeRobotRemotePolicy):
    pass


class SmolVLAPolicy(LeRobotRemotePolicy):
    pass


class HriDepthPolicy(BaseVLAPolicy):
    """Observation-only depth/RGB heuristic for simple HRI baseline probes.

    The policy uses rendered camera observations plus proprio. It does not read
    task objects, avatar state, simulator entities, or set robot state.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_dim = 8
        self._phase = 0
        self._phase_tick = 0
        self._target_xyz = None
        self._quat = None

    def load_model(self):
        print("[HriDepthPolicy] using observation-only depth/RGB heuristic")

    def predict_raw(self, processed_obs: dict) -> np.ndarray:
        return np.zeros(self.action_dim, dtype=np.float64)

    def reset(self):
        super().reset()
        self._phase = 0
        self._phase_tick = 0
        self._target_xyz = None
        self._quat = None

    def _camera_name(self, obs: dict) -> str:
        rgb = obs.get("rgb", {})
        for name in ("head_camera", "side_camera", "front_camera"):
            if name in rgb and name in obs.get("depth", {}):
                return name
        for name in rgb:
            if name in obs.get("depth", {}):
                return name
        raise RuntimeError("HriDepthPolicy needs an RGB-D camera observation")

    def _estimate_dump_cube(self, obs: dict) -> np.ndarray:
        name = self._camera_name(obs)
        rgb = np.asarray(obs["rgb"][name])
        depth = np.asarray(obs["depth"][name])
        intrinsic = np.asarray(obs["camera_intrinsics"][name])
        extrinsic = np.asarray(obs["camera_extrinsics"][name])
        pts_rgb = depth_to_world_points(depth, intrinsic, extrinsic, rgb=rgb)
        if pts_rgb.size == 0:
            raise RuntimeError("empty RGB-D point cloud")
        xyz = pts_rgb[:, :3]
        colors = pts_rgb[:, 3:6] * 255.0
        r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
        # Dump-bin easy objects are orange/red cubes on a beige table. The
        # world crop removes the bin/floor; the color test removes the table.
        crop = (
            (xyz[:, 0] > -0.28) & (xyz[:, 0] < 0.32) &
            (xyz[:, 1] > -0.46) & (xyz[:, 1] < 0.08) &
            (xyz[:, 2] > 0.76) & (xyz[:, 2] < 0.86)
        )
        orange = (
            (r > 130.0) & (g > 35.0) & (g < 170.0) &
            (b < 135.0) & (r > g + 25.0) & (r > b + 50.0)
        )
        cand = xyz[crop & orange]
        if cand.shape[0] < 25:
            # Fallback: choose the highest non-table points in the spawn crop.
            spawn = xyz[crop]
            if spawn.shape[0] == 0:
                raise RuntimeError("could not find dump object points")
            z_cut = np.quantile(spawn[:, 2], 0.90)
            cand = spawn[spawn[:, 2] >= z_cut]
        center = np.median(cand[:, :3], axis=0)
        center[2] = 0.785
        return center.astype(np.float64)

    def _advance(self, hold_steps: int) -> None:
        self._phase_tick += 1
        if self._phase_tick >= hold_steps:
            self._phase += 1
            self._phase_tick = 0

    def __call__(self, obs: dict) -> np.ndarray:
        if self.action_type != "ee":
            raise ValueError("HriDepthPolicy expects --action-type ee")
        ee = np.asarray(obs["ee_pose"]["right"], dtype=np.float64).ravel()
        if self._quat is None:
            # Same top-down TCP orientation used by TopDownPickPlaceMixin.
            self._quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        if self._target_xyz is None:
            self._target_xyz = self._estimate_dump_cube(obs)

        pick = self._target_xyz
        # Genesis IK/PD consistently settles the TCP a few cm forward on the
        # vertical descent. Bias the commanded grasp point backward so the
        # observed TCP closes around the visually estimated cube center.
        grasp_xy = pick[:2] + np.array([-0.008, -0.030], dtype=np.float64)
        hover_z = 1.02
        grasp_z = float(pick[2] - 0.006)
        # The cube rides on the outside of the right finger during transport,
        # so releasing above the geometric bin center lands it past the +x rim.
        # Aim left of center; gravity and the carried offset put the cube into
        # the 26 cm x 39 cm opening.
        bin_xyz = np.array([0.48, -0.30, 0.46], dtype=np.float64)
        retreat_xyz = np.array([0.25, -0.35, 1.02], dtype=np.float64)

        phases = (
            (np.array([grasp_xy[0], grasp_xy[1], hover_z]), 1.0, 8),
            (np.array([grasp_xy[0], grasp_xy[1], grasp_z]), 1.0, 8),
            (np.array([grasp_xy[0], grasp_xy[1], grasp_z]), 0.00, 16),
            (np.array([grasp_xy[0], grasp_xy[1], hover_z]), 0.00, 10),
            (bin_xyz, 0.05, 12),
            (bin_xyz, 1.0, 10),
            (retreat_xyz, 1.0, 20),
        )
        idx = min(self._phase, len(phases) - 1)
        xyz, gripper, hold = phases[idx]
        self._advance(int(hold))
        return np.concatenate([xyz, self._quat, [float(gripper)]]).astype(np.float64)


class TemplatePolicy(BaseVLAPolicy):
    def load_model(self):
        pass

    def predict_raw(self, processed_obs):
        return np.zeros(self.action_dim, dtype=np.float64)


DEFAULT_BBOX_MIN = np.array([-0.6, -1.0, 0.74], dtype=np.float32)
DEFAULT_BBOX_MAX = np.array([0.6, 0.4, 1.6], dtype=np.float32)


def depth_to_world_points(depth, intrinsic, extrinsic, rgb=None, depth_min=0.05, depth_max=5.0):
    height, width = depth.shape[:2]
    depth = np.asarray(depth, dtype=np.float32)
    valid = (depth > depth_min) & (depth < depth_max) & np.isfinite(depth)
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    us, vs = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    z = depth[valid]
    u = us[valid].astype(np.float32)
    v = vs[valid].astype(np.float32)
    pts_cam = np.stack(
        [(u - cx) * z / fx, -(v - cy) * z / fy, -z, np.ones_like(z)],
        axis=1,
    )
    pts_world = (extrinsic @ pts_cam.T).T[:, :3].astype(np.float32)
    if rgb is None:
        return pts_world
    colors = np.asarray(rgb, dtype=np.float32)[..., :3][valid] / 255.0
    return np.concatenate([pts_world, colors], axis=1).astype(np.float32)


def crop_bbox(points, bbox_min=None, bbox_max=None):
    bbox_min = DEFAULT_BBOX_MIN if bbox_min is None else bbox_min
    bbox_max = DEFAULT_BBOX_MAX if bbox_max is None else bbox_max
    mask = np.all((points[:, :3] >= bbox_min) & (points[:, :3] <= bbox_max), axis=1)
    return points[mask]


def voxel_downsample(points, voxel_size=0.005):
    if points.shape[0] == 0:
        return points
    keys = np.floor(points[:, :3] / voxel_size).astype(np.int64)
    hashes = keys[:, 0] * 73856093 ^ keys[:, 1] * 19349663 ^ keys[:, 2] * 83492791
    _, idx = np.unique(hashes, return_index=True)
    return points[idx]


def farthest_point_sample(points, n_points, seed=0):
    count = points.shape[0]
    if count == 0:
        dim = points.shape[1] if points.ndim == 2 else 3
        return np.zeros((n_points, dim), dtype=np.float32)
    if count <= n_points:
        rng = np.random.default_rng(seed)
        extra = rng.integers(0, count, size=n_points - count)
        return np.concatenate([points, points[extra]], axis=0)
    rng = np.random.default_rng(seed)
    xyz = points[:, :3]
    sampled = np.empty(n_points, dtype=np.int64)
    sampled[0] = rng.integers(0, count)
    dist = np.linalg.norm(xyz - xyz[sampled[0]], axis=1)
    for i in range(1, n_points):
        sampled[i] = int(np.argmax(dist))
        dist = np.minimum(dist, np.linalg.norm(xyz - xyz[sampled[i]], axis=1))
    return points[sampled]


def lift_pointcloud(
    depth,
    intrinsic,
    extrinsic,
    rgb=None,
    n_points=1024,
    use_color=False,
    bbox_min=None,
    bbox_max=None,
    voxel_size=0.005,
):
    pts = depth_to_world_points(
        depth,
        intrinsic,
        extrinsic,
        rgb=rgb if use_color else None,
    )
    pts = crop_bbox(pts, bbox_min=bbox_min, bbox_max=bbox_max)
    if pts.shape[0] == 0:
        return np.zeros((n_points, 6 if use_color else 3), dtype=np.float32)
    return farthest_point_sample(voxel_downsample(pts, voxel_size=voxel_size), n_points)


class DP3Policy(BaseVLAPolicy):
    def load_model(self):
        import requests

        self.server_url = self.config.get("server_url", "http://127.0.0.1:8774").rstrip("/")
        self.request_timeout = float(self.config.get("server_timeout", 60))
        self.primary_cam = self.config.get("primary_camera", "head_camera")
        self.n_points = int(self.config.get("dp3_num_points", 1024))
        self.use_color = bool(self.config.get("dp3_use_color", False))
        bbox = self.config.get("dp3_workspace_bbox")
        self.bbox_min = np.asarray(bbox[0], dtype=np.float32) if bbox is not None else None
        self.bbox_max = np.asarray(bbox[1], dtype=np.float32) if bbox is not None else None
        r = requests.get(f"{self.server_url}/health", timeout=5)
        r.raise_for_status()
        info = r.json()
        print(f"[DP3Policy] connected to {self.server_url}: {info}")
        self.n_points = int(info.get("n_points", self.n_points))
        self.use_color = bool(info.get("use_color", self.use_color))

    def preprocess_obs(self, obs):
        if "depth" not in obs or self.primary_cam not in obs.get("depth", {}):
            raise RuntimeError(f"DP3 needs obs['depth']['{self.primary_cam}'].")
        depth = np.asarray(obs["depth"][self.primary_cam])
        intrinsic = np.asarray(obs["camera_intrinsics"][self.primary_cam])
        extrinsic = np.asarray(obs["camera_extrinsics"][self.primary_cam])
        rgb = obs["rgb"].get(self.primary_cam) if self.use_color else None
        return {
            "point_cloud": lift_pointcloud(
                depth=depth,
                intrinsic=intrinsic,
                extrinsic=extrinsic,
                rgb=rgb,
                n_points=self.n_points,
                use_color=self.use_color,
                bbox_min=self.bbox_min,
                bbox_max=self.bbox_max,
            ),
            "proprio": extract_proprio(obs, self.action_type, self.dual_arm),
            "instruction": self.task_instruction,
        }

    def predict_raw(self, processed_obs):
        import requests

        payload = {
            "instruction": self.task_instruction,
            "point_cloud": processed_obs["point_cloud"].astype(np.float32).tolist(),
            "proprio": processed_obs["proprio"].astype(np.float32).tolist(),
        }
        r = requests.post(
            f"{self.server_url}/predict",
            json=payload,
            timeout=self.request_timeout,
        )
        r.raise_for_status()
        return np.asarray(r.json()["action"], dtype=np.float64)


VLA_REGISTRY = {
    "openvla_oft": OpenVLAOFTPolicy,
    "pi0": Pi0Policy,
    "pi05": Pi05Policy,
    "pi0_fast": Pi0FastPolicy,
    "rdt": RDTPolicy,
    "diffusion_policy": DiffusionPolicyPolicy,
    "dp3": DP3Policy,
    "act": ACTPolicy,
    "vqbet": VQBeTPolicy,
    "lerobot_diffusion": LeRobotDiffusionPolicy,
    "smolvla": SmolVLAPolicy,
    "hri_depth": HriDepthPolicy,
    "template": TemplatePolicy,
}
