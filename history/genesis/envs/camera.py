"""Camera module: head, wrist, and recording cameras for Genesis."""

import os

import numpy as np
import math
import transforms3d as t3d
import genesis as gs
from .exr_envmap import offset_env_texture


def _intrinsic_from_fov(fov_deg, w, h):
    """3x3 intrinsic matrix from vertical FOV (degrees)."""
    fov_rad = np.deg2rad(float(fov_deg))
    fy = (h / 2.0) / math.tan(fov_rad / 2.0)
    fx = fy
    return np.array([[fx, 0, w / 2.0], [0, fy, h / 2.0], [0, 0, 1]], dtype=np.float64)


def _extrinsic_from_pos_lookat(pos, lookat, up=None):
    """4x4 cam-to-world matrix from position and lookat point."""
    pos = np.asarray(pos, dtype=np.float64).ravel()[:3]
    lookat = np.asarray(lookat, dtype=np.float64).ravel()[:3]
    forward = lookat - pos
    n = np.linalg.norm(forward)
    forward = forward / n if n > 1e-9 else np.array([0, 0, -1.0])
    up = np.asarray(up or [0, 0, 1], dtype=np.float64)
    left = np.cross(forward, up)
    n = np.linalg.norm(left)
    left = left / n if n > 1e-9 else np.array([1, 0, 0.0])
    up = np.cross(left, forward)
    up /= np.linalg.norm(up)
    R = np.eye(4)
    R[:3, :3] = np.stack([left, up, -forward], axis=1)
    R[:3, 3] = pos
    return R


def _pose_to_pos_lookat(pose):
    """Convert a Pose-like object to (position, lookat)."""
    if hasattr(pose, "p") and hasattr(pose, "q"):
        pos = np.asarray(pose.p, dtype=np.float64).ravel()[:3]
        R = t3d.quaternions.quat2mat(np.asarray(pose.q).ravel()[:4])
        return pos, pos - R[:, 2]
    if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
        pos = pose[:3, 3]
        return pos, pos - pose[:3, :3][:, 2]
    raise TypeError(f"Cannot convert {type(pose)} to pos/lookat")


def _render_to_uint8_rgb(raw):
    """Convert Genesis render output to HxWx3 uint8."""
    if raw is None:
        return None
    arr = raw.detach().cpu().numpy() if hasattr(raw, "detach") else (
        raw.cpu().numpy() if hasattr(raw, "cpu") else np.asarray(raw))
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] >= 4:
        arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.floating):
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr.astype(np.uint8))


def _resize_rgb(arr, w, h):
    """Resize an HxWx3 uint8 image to (w, h). No-op if already that size."""
    if arr is None or (arr.shape[1] == w and arr.shape[0] == h):
        return arr
    import cv2
    # INTER_AREA for downscale, INTER_LINEAR for upscale.
    interp = cv2.INTER_AREA if (w < arr.shape[1] or h < arr.shape[0]) else cv2.INTER_LINEAR
    return np.ascontiguousarray(cv2.resize(arr, (int(w), int(h)), interpolation=interp))


def _resize_depth(arr, w, h):
    """Resize an HxW depth map to (w, h) with nearest-neighbor (no blending)."""
    if arr is None:
        return arr
    a = np.asarray(arr)
    a2 = a[..., 0] if a.ndim == 3 and a.shape[-1] == 1 else a
    if a2.shape[1] == w and a2.shape[0] == h:
        return a2
    import cv2
    return np.ascontiguousarray(cv2.resize(a2, (int(w), int(h)), interpolation=cv2.INTER_NEAREST))


def _finalize_rgb(cam, raw):
    """Convert + resize a genesis camera's RGB output to its logical resolution.

    Used by the recorder and capture_frame, which call ``cam.render`` directly
    on the genesis camera object (bypassing ``Camera.render_all``). Under the
    batch renderer every camera is added to the scene at a common resolution
    and tagged with ``cam._logical_res = (w, h)``; this resizes back to it. For
    the rasterizer/raytracer path there is no tag and this is a plain convert.
    """
    rgb = _render_to_uint8_rgb(raw)
    res = getattr(cam, "_logical_res", None)
    if res is not None and rgb is not None:
        rgb = _resize_rgb(rgb, res[0], res[1])
    return rgb


def _finalize_depth(cam, raw):
    """Convert + resize a genesis camera's depth output to its logical resolution."""
    if raw is None:
        return None
    arr = raw.detach().cpu().numpy() if hasattr(raw, "detach") else (
        raw.cpu().numpy() if hasattr(raw, "cpu") else np.asarray(raw))
    res = getattr(cam, "_logical_res", None)
    if res is not None:
        arr = _resize_depth(arr, res[0], res[1])
    return np.ascontiguousarray(arr)


class Camera:
    """Genesis camera manager: head camera, wrist cameras, recording camera."""

    def __init__(self, scene: gs.Scene, static_cameras: list = None, camera_config: dict = None,
                 recording_pos=None, recording_lookat=None,
                 side_pos=None, side_lookat=None, recording_fov: float = 60.0,
                 side_fov: float = 60.0, spp: int = None,
                 recording_resolution=None, side_resolution=None,
                 enable_side: bool = True, denoise: bool | None = None,
                 batch_render: bool = False):
        """
        Args:
            scene: Genesis scene (cameras must be added before scene.build())
            static_cameras: list of dicts with {name, position, forward, [type, fov, res]}
            camera_config: dict of camera types -> {w, h, fovy}
            spp: samples per pixel for RayTracer rendering (ignored by rasterizer)
            batch_render: if True, all cameras are added to the scene at a single
                common resolution (Madrona BatchRenderer requires identical
                resolution for every camera) and each camera's render output is
                resized back to its configured (logical) resolution. The common
                resolution is the per-axis max over all cameras, so cameras at
                the max resolution incur no resize (exact parity).
        """
        self.camera_config = camera_config or {"default": {"w": 640, "h": 480, "fovy": 60}}
        self._cameras = {}  # name -> (gs_camera, logical_w, logical_h, fov, pos, lookat)
        self._cache_rgb = {}
        self._cache_depth = {}
        self._attached_wrist_cameras = set()
        self._batch_render = bool(batch_render)

        # Extra kwargs for add_camera (e.g. spp/denoise for RayTracer)
        extra = {}
        if spp is not None:
            extra["spp"] = spp
        if denoise is not None:
            extra["denoise"] = bool(denoise)

        default_cfg = self.camera_config.get("default", {"w": 640, "h": 480, "fovy": 60})

        # ---- Pass 1: gather every camera spec (name, logical res, fov, pose) ----
        specs = []  # list of dicts: name, w, h, fov, pos, lookat, up

        for cam_info in (static_cameras or []):
            name = cam_info["name"]
            cam_type = cam_info.get("type", "default")
            cfg = self.camera_config.get(cam_type, default_cfg)
            w, h = cfg["w"], cfg["h"]
            fov = float(cam_info.get("fov", cam_info.get("fovy", cfg.get("fovy", 60))))
            pos = np.array(cam_info["position"], dtype=np.float64)
            if "lookat" in cam_info:
                lookat = np.array(cam_info["lookat"], dtype=np.float64)
            else:
                forward = np.array(cam_info.get("forward", -pos), dtype=np.float64)
                forward = forward / (np.linalg.norm(forward) + 1e-9)
                lookat = pos + forward
            specs.append(dict(name=name, w=int(w), h=int(h), fov=fov,
                              pos=pos.copy(), lookat=lookat.copy(), up=cam_info.get("up")))

        for name in ("left_wrist", "right_wrist"):
            specs.append(dict(name=name, w=int(default_cfg["w"]), h=int(default_cfg["h"]),
                              fov=float(default_cfg.get("fovy", 60)),
                              pos=np.array([0, 0, 1.0]), lookat=np.array([0, 0, 0.0]), up=None))

        rec_pos = np.array(recording_pos if recording_pos is not None else [0.0, 0.2, 2.0])
        rec_lookat = np.array(recording_lookat if recording_lookat is not None else [0.0, 0.0, 0.9])
        rec_w, rec_h = recording_resolution if recording_resolution is not None else (640, 480)
        specs.append(dict(name="recording", w=int(rec_w), h=int(rec_h),
                          fov=float(recording_fov),
                          pos=rec_pos, lookat=rec_lookat, up=None))

        if enable_side:
            side_pos_arr = np.array(side_pos if side_pos is not None else [1.5, -0.3, 1.2])
            side_lookat_arr = np.array(side_lookat if side_lookat is not None else [0.0, 0.0, 0.85])
            side_w, side_h = side_resolution if side_resolution is not None else (640, 480)
            specs.append(dict(name="side", w=int(side_w), h=int(side_h),
                              fov=float(side_fov),
                              pos=side_pos_arr, lookat=side_lookat_arr, up=None))

        # ---- Common render resolution (batch renderer needs identical res) ----
        if self._batch_render:
            common_w = max(s["w"] for s in specs)
            common_h = max(s["h"] for s in specs)
            aspects = {round(s["w"] / s["h"], 3) for s in specs}
            if len(aspects) > 1:
                import warnings
                warnings.warn(
                    "batch_render: cameras have differing aspect ratios "
                    f"{aspects}; rendering all at {common_w}x{common_h} then "
                    "resizing may distort cameras whose aspect differs.")
            self._render_res = (common_w, common_h)
        else:
            self._render_res = None

        # ---- Pass 2: add cameras to the scene ----
        for s in specs:
            render_w, render_h = (self._render_res if self._batch_render else (s["w"], s["h"]))
            cam_kwargs = dict(res=(render_w, render_h), pos=tuple(s["pos"]),
                              lookat=tuple(s["lookat"]), fov=s["fov"], **extra)
            if s["up"] is not None:
                cam_kwargs["up"] = tuple(np.array(s["up"], dtype=np.float64))
            cam = scene.add_camera(**cam_kwargs)
            # Tag with logical resolution so direct cam.render callers
            # (recorder, capture_frame) can resize via _finalize_rgb/_depth.
            if self._batch_render and (s["w"], s["h"]) != (render_w, render_h):
                cam._logical_res = (s["w"], s["h"])
            self._cameras[s["name"]] = (cam, s["w"], s["h"], s["fov"], s["pos"], s["lookat"])

    def update_wrist_cameras(self, left_pose, right_pose):
        """Update wrist camera positions from robot EE poses.

        Camera is mounted 8 cm behind the TCP along the gripper's -z axis and
        2 cm above along +y, looking forward along +z (the grasp approach
        direction). This matches a typical RealSense-D435 wrist mount.
        """
        for name, pose in [("left_wrist", left_pose), ("right_wrist", right_pose)]:
            if pose is None or name not in self._cameras:
                continue
            try:
                if name in self._attached_wrist_cameras:
                    cam, w, h, fov, _, _ = self._cameras[name]
                    cam.move_to_attach()
                    transform = np.asarray(cam.transform, dtype=np.float64)
                    pos = transform[:3, 3]
                    lookat = pos - transform[:3, :3][:, 2]
                    self._cameras[name] = (cam, w, h, fov, pos, lookat)
                    continue
                tcp = np.asarray(pose.p, dtype=np.float64).ravel()[:3]
                R = t3d.quaternions.quat2mat(np.asarray(pose.q).ravel()[:4])
                z_axis = R[:, 2]
                y_axis = R[:, 1]
                cam_pos = tcp - 0.08 * z_axis + 0.02 * y_axis
                cam_lookat = tcp + 0.5 * z_axis
                cam, w, h, fov, _, _ = self._cameras[name]
                cam.set_pose(pos=tuple(cam_pos), lookat=tuple(cam_lookat))
                self._cameras[name] = (cam, w, h, fov, cam_pos, cam_lookat)
            except Exception:
                pass

    def attach_franka_wrist_camera(self, robot):
        """Attach the Franka right wrist camera to link7 with the calibrated mount."""
        if "right_wrist" not in self._cameras:
            return False
        arm = getattr(robot, "right_arm", None)
        entity = getattr(arm, "entity", None)
        if entity is None:
            return False

        link = None
        for link_name in ("link7", "panda_link7", "hand"):
            for candidate in getattr(entity, "links", []):
                if getattr(candidate, "name", None) == link_name:
                    link = candidate
                    break
            if link is not None:
                break
        if link is None:
            return False

        wrist_cam_T = np.array([
            [0.0, 0.93253449, -0.36108092, 0.08890143],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.36108092, 0.93253449, 0.03220073],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        rot180z = np.diag([-1.0, 1.0, -1.0, 1.0])

        cam, w, h, fov, _, _ = self._cameras["right_wrist"]
        try:
            cam.attach(link, wrist_cam_T @ rot180z)
            self._attached_wrist_cameras.add("right_wrist")
            cam.move_to_attach()
            transform = np.asarray(cam.transform, dtype=np.float64)
            pos = transform[:3, 3]
            lookat = pos - transform[:3, :3][:, 2]
            self._cameras["right_wrist"] = (cam, w, h, fov, pos, lookat)
        except Exception:
            return False
        return True

    def render_all(self):
        """Render all cameras and cache results."""
        self._cache_rgb.clear()
        self._cache_depth.clear()
        for name, (cam, w, h, fov, pos, lookat) in self._cameras.items():
            out = cam.render(rgb=True, depth=True)
            rgb_raw = out[0] if isinstance(out, (list, tuple)) and len(out) > 0 else None
            depth_raw = out[1] if isinstance(out, (list, tuple)) and len(out) > 1 else None
            # _finalize_* resize to the camera's logical res under batch render
            # (no-op for the rasterizer/raytracer path).
            self._cache_rgb[name] = _finalize_rgb(cam, rgb_raw)
            if depth_raw is not None:
                self._cache_depth[name] = _finalize_depth(cam, depth_raw)

    def get_rgb(self, name: str) -> np.ndarray | None:
        """Get cached RGB image for a camera."""
        return self._cache_rgb.get(name)

    def get_depth(self, name: str) -> np.ndarray | None:
        """Get cached depth image for a camera."""
        return self._cache_depth.get(name)

    def get_all_rgb(self) -> dict[str, np.ndarray]:
        """Get all cached RGB images."""
        return {k: v for k, v in self._cache_rgb.items() if v is not None}

    def get_all_depth(self) -> dict[str, np.ndarray]:
        """Get all cached depth maps."""
        return {k: v for k, v in self._cache_depth.items() if v is not None}

    def get_intrinsic(self, name: str) -> np.ndarray:
        """Get 3x3 intrinsic matrix for a camera."""
        if name not in self._cameras:
            raise KeyError(f"Camera '{name}' not found")
        _, w, h, fov, _, _ = self._cameras[name]
        return _intrinsic_from_fov(fov, w, h)

    def get_extrinsic(self, name: str) -> np.ndarray:
        """Get 4x4 cam-to-world matrix for a camera."""
        if name not in self._cameras:
            raise KeyError(f"Camera '{name}' not found")
        _, w, h, fov, pos, lookat = self._cameras[name]
        return _extrinsic_from_pos_lookat(pos, lookat)

    def get_camera_names(self) -> list[str]:
        return list(self._cameras.keys())


class _NyxCameraAdapter:
    """Compatibility shim exposing the ``render`` API used by BaseTask."""

    def __init__(self, sensor):
        self.sensor = sensor

    def render(self, rgb: bool = True, depth: bool = False):
        data = self.sensor.read()
        rgb_out = getattr(data, "rgb", None) if rgb else None
        depth_out = getattr(data, "depth", None) if depth else None
        if depth:
            return rgb_out, depth_out
        return (rgb_out,)


class NyxCameraManager:
    """NYX sensor-backed camera manager with the same task-facing API as Camera."""

    def __init__(
        self,
        scene: gs.Scene,
        static_cameras: list = None,
        camera_config: dict = None,
        recording_pos=None,
        recording_lookat=None,
        side_pos=None,
        side_lookat=None,
        recording_fov: float = 60.0,
        side_fov: float = 60.0,
        recording_resolution=None,
        side_resolution=None,
        enable_side: bool = True,
        nyx_options: dict | None = None,
    ):
        from gs_nyx_plugin import nyx_camera_sensor as _nyx_camera_sensor  # noqa: F401
        from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
        from .nyx_kinematic_patch import apply_nyx_kinematic_patch

        apply_nyx_kinematic_patch()

        self.camera_config = camera_config or {"default": {"w": 640, "h": 480, "fovy": 60}}
        self._cameras = {}
        self._cache_rgb = {}
        self._cache_depth = {}
        self._scene = scene
        self._NyxCameraOptions = NyxCameraOptions
        self._nyx_options = dict(nyx_options or {})
        # Environment-map controls are ours, not NyxCameraOptions kwargs — pop
        # them before the option dict is splatted into sensors.
        self._pending_env_maps = self._build_env_maps(
            self._nyx_options.pop("env_texture", None),
            float(self._nyx_options.pop("env_rotation", 0.0) or 0.0),
            float(self._nyx_options.pop("env_multiplier", 1.0) or 1.0),
            self._nyx_options.pop("env_offset", (0.0, 0.0)),
        )

        default_cfg = self.camera_config.get("default", {"w": 640, "h": 480, "fovy": 60})

        for cam_info in (static_cameras or []):
            name = cam_info["name"]
            cam_type = cam_info.get("type", "default")
            cfg = self.camera_config.get(cam_type, default_cfg)
            w, h = int(cfg["w"]), int(cfg["h"])
            fov = float(cam_info.get("fov", cam_info.get("fovy", cfg.get("fovy", 60))))
            pos = np.asarray(cam_info["position"], dtype=np.float64)
            if "lookat" in cam_info:
                lookat = np.asarray(cam_info["lookat"], dtype=np.float64)
            else:
                forward = np.asarray(cam_info.get("forward", -pos), dtype=np.float64)
                forward = forward / (np.linalg.norm(forward) + 1e-9)
                lookat = pos + forward
            self._add_camera(name, w, h, fov, pos, lookat, cam_info.get("up"))

        # Wrist cameras (placeholder pose at build; moved to the EE each step by
        # update_wrist_cameras). Mirrors the rasterizer Camera so nyx provides
        # the same 5-camera set (head + 2 wrist + recording + side) that VLA
        # collection/eval consume.
        wrist_w, wrist_h = int(default_cfg["w"]), int(default_cfg["h"])
        wrist_fov = float(default_cfg.get("fovy", 60))
        for name in ("left_wrist", "right_wrist"):
            self._add_camera(name, wrist_w, wrist_h, wrist_fov,
                             np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0]))

        rec_pos = np.asarray(recording_pos if recording_pos is not None else [0.0, 0.2, 2.0])
        rec_lookat = np.asarray(recording_lookat if recording_lookat is not None else [0.0, 0.0, 0.9])
        rec_w, rec_h = recording_resolution if recording_resolution is not None else (640, 480)
        self._add_camera(
            "recording", int(rec_w), int(rec_h), float(recording_fov),
            rec_pos, rec_lookat,
        )

        if enable_side:
            side_pos_arr = np.asarray(side_pos if side_pos is not None else [1.5, -0.3, 1.2])
            side_lookat_arr = np.asarray(side_lookat if side_lookat is not None else [0.0, 0.0, 0.85])
            side_w, side_h = side_resolution if side_resolution is not None else (640, 480)
            self._add_camera(
                "side", int(side_w), int(side_h), float(side_fov),
                side_pos_arr, side_lookat_arr,
            )

    @staticmethod
    def _build_env_maps(env_texture, rotation_deg, multiplier, offset=(0.0, 0.0)):
        """Build the HDRI environment map (background + image-based lighting).

        Returns a tuple of gs_nyx EnvironmentMapAsset for NyxCameraOptions.
        The scene exporter concatenates env_maps across ALL sensors, so the
        result must be attached to exactly one sensor (see _add_camera).
        """
        if not env_texture:
            return ()
        import gs_nyx.nyx_py_sdk as nps
        from .utils import ROOT_PATH

        tex_path = str(env_texture)
        if not os.path.isabs(tex_path):
            tex_path = str(ROOT_PATH / tex_path)
        if not os.path.isfile(tex_path):
            print(f"[WARNING] nyx env_texture not found, skipping: {tex_path}")
            return ()
        tex_path = offset_env_texture(tex_path, offset)
        env_map = nps.EnvironmentMapAsset()
        env_map.texture = tex_path
        env_map.layout = nps.EEnvMapLayout.LongLat
        env_map.rotation = math.radians(rotation_deg)
        env_map.multiplier = multiplier
        return (env_map,)

    def _add_camera(self, name, w, h, fov, pos, lookat, up=None):
        kwargs = dict(
            res=(int(w), int(h)),
            pos=tuple(np.asarray(pos, dtype=np.float64).ravel()[:3]),
            lookat=tuple(np.asarray(lookat, dtype=np.float64).ravel()[:3]),
            fov=float(fov),
        )
        if up is not None:
            kwargs["up"] = tuple(np.asarray(up, dtype=np.float64).ravel()[:3])
        kwargs.update(self._nyx_options)
        if self._pending_env_maps:
            # Attach to the first sensor only — the exporter gathers env_maps
            # from every sensor, so per-sensor attachment would duplicate it.
            kwargs["env_maps"] = self._pending_env_maps
            self._pending_env_maps = ()
        sensor = self._scene.add_sensor(self._NyxCameraOptions(**kwargs))
        self._cameras[name] = (
            _NyxCameraAdapter(sensor),
            int(w),
            int(h),
            float(fov),
            np.asarray(pos, dtype=np.float64).copy(),
            np.asarray(lookat, dtype=np.float64).copy(),
        )

    def update_wrist_cameras(self, left_pose, right_pose):
        """Move the wrist nyx sensors to the current EE poses.

        Same RealSense-D435 mount as the rasterizer Camera: 8 cm behind the TCP
        along -z, 2 cm above along +y, looking forward along +z. Updates the nyx
        sensor pose (applied on next render) and the cached pose used by
        get_extrinsic.
        """
        for name, pose in [("left_wrist", left_pose), ("right_wrist", right_pose)]:
            if pose is None or name not in self._cameras:
                continue
            try:
                tcp = np.asarray(pose.p, dtype=np.float64).ravel()[:3]
                R = t3d.quaternions.quat2mat(np.asarray(pose.q).ravel()[:4])
                z_axis = R[:, 2]
                y_axis = R[:, 1]
                cam_pos = tcp - 0.08 * z_axis + 0.02 * y_axis
                cam_lookat = tcp + 0.5 * z_axis
                cam, w, h, fov, _, _ = self._cameras[name]
                cam.sensor.update_camera_pose(pos=tuple(cam_pos), lookat=tuple(cam_lookat))
                self._cameras[name] = (cam, w, h, fov, cam_pos, cam_lookat)
            except Exception:
                pass

    def attach_franka_wrist_camera(self, robot):
        # nyx wrist cameras track the EE via update_wrist_cameras each step
        # (no rigid link attach needed); report no-attach like the static cams.
        return False

    def render_all(self):
        self._cache_rgb.clear()
        self._cache_depth.clear()
        for name, (cam, _w, _h, _fov, _pos, _lookat) in self._cameras.items():
            rgb_raw, depth_raw = cam.render(rgb=True, depth=True)
            rgb = _render_to_uint8_rgb(rgb_raw)
            if rgb is not None:
                self._cache_rgb[name] = rgb
            if depth_raw is not None:
                self._cache_depth[name] = np.asarray(
                    depth_raw.cpu().numpy() if hasattr(depth_raw, "cpu") else depth_raw
                )

    def get_rgb(self, name: str) -> np.ndarray | None:
        return self._cache_rgb.get(name)

    def get_depth(self, name: str) -> np.ndarray | None:
        return self._cache_depth.get(name)

    def get_all_rgb(self) -> dict[str, np.ndarray]:
        return {k: v for k, v in self._cache_rgb.items() if v is not None}

    def get_all_depth(self) -> dict[str, np.ndarray]:
        return {k: v for k, v in self._cache_depth.items() if v is not None}

    def get_intrinsic(self, name: str) -> np.ndarray:
        if name not in self._cameras:
            raise KeyError(f"Camera '{name}' not found")
        _, w, h, fov, _, _ = self._cameras[name]
        return _intrinsic_from_fov(fov, w, h)

    def get_extrinsic(self, name: str) -> np.ndarray:
        if name not in self._cameras:
            raise KeyError(f"Camera '{name}' not found")
        _, _, _, _, pos, lookat = self._cameras[name]
        return _extrinsic_from_pos_lookat(pos, lookat)

    def get_camera_names(self) -> list[str]:
        return list(self._cameras.keys())
