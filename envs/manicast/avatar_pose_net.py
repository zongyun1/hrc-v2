"""AvatarPoseNet: head-camera RGB -> avatar 3D joints (camera frame).

Observation-only perception front-end for the ManiCast-ACT baseline. The
network regresses J avatar joint positions in the *camera* frame plus a
presence logit; the runtime wrapper maps to world frame using the camera
extrinsics provided in the observation dict (cam-to-world, part of obs —
not privileged state). Ground-truth joints are used as supervision at
training time only.
"""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AvatarPoseNet(nn.Module):
    """~2M-param CNN. Input (N,3,H,W) in [0,1]; output (N, J*3 + 1)."""

    def __init__(self, n_joints: int, width: int = 32):
        super().__init__()
        w = width
        self.backbone = nn.Sequential(
            ConvBlock(3, w),        # /2
            ConvBlock(w, w * 2),    # /4
            ConvBlock(w * 2, w * 4),  # /8
            ConvBlock(w * 4, w * 8),  # /16
            ConvBlock(w * 8, w * 8),  # /32
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(w * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_joints * 3 + 1),
        )
        self.n_joints = n_joints

    def forward(self, x):
        f = self.pool(self.backbone(x))
        out = self.head(f)
        joints = out[:, : self.n_joints * 3].view(-1, self.n_joints, 3)
        presence_logit = out[:, -1]
        return joints, presence_logit


def preprocess_rgb(rgb_uint8: np.ndarray, size_hw=(192, 256)) -> torch.Tensor:
    """(H,W,3) uint8 -> (3,h,w) float tensor in [0,1], resized."""
    import cv2

    img = cv2.resize(rgb_uint8, (size_hw[1], size_hw[0]),
                     interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)


class AvatarPoseEstimator:
    """Runtime wrapper: obs dict -> (J,3) world-frame joints + presence."""

    def __init__(self, ckpt_path: str, camera: str = "head_camera",
                 device: str | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(ckpt_path, map_location=self.device,
                          weights_only=False)
        self.bones = list(ckpt["bones"])
        self.size_hw = tuple(ckpt.get("size_hw", (192, 256)))
        self.model = AvatarPoseNet(len(self.bones),
                                   width=int(ckpt.get("width", 32)))
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        self.camera = camera

    @torch.no_grad()
    def estimate(self, obs: dict):
        """Returns (joints_world (J,3) float64 | None, presence float)."""
        rgb = obs.get("rgb", {}).get(self.camera)
        ext = obs.get("camera_extrinsics", {}).get(self.camera)
        if rgb is None or ext is None:
            return None, 0.0
        rgb = np.asarray(rgb)
        if rgb.ndim == 4:
            rgb = rgb[0]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * 255.0 if rgb.max() <= 1.5 else rgb,
                          0, 255).astype(np.uint8)
        x = preprocess_rgb(rgb[..., :3], self.size_hw)[None].to(self.device)
        joints_cam, logit = self.model(x)
        joints_cam = joints_cam[0].cpu().numpy().astype(np.float64)
        presence = float(torch.sigmoid(logit[0]))
        E = np.asarray(ext, dtype=np.float64)  # cam-to-world
        world = joints_cam @ E[:3, :3].T + E[:3, 3]
        return world, presence


def joints_world_to_cam(joints_world: np.ndarray, extrinsic: np.ndarray):
    """World (.., 3) -> camera frame using cam-to-world extrinsic."""
    E = np.asarray(extrinsic, dtype=np.float64)
    R, t = E[:3, :3], E[:3, 3]
    return (np.asarray(joints_world, np.float64) - t) @ R


def joint_visibility(joints_cam: np.ndarray, intrinsic: np.ndarray,
                     resolution_wh, margin: float = 0.0):
    """Per-joint in-frustum mask for OpenGL-style camera frames (-z forward).

    joints_cam: (..., J, 3) camera-frame positions.
    Returns boolean (..., J).
    """
    K = np.asarray(intrinsic, dtype=np.float64)
    W, H = resolution_wh
    x = joints_cam[..., 0]
    y = -joints_cam[..., 1]
    z = -joints_cam[..., 2]
    in_front = z > 0.05
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * x / z + K[0, 2]
        v = K[1, 1] * y / z + K[1, 2]
    inside = ((u >= -margin) & (u < W + margin)
              & (v >= -margin) & (v < H + margin))
    return in_front & inside


def load_cam_meta(h5_attrs) -> dict:
    return json.loads(h5_attrs["cam_meta"])
