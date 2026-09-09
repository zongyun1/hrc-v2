"""Small visual progress model for PACE-like interrupt triggering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn


class VisualProgressNet(nn.Module):
    def __init__(self, in_channels: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.head(self.features(x))


@dataclass
class VisualPaceState:
    present: float = 0.0
    active: float = 0.0
    progress: float = 0.0
    best_progress: float = 0.0
    motion_score: float = 0.0
    fired: bool = False


def _extract_head_rgb(obs: dict[str, Any], camera: str = "head_camera") -> np.ndarray | None:
    rgb = obs.get("rgb") if isinstance(obs, dict) else None
    if isinstance(rgb, dict):
        img = rgb.get(camera)
        if img is None and rgb:
            img = next(iter(rgb.values()))
    else:
        img = obs.get(camera) if isinstance(obs, dict) else None
    if img is None:
        return None
    arr = np.asarray(img)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        return None
    if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if arr.max(initial=0) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def preprocess_pair(
    prev_rgb: np.ndarray,
    curr_rgb: np.ndarray,
    image_size: int = 128,
    crop_top_frac: float = 0.48,
) -> torch.Tensor:
    h = curr_rgb.shape[0]
    y1 = max(1, int(round(h * crop_top_frac)))
    prev = prev_rgb[:y1]
    curr = curr_rgb[:y1]
    prev = cv2.resize(prev, (image_size, image_size), interpolation=cv2.INTER_AREA)
    curr = cv2.resize(curr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    x = np.concatenate([prev, curr], axis=2).astype(np.float32) / 255.0
    x = np.moveaxis(x, -1, 0)
    return torch.from_numpy(x)


class VisualProgressDetector:
    def __init__(
        self,
        checkpoint_path: str,
        device: str | None = None,
        camera: str = "head_camera",
        present_threshold: float = 0.5,
        active_threshold: float = 0.5,
        trigger_progress: float = 0.35,
        image_size: int = 128,
        crop_top_frac: float = 0.48,
    ):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.image_size = int(ckpt.get("image_size", image_size))
        self.crop_top_frac = float(ckpt.get("crop_top_frac", crop_top_frac))
        self.camera = camera
        self.present_threshold = float(present_threshold)
        self.active_threshold = float(active_threshold)
        self.trigger_progress = float(trigger_progress)
        self.motion_threshold = float(ckpt.get("motion_threshold", 0.018))
        self.motion_progress_gain = 1.0 / max(1.0, float(ckpt.get("motion_steps_to_full", 16.0)))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = VisualProgressNet()
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        self.prev_rgb = None
        self.state = VisualPaceState()

    def reset(self):
        self.prev_rgb = None
        self.state = VisualPaceState()

    @torch.no_grad()
    def update(self, obs: dict[str, Any]) -> VisualPaceState:
        curr = _extract_head_rgb(obs, self.camera)
        if curr is None:
            return self.state
        if self.prev_rgb is None:
            self.prev_rgb = curr.copy()
            return self.state
        h = curr.shape[0]
        y1 = max(1, int(round(h * self.crop_top_frac)))
        prev_gray = cv2.cvtColor(self.prev_rgb[:y1], cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr[:y1], cv2.COLOR_RGB2GRAY)
        motion_score = float(
            np.mean(np.abs(prev_gray.astype(np.float32) - curr_gray.astype(np.float32))) / 255.0
        )
        x = preprocess_pair(
            self.prev_rgb,
            curr,
            image_size=self.image_size,
            crop_top_frac=self.crop_top_frac,
        ).unsqueeze(0).to(self.device)
        logits = self.model(x)[0]
        present = float(torch.sigmoid(logits[0]).item())
        active = float(torch.sigmoid(logits[1]).item())
        progress = float(torch.sigmoid(logits[2]).item())
        self.prev_rgb = curr.copy()
        self.state.present = present
        self.state.motion_score = motion_score
        motion_active = 1.0 if motion_score >= self.motion_threshold else 0.0
        self.state.active = max(active, motion_active)
        if present >= self.present_threshold and motion_active:
            self.state.progress = min(
                1.0,
                max(float(self.state.progress), progress) + self.motion_progress_gain,
            )
            self.state.best_progress = max(self.state.best_progress, self.state.progress)
        else:
            self.state.progress = max(float(self.state.progress) * 0.98, progress * 0.25)
        return self.state

    def should_trigger(self) -> tuple[bool, str | None]:
        state = self.state
        if state.fired:
            return False, None
        active = state.active >= self.active_threshold
        present = state.present >= self.present_threshold
        enough = state.best_progress >= self.trigger_progress
        if present and active and enough:
            state.fired = True
            return True, "visual_pace_progress"
        return False, None
