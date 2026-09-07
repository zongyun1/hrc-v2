"""Fast NumPy visual detector for observation-only PACE-like triggering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class FastVisualPaceState:
    present: float = 0.0
    active: float = 0.0
    progress: float = 0.0
    best_progress: float = 0.0
    motion_score: float = 0.0
    fired: bool = False


def extract_head_rgb(obs: dict[str, Any], camera: str = "head_camera") -> np.ndarray | None:
    rgb = obs.get("rgb") if isinstance(obs, dict) else None
    img = rgb.get(camera) if isinstance(rgb, dict) else None
    if img is None and isinstance(rgb, dict) and rgb:
        img = next(iter(rgb.values()))
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


def top_features(rgb: np.ndarray, crop_top_frac: float) -> np.ndarray:
    h = rgb.shape[0]
    y1 = max(1, int(round(h * crop_top_frac)))
    crop = rgb[:y1].astype(np.float32) / 255.0
    hsv = cv2.cvtColor((crop * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    mean = crop.reshape(-1, 3).mean(axis=0)
    std = crop.reshape(-1, 3).std(axis=0)
    hsv_mean = hsv.reshape(-1, 3).mean(axis=0) / np.array([179.0, 255.0, 255.0], dtype=np.float32)
    non_gray = np.mean(hsv[..., 1] > 28.0)
    dark = np.mean(hsv[..., 2] < 80.0)
    return np.concatenate([mean, std, hsv_mean, [non_gray, dark]]).astype(np.float32)


class FastVisualProgressDetector:
    def __init__(
        self,
        checkpoint_path: str,
        camera: str = "head_camera",
        present_threshold: float = 0.5,
        active_threshold: float = 0.5,
        trigger_progress: float = 0.35,
        motion_threshold: float | None = None,
        motion_steps_to_full: float | None = None,
        min_updates: int = 0,
    ):
        ckpt = np.load(checkpoint_path)
        self.camera = camera
        self.crop_top_frac = float(ckpt["crop_top_frac"])
        self.weight = ckpt["weight"].astype(np.float32)
        self.bias = float(ckpt["bias"])
        self.motion_threshold = (
            float(ckpt["motion_threshold"])
            if motion_threshold is None
            else float(motion_threshold)
        )
        self.motion_steps_to_full = max(
            1.0,
            float(ckpt["motion_steps_to_full"])
            if motion_steps_to_full is None
            else float(motion_steps_to_full),
        )
        self.min_updates = max(0, int(min_updates))
        self.updates = 0
        self.present_threshold = float(present_threshold)
        self.active_threshold = float(active_threshold)
        self.trigger_progress = float(trigger_progress)
        self.prev_rgb = None
        self.state = FastVisualPaceState()

    def reset(self):
        self.prev_rgb = None
        self.state = FastVisualPaceState()
        self.updates = 0

    def update(self, obs: dict[str, Any]) -> FastVisualPaceState:
        curr = extract_head_rgb(obs, self.camera)
        if curr is None:
            return self.state
        self.updates += 1
        features = top_features(curr, self.crop_top_frac)
        logit = float(np.dot(features, self.weight) + self.bias)
        present = float(1.0 / (1.0 + np.exp(-np.clip(logit, -50.0, 50.0))))
        motion_score = 0.0
        if self.prev_rgb is not None:
            h = curr.shape[0]
            y1 = max(1, int(round(h * self.crop_top_frac)))
            prev_gray = cv2.cvtColor(self.prev_rgb[:y1], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(curr[:y1], cv2.COLOR_RGB2GRAY)
            motion_score = float(
                np.mean(np.abs(prev_gray.astype(np.float32) - curr_gray.astype(np.float32))) / 255.0
            )
        self.prev_rgb = curr.copy()
        motion_active = motion_score >= self.motion_threshold
        self.state.present = present
        self.state.motion_score = motion_score
        self.state.active = 1.0 if motion_active else 0.0
        if present >= self.present_threshold and motion_active:
            self.state.progress = min(1.0, self.state.progress + 1.0 / self.motion_steps_to_full)
            self.state.best_progress = max(self.state.best_progress, self.state.progress)
        else:
            self.state.progress *= 0.98
        return self.state

    def should_trigger(self) -> tuple[bool, str | None]:
        if self.state.fired:
            return False, None
        if self.updates < self.min_updates:
            return False, None
        if (
            self.state.present >= self.present_threshold
            and self.state.active >= self.active_threshold
            and self.state.best_progress >= self.trigger_progress
        ):
            self.state.fired = True
            return True, "fast_visual_pace_progress"
        return False, None
