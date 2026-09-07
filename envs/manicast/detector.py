"""ManiCast-ACT trigger: anticipatory yield from forecasted avatar motion.

Observation-only (no privileged sim state):
  obs RGB -> AvatarPoseNet -> world-frame joints (via obs camera extrinsics)
  joint history -> STS-GCN forecaster -> 1 s wrist forecast
  trigger when the predicted wrist comes within ``trigger_dist`` of the
  robot's current TCP (from obs proprio) within the forecast horizon.

Duck-types the visual-PACE detector interface used by scripts/eval.py
(``update(obs) -> state``, ``should_trigger() -> (bool, reason)``) and adds
``human_clear()`` so the generic yield wrapper can end its wait phase from
observations instead of privileged ``avatar.spare()``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from .avatar_pose_net import AvatarPoseEstimator
from .stsgcn import STSGCN


@dataclass
class ManiCastState:
    present: float = 0.0
    active: float = 0.0
    progress: float = 0.0
    best_progress: float = 0.0
    motion_score: float = 0.0
    wrist_dist_now: float | None = None
    wrist_dist_pred_min: float | None = None
    clear_streak: int = 0
    history_len: int = 0
    fired: bool = False


class ManiCastDetector:
    def __init__(
        self,
        forecaster_path: str,
        pose_path: str,
        camera: str = "head_camera",
        trigger_dist: float = 0.30,
        clear_dist: float = 0.35,
        clear_consecutive: int = 5,
        presence_threshold: float = 0.5,
        min_updates: int = 4,
        horizon_steps: int | None = None,
        device: str | None = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.pose = AvatarPoseEstimator(pose_path, camera=camera,
                                        device=str(self.device))
        ckpt = torch.load(forecaster_path, map_location=self.device,
                          weights_only=False)
        self.bones = list(ckpt["bones"])
        self.t_in = int(ckpt["t_in"])
        self.t_out = int(ckpt["t_out"])
        self.root_idx = self.bones.index(ckpt.get("root_bone", "Hips"))
        self.wrist_idx = [self.bones.index(b)
                          for b in ckpt.get("wrist_bones",
                                            ("LeftHand", "RightHand"))
                          if b in self.bones]
        self.model = STSGCN(input_time_frame=self.t_in,
                            output_time_frame=self.t_out,
                            joints_to_consider=len(self.bones))
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        if self.pose.bones != self.bones:
            raise ValueError(
                f"pose bones {self.pose.bones} != forecaster bones {self.bones}")

        self.trigger_dist = float(trigger_dist)
        self.clear_dist = float(clear_dist)
        self.clear_consecutive = int(clear_consecutive)
        self.presence_threshold = float(presence_threshold)
        self.min_updates = int(min_updates)
        self.horizon_steps = (int(horizon_steps) if horizon_steps
                              else self.t_out)

        self.history: deque = deque(maxlen=self.t_in)
        self.updates = 0
        self.saved_ee: np.ndarray | None = None
        self.state = ManiCastState()
        self._pending_reason: str | None = None

    def reset(self):
        self.history.clear()
        self.updates = 0
        self.saved_ee = None
        self.state = ManiCastState()
        self._pending_reason = None

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _forecast(self) -> np.ndarray | None:
        if len(self.history) < self.t_in:
            return None
        seq = np.stack(self.history)  # (T_in, J, 3)
        center = seq[-1, self.root_idx, :].copy()
        x = (seq - center).transpose(2, 0, 1)[None]  # (1,3,T,J)
        x_t = torch.from_numpy(x.astype(np.float32)).to(self.device)
        pred = self.model(x_t)[0].permute(0, 2, 1).cpu().numpy()  # (T_out,J,3)
        return pred + center

    @staticmethod
    def _robot_tcp(obs) -> np.ndarray | None:
        ee = obs.get("ee_pose", {}) if isinstance(obs, dict) else {}
        p = ee.get("right")
        if p is None and ee:
            p = next(iter(ee.values()))
        if p is None:
            return None
        return np.asarray(p, dtype=np.float64).ravel()[:3]

    def _wrist_min_dist(self, joints_seq: np.ndarray, ref: np.ndarray) -> float:
        wr = joints_seq[..., self.wrist_idx, :]  # (...,W,3)
        return float(np.linalg.norm(wr - ref, axis=-1).min())

    # ------------------------------------------------------------------
    def update(self, obs) -> ManiCastState:
        self.updates += 1
        joints, presence = self.pose.estimate(obs)
        self.state.present = float(presence)
        if joints is None or presence < self.presence_threshold:
            # Avatar out of view: pose output is unconstrained — do not feed
            # the forecaster. Drop stale history so a later re-entry does not
            # splice across the invisibility gap. Out-of-view also counts
            # toward wait-phase clearance.
            self._absent_streak = getattr(self, "_absent_streak", 0) + 1
            if self._absent_streak >= 3:
                self.history.clear()
            if self.saved_ee is not None:
                self.state.clear_streak += 1
            self.state.history_len = len(self.history)
            return self.state
        self._absent_streak = 0
        prev = self.history[-1] if self.history else None
        self.history.append(joints)
        self.state.history_len = len(self.history)
        if prev is not None:
            self.state.motion_score = float(
                np.linalg.norm(joints[self.wrist_idx] - prev[self.wrist_idx],
                               axis=-1).max())

        tcp = self._robot_tcp(obs)
        ref = self.saved_ee if self.saved_ee is not None else tcp
        if ref is None:
            return self.state

        self.state.wrist_dist_now = self._wrist_min_dist(joints[None], ref)
        forecast = self._forecast()
        if forecast is not None:
            fc = forecast[: self.horizon_steps]
            pred_min = min(self._wrist_min_dist(fc, ref),
                           self.state.wrist_dist_now)
            self.state.wrist_dist_pred_min = pred_min
            self.state.active = 1.0 if pred_min < self.trigger_dist else 0.0
            # progress-style scalar for trace compatibility: 1 = contact
            self.state.progress = float(
                np.clip(1.0 - pred_min / max(1e-6, 2 * self.trigger_dist),
                        0.0, 1.0))
            self.state.best_progress = max(self.state.best_progress,
                                           self.state.progress)
            # wait-phase clearance bookkeeping
            clear_now = (self.state.wrist_dist_now > self.clear_dist
                         and pred_min > self.clear_dist)
            self.state.clear_streak = (self.state.clear_streak + 1
                                       if clear_now else 0)

            if (not self.state.fired
                    and self.updates >= self.min_updates
                    and self.state.present >= self.presence_threshold
                    and pred_min < self.trigger_dist):
                self._pending_reason = "manicast_forecast_proximity"
        return self.state

    def should_trigger(self):
        if self.state.fired or self._pending_reason is None:
            return False, None
        self.state.fired = True
        reason = self._pending_reason
        self._pending_reason = None
        return True, reason

    # -- wait-phase support (observation-based, replaces avatar.spare()) ---
    def note_yield_started(self, saved_ee_pos=None):
        """Freeze the reference point used for clearance checks."""
        if saved_ee_pos is not None:
            self.saved_ee = np.asarray(saved_ee_pos, dtype=np.float64).ravel()[:3]
        elif self.history:
            self.saved_ee = None  # fall back to live TCP
        self.state.clear_streak = 0

    def human_clear(self) -> bool:
        return self.state.clear_streak >= self.clear_consecutive
