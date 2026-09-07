"""Policy evaluation: run a policy on a task and report success rate."""

import argparse
import hashlib
import os
import sys
import yaml
import time
import numpy as np
import json
import traceback
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.tasks import TASK_MAP, resolve_task_class
from envs.utils import load_config
from envs.scene_init import (
    episode_scene_init,
    load_scene_init_file,
    prepare_config_for_scene_init,
)


def _resolve_eval_task_class(task_name: str, config: dict):
    static_avatar = bool((config or {}).get("static_avatar", False))
    resolved_task_name, TaskClass = resolve_task_class(
        task_name,
        no_human=bool((config or {}).get("no_human", False) or static_avatar),
    )
    if static_avatar:
        from scripts.vla_data.collect_vla import with_static_avatar
        TaskClass = with_static_avatar(TaskClass)
    return resolved_task_name, TaskClass


def _array_digest(arr) -> dict:
    a = np.asarray(arr)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "sha1": hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest(),
    }


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _host_array(value):
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    try:
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


def _vec3(value):
    return np.asarray(_host_array(value), dtype=float).ravel()[:3].tolist()


def _merge_config(base: dict, override: dict | None) -> dict:
    out = deepcopy(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_config(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _flat_public_state(obj) -> dict:
    if obj is None:
        return {}
    out = {}
    for key, value in vars(obj).items():
        if key.startswith("_"):
            continue
        if isinstance(value, (bool, int, float, str)) or value is None:
            out[key] = value
    return out


def _summarize_policy_obs(obs: dict) -> dict:
    summary = {}
    if not isinstance(obs, dict):
        return summary
    rgb = obs.get("rgb")
    if isinstance(rgb, dict):
        summary["rgb"] = {
            str(k): _array_digest(v)
            for k, v in rgb.items()
            if v is not None
        }
    for key in (
        "qpos", "joint_pos", "gripper_val", "ee_pose", "state",
        "robot_state",
    ):
        if key in obs:
            try:
                summary[key] = np.asarray(obs[key], dtype=float).ravel().tolist()
            except Exception:
                pass
    return summary


def _reserve_video_path(video_dir: str, filename: str, max_videos: int | None) -> tuple[str | None, str | None]:
    if max_videos is None or max_videos < 0:
        return os.path.join(video_dir, filename), None

    os.makedirs(video_dir, exist_ok=True)
    lock_dir = os.path.join(video_dir, ".video_slots.lock")
    for _ in range(200):
        try:
            os.mkdir(lock_dir)
            break
        except FileExistsError:
            time.sleep(0.05)
    else:
        return None, None

    marker = None
    try:
        videos = [name for name in os.listdir(video_dir) if name.endswith(".mp4")]
        reservations = [name for name in os.listdir(video_dir) if name.startswith(".video_reserve_")]
        if len(videos) + len(reservations) >= max_videos:
            return None, None
        marker = os.path.join(video_dir, f".video_reserve_{filename}")
        with open(marker, "w") as f:
            f.write(str(time.time()))
        return os.path.join(video_dir, filename), marker
    finally:
        try:
            os.rmdir(lock_dir)
        except OSError:
            pass


def _release_video_reservation(marker: str | None) -> None:
    if not marker:
        return
    try:
        os.remove(marker)
    except OSError:
        pass


class ExpertFullStatePolicy:
    """Privileged scripted expert used for data rollouts.

    This intentionally receives the task object instead of only observation
    tensors: the baseline is the full-state scripted policy, not a learned
    observation-only controller.
    """

    def reset(self):
        pass

    def run_episode(self, task, obs, max_steps: int, action_type: str):
        return bool(task.play_once()), None


def _should_save_video(ep_index: int, video_dir: str | None, video_limit: int | None) -> bool:
    if not video_dir:
        return False
    if video_limit is None:
        return True
    return ep_index < max(0, int(video_limit))


def _reset_failure_result(seed: int, elapsed_s: float, error: str) -> dict:
    return {
        "seed": seed,
        "success": False,
        "steps": 0,
        "elapsed_s": elapsed_s,
        "error": error,
        "stage_failed": "reset",
    }


def _scene_init_failure_result(
    seed: int,
    scene_init: dict | None,
    elapsed_s: float,
    error: str,
) -> dict:
    result = _reset_failure_result(seed, elapsed_s, error)
    if scene_init:
        result["scene_init_id"] = scene_init.get("id")
        result["scene_init_source_seed"] = scene_init.get("source_seed")
    return result


def _policy_metrics_dict(policy_fn):
    """Return serializable diagnostics from privileged baseline policies."""
    metrics = getattr(policy_fn, "last_metrics", None)
    if metrics is None:
        return None
    if hasattr(metrics, "as_dict"):
        return metrics.as_dict()
    if hasattr(metrics, "__dataclass_fields__"):
        out = {}
        for key in metrics.__dataclass_fields__:
            value = getattr(metrics, key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                out[key] = value
            elif isinstance(value, (list, tuple)):
                out[key] = list(value)
            elif isinstance(value, dict):
                out[key] = dict(value)
            elif isinstance(value, set):
                out[key] = sorted(value)
            else:
                out[key] = repr(value)
        return out
    return None


def _print_policy_metrics(policy_name: str, policy_fn):
    metrics = _policy_metrics_dict(policy_fn)
    if not metrics:
        return
    keys = (
        "success",
        "steps",
        "fallback_used",
        "time_progress_fallback_used",
        "first_assist_step",
        "robot_idle_time",
        "human_idle_time",
        "ticks",
        "skill_calls",
        "successes",
        "failures",
        "recoveries",
    )
    parts = []
    for key in keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]}")
    if not parts:
        parts.append(f"keys={sorted(metrics)}")
    print(f"  {policy_name}-metrics: " + ", ".join(parts))


def _privileged_policy_step_count(policy_fn):
    """Best-effort episode step count for privileged full-state baselines."""
    last_metrics = getattr(policy_fn, "last_metrics", None)
    if last_metrics is not None:
        steps = getattr(last_metrics, "steps", None)
        if steps:
            return int(steps)
        events = getattr(last_metrics, "events", None)
        if events:
            if isinstance(events, (list, tuple, set, dict)):
                return len(events)
            return int(events)
        event_log = getattr(last_metrics, "event_log", None)
        if event_log:
            return len(event_log)

    last_context = getattr(policy_fn, "last_context", None)
    steps = getattr(last_context, "step_count", None) if last_context else None
    return int(steps) if steps else None


def _make_visual_pace_detector(config: dict):
    manicast_path = config.get("manicast_model")
    if manicast_path:
        from envs.manicast.detector import ManiCastDetector

        return ManiCastDetector(
            manicast_path,
            config["manicast_pose_model"],
            camera=str(config.get("pace_visual_camera", "head_camera")),
            trigger_dist=float(config.get("manicast_trigger_dist", 0.30)),
            clear_dist=float(config.get("manicast_clear_dist", 0.35)),
            clear_consecutive=int(config.get("manicast_clear_consecutive", 5)),
            presence_threshold=float(config.get("manicast_presence_threshold", 0.5)),
            min_updates=int(config.get("manicast_min_updates", 4)),
            horizon_steps=config.get("manicast_horizon_steps"),
        )
    model_path = config.get("pace_visual_model")
    if not model_path:
        return None
    if str(model_path).endswith(".npz"):
        from envs.pace.fast_visual_progress import FastVisualProgressDetector

        return FastVisualProgressDetector(
            model_path,
            camera=str(config.get("pace_visual_camera", "head_camera")),
            present_threshold=float(config.get("pace_visual_present_threshold", 0.5)),
            active_threshold=float(config.get("pace_visual_active_threshold", 0.5)),
            trigger_progress=float(config.get("pace_visual_trigger_progress", 0.35)),
            motion_threshold=(
                None if config.get("pace_visual_motion_threshold") is None
                else float(config.get("pace_visual_motion_threshold"))
            ),
            motion_steps_to_full=(
                None if config.get("pace_visual_motion_steps_to_full") is None
                else float(config.get("pace_visual_motion_steps_to_full"))
            ),
            min_updates=int(config.get("pace_visual_min_updates", 0)),
        )
    from envs.pace.visual_progress import VisualProgressDetector

    return VisualProgressDetector(
        model_path,
        camera=str(config.get("pace_visual_camera", "head_camera")),
        present_threshold=float(config.get("pace_visual_present_threshold", 0.5)),
        active_threshold=float(config.get("pace_visual_active_threshold", 0.5)),
        trigger_progress=float(config.get("pace_visual_trigger_progress", 0.35)),
    )


def _make_hri_phase_controller(config: dict):
    model_path = config.get("hri_phase_controller")
    if not model_path:
        return None
    from scripts.hri.phase_controller import HRIPhaseControllerRuntime

    raw_labels = config.get("hri_phase_trigger_labels", ("yield",))
    if isinstance(raw_labels, str):
        trigger_labels = tuple(x.strip() for x in raw_labels.split(",") if x.strip())
    else:
        trigger_labels = tuple(str(x) for x in raw_labels)
    return HRIPhaseControllerRuntime(
        model_path,
        min_confidence=float(config.get("hri_phase_min_confidence", 0.0)),
        min_step=int(config.get("hri_phase_min_step", 0)),
        trigger_labels=trigger_labels or ("yield",),
        consecutive=int(config.get("hri_phase_consecutive", 1)),
    )


class _HriInitiativeTrigger:
    """Baraglia-style initiative trigger adapter for the generic yield wrapper."""

    MODES = {"none", "oracle", "reactive", "proactive"}

    def __init__(self, config: dict):
        mode = str(config.get("hri_initiative_mode", "none")).lower().strip()
        if mode not in self.MODES:
            raise ValueError(
                f"unknown hri_initiative_mode={mode!r}; expected {sorted(self.MODES)}"
            )
        self.mode = mode
        self.enabled = mode != "none"
        self.min_steps = int(config.get("hri_initiative_min_steps", 0))
        self.proactive_step = int(config.get(
            "hri_initiative_step",
            config.get("pace_visual_human_start_step", 20),
        ))
        self.oracle_offset_steps = int(config.get("hri_initiative_oracle_offset_steps", 0))
        self.motion_threshold = float(config.get(
            "hri_initiative_motion_threshold",
            config.get("pace_visual_motion_threshold", 0.006),
        ))
        self.progress_threshold = float(config.get(
            "hri_initiative_progress_threshold",
            config.get("pace_visual_trigger_progress", 0.35),
        ))
        self.proximity_threshold = float(config.get("hri_initiative_proximity_threshold", 0.12))
        self.fired = False
        self.trigger_step = None
        self.trigger_reason = None
        self.trigger_signal = None
        self.last_signal = {}

    def _visual_signal(self, visual_state) -> dict:
        if visual_state is None:
            return {}
        signal = _flat_public_state(visual_state)
        motion = signal.get("motion_score")
        progress = signal.get("best_progress", signal.get("progress"))
        active = signal.get("active")
        signal["reactive_motion_ready"] = (
            motion is not None and float(motion) >= self.motion_threshold
        )
        signal["reactive_progress_ready"] = (
            progress is not None and float(progress) >= self.progress_threshold
        )
        signal["reactive_active_ready"] = active is not None and float(active) > 0.0
        return signal

    def _safe_distance_signal(self, task) -> dict:
        if not hasattr(task, "safe_distance_summary"):
            return {}
        try:
            safe = task.safe_distance_summary()
        except Exception:
            return {}
        if not isinstance(safe, dict) or not safe.get("enabled"):
            return {}
        dist = safe.get("safe_distance_m")
        if dist is None:
            return {}
        return {
            "safe_distance_m": float(dist),
            "reactive_proximity_ready": float(dist) <= self.proximity_threshold,
        }

    def _oracle_event_step(self, task):
        for name in ("_eval_trigger_step", "_pace_trigger_step"):
            value = getattr(task, name, None)
            if value is not None:
                try:
                    return int(value) + self.oracle_offset_steps
                except Exception:
                    pass
        return self.proactive_step

    def _oracle_ready(self, task, step: int) -> tuple[bool, str, dict]:
        event_flags = {
            "eval_interrupt_fired": bool(getattr(task, "_eval_interrupt_fired", False)),
            "eval_glide_active": bool(getattr(task, "_eval_glide_active", False)),
            "eval_waiting_avatar_done": bool(getattr(task, "_eval_waiting_avatar_done", False)),
            "eval_return_active": bool(getattr(task, "_eval_return_active", False)),
        }
        if any(event_flags.values()):
            return True, "initiative_oracle_task_event", event_flags
        event_step = self._oracle_event_step(task)
        signal = {**event_flags, "oracle_event_step": event_step}
        return step >= event_step, "initiative_oracle_event_step", signal

    def should_trigger(self, task, step: int, visual_state=None) -> tuple[bool, str | None]:
        if not self.enabled or self.fired or step < self.min_steps:
            return False, None

        visual_signal = self._visual_signal(visual_state)
        proximity_signal = self._safe_distance_signal(task)
        self.last_signal = {**visual_signal, **proximity_signal}

        if self.mode == "proactive":
            self.last_signal["proactive_step"] = self.proactive_step
            ready = step >= self.proactive_step
            reason = "initiative_proactive_fixed_step"
        elif self.mode == "oracle":
            ready, reason, oracle_signal = self._oracle_ready(task, step)
            self.last_signal.update(oracle_signal)
        elif self.mode == "reactive":
            visual_ready = (
                bool(visual_signal.get("reactive_motion_ready"))
                and (
                    bool(visual_signal.get("reactive_active_ready"))
                    or bool(visual_signal.get("reactive_progress_ready"))
                )
            )
            proximity_ready = bool(proximity_signal.get("reactive_proximity_ready"))
            ready = visual_ready or proximity_ready
            reason = (
                "initiative_reactive_visual_motion"
                if visual_ready else "initiative_reactive_proximity"
            )
        else:
            ready = False
            reason = None

        if not ready:
            return False, None
        self.fired = True
        self.trigger_step = int(step)
        self.trigger_reason = str(reason)
        self.trigger_signal = dict(self.last_signal)
        return True, self.trigger_reason

    def metrics(self) -> dict:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "fired": self.fired,
            "trigger_step": self.trigger_step,
            "trigger_reason": self.trigger_reason,
            "trigger_signal": self.trigger_signal,
            "last_signal": self.last_signal,
            "min_steps": self.min_steps,
            "proactive_step": self.proactive_step,
            "oracle_offset_steps": self.oracle_offset_steps,
            "motion_threshold": self.motion_threshold,
            "progress_threshold": self.progress_threshold,
            "proximity_threshold": self.proximity_threshold,
        }


class _GenericPaceYield:
    def __init__(self, config: dict):
        self.hold_position = bool(config.get("pace_generic_hold_position", False))
        self.enabled = (
            bool(config.get("pace_generic_retreat_return", False))
            or self.hold_position
        )
        self.retreat_steps = int(config.get("pace_generic_retreat_steps", 10))
        self.return_steps = int(config.get("pace_generic_return_steps", 10))
        self.return_max_steps = int(config.get("pace_generic_return_max_steps", 100))
        self.return_qpos_tol = float(config.get("pace_generic_return_qpos_tol", 0.01))
        self.return_gripper_tol = float(config.get("pace_generic_return_gripper_tol", 0.02))
        self.return_correction_gain = float(config.get("pace_generic_return_correction_gain", 1.0))
        self.return_correction_clip = float(config.get("pace_generic_return_correction_clip", 0.05))
        self.wait_avatar_done = bool(config.get("pace_generic_wait_avatar_done", True))
        self.wait_max_steps = int(config.get("pace_generic_wait_max_steps", 300))
        self.return_to_saved_qpos = bool(config.get("pace_generic_return_to_saved_qpos", True))
        self.retreat_blend = float(config.get("pace_generic_retreat_blend", 0.5))
        self.return_blend = float(config.get("pace_generic_return_blend", 0.5))
        self.reset_policy_on_return = bool(config.get("pace_generic_reset_policy_after_return", True))
        self.resume_with_snapshot_obs = bool(config.get("pace_generic_resume_snapshot_obs", False))
        self.restore_scene_state = bool(config.get("pace_generic_restore_scene_state", False))
        self.replay_snapshot_action = bool(config.get("pace_generic_replay_snapshot_action", True))
        self.capture_trigger_action = bool(config.get("pace_generic_capture_trigger_action", True))
        # "avatar_done" (default, task avatar state) or "detector"
        # (observation-only clearance from the trigger detector).
        self.wait_mode = str(config.get("pace_generic_wait_mode", "avatar_done"))
        self.wait_detector = None
        self.phase = None
        self.remaining = 0
        self.wait_elapsed = 0
        self.return_elapsed = 0
        self.return_start_qpos = None
        self.saved_qpos = None
        self.saved_gripper = 1.0
        self.saved_obs = None
        self.saved_action = None
        self.saved_scene_state = None
        self.retreat_target = None
        self.fired = False
        self.trigger_step = None
        self.trigger_reason = None
        self.reset_policy_after = False
        self.resume_snapshot_pending = False
        self.resume_snapshot_used = False
        self.resume_action_pending = False
        self.resume_action_used = False
        self.wait_completed = False
        self.wait_timed_out = False
        self.return_converged = False
        self.return_qpos_error = None
        self.return_gripper_error = None
        self.scene_state_restored = False
        self.scene_state_restore_count = 0

    def _scene_entities(self, task):
        entities = []
        seen = set()

        def add_entity(entity):
            if entity is None:
                return
            key = id(entity)
            if key in seen:
                return
            if not hasattr(entity, "get_pos") or not hasattr(entity, "set_pos"):
                return
            seen.add(key)
            entities.append(entity)

        for attr in ("bowls", "objects"):
            for obj in getattr(task, attr, []) or []:
                add_entity(getattr(obj, "entity", obj))
        obj = getattr(task, "object", None)
        add_entity(getattr(obj, "entity", obj))
        return entities

    def _capture_scene_state(self, task):
        if not self.restore_scene_state:
            return None
        state = []
        for entity in self._scene_entities(task):
            try:
                state.append(
                    (
                        entity,
                        np.asarray(_host_array(entity.get_pos()), dtype=np.float64).copy(),
                        np.asarray(_host_array(entity.get_quat()), dtype=np.float64).copy()
                        if hasattr(entity, "get_quat")
                        else None,
                    )
                )
            except Exception:
                continue
        return state

    def _restore_scene_state(self) -> None:
        if not self.restore_scene_state or not self.saved_scene_state:
            return
        restored = 0
        for entity, pos, quat in self.saved_scene_state:
            try:
                entity.set_pos(pos.astype(float))
                if quat is not None and hasattr(entity, "set_quat"):
                    entity.set_quat(quat.astype(float))
                if hasattr(entity, "set_dofs_velocity"):
                    entity.set_dofs_velocity(np.zeros(6, dtype=np.float64))
                restored += 1
            except Exception:
                continue
        self.scene_state_restored = restored > 0
        self.scene_state_restore_count = restored

    def start(self, task, obs, step: int, reason: str) -> bool:
        if not self.enabled or self.fired:
            return False
        arm = task.robot.get_arm("right")
        self.saved_qpos = np.asarray(arm.get_arm_qpos(), dtype=np.float64).copy()
        self.saved_gripper = float(getattr(arm, "gripper_val", 1.0))
        self.saved_obs = deepcopy(obs) if self.resume_with_snapshot_obs else None
        self.saved_scene_state = self._capture_scene_state(task)
        if self.hold_position:
            self.retreat_target = self.saved_qpos.copy()
            self.phase = "wait"
            self.wait_elapsed = 0
            self.remaining = max(1, self.wait_max_steps)
        else:
            self.retreat_target = np.asarray(arm.homestate, dtype=np.float64).copy()
            self.phase = "retreat"
            self.remaining = max(0, self.retreat_steps)
        self.fired = True
        self.trigger_step = int(step)
        self.trigger_reason = str(reason)
        return True

    def maybe_start(self, task, obs, step: int, detector) -> bool:
        if detector is None:
            return False
        should_trigger, reason = detector.should_trigger()
        if not should_trigger:
            return False
        return self.start(task, obs, step, str(reason))

    def save_resume_action(self, action) -> None:
        if not self.enabled or not self.fired or self.saved_action is not None:
            return
        self.saved_action = np.asarray(action, dtype=np.float64).copy()

    def active(self) -> bool:
        return self.enabled and self.phase is not None

    def consume_reset(self) -> bool:
        flag = self.reset_policy_after
        self.reset_policy_after = False
        return flag

    def consume_resume_obs(self, obs):
        if not self.resume_snapshot_pending:
            return obs
        self.resume_snapshot_pending = False
        self.resume_snapshot_used = self.saved_obs is not None
        return self.saved_obs if self.saved_obs is not None else obs

    def consume_resume_action(self):
        if not self.resume_action_pending or self.saved_action is None:
            return None
        self.resume_action_pending = False
        self.resume_action_used = True
        return np.asarray(self.saved_action, dtype=np.float64).copy()

    def action(self, task) -> np.ndarray:
        arm = task.robot.get_arm("right")
        current = np.asarray(arm.get_arm_qpos(), dtype=np.float64)
        if self.phase == "retreat":
            target = self.retreat_target
            alpha = float(np.clip(self.retreat_blend, 0.0, 1.0))
            cmd = current * (1.0 - alpha) + np.asarray(target, dtype=np.float64) * alpha
        elif self.phase == "wait":
            cmd = np.asarray(self.retreat_target, dtype=np.float64)
        elif self.phase == "return":
            if self.return_start_qpos is None:
                self.return_start_qpos = current.copy()
            steps = max(1, int(self.return_steps))
            frac = float(np.clip((self.return_elapsed + 1) / steps, 0.0, 1.0))
            cmd = (
                np.asarray(self.return_start_qpos, dtype=np.float64) * (1.0 - frac)
                + np.asarray(self.saved_qpos, dtype=np.float64) * frac
            )
            if frac >= 1.0 and self.return_correction_gain > 0.0:
                error = current - np.asarray(self.saved_qpos, dtype=np.float64)
                correction = -self.return_correction_gain * error
                clip = max(0.0, self.return_correction_clip)
                if clip > 0.0:
                    correction = np.clip(correction, -clip, clip)
                cmd = np.asarray(self.saved_qpos, dtype=np.float64) + correction
        else:
            cmd = current
        return np.concatenate([cmd, [self.saved_gripper]])

    def _avatar_wait_required(self, task) -> bool:
        if (
            self.wait_mode == "detector"
            and self.wait_detector is not None
            and hasattr(self.wait_detector, "human_clear")
        ):
            return not self.wait_detector.human_clear()
        if not self.wait_avatar_done:
            return False
        avatar = getattr(task, "avatar", None)
        if avatar is not None and hasattr(avatar, "spare"):
            if not avatar.spare():
                return True
            if not bool(getattr(task, "config", {}).get("eval_robot_yield_enabled", True)):
                return False
        has_interrupt_state = any(
            hasattr(task, name)
            for name in (
                "_eval_interrupt_fired",
                "_eval_glide_active",
                "_eval_waiting_avatar_done",
                "_eval_return_active",
            )
        )
        if not has_interrupt_state:
            return False
        if not bool(getattr(task, "_eval_interrupt_fired", False)):
            return True
        if bool(getattr(task, "_eval_glide_active", False)):
            return True
        if bool(getattr(task, "_eval_waiting_avatar_done", False)):
            return True
        if bool(getattr(task, "_eval_return_active", False)):
            return True
        return False

    def _return_errors(self, task) -> tuple[float | None, float]:
        arm = task.robot.get_arm("right")
        qpos_error = None
        if self.saved_qpos is not None:
            current = np.asarray(arm.get_arm_qpos(), dtype=np.float64)
            qpos_error = float(np.max(np.abs(current - self.saved_qpos)))
        gripper_error = float(abs(float(getattr(arm, "gripper_val", 1.0)) - self.saved_gripper))
        return qpos_error, gripper_error

    def _finish_return(self, task, converged: bool):
        self.return_qpos_error, self.return_gripper_error = self._return_errors(task)
        self.return_converged = bool(converged)
        self._restore_scene_state()
        self.phase = None
        self.resume_snapshot_pending = self.resume_with_snapshot_obs
        self.resume_action_pending = self.replay_snapshot_action and self.saved_action is not None
        self.reset_policy_after = self.reset_policy_on_return

    def _finish_without_return(self, task):
        self.return_qpos_error, self.return_gripper_error = self._return_errors(task)
        self.return_converged = False
        self._restore_scene_state()
        self.phase = None
        self.resume_snapshot_pending = False
        self.resume_action_pending = False
        self.reset_policy_after = self.reset_policy_on_return

    def _start_return(self, task):
        if not self.return_to_saved_qpos:
            self._finish_without_return(task)
            return
        self.phase = "return"
        self.return_start_qpos = np.asarray(
            task.robot.get_arm("right").get_arm_qpos(), dtype=np.float64,
        ).copy()
        self.return_elapsed = 0
        self.remaining = max(1, self.return_max_steps)

    def step_after_action(self, task):
        if self.phase is None:
            return
        if self.phase == "retreat":
            self.remaining -= 1
            if self.remaining > 0:
                return
            if self._avatar_wait_required(task):
                self.phase = "wait"
                self.wait_elapsed = 0
                self.remaining = max(1, self.wait_max_steps)
            else:
                self.wait_completed = True
                self._start_return(task)
        elif self.phase == "wait":
            self.remaining -= 1
            self.wait_elapsed += 1
            if not self._avatar_wait_required(task):
                self.wait_completed = True
                self._start_return(task)
            elif self.remaining <= 0:
                self.wait_timed_out = True
                self._start_return(task)
        elif self.phase == "return":
            self.remaining -= 1
            self.return_elapsed += 1
            qpos_error, gripper_error = self._return_errors(task)
            self.return_qpos_error = qpos_error
            self.return_gripper_error = gripper_error
            converged = (
                qpos_error is not None
                and qpos_error <= self.return_qpos_tol
                and gripper_error <= self.return_gripper_tol
            )
            if converged or self.remaining <= 0:
                self._finish_return(task, converged=converged)

    def metrics(self) -> dict:
        return {
            "enabled": self.enabled,
            "fired": self.fired,
            "trigger_step": self.trigger_step,
            "trigger_reason": self.trigger_reason,
            "phase": self.phase,
            "hold_position": self.hold_position,
            "reset_policy_on_return": self.reset_policy_on_return,
            "resume_with_snapshot_obs": self.resume_with_snapshot_obs,
            "restore_scene_state": self.restore_scene_state,
            "scene_state_restored": self.scene_state_restored,
            "scene_state_restore_count": self.scene_state_restore_count,
            "resume_snapshot_pending": self.resume_snapshot_pending,
            "resume_snapshot_used": self.resume_snapshot_used,
            "replay_snapshot_action": self.replay_snapshot_action,
            "capture_trigger_action": self.capture_trigger_action,
            "resume_action_pending": self.resume_action_pending,
            "resume_action_used": self.resume_action_used,
            "wait_avatar_done": self.wait_avatar_done,
            "wait_mode": self.wait_mode,
            "wait_max_steps": self.wait_max_steps,
            "wait_elapsed": self.wait_elapsed,
            "wait_completed": self.wait_completed,
            "wait_timed_out": self.wait_timed_out,
            "return_to_saved_qpos": self.return_to_saved_qpos,
            "return_converged": self.return_converged,
            "return_elapsed": self.return_elapsed,
            "return_qpos_tol": self.return_qpos_tol,
            "return_qpos_linf_error": self.return_qpos_error,
            "return_gripper_error": self.return_gripper_error,
            "return_correction_gain": self.return_correction_gain,
            "return_correction_clip": self.return_correction_clip,
        }


def eval_policy(
    task_name: str,
    config: dict,
    policy_fn,
    num_episodes: int,
    max_steps: int = 300,
    start_seed: int = 0,
    action_type: str = "qpos",
    video_dir: str = None,
    video_limit: int | None = None,
    return_details: bool = False,
    scene_inits: list[dict] | None = None,
    scene_init_start_index: int = 0,
):
    """Evaluate a policy on a task.

    Args:
        task_name: task name from TASK_MAP
        config: task config dict
        policy_fn: callable(obs) -> action
        num_episodes: number of evaluation episodes
        max_steps: max steps per episode
        start_seed: starting seed
        action_type: "qpos" or "ee"
        video_dir: if set, write one MP4 per episode to this directory
        video_limit: when video_dir is set, save videos for only the first N episodes

    Returns:
        success_rate: float
    """
    success_count = 0
    episode_results = []

    if video_dir:
        os.makedirs(video_dir, exist_ok=True)
    max_videos_per_dir = config.get("max_videos_per_dir")
    if max_videos_per_dir is not None:
        max_videos_per_dir = int(max_videos_per_dir)

    for ep in range(num_episodes):
        seed = start_seed + ep
        scene_init_entry = None
        scene_init = None
        if scene_inits is not None:
            idx = int(scene_init_start_index) + ep
            if idx >= len(scene_inits):
                raise IndexError(
                    f"Scene-init index {idx} out of range for {len(scene_inits)} entries"
                )
            scene_init_entry = scene_inits[idx]
            ep_base_config = _merge_config(config, scene_init_entry.get("config_overrides"))
            resolved_task_name, TaskClass = _resolve_eval_task_class(
                task_name,
                ep_base_config,
            )
            entry_task = scene_init_entry.get("task")
            if entry_task and entry_task not in {task_name, resolved_task_name}:
                raise ValueError(
                    f"Scene-init task mismatch at index {idx}: "
                    f"{entry_task!r} not in {{{task_name!r}, {resolved_task_name!r}}}"
                )
            scene_init = episode_scene_init(scene_init_entry)
            seed = int(scene_init.get("source_seed", scene_init_entry.get("seed", seed)))
        else:
            ep_base_config = config
            resolved_task_name, TaskClass = _resolve_eval_task_class(
                task_name,
                ep_base_config,
            )
        label = f"scene_init={scene_init.get('id')}" if scene_init else f"seed={seed}"
        print(f"\nEpisode {ep + 1}/{num_episodes} ({label})")

        reset_t0 = time.time()
        try:
            ep_config = prepare_config_for_scene_init(ep_base_config, scene_init)
            task = TaskClass(ep_config)
            obs = task.reset(seed=seed)
        except Exception:
            elapsed = time.time() - reset_t0
            err = traceback.format_exc()
            print(f"  ERROR during reset ({elapsed:.1f}s)")
            print(err, flush=True)
            episode_results.append(_scene_init_failure_result(seed, scene_init, elapsed, err))
            continue

        save_episode_video = _should_save_video(ep, video_dir, video_limit)
        video_marker = None
        video_path = None
        if save_episode_video:
            video_path, video_marker = _reserve_video_path(
                video_dir,
                f"{resolved_task_name}_seed_{seed}.mp4",
                max_videos_per_dir,
            )
        if video_path:
            task.start_video(video_path)

        if ep_config.get("eval_avatar_pre_policy", True) and hasattr(task, "eval_pre_policy_warmup"):
            print("  pre-policy warmup: begin", flush=True)
            maybe_obs = task.eval_pre_policy_warmup()
            obs = maybe_obs if maybe_obs is not None else task.get_obs()
            print("  pre-policy warmup: done", flush=True)

        policy_name = getattr(policy_fn, "baseline_name", None)
        if policy_name:
            print(f"  warning: ignoring removed privileged baseline policy {policy_name!r}")

        if hasattr(policy_fn, "reset"):
            policy_fn.reset()
        # Feed the per-episode instruction to the policy: tasks that sample
        # their target object each reset (object catalog) template the object
        # name into task.instruction, which must reach the language-conditioned
        # model for this episode.
        if hasattr(policy_fn, "set_instruction"):
            policy_fn.set_instruction(getattr(task, "instruction", None))

        visual_pace = None
        if ep_config.get("pace_visual_model") or ep_config.get("manicast_model"):
            visual_pace = _make_visual_pace_detector(ep_config)
            if hasattr(task, "set_eval_visual_pace_detector"):
                task.set_eval_visual_pace_detector(visual_pace)
        generic_pace = _GenericPaceYield(ep_config)
        if generic_pace.wait_mode == "detector" and visual_pace is not None:
            generic_pace.wait_detector = visual_pace
        hri_initiative = _HriInitiativeTrigger(ep_config)
        hri_phase = _make_hri_phase_controller(ep_config)
        if hri_phase is not None:
            hri_phase.reset(task, max_steps=max_steps)

        t0 = time.time()
        step = 0
        episode_error = None
        if hasattr(policy_fn, "run_episode"):
            try:
                play_success, obs = policy_fn.run_episode(
                    task, obs, max_steps=max_steps, action_type=action_type,
                )
                policy_steps = _privileged_policy_step_count(policy_fn)
                step = max(0, int(policy_steps) - 1) if policy_steps else 0
                if not play_success:
                    task.plan_success = False
            except Exception:
                episode_error = traceback.format_exc()
                print(f"  ERROR during rollout ({time.time() - t0:.1f}s)")
                print(episode_error, flush=True)
                task.plan_success = False
        else:
            trace_rows = []
            trace_dir = ep_config.get("trace_actions_dir")
            trace_eval_metrics = bool(ep_config.get("trace_eval_metrics", False))
            stop_on_success = bool(ep_config.get("eval_stop_on_success", True))
            success_hold_steps = max(1, int(ep_config.get("eval_success_hold_steps", 1)))
            consecutive_success = 0
            last_action = None
            visual_state = None
            hri_phase_state = None
            progress_interval = int(ep_config.get("debug_eval_progress_interval", 0) or 0)
            for step in range(max_steps):
                if progress_interval > 0 and step % progress_interval == 0:
                    print(f"  step {step}/{max_steps}", flush=True)
                visual_stride = max(1, int(ep_config.get("pace_visual_update_stride", 1)))
                if visual_pace is not None and step % visual_stride == 0:
                    state = visual_pace.update(obs)
                    visual_state = state
                    if hasattr(task, "set_eval_visual_pace_state"):
                        task.set_eval_visual_pace_state(state)
                    if (
                        hri_phase is None
                        and
                        not hri_initiative.enabled
                        and generic_pace.maybe_start(task, obs, step, visual_pace)
                    ):
                        if hasattr(visual_pace, "note_yield_started"):
                            try:
                                visual_pace.note_yield_started(
                                    np.asarray(
                                        task.robot.right_arm.get_ee_pose(),
                                        dtype=np.float64,
                                    ).ravel()[:3]
                                )
                            except Exception:
                                visual_pace.note_yield_started(None)
                        if hasattr(task, "notify_eval_visual_pace_trigger"):
                            task.notify_eval_visual_pace_trigger(
                                step=step,
                                reason=generic_pace.trigger_reason,
                                state=state,
                            )
                        # Preserve the exact ACT output that would have been
                        # executed at the interrupt step. The policy call also
                        # advances ACT's local/server chunk state, so replaying
                        # this action after return keeps the post-HRI sequence
                        # aligned with the uninterrupted rollout.
                        if generic_pace.capture_trigger_action:
                            trigger_obs = generic_pace.consume_resume_obs(obs)
                            trigger_action = policy_fn(trigger_obs)
                            generic_pace.save_resume_action(trigger_action)
                if hri_phase is not None and generic_pace.enabled and not generic_pace.fired:
                    hri_phase_state = hri_phase.update(
                        task, obs, step, visual_state, generic_pace.metrics(),
                    )
                    should_trigger, reason = hri_phase.should_trigger()
                    if should_trigger and generic_pace.start(task, obs, step, reason):
                        if hasattr(task, "notify_eval_visual_pace_trigger"):
                            task.notify_eval_visual_pace_trigger(
                                step=step,
                                reason=generic_pace.trigger_reason,
                                state=visual_state,
                            )
                        if generic_pace.capture_trigger_action:
                            trigger_obs = generic_pace.consume_resume_obs(obs)
                            trigger_action = policy_fn(trigger_obs)
                            generic_pace.save_resume_action(trigger_action)
                if hri_initiative.enabled and generic_pace.enabled and not generic_pace.fired:
                    should_trigger, reason = hri_initiative.should_trigger(
                        task, step, visual_state,
                    )
                    if should_trigger and generic_pace.start(task, obs, step, reason):
                        if hasattr(task, "notify_eval_visual_pace_trigger"):
                            task.notify_eval_visual_pace_trigger(
                                step=step,
                                reason=generic_pace.trigger_reason,
                                state=visual_state,
                            )
                        if generic_pace.capture_trigger_action:
                            trigger_obs = generic_pace.consume_resume_obs(obs)
                            trigger_action = policy_fn(trigger_obs)
                            generic_pace.save_resume_action(trigger_action)
                if (
                    hasattr(task, "consume_eval_policy_reset_requested")
                    and task.consume_eval_policy_reset_requested()
                    and hasattr(policy_fn, "reset")
                ):
                    policy_fn.reset()
                if generic_pace.consume_reset() and hasattr(policy_fn, "reset"):
                    policy_fn.reset()
                skip_policy = (
                    hasattr(task, "eval_skip_policy_action")
                    and task.eval_skip_policy_action()
                ) or generic_pace.active()
                try:
                    action_source = "policy"
                    if skip_policy:
                        if generic_pace.active():
                            action = generic_pace.action(task)
                            action_source = f"generic_pace:{generic_pace.phase}"
                            last_action = np.asarray(action, dtype=np.float64).copy()
                        elif last_action is None:
                            action_dim = int(getattr(policy_fn, "action_dim", 8))
                            last_action = np.zeros(action_dim, dtype=np.float64)
                            action = np.asarray(last_action, dtype=np.float64).copy()
                            action_source = "zero_hold"
                        else:
                            action = np.asarray(last_action, dtype=np.float64).copy()
                            action_source = "last_action_hold"
                    else:
                        action = generic_pace.consume_resume_action()
                        if action is not None:
                            action_source = "generic_pace:snapshot_action"
                        else:
                            policy_obs = generic_pace.consume_resume_obs(obs)
                            action = policy_fn(policy_obs)
                        last_action = np.asarray(action, dtype=np.float64).copy()
                except Exception:
                    episode_error = traceback.format_exc()
                    print(f"  ERROR during policy step {step} ({time.time() - t0:.1f}s)")
                    print(episode_error, flush=True)
                    task.plan_success = False
                    break
                if trace_dir:
                    row = {
                        "step": int(step),
                        "policy_skipped": bool(skip_policy),
                        "action_source": action_source,
                        "visual_pace": _flat_public_state(visual_state),
                        "hri_initiative": hri_initiative.metrics(),
                        "hri_phase_controller": (
                            hri_phase.metrics() if hri_phase is not None else {"enabled": False}
                        ),
                        "generic_pace": generic_pace.metrics(),
                        "eval_policy_step_count": int(getattr(task, "_policy_step_count", -1)),
                        "eval_interrupt_fired": bool(getattr(task, "_eval_interrupt_fired", False)),
                        "eval_glide_active": bool(getattr(task, "_eval_glide_active", False)),
                        "eval_retreat_active": bool(getattr(task, "_eval_retreat_active", False)),
                        "eval_return_active": bool(getattr(task, "_eval_return_active", False)),
                        "eval_waiting_avatar_done": bool(getattr(task, "_eval_waiting_avatar_done", False)),
                        "eval_pending_interrupt_idx": getattr(task, "_eval_pending_interrupt_idx", None),
                        "pace_trigger_step": getattr(task, "_pace_trigger_step", None),
                        "pace_trigger_reason": getattr(task, "_pace_trigger_reason", None),
                        "policy_obs_before": _summarize_policy_obs(obs),
                        "action": np.asarray(action, dtype=float).ravel().tolist(),
                        "ee_pose_before": np.asarray(
                            task.robot.right_arm.get_ee_pose(), dtype=float,
                        ).ravel().tolist(),
                        "qpos_before": np.asarray(
                            task.robot.right_arm.get_arm_qpos(), dtype=float,
                        ).ravel().tolist(),
                        "gripper_before": float(task.robot.right_arm.gripper_val),
                    }
                    if hasattr(task, "objects"):
                        row["object_positions_before"] = [
                            _vec3(task._object_pos(obj))
                            if hasattr(task, "_object_pos")
                            else _vec3(obj.get_pos())
                            for obj in getattr(task, "objects", [])
                        ]
                    if hasattr(task, "bowls"):
                        row["bowl_positions_before"] = [
                            _vec3(bowl.entity.get_pos())
                            for bowl in getattr(task, "bowls", [])
                            if bowl is not None
                        ]
                    if hasattr(task, "object") and getattr(task, "object") is not None:
                        entity = getattr(task.object, "entity", task.object)
                        row["object_position_before"] = _vec3(entity.get_pos())
                try:
                    obs = task.take_action(action, action_type=action_type)
                    generic_pace.step_after_action(task)
                except Exception:
                    episode_error = traceback.format_exc()
                    print(f"  ERROR during env step {step} ({time.time() - t0:.1f}s)")
                    print(episode_error, flush=True)
                    task.plan_success = False
                    break
                if trace_dir:
                    row["qpos_after"] = np.asarray(
                        task.robot.right_arm.get_arm_qpos(), dtype=float,
                    ).ravel().tolist()
                    row["ee_pose_after"] = np.asarray(
                        task.robot.right_arm.get_ee_pose(), dtype=float,
                    ).ravel().tolist()
                    row["gripper_after"] = float(task.robot.right_arm.gripper_val)
                    if getattr(task, "target", None) is not None:
                        metrics_now = task.evaluate()
                        for key in (
                            "target_dist_xy",
                            "target_dist_3d",
                            "target_dz",
                            "target_tilt_deg",
                        ):
                            if key in metrics_now:
                                row[key] = float(metrics_now[key])
                    if hasattr(task, "objects"):
                        row["object_positions_after"] = [
                            _vec3(task._object_pos(obj))
                            if hasattr(task, "_object_pos")
                            else _vec3(obj.get_pos())
                            for obj in getattr(task, "objects", [])
                        ]
                    if hasattr(task, "bowls"):
                        row["bowl_positions_after"] = [
                            _vec3(bowl.entity.get_pos())
                            for bowl in getattr(task, "bowls", [])
                            if bowl is not None
                        ]
                    if hasattr(task, "object") and getattr(task, "object") is not None:
                        entity = getattr(task.object, "entity", task.object)
                        row["object_position_after"] = _vec3(entity.get_pos())
                    if trace_eval_metrics:
                        step_metrics = task.evaluate()
                        for key, value in step_metrics.items():
                            if isinstance(value, (bool, int, float, str)) or value is None:
                                row[f"metric_{key}"] = value
                    trace_rows.append(row)

                if task.eval_success:
                    break
                if stop_on_success:
                    step_metrics = task.evaluate()
                    if step_metrics.get("success", False):
                        consecutive_success += 1
                        if consecutive_success >= success_hold_steps:
                            task.eval_success = True
                            print(
                                f"  early-stop: success stable for "
                                f"{consecutive_success} step(s) at step={step + 1}",
                                flush=True,
                            )
                            break
                    else:
                        consecutive_success = 0
            if trace_dir:
                os.makedirs(trace_dir, exist_ok=True)
                trace_path = os.path.join(
                    trace_dir,
                    f"{resolved_task_name}_seed_{seed}_trace.json",
                )
                with open(trace_path, "w") as f:
                    json.dump(trace_rows, f, indent=2)
                print(f"  trace saved: {trace_path} ({len(trace_rows)} rows)")

        metrics = task.evaluate()
        if episode_error is not None:
            metrics["stage_failed"] = "rollout"
        if generic_pace.enabled:
            metrics["generic_pace"] = generic_pace.metrics()
        if hri_initiative.enabled:
            metrics["hri_initiative"] = hri_initiative.metrics()
        if hri_phase is not None:
            metrics["hri_phase_controller"] = hri_phase.metrics()
        task_success = bool(metrics.get("success", False)) and episode_error is None
        policy_metrics = _policy_metrics_dict(policy_fn)
        if policy_metrics is not None:
            metrics["hrc_baseline"] = policy_metrics
        elapsed = time.time() - t0

        if video_path:
            task.save_video()
            _release_video_reservation(video_marker)

        _print_policy_metrics(getattr(policy_fn, "__class__", type(policy_fn)).__name__, policy_fn)

        if "target_dist_xy" in metrics:
            tilt = metrics.get("target_tilt_deg")
            tilt_s = f", tilt={tilt:.1f}°" if tilt is not None else ""
            print(f"  target[{metrics.get('target_label','?')}]: "
                  f"dist_xy={metrics['target_dist_xy']:.3f}m, "
                  f"dist_3d={metrics['target_dist_3d']:.3f}m, "
                  f"dz={metrics['target_dz']:+.3f}m{tilt_s}")
        # Per-episode avatar-collision report (if tracking enabled).
        coll = metrics.get("avatar_collision", {})
        if coll.get("enabled"):
            n = coll["n_checks"]; hits = coll["n_collisions"]
            dd = coll.get("deepest_depth_m")
            dd_str = f"{dd*100:+.1f} cm" if dd is not None else "—"
            print(f"  avatar-collision: {hits}/{n} checks hit, "
                  f"deepest={dd_str}, pair={coll.get('deepest_pair')}")
        safe_dist = metrics.get("safe_distance", {})
        if safe_dist.get("enabled"):
            sd = safe_dist.get("safe_distance_m")
            sd_str = f"{sd*100:+.1f} cm" if sd is not None else "—"
            print(f"  safe-distance: min={sd_str}, "
                  f"pair={safe_dist.get('closest_pair')}")
        intervention = metrics.get("avatar_safety_intervention", {})
        if intervention.get("enabled"):
            md = intervention.get("min_distance_m")
            md_str = f"{md*100:+.1f} cm" if md is not None else "—"
            print(f"  avatar-retreat: {intervention['n_interventions']} interventions, "
                  f"min_dist={md_str}")
        pace = metrics.get("pace_interrupt", {})
        if pace:
            dist = pace.get("trigger_dist_xy")
            dist_s = f"{dist:.3f}m" if dist is not None else "—"
            prog = pace.get("trigger_progress")
            prog_s = f"{prog:.2f}" if prog is not None else "—"
            print(f"  pace-interrupt: mode={pace.get('trigger_mode')}, "
                  f"fired={pace.get('fired')}, step={pace.get('trigger_step')}, "
                  f"reason={pace.get('trigger_reason')}, "
                  f"progress={prog_s}, dist_xy={dist_s}")
        gp = metrics.get("generic_pace", {})
        if gp.get("enabled"):
            print(f"  generic-pace: fired={gp.get('fired')}, "
                  f"step={gp.get('trigger_step')}, "
                  f"reason={gp.get('trigger_reason')}, "
                  f"phase={gp.get('phase')}")
        initiative = metrics.get("hri_initiative", {})
        if initiative.get("enabled"):
            print(f"  hri-initiative: mode={initiative.get('mode')}, "
                  f"fired={initiative.get('fired')}, "
                  f"step={initiative.get('trigger_step')}, "
                  f"reason={initiative.get('trigger_reason')}")
        phase_ctl = metrics.get("hri_phase_controller", {})
        if phase_ctl.get("enabled"):
            print(f"  hri-phase: phase={phase_ctl.get('phase')}, "
                  f"conf={phase_ctl.get('confidence')}, "
                  f"triggered={phase_ctl.get('triggered')}, "
                  f"counts={phase_ctl.get('prediction_counts')}")
        handoff = metrics.get("eval_handoff_debug")
        if handoff is not None:
            print(f"  eval-handoff: step={handoff.get('step_idx')}, "
                  f"gripper={handoff.get('gripper'):.3f}, "
                  f"ee_apple_dist={handoff.get('ee_apple_dist'):.3f}m")
        elif "eval_apple_attached_to_avatar" in metrics:
            print(f"  eval-handoff: no detach, "
                  f"attached_to_avatar={metrics['eval_apple_attached_to_avatar']}")

        if task_success:
            success_count += 1
            print(f"  SUCCESS (step={step + 1}, {elapsed:.1f}s)")
        else:
            print(f"  FAIL (step={step + 1}, {elapsed:.1f}s)")
        detail = {
            "seed": seed,
            "success": task_success,
            "steps": step + 1,
            "elapsed_s": elapsed,
            "avatar_collision": metrics.get("avatar_collision"),
            "safe_distance": metrics.get("safe_distance"),
            "avatar_safety_intervention": metrics.get("avatar_safety_intervention"),
            "pace_interrupt": metrics.get("pace_interrupt"),
            "hrc_baseline": policy_metrics,
            "generic_pace": metrics.get("generic_pace"),
            "hri_initiative": metrics.get("hri_initiative"),
            "hri_phase_controller": metrics.get("hri_phase_controller"),
        }
        if scene_init:
            detail["scene_init_id"] = scene_init.get("id")
            detail["scene_init_source_seed"] = scene_init.get("source_seed")
            detail["scene_init_replay"] = getattr(task, "scene_init_replay", None)
        if episode_error is not None:
            detail["error"] = episode_error
        detail_keys = (
            "target_label",
            "target_dist_xy",
            "target_dist_3d",
            "target_dz",
            "target_tilt_deg",
            "object_name",
            "object_model_id",
            "grasp_method",
            "stage_failed",
            "grasp_used",
            "object_bottom_z",
            "object_table_clearance",
            "object_table_clearance_threshold",
            "object_still_held",
            "object_hold_metric_error",
            "object_in_drawer",
            "object_in_drawer_xy",
            "object_in_drawer_aabb_xy",
            "object_in_drawer_exposed",
            "object_in_drawer_z",
            "object_center",
            "drawer_center",
            "drawer_aabb_min",
            "drawer_aabb_max",
            "drawer_floor_z",
            "drawer_rim_z",
            "drawer_qpos",
            "cabinet_front_y",
            "ee_pose",
            "ee_in_drawer_aabb",
            "ee_to_drawer_center",
            "gripper_open_value",
            "door_angle",
            "max_door_angle",
            "success_open_angle",
            "robot_plate_collision",
            "robot_plate_collision_events",
            "robot_plate_min_clearance",
            "robot_blocks_load_path",
            "robot_load_path_clearance",
            "avatar_plate_attached",
            "avatar_plate_two_hand_attached",
            "avatar_plate_hand_distance",
            "avatar_plate_hand_max_dist",
            "avatar_plate_carried_ok",
            "eval_avatar_trigger_step",
            "eval_avatar_started",
            "eval_avatar_failed",
        )
        for key in detail_keys:
            if key in metrics:
                detail[key] = metrics[key]
        episode_results.append(detail)

    success_rate = success_count / num_episodes if num_episodes > 0 else 0.0
    print(f"\nSuccess rate: {success_count}/{num_episodes} = {success_rate:.1%}")
    if return_details:
        return {
            "success_rate": success_rate,
            "successes": success_count,
            "episodes": num_episodes,
            "episode_results": episode_results,
        }
    return success_rate


def main():
    parser = argparse.ArgumentParser(description="Evaluate a policy on a task")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--policy", type=str, default="random",
                        choices=[
                            "random",
                            "expert_full_state",
                            "expert",
                            "expert_no_human",
                        ],
                        help="Policy baseline. expert_full_state is the scripted "
                             "rollout policy with privileged task state; expert "
                             "runs the same scripted policy with the avatar "
                             "disabled. expert_no_human is kept as an alias.")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--start-seed", type=int, default=0, help="Starting seed")
    parser.add_argument("--scene-init-file", type=str, default=None,
                        help="YAML/JSON/JSONL deterministic scene-init file. "
                             "When set, eval episodes are selected by index "
                             "instead of start-seed-only initialization.")
    parser.add_argument("--scene-init-start-index", type=int, default=0,
                        help="First scene-init entry to evaluate.")
    parser.add_argument("--renderer", type=str, default=None,
                        choices=["rasterizer", "raytracer", "nyx"],
                        help="Override renderer (nyx/raytracer require GPU)")
    parser.add_argument("--no-human", action="store_true",
                        help="Collapse an assist/interrupt/neutral family task to "
                             "its canonical robot-only no-human task. Intent tasks "
                             "do not support this flag.")
    parser.add_argument("--table-random-objects",
                        "--task-irrelevant-objects",
                        "--with-irrelevant-objects",
                        dest="table_random_objects", action="store_true",
                        help="Enable random background tabletop objects for tasks "
                             "that define decorative object regions. Default: disabled.")
    parser.add_argument("--hard", action="store_true",
                        help="Use the legacy multi-object task variant where supported.")
    parser.add_argument("--no-table-random-objects",
                        "--no-task-irrelevant-objects",
                        dest="no_table_random_objects",
                        action="store_true",
                        help="Disable random background tabletop objects, overriding config.")
    parser.add_argument("--track-avatar-collision", action="store_true",
                        help="Run the analytic robot↔avatar-capsule collision checker "
                             "and print per-episode summary. Physics untouched.")
    parser.add_argument("--collision-stride", type=int, default=None,
                        help="Collision check every N sim steps (default 30).")
    parser.add_argument("--collision-margin", type=float, default=None,
                        help="Flag within M metres of capsule surface (default 0).")
    parser.add_argument("--avatar-safety-margin", type=float, default=None,
                        help="Log inflated avatar safety envelope violations at this margin.")
    parser.add_argument("--avatar-retreat-enabled", action="store_true",
                        help="Override nominal robot motion with an up-and-away retreat "
                             "when the avatar safety threshold is breached.")
    parser.add_argument("--avatar-retreat-margin", type=float, default=None,
                        help="Unsafe distance threshold for active retreat. Defaults to "
                             "--avatar-safety-margin, then 0.10m.")
    parser.add_argument("--show-collider", action="store_true",
                        help="Render the bone cylinders — debug only.")
    parser.add_argument("--eval-mode", action="store_true",
                        help="Set config['eval_mode']=True. Tasks that distinguish "
                             "scripted vs eval flow (e.g. dump_bin_interrupt) drive "
                             "the avatar via a per-step state machine instead of "
                             "the choreographed play_once timeline.")
    parser.add_argument("--disable-stop-on-success", action="store_true",
                        help="Do not terminate rollout early when check_success() "
                             "becomes true.")
    parser.add_argument("--success-hold-steps", type=int, default=None,
                        help="Require success for N consecutive policy steps before "
                             "early termination. Default: 1.")
    parser.add_argument("--inspect-motion-name", type=str, default=None,
                        help="(interrupt tasks) Force a specific inspect motion by name.")
    parser.add_argument("--video-dir", type=str, default=None,
                        help="If set, save one MP4 per episode under this directory.")
    parser.add_argument("--video-limit", type=int, default=None,
                        help="When --video-dir is set, save videos only for the first N episodes.")
    parser.add_argument("--video-stride", type=int, default=None,
                        help="Capture 1 video frame every N sim steps when "
                             "--video-dir is set (default: task VIDEO_STRIDE).")
    parser.add_argument("--output-json", type=str, default=None,
                        help="If set, write aggregate and per-episode eval results to this JSON file.")
    parser.add_argument("--random-action-scale", type=float, default=0.0,
                        help="Magnitude of random arm-delta per step in the "
                             "default random_policy (only used if no real "
                             "policy is supplied). 0 = zero policy. Small "
                             "values like 0.02 produce visible joint jitter.")
    args = parser.parse_args()

    # config/default.yml is the base (renderer, nyx spp/denoise/env_texture,
    # camera resolutions); --config overlays it.
    config = load_config(args.config)

    if args.renderer:
        config["renderer"] = args.renderer
        if args.renderer in ("raytracer", "nyx"):
            os.environ.setdefault("GENESIS_BACKEND", "gpu")
        if args.renderer == "nyx":
            config.setdefault("nyx", {"spp": 1, "denoise": False, "open_window": False})
        if args.renderer == "raytracer":
            config.setdefault("raytracer", {
                "spp": 128, "tracing_depth": 32, "env_radius": 1000.0,
                "lights": [{"pos": [0, 0, 10], "color": [1, 1, 1], "intensity": 10, "radius": 4}],
            })
    if args.table_random_objects:
        config["table_random_objects"] = True
        config["task_irrelevant_objects"] = True
    if args.policy == "expert_no_human":
        args.no_human = True

    if args.no_human:
        config["no_human"] = True
        config["randomize_avatar"] = False
        config["track_avatar_collision"] = False
        config["avatar_retreat_enabled"] = False
        config["eval_mode"] = False
    else:
        # Policy eval bypasses scripted play_once(), so avatar tasks need
        # their eval-mode state machines to reproduce assist / interrupt /
        # neutral avatar behaviour during take_action() rollouts.
        config.setdefault("eval_mode", True)
        if args.policy == "expert_full_state":
            # Full-state expert is allowed privileged scheduling / safety
            # feedback.  Keep plain `expert` as the nominal scripted baseline.
            config.setdefault("track_avatar_collision", True)
            config.setdefault("use_avatar_collider", True)
            config.setdefault("collision_check_stride", 5)
            config.setdefault("neutral_avatar_start_delay_steps", 2500)
    if args.hard:
        config["hard"] = True
        config["difficulty"] = "hard"
    if args.no_table_random_objects:
        config["table_random_objects"] = False
        config["task_irrelevant_objects"] = False

    if args.track_avatar_collision:
        config["track_avatar_collision"] = True
    if args.collision_stride is not None:
        config["collision_check_stride"] = max(1, args.collision_stride)
    if args.collision_margin is not None:
        config["collision_margin"] = float(args.collision_margin)
    if args.avatar_safety_margin is not None:
        config["avatar_safety_margin"] = float(args.avatar_safety_margin)
    if args.avatar_retreat_enabled:
        config["avatar_retreat_enabled"] = True
    if args.avatar_retreat_margin is not None:
        config["avatar_retreat_margin"] = float(args.avatar_retreat_margin)
    if args.show_collider:
        config["use_avatar_collider"] = True
        config["collider_visualization"] = True
    if args.eval_mode:
        config["eval_mode"] = True
    if args.disable_stop_on_success:
        config["eval_stop_on_success"] = False
    if args.success_hold_steps is not None:
        config["eval_success_hold_steps"] = max(1, int(args.success_hold_steps))
    if args.inspect_motion_name:
        config["inspect_motion_name"] = args.inspect_motion_name
    if args.video_stride is not None:
        config["video_stride"] = max(1, args.video_stride)

    # Random policy for testing — replace with actual policy.
    rng = np.random.default_rng(0)

    def random_policy(obs):
        joint_state = obs.get("joint_state") if isinstance(obs, dict) else None
        if not isinstance(joint_state, dict):
            joint_state = {}
        right_raw = joint_state.get("right")
        left_q = joint_state.get("left")
        if right_raw is None:
            right_q = np.zeros(8, dtype=np.float64)
        else:
            right_q = np.asarray(right_raw, dtype=np.float64).ravel()
        # Observed joint_state includes arm qpos plus the scalar gripper
        # value; BaseTask.take_action expects arm deltas and separate
        # absolute gripper targets.
        n_right = max(0, int(right_q.size) - 1)
        if left_q is not None:
            n_left = max(0, int(np.asarray(left_q, dtype=np.float64).ravel().size) - 1)
            a = np.zeros(n_left + 1 + n_right + 1, dtype=np.float64)
        else:
            a = np.zeros(n_right + 1, dtype=np.float64)
        if args.random_action_scale > 0:
            if left_q is not None:
                a[:n_left] = rng.uniform(
                    -args.random_action_scale,
                    +args.random_action_scale,
                    size=n_left,
                )
                a[n_left + 1:n_left + 1 + n_right] = rng.uniform(
                    -args.random_action_scale,
                    +args.random_action_scale,
                    size=n_right,
                )
            else:
                a[:n_right] = rng.uniform(
                    -args.random_action_scale,
                    +args.random_action_scale,
                    size=n_right,
                )
        # Gripper slots are absolute targets in qpos mode. Hold open so
        # random-policy smoke tests perturb only the arm joints.
        if left_q is not None:
            a[n_left] = 1.0
            a[n_left + 1 + n_right] = 1.0
        else:
            a[n_right] = 1.0
        return a

    if args.policy in {"expert_full_state", "expert", "expert_no_human"}:
        policy = ExpertFullStatePolicy()
    else:
        policy = random_policy

    scene_inits = load_scene_init_file(args.scene_init_file) if args.scene_init_file else None
    result = eval_policy(
        args.task,
        config,
        policy,
        args.episodes,
        args.max_steps,
        args.start_seed,
        video_dir=args.video_dir,
        video_limit=args.video_limit,
        return_details=bool(args.output_json),
        scene_inits=scene_inits,
        scene_init_start_index=args.scene_init_start_index,
    )
    if args.output_json:
        if isinstance(result, dict) and args.scene_init_file:
            result["scene_init_file"] = args.scene_init_file
            result["scene_init_start_index"] = args.scene_init_start_index
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True, default=_json_default)
        print(f"Results JSON: {args.output_json}")


if __name__ == "__main__":
    main()
