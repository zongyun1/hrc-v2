"""Small PbD-style phase controller for ACT HRI eval.

The controller learns a discrete phase label from existing action traces.  At
runtime it is used as a trigger source for the existing generic PACE
retreat/wait/return controller; low-level robot motion stays deterministic.
"""

from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import json
import pickle
from typing import Any

import numpy as np


PHASES = ("continue", "yield", "wait", "return", "resume")
FEATURE_NAMES = (
    "step_frac",
    "visual_present",
    "visual_active",
    "visual_progress",
    "visual_best_progress",
    "visual_motion_score",
    "visual_fired",
    "eval_interrupt_fired",
    "eval_glide_active",
    "eval_retreat_active",
    "eval_return_active",
    "eval_waiting_avatar_done",
    "qpos_home_l2",
    "qpos_step_l2",
    "qpos_linf",
    "gripper",
    "ee_z",
)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        v = float(value)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _bool_float(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    return arr


def phase_label_from_trace_row(row: dict) -> str | None:
    """Derive the phase label described in the pipeline note."""
    source = str(row.get("action_source") or "")
    generic = row.get("generic_pace") if isinstance(row.get("generic_pace"), dict) else {}
    phase = str(generic.get("phase") or "")

    if source == "policy":
        return "continue"
    if source == "generic_pace:snapshot_action":
        return "resume"
    if source.startswith("generic_pace:"):
        phase = source.split(":", 1)[1]
    if phase == "retreat":
        return "yield"
    if phase in {"wait", "return"}:
        return phase
    return None


def trace_row_features(
    row: dict,
    *,
    max_step: int,
    home_qpos: np.ndarray | None,
    prev_qpos: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    visual = row.get("visual_pace") if isinstance(row.get("visual_pace"), dict) else {}
    qpos = _array(row.get("qpos_before"))
    ee = _array(row.get("ee_pose_before"))
    home = home_qpos if home_qpos is not None else qpos
    prev = prev_qpos if prev_qpos is not None else qpos
    if qpos is None:
        qpos_home_l2 = 0.0
        qpos_step_l2 = 0.0
        qpos_linf = 0.0
    else:
        qpos_home_l2 = float(np.linalg.norm(qpos - home)) if home is not None else 0.0
        qpos_step_l2 = float(np.linalg.norm(qpos - prev)) if prev is not None else 0.0
        qpos_linf = float(np.max(np.abs(qpos)))
    features = np.array(
        [
            _float(row.get("step")) / max(1.0, float(max_step)),
            _float(visual.get("present")),
            _float(visual.get("active")),
            _float(visual.get("progress")),
            _float(visual.get("best_progress")),
            _float(visual.get("motion_score")),
            _bool_float(visual.get("fired")),
            _bool_float(row.get("eval_interrupt_fired")),
            _bool_float(row.get("eval_glide_active")),
            _bool_float(row.get("eval_retreat_active")),
            _bool_float(row.get("eval_return_active")),
            _bool_float(row.get("eval_waiting_avatar_done")),
            qpos_home_l2,
            qpos_step_l2,
            qpos_linf,
            _float(row.get("gripper_before"), 1.0),
            _float(ee[2] if ee is not None and ee.size >= 3 else None),
        ],
        dtype=np.float64,
    )
    return features, qpos


def load_trace_examples(
    paths: list[str | Path],
    *,
    drop_visual: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    xs: list[np.ndarray] = []
    ys: list[str] = []
    files: list[str] = []
    for path in paths:
        p = Path(path)
        try:
            rows = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(rows, list) or not rows:
            continue
        max_step = max(1, max(int(_float(r.get("step"))) for r in rows if isinstance(r, dict)))
        home_qpos = None
        prev_qpos = None
        n_file = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            label = phase_label_from_trace_row(row)
            if label is None:
                continue
            if drop_visual:
                row = dict(row)
                row["visual_pace"] = {}
            features, qpos = trace_row_features(
                row,
                max_step=max_step,
                home_qpos=home_qpos,
                prev_qpos=prev_qpos,
            )
            if home_qpos is None and qpos is not None:
                home_qpos = qpos.copy()
            prev_qpos = qpos.copy() if qpos is not None else prev_qpos
            xs.append(features)
            ys.append(label)
            n_file += 1
        if n_file:
            files.append(str(p))
    if not xs:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float64), np.array([], dtype=object), files
    return np.vstack(xs), np.asarray(ys, dtype=object), files


def find_trace_files(roots: list[str | Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        p = Path(root)
        if p.is_file():
            out.append(p)
        elif p.exists():
            out.extend(sorted(p.rglob("*_trace.json")))
    return sorted(dict.fromkeys(out))


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    return (x - mean) / std, mean, std


def train_nearest_centroid(x: np.ndarray, y: np.ndarray) -> dict:
    z, mean, std = _standardize(x)
    labels = [label for label in PHASES if np.any(y == label)]
    centroids = []
    for label in labels:
        centroids.append(z[y == label].mean(axis=0))
    return {
        "model_type": "nearest_centroid",
        "feature_names": list(FEATURE_NAMES),
        "labels": labels,
        "mean": mean,
        "std": std,
        "centroids": np.vstack(centroids),
        "class_counts": dict(Counter(map(str, y))),
    }


def _predict_one(model: dict, x: np.ndarray) -> tuple[str, float, dict[str, float]]:
    z = (np.asarray(x, dtype=np.float64) - model["mean"]) / model["std"]
    labels = list(model["labels"])
    if model["model_type"] == "nearest_centroid":
        centroids = np.asarray(model["centroids"], dtype=np.float64)
        d = np.linalg.norm(centroids - z[None, :], axis=1)
        logits = -d
    else:
        w = np.asarray(model["weights"], dtype=np.float64)
        b = np.asarray(model["bias"], dtype=np.float64)
        logits = z @ w + b
    logits = logits - float(np.max(logits))
    probs = np.exp(logits)
    probs = probs / max(1e-12, float(np.sum(probs)))
    idx = int(np.argmax(probs))
    prob_map = {labels[i]: float(probs[i]) for i in range(len(labels))}
    return labels[idx], float(probs[idx]), prob_map


def predict_batch(model: dict, x: np.ndarray) -> np.ndarray:
    return np.asarray([_predict_one(model, row)[0] for row in x], dtype=object)


def save_model(path: str | Path, model: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(model, f)


def load_model(path: str | Path) -> dict:
    with Path(path).open("rb") as f:
        return pickle.load(f)


class HRIPhaseControllerRuntime:
    """Runtime wrapper used by ``scripts.eval`` as a generic-yield trigger."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        min_confidence: float = 0.0,
        min_step: int = 0,
        trigger_labels: tuple[str, ...] = ("yield",),
        consecutive: int = 1,
    ):
        self.checkpoint = str(checkpoint)
        self.model = load_model(checkpoint)
        self.min_confidence = float(min_confidence)
        self.min_step = int(min_step)
        self.trigger_labels = tuple(trigger_labels)
        self.consecutive = max(1, int(consecutive))
        self._recent: deque[str] = deque(maxlen=self.consecutive)
        self.home_qpos = None
        self.prev_qpos = None
        self.max_steps = 1
        self.last_state: dict[str, Any] = {}
        self.prediction_counts: Counter[str] = Counter()
        self.triggered = False

    def reset(self, task=None, max_steps: int = 1) -> None:
        self._recent.clear()
        self.max_steps = max(1, int(max_steps))
        self.prev_qpos = None
        self.last_state = {}
        self.prediction_counts.clear()
        self.triggered = False
        self.home_qpos = None
        if task is not None:
            try:
                self.home_qpos = np.asarray(
                    task.robot.get_arm("right").get_arm_qpos(),
                    dtype=np.float64,
                ).ravel().copy()
            except Exception:
                self.home_qpos = None

    def update(self, task, obs, step: int, visual_state=None, generic_pace=None) -> dict:
        row = {
            "step": int(step),
            "visual_pace": visual_state if isinstance(visual_state, dict) else {},
            "eval_interrupt_fired": bool(getattr(task, "_eval_interrupt_fired", False)),
            "eval_glide_active": bool(getattr(task, "_eval_glide_active", False)),
            "eval_retreat_active": bool(getattr(task, "_eval_retreat_active", False)),
            "eval_return_active": bool(getattr(task, "_eval_return_active", False)),
            "eval_waiting_avatar_done": bool(getattr(task, "_eval_waiting_avatar_done", False)),
        }
        try:
            arm = task.robot.get_arm("right")
            row["qpos_before"] = np.asarray(arm.get_arm_qpos(), dtype=float).ravel().tolist()
            row["gripper_before"] = float(getattr(arm, "gripper_val", 1.0))
            row["ee_pose_before"] = np.asarray(arm.get_ee_pose(), dtype=float).ravel().tolist()
        except Exception:
            pass
        features, qpos = trace_row_features(
            row,
            max_step=self.max_steps,
            home_qpos=self.home_qpos,
            prev_qpos=self.prev_qpos,
        )
        if self.home_qpos is None and qpos is not None:
            self.home_qpos = qpos.copy()
        self.prev_qpos = qpos.copy() if qpos is not None else self.prev_qpos
        phase, confidence, probs = _predict_one(self.model, features)
        self._recent.append(phase)
        self.prediction_counts[phase] += 1
        self.last_state = {
            "enabled": True,
            "checkpoint": self.checkpoint,
            "step": int(step),
            "phase": phase,
            "confidence": confidence,
            "probabilities": probs,
            "trigger_labels": list(self.trigger_labels),
            "min_confidence": self.min_confidence,
            "min_step": self.min_step,
            "consecutive": self.consecutive,
            "recent": list(self._recent),
        }
        return self.last_state

    def should_trigger(self) -> tuple[bool, str]:
        if self.triggered or not self.last_state:
            return False, "hri_phase:not_ready"
        phase = str(self.last_state.get("phase"))
        step = int(self.last_state.get("step", 0))
        confidence = float(self.last_state.get("confidence", 0.0))
        stable = (
            len(self._recent) >= self.consecutive
            and all(p in self.trigger_labels for p in self._recent)
        )
        should = (
            stable
            and phase in self.trigger_labels
            and confidence >= self.min_confidence
            and step >= self.min_step
        )
        if should:
            self.triggered = True
            return True, f"hri_phase:{phase}:{confidence:.3f}"
        return False, f"hri_phase:{phase}:{confidence:.3f}"

    def metrics(self) -> dict:
        out = dict(self.last_state) if self.last_state else {"enabled": True}
        out["prediction_counts"] = dict(self.prediction_counts)
        out["triggered"] = self.triggered
        return out
