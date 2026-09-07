#!/usr/bin/env python3
"""Train the lightweight NumPy visual PACE detector from cached LeRobot data."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs.pace.fast_visual_progress import top_features


DEFAULT_DATA_ROOT = (
    "/scratch4/workspace/qinhongzhou_umass_edu-simple/cache/huggingface/lerobot/"
    "genesis-hr-bench/genesis_hr_bench_lerobot_qpos_target"
)


def _decode_png(buf: bytes) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("failed to decode PNG frame")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _is_human_task(tasks: list[str]) -> bool:
    text = " ".join(tasks).lower()
    return "human" in text or "person" in text


def _read_episodes(data_root: Path) -> list[dict]:
    paths = sorted((data_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no episode metadata under {data_root / 'meta' / 'episodes'}")
    cols = [
        "episode_index",
        "tasks",
        "length",
        "data/chunk_index",
        "data/file_index",
    ]
    rows: list[dict] = []
    for path in paths:
        data = pq.read_table(path, columns=cols).to_pydict()
        n = len(data["episode_index"])
        for i in range(n):
            tasks = data["tasks"][i]
            rows.append(
                {
                    "episode_index": int(data["episode_index"][i]),
                    "tasks": tasks,
                    "label": int(_is_human_task(tasks)),
                    "length": int(data["length"][i]),
                    "chunk": int(data["data/chunk_index"][i]),
                    "file": int(data["data/file_index"][i]),
                }
            )
    return rows


def _balanced_split(rows: list[dict], episodes_per_class: int, test_frac: float, seed: int):
    rng = random.Random(seed)
    by_label: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if row["length"] >= 2:
            by_label[int(row["label"])].append(row)
    if not by_label[0] or not by_label[1]:
        raise RuntimeError("need at least one positive and one negative episode")
    n = min(episodes_per_class, len(by_label[0]), len(by_label[1]))
    selected = []
    for label in (0, 1):
        candidates = list(by_label[label])
        rng.shuffle(candidates)
        selected.extend(candidates[:n])
    train_rows = []
    val_rows = []
    for label in (0, 1):
        label_rows = [r for r in selected if int(r["label"]) == label]
        rng.shuffle(label_rows)
        n_val = max(1, int(round(len(label_rows) * test_frac)))
        val_rows.extend(label_rows[:n_val])
        train_rows.extend(label_rows[n_val:])
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def _sample_frame_indices(length: int, frames_per_episode: int) -> set[int]:
    if length <= frames_per_episode:
        return set(range(length))
    # Skip the first frame; runtime triggering also needs at least one prior
    # frame for motion, and the first rendered frame is often a reset view.
    return set(np.linspace(1, length - 1, frames_per_episode).round().astype(int).tolist())


def _motion_score(prev: np.ndarray, curr: np.ndarray, crop_top_frac: float) -> float:
    h = curr.shape[0]
    y1 = max(1, int(round(h * crop_top_frac)))
    a = cv2.cvtColor(prev[:y1], cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(curr[:y1], cv2.COLOR_RGB2GRAY)
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)


def _collect_features(data_root: Path, rows: list[dict], frames_per_episode: int, crop_top_frac: float):
    by_file: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_file[(row["chunk"], row["file"])].append(row)

    features = []
    labels = []
    episode_eval = []
    motion_positive = []
    for (chunk_idx, file_idx), file_rows in sorted(by_file.items()):
        path = data_root / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"
        table = pq.read_table(
            path,
            columns=["observation.image", "episode_index", "frame_index"],
        ).to_pydict()
        wanted = {int(r["episode_index"]): r for r in file_rows}
        wanted_frames = {
            ep: _sample_frame_indices(row["length"], frames_per_episode)
            for ep, row in wanted.items()
        }
        prev_by_ep: dict[int, np.ndarray] = {}
        eval_by_ep = {
            ep: {"label": int(row["label"]), "scores": [], "sample_motions": []}
            for ep, row in wanted.items()
        }
        for image_ref, ep_raw, frame_raw in zip(
            table["observation.image"],
            table["episode_index"],
            table["frame_index"],
        ):
            ep = int(ep_raw)
            if ep not in wanted:
                continue
            frame_idx = int(frame_raw)
            img = _decode_png(image_ref["bytes"])
            prev = prev_by_ep.get(ep)
            motion = 0.0
            if prev is not None:
                motion = _motion_score(prev, img, crop_top_frac)
                if eval_by_ep[ep]["label"]:
                    motion_positive.append(motion)
            prev_by_ep[ep] = img
            if frame_idx not in wanted_frames[ep]:
                continue
            features.append(top_features(img, crop_top_frac))
            labels.append(eval_by_ep[ep]["label"])
            eval_by_ep[ep]["scores"].append(len(features) - 1)
            eval_by_ep[ep]["sample_motions"].append(motion)
        episode_eval.extend(eval_by_ep.values())
    if not features:
        raise RuntimeError("no frame features collected")
    return np.vstack(features).astype(np.float32), np.asarray(labels, dtype=np.int64), episode_eval, motion_positive


def _fit_linear(features: np.ndarray, labels: np.ndarray, seed: int):
    scaler = StandardScaler()
    x = scaler.fit_transform(features)
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    )
    clf.fit(x, labels)
    # Fold StandardScaler into a single runtime dot product over raw features.
    raw_weight = clf.coef_[0] / scaler.scale_
    raw_bias = float(clf.intercept_[0] - np.sum(clf.coef_[0] * scaler.mean_ / scaler.scale_))
    return clf, raw_weight.astype(np.float32), raw_bias


def _metrics(clf, features: np.ndarray, labels: np.ndarray):
    probs = clf.predict_proba(StandardScaler().fit(features).transform(features))[:, 1]
    # This helper is intentionally not used for reporting because the scaler
    # fitted here differs from training; report with folded weights instead.
    return probs


def _predict_raw(features: np.ndarray, weight: np.ndarray, bias: float) -> np.ndarray:
    logits = np.clip(features @ weight + bias, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _report_split(name: str, features: np.ndarray, labels: np.ndarray, weight: np.ndarray, bias: float):
    probs = _predict_raw(features, weight, bias)
    pred = probs >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        pred.astype(np.int64),
        average="binary",
        zero_division=0,
    )
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float("nan")
    return {
        "split": name,
        "frames": int(len(labels)),
        "positive_frames": int(labels.sum()),
        "accuracy": float(accuracy_score(labels, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": auc,
    }


def _episode_trigger_metrics(
    episode_eval: list[dict],
    features: np.ndarray,
    weight: np.ndarray,
    bias: float,
    present_threshold: float,
    motion_threshold: float,
    motion_steps_to_full: float,
    trigger_progress: float,
):
    triggered = []
    for episode in episode_eval:
        progress = 0.0
        best = 0.0
        fired = False
        for feature_idx, motion in zip(episode["scores"], episode["sample_motions"]):
            present = float(_predict_raw(features[feature_idx : feature_idx + 1], weight, bias)[0])
            if present >= present_threshold and motion >= motion_threshold:
                progress = min(1.0, progress + 1.0 / max(1.0, motion_steps_to_full))
                best = max(best, progress)
            else:
                progress *= 0.98
            if present >= present_threshold and best >= trigger_progress and motion >= motion_threshold:
                fired = True
                break
        triggered.append((int(episode["label"]), fired))
    labels = np.asarray([x[0] for x in triggered], dtype=np.int64)
    fires = np.asarray([x[1] for x in triggered], dtype=np.int64)
    return {
        "episodes": int(len(triggered)),
        "positive_episodes": int(labels.sum()),
        "trigger_rate_positive": float(fires[labels == 1].mean()) if np.any(labels == 1) else float("nan"),
        "trigger_rate_negative": float(fires[labels == 0].mean()) if np.any(labels == 0) else float("nan"),
    }


def train(args):
    data_root = Path(args.data_root)
    rows = _read_episodes(data_root)
    train_rows, val_rows = _balanced_split(
        rows,
        args.episodes_per_class,
        args.test_frac,
        args.seed,
    )
    train_x, train_y, train_ep, train_motion_pos = _collect_features(
        data_root,
        train_rows,
        args.frames_per_episode,
        args.crop_top_frac,
    )
    val_x, val_y, val_ep, val_motion_pos = _collect_features(
        data_root,
        val_rows,
        args.frames_per_episode,
        args.crop_top_frac,
    )
    _, weight, bias = _fit_linear(train_x, train_y, args.seed)
    motion_values = np.asarray(train_motion_pos + val_motion_pos, dtype=np.float32)
    if len(motion_values):
        positive = motion_values[motion_values > 0]
        motion_threshold = float(np.quantile(positive, args.motion_quantile)) if len(positive) else 0.006
    else:
        motion_threshold = 0.006
    motion_threshold = max(args.motion_floor, motion_threshold)

    train_report = _report_split("train", train_x, train_y, weight, bias)
    val_report = _report_split("val", val_x, val_y, weight, bias)
    trigger_report = _episode_trigger_metrics(
        val_ep,
        val_x,
        weight,
        bias,
        args.present_threshold,
        motion_threshold,
        args.motion_steps_to_full,
        args.trigger_progress,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        crop_top_frac=np.asarray(args.crop_top_frac, dtype=np.float32),
        weight=weight.astype(np.float32),
        bias=np.asarray(bias, dtype=np.float32),
        motion_threshold=np.asarray(motion_threshold, dtype=np.float32),
        motion_steps_to_full=np.asarray(args.motion_steps_to_full, dtype=np.float32),
    )
    report = {
        "data_root": str(data_root),
        "output": str(output),
        "episodes_per_class": int(args.episodes_per_class),
        "train_episodes": len(train_rows),
        "val_episodes": len(val_rows),
        "frames_per_episode": int(args.frames_per_episode),
        "crop_top_frac": float(args.crop_top_frac),
        "motion_threshold": motion_threshold,
        "motion_steps_to_full": float(args.motion_steps_to_full),
        "train": train_report,
        "val": val_report,
        "val_trigger": trigger_report,
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", default="data/hri2_visual_yield/checkpoints/hri2_fast_visual_lerobot.npz")
    parser.add_argument("--report", default="data/hri2_visual_yield/checkpoints/hri2_fast_visual_lerobot_report.json")
    parser.add_argument("--episodes-per-class", type=int, default=80)
    parser.add_argument("--frames-per-episode", type=int, default=16)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--crop-top-frac", type=float, default=0.48)
    parser.add_argument("--motion-quantile", type=float, default=0.70)
    parser.add_argument("--motion-floor", type=float, default=0.006)
    parser.add_argument("--motion-steps-to-full", type=float, default=4.0)
    parser.add_argument("--present-threshold", type=float, default=0.5)
    parser.add_argument("--trigger-progress", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
