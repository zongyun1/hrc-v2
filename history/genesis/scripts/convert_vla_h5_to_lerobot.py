"""Convert raw VLA steps.h5 episodes directly to a LeRobot dataset.

This is the faster ACT path for data already collected in this repo. It avoids
building TFDS/RLDS first, which is unnecessary for native LeRobot policies.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
try:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
except ModuleNotFoundError:
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from PIL import Image


def _decode_jpeg_resize(blob, size: int) -> np.ndarray:
    if isinstance(blob, np.ndarray):
        raw = blob.tobytes()
    else:
        raw = bytes(blob)
    if not raw:
        return np.zeros((size, size, 3), dtype=np.uint8)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _read_action_state(f: h5py.File, mode: str, t: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    qpos = f["qpos"]
    grip_meas = f["gripper_meas"]
    grip_action = f["gripper_action"]

    state = np.concatenate(
        [np.asarray(qpos[t], dtype=np.float32), [np.float32(grip_meas[t])]]
    ).astype(np.float32)

    if mode == "qpos_abs":
        target_qpos = qpos[t + 1] if t < n - 1 else qpos[t]
        target_gripper = grip_action[t] if t < n - 1 else grip_meas[t]
    elif mode == "qpos_target_abs":
        if "qpos_target_action" not in f:
            raise KeyError("steps.h5 missing qpos_target_action")
        qpos_target = f["qpos_target_action"]
        target_qpos = qpos_target[t] if t < n - 1 else qpos[t]
        target_gripper = grip_action[t] if t < n - 1 else grip_meas[t]
    elif mode == "qpos":
        qpos_delta = f["qpos_delta_action"]
        target_qpos = qpos_delta[t] if t < n - 1 else np.zeros_like(qpos[t], dtype=np.float32)
        target_gripper = grip_action[t] if t < n - 1 else grip_meas[t]
    else:
        raise ValueError(f"unsupported control mode: {mode}")

    action = np.concatenate(
        [np.asarray(target_qpos, dtype=np.float32), [np.float32(target_gripper)]]
    ).astype(np.float32)
    return state, action


def _episode_paths(raw_root: Path, renderer: str, max_episodes: int | None) -> list[Path]:
    paths: list[Path] = []
    for task_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        renderer_dir = task_dir / renderer
        if not renderer_dir.exists():
            continue
        for seed_dir in sorted(renderer_dir.glob("seed_*"), key=lambda p: int(p.name.split("_", 1)[1])):
            if (seed_dir / "steps.h5").exists() and (seed_dir / "meta.json").exists():
                paths.append(seed_dir)
                if max_episodes is not None and len(paths) >= max_episodes:
                    return paths
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, help="Root with <task>/<renderer>/seed_*/steps.h5")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--robot-type", default="xarm7")
    parser.add_argument("--renderer", default="rasterizer")
    parser.add_argument("--control-mode", choices=["qpos", "qpos_abs", "qpos_target_abs"], default="qpos_abs")
    parser.add_argument("--primary-cam", default="head_camera")
    parser.add_argument("--wrist-cam", default="right_wrist")
    parser.add_argument("--no-wrist-cam", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--image-writer-threads", type=int, default=10)
    parser.add_argument("--image-writer-processes", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    output_path = Path(HF_LEROBOT_HOME) / args.repo_id
    if output_path.exists():
        if not args.overwrite:
            raise SystemExit(f"refusing to overwrite existing dataset at {output_path}; pass --overwrite")
        shutil.rmtree(output_path)

    features = {
        "observation.image": {
            "dtype": "image",
            "shape": (args.image_size, args.image_size, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["actions"],
        },
    }
    if not args.no_wrist_cam:
        features["observation.wrist_image"] = {
            "dtype": "image",
            "shape": (args.image_size, args.image_size, 3),
            "names": ["height", "width", "channel"],
        }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        fps=args.fps,
        features=features,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )

    episodes = _episode_paths(raw_root, args.renderer, args.max_episodes)
    print(f"[h5_to_lerobot] raw_root={raw_root}", flush=True)
    print(f"[h5_to_lerobot] repo_id={args.repo_id}", flush=True)
    print(f"[h5_to_lerobot] episodes={len(episodes)}", flush=True)

    n_frames = 0
    for ep_i, seed_dir in enumerate(episodes, start=1):
        meta = json.load(open(seed_dir / "meta.json"))
        instruction = str(meta.get("instruction") or "")
        with h5py.File(seed_dir / "steps.h5", "r") as f:
            n = int(f["t"].shape[0])
            if n < 2:
                continue
            images = f["images"]
            if args.primary_cam not in images:
                raise KeyError(f"{seed_dir} missing camera {args.primary_cam!r}")
            primary = images[args.primary_cam]
            wrist = None if args.no_wrist_cam else images[args.wrist_cam] if args.wrist_cam in images else None
            for t in range(n):
                state, action = _read_action_state(f, args.control_mode, t, n)
                frame = {
                    "observation.image": _decode_jpeg_resize(primary[t], args.image_size),
                    "observation.state": state,
                    "action": action,
                    "task": instruction,
                }
                if not args.no_wrist_cam:
                    frame["observation.wrist_image"] = (
                        _decode_jpeg_resize(wrist[t], args.image_size)
                        if wrist is not None
                        else np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8)
                    )
                dataset.add_frame(frame)
                n_frames += 1
        dataset.save_episode()
        if ep_i % 10 == 0:
            print(f"[h5_to_lerobot] episode {ep_i} written ({n_frames} frames)", flush=True)

    print(f"[h5_to_lerobot] done. {len(episodes)} episodes, {n_frames} frames -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
