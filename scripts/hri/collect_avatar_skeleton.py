"""Collect (head-camera RGB, avatar 3D joint) pairs from scripted rollouts.

Purpose: training data for the ManiCast-ACT baseline —
  1. AvatarPoseNet: RGB -> avatar upper-body joints (GT joints are labels
     here only; never read at eval time).
  2. STS-GCN forecaster: joint-history -> joint-future trajectories.

Per episode (one ``seed_<N>/`` dir under ``--out-root``):
  steps.h5
    images/<cam>      vlen JPEG bytes, one per policy step (20 Hz)
    avatar/joints     (T, J, 3) float32 world-frame bone positions
    avatar/present    (T,) uint8  — avatar exists and skin resolved
    sim_step          (T,) int64
    qpos              (T, n_arm) float32 robot arm joints (context only)
    ee                (T, 7) float32 TCP pose
    attrs: bones, capture_stride, camera intrinsics/extrinsics JSON
  meta.json           task, seed, success, wall_time, bones, cams

Run (one process loops seeds to amortize Genesis startup):
  python scripts/hri/collect_avatar_skeleton.py \
      --task stack_bowls_three_interrupt --start-seed 0 --seeds 40 \
      --out-root data/hri_manicast/skeleton_data/stack_bowls_three_interrupt
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from envs.tasks import resolve_task_class

# Mixamo bone names resolvable via AvatarRobot.skin.get_global_translation.
# 7-joint upper-body set following ManiCast (wrists/elbows/shoulders/back),
# plus Hips + Head for context/debug.
BONES = (
    "Hips",
    "Spine2",
    "Head",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightArm",
    "RightForeArm",
    "RightHand",
)


def _encode_jpeg(rgb_uint8: np.ndarray, quality: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb_uint8).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class AvatarSkeletonRecorder:
    """Lightweight VLARecorder-compatible hook (``task.vla_recorder = self``).

    ``BaseTask.step_sim`` calls ``tick(task)`` every physics step; we capture
    every ``capture_stride`` ticks (= policy cadence).
    """

    def __init__(self, out_dir, cams=("head_camera",), capture_stride=25,
                 jpeg_quality=85):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cams = tuple(cams)
        self.capture_stride = max(1, int(capture_stride))
        self.jpeg_quality = int(jpeg_quality)
        self._tick = 0
        self._rows: list[dict] = []
        self.cam_meta: dict[str, dict] = {}
        self._t0 = time.time()
        self._end_tick = None
        self._end_reason = None

    # -- VLARecorder-compatible surface ---------------------------------
    def tick(self, task) -> None:
        self._tick += 1
        if self._tick % self.capture_stride == 0:
            self._capture(task)

    def mark_episode_end(self, reason: str = "task_end") -> None:
        self._end_tick = int(self._tick)
        self._end_reason = str(reason)

    def discard(self) -> dict:
        n = len(self._rows)
        self._rows.clear()
        for name in ("steps.h5", "meta.json"):
            try:
                (self.out_dir / name).unlink()
            except FileNotFoundError:
                pass
        return {"n_captures": n}

    # -- capture ----------------------------------------------------------
    def _avatar_joints(self, task) -> tuple[np.ndarray, bool]:
        avatar = getattr(task, "avatar", None)
        skin = getattr(getattr(avatar, "robot", None), "skin", None)
        if skin is None:
            return np.full((len(BONES), 3), np.nan, dtype=np.float32), False
        out = np.empty((len(BONES), 3), dtype=np.float32)
        try:
            for i, bone in enumerate(BONES):
                out[i] = np.asarray(
                    skin.get_global_translation(bone)[0], dtype=np.float32
                ).ravel()[:3]
        except Exception:
            return np.full((len(BONES), 3), np.nan, dtype=np.float32), False
        return out, True

    def _capture(self, task) -> None:
        from envs.camera import _render_to_uint8_rgb

        images = {}
        for cam_name in self.cams:
            entry = task.cameras._cameras.get(cam_name)
            if entry is None:
                continue
            out = entry[0].render(rgb=True, depth=False)
            rgb_raw = out[0] if isinstance(out, (list, tuple)) else out
            rgb = _render_to_uint8_rgb(rgb_raw)
            if rgb is None:
                continue
            images[cam_name] = _encode_jpeg(rgb, self.jpeg_quality)
            if cam_name not in self.cam_meta:
                try:
                    self.cam_meta[cam_name] = {
                        "intrinsic": task.cameras.get_intrinsic(cam_name).tolist(),
                        "extrinsic": task.cameras.get_extrinsic(cam_name).tolist(),
                        "resolution": list(rgb.shape[:2][::-1]),
                    }
                except Exception:
                    pass

        joints, present = self._avatar_joints(task)
        arm = task.robot.right_arm
        self._rows.append({
            "sim_step": int(self._tick),
            "images": images,
            "joints": joints,
            "present": bool(present),
            "qpos": np.asarray(arm.get_arm_qpos(), dtype=np.float32),
            "ee": np.asarray(arm.get_ee_pose(), dtype=np.float32),
        })

    # -- write -------------------------------------------------------------
    def close(self, meta: dict) -> dict:
        import h5py

        rows = self._rows
        if self._end_tick is not None:
            rows = [r for r in rows if r["sim_step"] <= self._end_tick]
        n = len(rows)
        if n == 0:
            (self.out_dir / "meta.json").write_text(
                json.dumps({**meta, "n_steps": 0}, indent=1))
            return {"n_steps": 0}

        path = self.out_dir / "steps.h5"
        with h5py.File(path, "w") as f:
            f.attrs["schema"] = "hri_avatar_skeleton_v1"
            f.attrs["bones"] = json.dumps(list(BONES))
            f.attrs["capture_stride"] = self.capture_stride
            f.attrs["cam_meta"] = json.dumps(self.cam_meta)
            f.create_dataset("sim_step", data=np.array(
                [r["sim_step"] for r in rows], dtype=np.int64))
            f.create_dataset("avatar/joints", data=np.stack(
                [r["joints"] for r in rows]))
            f.create_dataset("avatar/present", data=np.array(
                [r["present"] for r in rows], dtype=np.uint8))
            f.create_dataset("qpos", data=np.stack([r["qpos"] for r in rows]))
            f.create_dataset("ee", data=np.stack([r["ee"] for r in rows]))
            img_grp = f.create_group("images")
            dt = h5py.vlen_dtype(np.dtype("uint8"))
            for cam in self.cams:
                blobs = [
                    np.frombuffer(r["images"].get(cam, b""), dtype=np.uint8)
                    for r in rows
                ]
                if any(len(b) for b in blobs):
                    ds = img_grp.create_dataset(cam, (n,), dtype=dt)
                    for i, b in enumerate(blobs):
                        ds[i] = b
        meta = {**meta, "n_steps": n, "bones": list(BONES),
                "cam_meta": self.cam_meta,
                "end_reason": self._end_reason}
        (self.out_dir / "meta.json").write_text(
            json.dumps(meta, indent=1, default=str))
        return {"n_steps": n}


def run_episode(task_name, TaskClass, config, seed, out_dir, args) -> dict:
    ep_t0 = time.time()
    recorder = AvatarSkeletonRecorder(
        out_dir,
        cams=tuple(args.cams.split(",")),
        capture_stride=int(config.get("action_substeps", 25)),
        jpeg_quality=args.jpeg_quality,
    )
    success = False
    error = None
    task = TaskClass(config)
    try:
        task.reset(seed=seed)
        task.vla_recorder = recorder  # captures everything in play_once
        success = bool(task.play_once())
    except Exception as e:
        error = repr(e)
        traceback.print_exc()
    stats = recorder.close({
        "task": task_name,
        "seed": seed,
        "success": success,
        "error": error,
        "wall_time": time.time() - ep_t0,
        "cams": args.cams,
    })
    return {"seed": seed, "success": success, "error": error, **stats}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=1, help="number of seeds")
    p.add_argument("--out-root", required=True)
    p.add_argument("--cams", default="head_camera")
    p.add_argument("--jpeg-quality", type=int, default=85)
    p.add_argument("--randomize-avatar", action="store_true")
    args = p.parse_args()

    task_name, TaskClass = resolve_task_class(args.task, no_human=False)
    config = {
        "track_avatar_collision": True,
        "table_random_objects": False,
        "task_irrelevant_objects": False,
    }
    if args.randomize_avatar:
        config["randomize_avatar"] = True

    out_root = Path(args.out_root)
    results = []
    for seed in range(args.start_seed, args.start_seed + args.seeds):
        ep_dir = out_root / f"seed_{seed}"
        if (ep_dir / "meta.json").exists():
            print(f"[skeleton] seed={seed} exists, skip", flush=True)
            continue
        print(f"[skeleton] task={task_name} seed={seed}", flush=True)
        try:
            res = run_episode(task_name, TaskClass, dict(config), seed, ep_dir, args)
        except Exception as e:
            traceback.print_exc()
            res = {"seed": seed, "success": False, "error": repr(e), "n_steps": 0}
        print(f"[skeleton] seed={seed} done: {res}", flush=True)
        results.append(res)

    ok = sum(1 for r in results if r.get("n_steps", 0) > 0)
    print(f"[skeleton] wrote {ok}/{len(results)} episodes under {out_root}")
    return 0


if __name__ == "__main__":
    main()
