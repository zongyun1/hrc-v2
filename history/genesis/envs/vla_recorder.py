"""VLA training-data recorder.

Hooks into ``BaseTask.step_sim`` and snapshots state every
``action_substeps`` sim steps (matches the policy frequency that
``BaseTask.take_action`` runs at -- default 25 sim steps = 50 ms = 20 Hz).

What gets logged per capture:
  qpos_meas        right-arm joint positions (radians) [N_arm]
  qpos_target      right-arm commanded PD target (radians) [N_arm], when available
  qpos_target_valid whether qpos_target came from a cached commanded target
  ee_meas          [x, y, z, qw, qx, qy, qz] world-frame TCP pose
  gripper_meas     legacy scalar in [0, 1], currently the commanded target
  gripper_target   commanded scalar in [0, 1], 1=open and 0=closed
  images           JPEG-encoded bytes per chosen camera

Action labels are back-filled when the *next* capture lands -- the action
at step t is what the policy would output to move from t to t+stride.  The
file stores both legacy measured deltas (qpos_delta_action, ee_delta_action)
and the absolute commanded joint target for ACT-style training
(qpos_target_action), plus a validity mask for target labels.

The final row gets None for the action label (no future state to diff
against) and is dropped at close() time so the saved (obs, action) pairs
are always well-defined.

Output: one ``steps.h5`` per episode containing the arrays plus
JPEG-encoded image bytes (variable-length), plus a ``meta.json`` with
instruction, seed, success, evaluate() output, and camera intrinsics +
extrinsics.  Layout chosen because tiny-files-per-frame eats the parallel
filesystem inode budget and slows downstream loaders.
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

SCHEMA_VERSION = "vla_steps_v2_qpos_target"
ACTION_LABEL_KEYS = (
    "qpos_target_action",
    "qpos_delta_action",
    "ee_delta_action",
    "gripper_target_action",
    "gripper_action",
)


def _arm_qpos_from_full_target(arm) -> Optional[np.ndarray]:
    """Extract arm-joint target values from the robot's cached full qpos."""
    target = getattr(arm, "_cached_target", None)
    if target is None:
        return None
    try:
        from .robot.franka_robot import _get_dof_idx, to_numpy
    except Exception:
        return None
    target = np.asarray(to_numpy(target), dtype=np.float64).ravel()
    vals = []
    for joint in getattr(arm, "arm_joints", []):
        if joint is None:
            continue
        idx = _get_dof_idx(joint)
        if idx is None or idx >= len(target):
            return None
        vals.append(float(target[idx]))
    return np.asarray(vals, dtype=np.float64)


def _quat_inv(q):
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    return np.array([w, -x, -y, -z]) / max(n, 1e-12)


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _quat_to_axis_angle(q):
    """Return 3-vec axis*angle (rad).  Sign-normalized to short rotation."""
    q = np.asarray(q, dtype=np.float64)
    if q[0] < 0:
        q = -q
    w = float(np.clip(q[0], -1.0, 1.0))
    s = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if s < 1e-8:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arccos(w)
    return q[1:4] / s * angle


def _encode_jpeg(rgb_uint8: np.ndarray, quality: int) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb_uint8).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class VLARecorder:
    """Streaming recorder.  One instance per episode.

    Attach by giving ``BaseTask`` access to the recorder; ``step_sim`` then
    calls ``recorder.tick(task)`` after every physics step.  The recorder
    decides when to capture based on its own substep counter.
    """

    def __init__(
        self,
        out_dir: str,
        cams: Iterable[str] = ("recording", "right_wrist"),
        jpeg_quality: int = 92,
        capture_stride: int = 25,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cams = tuple(cams)
        self.jpeg_quality = int(jpeg_quality)
        self.capture_stride = max(1, int(capture_stride))

        self._tick = 0
        self._captures: list[dict] = []
        # Cached per-camera intrinsics/extrinsics (filled on first capture).
        self.cam_meta: dict[str, dict] = {}
        self._t0 = time.time()
        self._end_tick: int | None = None
        self._end_reason: str | None = None

    # ------------------------------------------------------------------
    # Hook called from BaseTask.step_sim()
    # ------------------------------------------------------------------

    def tick(self, task) -> None:
        self._tick += 1
        if self._tick % self.capture_stride == 0:
            self._capture(task)

    def mark_episode_end(self, reason: str = "task_end") -> None:
        """Mark the last simulation tick that should be kept in the dataset.

        Tasks can call this when the useful policy segment has ended but they
        still need extra physics settling for a reliable success check.
        """
        self._end_tick = int(self._tick)
        self._end_reason = str(reason)

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def _capture(self, task) -> None:
        right_arm = task.robot.right_arm
        qpos = np.asarray(right_arm.get_arm_qpos(), dtype=np.float64)
        qpos_target = _arm_qpos_from_full_target(right_arm)
        qpos_target_valid = qpos_target is not None and qpos_target.shape == qpos.shape
        if not qpos_target_valid:
            qpos_target = qpos.copy()
        ee = np.asarray(right_arm.get_ee_pose(), dtype=np.float64)
        gripper_target = float(right_arm.gripper_val)

        # Render only the chosen cams.  We deliberately avoid render_all()
        # to skip cams we don't persist (each render call is a real GPU
        # cost, especially under raytracer).
        #
        # Under the Madrona batch renderer the first cam.render() of a sim
        # step renders ALL cameras at once into a per-timestep cache, so the
        # wrist cameras must be repositioned BEFORE any render this tick —
        # otherwise a non-wrist cam rendered first would freeze the cache at
        # the wrist cams' stale poses. So hoist the wrist update out of the
        # loop. (Harmless for the rasterizer/raytracer path.)
        from .camera import _finalize_rgb, _finalize_depth
        from .utils import Pose
        if any(c in ("left_wrist", "right_wrist") for c in self.cams):
            left_ee = task.robot.get_ee_pose("left") if task.robot.left_arm else None
            right_ee = task.robot.get_ee_pose("right")
            left_pose = Pose.from_pose7(left_ee) if left_ee is not None else None
            right_pose = Pose.from_pose7(right_ee) if right_ee is not None else None
            task.cameras.update_wrist_cameras(left_pose, right_pose)

        images: dict[str, bytes] = {}
        depths: dict[str, np.ndarray] = {}
        for cam_name in self.cams:
            entry = task.cameras._cameras.get(cam_name)
            if entry is None:
                continue
            cam = entry[0]
            out = cam.render(rgb=True, depth=True)
            rgb_raw = out[0] if isinstance(out, (list, tuple)) and len(out) > 0 else None
            depth_raw = out[1] if isinstance(out, (list, tuple)) and len(out) > 1 else None
            rgb = _finalize_rgb(cam, rgb_raw)
            if rgb is None:
                continue
            images[cam_name] = _encode_jpeg(rgb, self.jpeg_quality)
            if depth_raw is not None:
                depth_arr = _finalize_depth(cam, depth_raw)
                while depth_arr.ndim > 2 and depth_arr.shape[0] == 1:
                    depth_arr = depth_arr[0]
                depths[cam_name] = depth_arr.astype(np.float16)
            if cam_name not in self.cam_meta:
                try:
                    self.cam_meta[cam_name] = {
                        "intrinsic": task.cameras.get_intrinsic(cam_name).tolist(),
                        "extrinsic": task.cameras.get_extrinsic(cam_name).tolist(),
                        "resolution": list(rgb.shape[:2][::-1]),  # [W, H]
                    }
                except Exception:
                    pass

        self._captures.append({
            "sim_step": int(self._tick),
            "wall_time": float(time.time() - self._t0),
            "qpos": qpos,
            "qpos_target": qpos_target,
            "qpos_target_valid": bool(qpos_target_valid),
            "ee": ee,
            "gripper": gripper_target,
            "gripper_target": gripper_target,
            "images": images,
            "depths": depths,
        })

    # ------------------------------------------------------------------
    # Close: back-fill action labels and write to disk
    # ------------------------------------------------------------------

    def close(self, meta: dict) -> dict:
        """Write steps.h5 + meta.json.  Returns summary stats."""
        captures = self._captures
        if self._end_tick is not None:
            captures = [c for c in captures if int(c["sim_step"]) <= self._end_tick]
            meta = dict(meta)
            meta["vla_trim"] = {
                "end_tick": int(self._end_tick),
                "reason": self._end_reason,
                "dropped_captures": len(self._captures) - len(captures),
            }

        if len(captures) < 2:
            # Nothing meaningful to label.  Still emit meta so the dir is
            # marked complete (avoids retry storms for empty rollouts).
            self._write_meta(meta, n_steps=0)
            return {"n_steps": 0, "n_pairs": 0}

        # Action[t] = state[t+1] - state[t] (qpos delta + ee delta in
        # local-EE axis-angle).  Drop the last capture (no successor).
        rows = []
        for i in range(len(captures) - 1):
            cur = captures[i]
            nxt = captures[i + 1]
            qpos_delta = (nxt["qpos"] - cur["qpos"]).astype(np.float64)

            cur_pos, cur_quat = cur["ee"][:3], cur["ee"][3:7]
            nxt_pos, nxt_quat = nxt["ee"][:3], nxt["ee"][3:7]
            ee_pos_delta = (nxt_pos - cur_pos).astype(np.float64)
            ee_rot_delta = _quat_to_axis_angle(_quat_mul(_quat_inv(cur_quat), nxt_quat))
            ee_delta = np.concatenate([ee_pos_delta, ee_rot_delta]).astype(np.float64)
            gripper_action = float(nxt["gripper_target"])

            rows.append({
                "t": cur["sim_step"],
                "qpos": cur["qpos"],
                "qpos_target": cur["qpos_target"],
                "qpos_target_valid": cur["qpos_target_valid"],
                "ee": cur["ee"],
                "gripper_meas": cur["gripper"],
                "qpos_target_action": nxt["qpos_target"],
                "qpos_target_action_valid": nxt["qpos_target_valid"],
                "qpos_delta_action": qpos_delta,
                "ee_delta_action": ee_delta,
                "gripper_target_action": gripper_action,
                "gripper_action": gripper_action,
                "images": cur["images"],
                "depths": cur.get("depths", {}),
            })

        self._write_h5(rows)
        self._write_meta(meta, n_steps=len(rows))
        return {"n_steps": len(rows), "n_pairs": len(rows)}

    def discard(self) -> dict:
        """Drop captured frames and remove any recorder products for this episode."""
        n_captures = len(self._captures)
        self._captures.clear()
        removed: list[str] = []
        for name in ("steps.h5", "meta.json"):
            path = self.out_dir / name
            try:
                path.unlink()
                removed.append(name)
            except FileNotFoundError:
                pass
        return {"n_captures": n_captures, "removed": removed}

    # ------------------------------------------------------------------
    # Disk writes
    # ------------------------------------------------------------------

    def _write_h5(self, rows: list[dict]) -> None:
        import h5py
        n = len(rows)
        n_arm = rows[0]["qpos"].shape[0]
        path = self.out_dir / "steps.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("t", data=np.array([r["t"] for r in rows], dtype=np.int64))
            f.create_dataset(
                "qpos",
                data=np.stack([r["qpos"] for r in rows]).astype(np.float32),
                compression="gzip", compression_opts=4,
            )
            f.create_dataset(
                "qpos_target",
                data=np.stack([r["qpos_target"] for r in rows]).astype(np.float32),
                compression="gzip", compression_opts=4,
            )
            f.create_dataset(
                "qpos_target_valid",
                data=np.array([r["qpos_target_valid"] for r in rows], dtype=np.bool_),
            )
            f.create_dataset(
                "ee",
                data=np.stack([r["ee"] for r in rows]).astype(np.float32),
                compression="gzip", compression_opts=4,
            )
            f.create_dataset(
                "gripper_meas",
                data=np.array([r["gripper_meas"] for r in rows], dtype=np.float32),
            )
            f.create_dataset(
                "gripper_target",
                data=np.array([r["gripper_meas"] for r in rows], dtype=np.float32),
            )
            f.create_dataset(
                "qpos_delta_action",
                data=np.stack([r["qpos_delta_action"] for r in rows]).astype(np.float32),
                compression="gzip", compression_opts=4,
            )
            f.create_dataset(
                "qpos_target_action",
                data=np.stack([r["qpos_target_action"] for r in rows]).astype(np.float32),
                compression="gzip", compression_opts=4,
            )
            f.create_dataset(
                "qpos_target_action_valid",
                data=np.array([r["qpos_target_action_valid"] for r in rows], dtype=np.bool_),
            )
            f.create_dataset(
                "ee_delta_action",
                data=np.stack([r["ee_delta_action"] for r in rows]).astype(np.float32),
                compression="gzip", compression_opts=4,
            )
            f.create_dataset(
                "gripper_action",
                data=np.array([r["gripper_action"] for r in rows], dtype=np.float32),
            )
            f.create_dataset(
                "gripper_target_action",
                data=np.array([r["gripper_target_action"] for r in rows], dtype=np.float32),
            )
            # JPEG bytes per cam, variable length.
            vlen = h5py.vlen_dtype(np.dtype("uint8"))
            img_grp = f.create_group("images")
            for cam in self.cams:
                blobs_arr = np.empty(n, dtype=object)
                nonempty = 0
                for i, r in enumerate(rows):
                    b = r["images"].get(cam, b"")
                    if b:
                        nonempty += 1
                    blobs_arr[i] = np.frombuffer(b, dtype="uint8")
                if nonempty == 0:
                    print(f"[vla_recorder] WARN: cam {cam!r} captured 0 frames; skipping dataset")
                    continue
                img_grp.create_dataset(cam, dtype=vlen, data=blobs_arr)

            # Depth per cam: (N, H, W) float16, gzipped. Skip cams with no
            # depth captures (e.g. raytracer-only paths that didn't return depth).
            depth_grp = f.create_group("depth")
            for cam in self.cams:
                arrays = [r["depths"].get(cam) for r in rows]
                arrays = [a for a in arrays if a is not None]
                if not arrays:
                    continue
                stacked = np.stack(arrays, axis=0).astype(np.float16)
                depth_grp.create_dataset(
                    cam, data=stacked, compression="gzip", compression_opts=4,
                )

            f.attrs["n_arm"] = n_arm
            f.attrs["capture_stride"] = self.capture_stride
            f.attrs["cams"] = list(self.cams)
            f.attrs["schema_version"] = SCHEMA_VERSION
            f.attrs["action_label_keys"] = json.dumps(list(ACTION_LABEL_KEYS))

    def _write_meta(self, meta: dict, n_steps: int) -> None:
        meta_out = dict(meta)
        meta_out["n_steps"] = n_steps
        meta_out["cameras"] = self.cam_meta
        meta_out["recorder"] = {
            "cams": list(self.cams),
            "jpeg_quality": self.jpeg_quality,
            "capture_stride": self.capture_stride,
            "schema_version": SCHEMA_VERSION,
            "action_label_keys": list(ACTION_LABEL_KEYS),
        }
        with open(self.out_dir / "meta.json", "w") as f:
            json.dump(meta_out, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (bytes,)):
        return None
    return str(o)
