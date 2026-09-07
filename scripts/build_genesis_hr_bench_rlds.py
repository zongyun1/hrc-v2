"""Build the genesis_hr_bench RLDS dataset for OpenVLA / OpenVLA-OFT fine-tuning.

Reads the streaming-recorder output written by ``scripts/vla_data/collect_vla.py``
(which lives at ``vla_data/<task>/<renderer>/seed_<N>/{steps.h5, meta.json,
verdict.json}``) and writes a TFDS RLDS dataset to
    ``openvla/data/genesis_hr_bench/1.0.0/``.

Per-step features (Bridge-V2-shaped EE by default, with extra wrist image for OFT):
  observation/image:        (256, 256, 3) uint8   primary RGB (head_camera, resized)
  observation/wrist_image:  (256, 256, 3) uint8   right-wrist RGB (resized)
  observation/state:        (7,) float32          EE mode: [x, y, z, roll, pitch, yaw, gripper]
                                                  gripper polarity 1 = closed (Bridge convention)
                            (8,) float32          qpos/qpos_abs mode: [7 qpos, gripper]
                                                  gripper polarity 1 = open (repo convention)
  action:                   (7,) float32          EE mode: [dx, dy, dz, droll, dpitch, dyaw, gripper]
                                                  gripper polarity 1 = closed (Bridge convention)
                            (8,) float32          qpos mode: [7 qpos_delta, gripper]
                                                  gripper polarity 1 = open (repo convention)
                            (8,) float32          qpos_abs mode: [7 next measured qpos, gripper]
                            (8,) float32          qpos_target_abs mode: [7 commanded target qpos, gripper]
                                                  gripper polarity 1 = open (repo convention)
  language_instruction:     string                from meta.instruction
  reward / discount / is_first / is_last / is_terminal: standard RLDS

The h5 already stores per-step ``ee_delta_action`` as 6-D ``[Δxyz, axis-angle]``.
We convert the axis-angle delta to an Euler-angle (sxyz) delta to match the
existing Bridge-V2 action schema, then append the bridge-convention gripper.

Only episodes with ``meta.success == true`` and at least 2 captured steps are
included. The trailing recorded step always has ``ee_delta_action == 0`` (the
recorder drops the final un-labelable capture; see ``envs/vla_recorder.py``)
so we synthesize a terminal step here with zero delta + held gripper, and mark
``is_last`` / ``is_terminal``.

Run:
    python scripts/build_genesis_hr_bench_rlds.py \\
        [--raw-root vla_data] \\
        [--renderer rasterizer] \\
        [--data-dir openvla/data] \\
        [--image-size 256] \\
        [--include-failures]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import transforms3d as t3d
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# Module-level config — set by main() before tfds.builder() instantiates the class.
_BUILDER_CFG = {
    "raw_root": str(REPO_ROOT / "vla_data"),
    "renderer": "rasterizer",
    "image_size": 256,
    "include_failures": False,
    "primary_cam": "head_camera",
    "wrist_cam": "right_wrist",
    "control_mode": "ee",
}


# ---------- helpers ---------------------------------------------------------

def _quat_wxyz_to_euler(q):
    """(w, x, y, z) → (roll, pitch, yaw) in radians (sxyz convention)."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(t3d.euler.quat2euler([w, x, y, z], axes="sxyz"), dtype=np.float32)


def _axis_angle_to_euler(aa):
    """3-vec axis*angle (rad) → (roll, pitch, yaw) sxyz (rad)."""
    aa = np.asarray(aa, dtype=np.float64)
    angle = float(np.linalg.norm(aa))
    if angle < 1e-8:
        return np.zeros(3, dtype=np.float32)
    axis = aa / angle
    q = t3d.quaternions.axangle2quat(axis, angle)  # (w, x, y, z)
    return np.array(t3d.euler.quat2euler(q, axes="sxyz"), dtype=np.float32)


def _decode_jpeg_resize(jpeg_blob, size):
    """JPEG bytes (or h5 vlen uint8 array) → uint8 H×W×3 resized to (size, size)."""
    # h5py vlen returns a 1-D uint8 ndarray; the dummy-fill path passes raw bytes.
    if isinstance(jpeg_blob, np.ndarray):
        if jpeg_blob.size == 0:
            return np.zeros((size, size, 3), dtype=np.uint8)
        raw = jpeg_blob.tobytes()
    else:
        if not jpeg_blob:
            return np.zeros((size, size, 3), dtype=np.uint8)
        raw = bytes(jpeg_blob)
    import io
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    if pil.size != (size, size):
        pil = pil.resize((size, size), Image.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def _episode_to_steps(h5_path, meta, image_size, primary_cam, wrist_cam, control_mode):
    """Convert a single seed_<N>/steps.h5 + meta.json into a list of step dicts.

    Returns None if the episode is too short or required cams are missing.
    """
    with h5py.File(h5_path, "r") as f:
        n = f["t"].shape[0]
        if n < 2:
            return None

        ee = f["ee"][:]                       # (n, 7)  [xyz, qwxyz]
        ee_delta = f["ee_delta_action"][:]    # (n, 6)  [Δxyz, axis-angle]
        qpos = f["qpos"][:]                   # (n, 7)  Franka arm joints
        qpos_delta = f["qpos_delta_action"][:]  # (n, 7) joint deltas
        qpos_target_action = f["qpos_target_action"][:] if "qpos_target_action" in f else None
        grip_meas = f["gripper_meas"][:]      # (n,)   1=open (repo convention)
        grip_act = f["gripper_action"][:]     # (n,)   1=open (repo convention)

        img_grp = f["images"]
        if primary_cam not in img_grp:
            print(f"  skip {h5_path} (missing {primary_cam!r}; have {list(img_grp.keys())})")
            return None
        primary_blobs = list(img_grp[primary_cam][:])
        wrist_blobs = list(img_grp[wrist_cam][:]) if wrist_cam in img_grp else [b""] * n

    instruction = str(meta.get("instruction") or "")

    steps = []
    for t in range(n):
        if control_mode in {"qpos", "qpos_abs", "qpos_target_abs"}:
            state = np.concatenate(
                [qpos[t].astype(np.float32), [np.float32(grip_meas[t])]]
            ).astype(np.float32)
            if control_mode == "qpos_abs":
                target_qpos = qpos[t + 1] if t < n - 1 else qpos[t]
                target_gripper = grip_act[t] if t < n - 1 else grip_meas[t]
                action = np.concatenate(
                    [target_qpos.astype(np.float32), [np.float32(target_gripper)]]
                ).astype(np.float32)
            elif control_mode == "qpos_target_abs":
                if qpos_target_action is None:
                    raise KeyError(
                        f"{h5_path} is missing qpos_target_action; "
                        "recollect with the updated VLARecorder"
                    )
                target_qpos = qpos_target_action[t] if t < n - 1 else qpos[t]
                target_gripper = grip_act[t] if t < n - 1 else grip_meas[t]
                action = np.concatenate(
                    [target_qpos.astype(np.float32), [np.float32(target_gripper)]]
                ).astype(np.float32)
            elif t < n - 1:
                action = np.concatenate(
                    [qpos_delta[t].astype(np.float32), [np.float32(grip_act[t])]]
                ).astype(np.float32)
            else:
                action = np.concatenate(
                    [np.zeros_like(qpos_delta[t], dtype=np.float32), [np.float32(grip_meas[t])]]
                ).astype(np.float32)
        else:
            pos = ee[t, :3].astype(np.float32)
            rpy = _quat_wxyz_to_euler(ee[t, 3:7])
            # Bridge convention: 1 = closed. Recorder logs 1 = open.
            grip_state_bridge = np.float32(1.0 - float(grip_meas[t]))
            state = np.concatenate([pos, rpy, [grip_state_bridge]]).astype(np.float32)

            # The h5 always has ee_delta_action[n-1] == 0 (recorder convention).
            # We emit an explicit terminal step at index n-1 with held gripper.
            if t < n - 1:
                d_xyz = ee_delta[t, :3].astype(np.float32)
                d_rpy = _axis_angle_to_euler(ee_delta[t, 3:6])
                grip_act_bridge = np.float32(1.0 - float(grip_act[t]))
            else:
                d_xyz = np.zeros(3, dtype=np.float32)
                d_rpy = np.zeros(3, dtype=np.float32)
                grip_act_bridge = grip_state_bridge
            action = np.concatenate([d_xyz, d_rpy, [grip_act_bridge]]).astype(np.float32)

        steps.append({
            "observation": {
                "image": _decode_jpeg_resize(primary_blobs[t], image_size),
                "wrist_image": _decode_jpeg_resize(wrist_blobs[t], image_size),
                "state": state,
            },
            "action": action,
            "discount": np.float32(1.0),
            "reward": np.float32(1.0 if t == n - 1 else 0.0),
            "is_first": t == 0,
            "is_last": t == n - 1,
            "is_terminal": t == n - 1,
            "language_instruction": instruction,
        })
    return steps


# ---------- TFDS builder ----------------------------------------------------

class GenesisHrBench(tfds.core.GeneratorBasedBuilder):
    """genesis_hr_bench — Bridge-V2-shaped RLDS dataset of scripted Franka
    rollouts from the Genesis HR Bench, for OpenVLA / OpenVLA-OFT fine-tuning."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release: scripted rollouts, head_camera + right_wrist."}

    def _info(self) -> tfds.core.DatasetInfo:
        size = _BUILDER_CFG["image_size"]
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "genesis_hr_bench: Bridge-V2-shaped scripted Franka rollouts "
                "from the Genesis HR Bench, for OpenVLA / OpenVLA-OFT fine-tuning."
            ),
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(
                            shape=(size, size, 3), dtype=np.uint8,
                            encoding_format="jpeg",
                            doc="head_camera RGB",
                        ),
                        "wrist_image": tfds.features.Image(
                            shape=(size, size, 3), dtype=np.uint8,
                            encoding_format="jpeg",
                            doc="right_wrist RGB",
                        ),
                        "state": tfds.features.Tensor(
                            shape=(8 if _BUILDER_CFG["control_mode"] in {"qpos", "qpos_abs", "qpos_target_abs"} else 7,),
                            dtype=np.float32,
                            doc=("[7 qpos, gripper] (gripper 1=open)"
                                 if _BUILDER_CFG["control_mode"] in {"qpos", "qpos_abs", "qpos_target_abs"}
                                 else "[x, y, z, roll, pitch, yaw, gripper] (gripper 1=closed)"),
                        ),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(8 if _BUILDER_CFG["control_mode"] in {"qpos", "qpos_abs", "qpos_target_abs"} else 7,),
                        dtype=np.float32,
                        doc=("[7 target qpos, gripper] (gripper 1=open)"
                             if _BUILDER_CFG["control_mode"] in {"qpos_abs", "qpos_target_abs"}
                             else "[7 qpos_delta, gripper] (gripper 1=open)"
                             if _BUILDER_CFG["control_mode"] == "qpos"
                             else "[dx, dy, dz, droll, dpitch, dyaw, gripper] (gripper 1=closed)"),
                    ),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(),
                    "task_name": tfds.features.Text(),
                    "seed": tfds.features.Scalar(dtype=np.int32),
                }),
            }),
        )

    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples()}

    def _generate_examples(self):
        raw_root = Path(_BUILDER_CFG["raw_root"])
        renderer = _BUILDER_CFG["renderer"]
        image_size = _BUILDER_CFG["image_size"]
        include_failures = _BUILDER_CFG["include_failures"]
        primary_cam = _BUILDER_CFG["primary_cam"]
        wrist_cam = _BUILDER_CFG["wrist_cam"]
        control_mode = _BUILDER_CFG["control_mode"]
        if not raw_root.exists():
            raise FileNotFoundError(f"raw_root not found: {raw_root}")

        n_total, n_skipped = 0, 0
        # Walk vla_data/<task>/<renderer>/seed_*/.
        for task_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
            renderer_dir = task_dir / renderer
            if not renderer_dir.exists():
                continue
            task_name = task_dir.name
            for seed_dir in sorted(renderer_dir.glob("seed_*")):
                meta_path = seed_dir / "meta.json"
                h5_path = seed_dir / "steps.h5"
                if not meta_path.exists() or not h5_path.exists():
                    continue
                with open(meta_path) as f:
                    meta = json.load(f)
                if not include_failures and not bool(meta.get("success", False)):
                    n_skipped += 1
                    continue

                steps = _episode_to_steps(
                    h5_path, meta, image_size, primary_cam, wrist_cam, control_mode
                )
                if steps is None:
                    n_skipped += 1
                    continue

                seed_str = seed_dir.name.split("_", 1)[1]
                try:
                    seed_int = int(seed_str)
                except ValueError:
                    seed_int = -1

                ep_id = f"{task_name}/{seed_dir.name}"
                # os.path.relpath handles both relative h5_path (anchored at cwd)
                # and absolute h5_path that may live outside REPO_ROOT (e.g. when
                # vla_data is a symlink to /lustre-storage/...). Path.relative_to
                # rejects the relative case and the symlink-resolved case alike.
                yield ep_id, {
                    "steps": steps,
                    "episode_metadata": {
                        "file_path": os.path.relpath(h5_path, REPO_ROOT),
                        "task_name": task_name,
                        "seed": np.int32(seed_int),
                    },
                }
                n_total += 1
                if n_total % 10 == 0:
                    print(f"  [{n_total}] {ep_id}  ({len(steps)} steps)")

        print(f"\nemitted {n_total} episodes; skipped {n_skipped} (failures / missing / too-short)")


# ---------- action statistics for OpenVLA -----------------------------------

def compute_action_stats(out_dir: Path):
    """OpenVLA normalizes actions per dataset using 1st/99th percentiles.
    Walk the just-built TFDS shards and dump dataset_statistics.json next to them."""
    builder = tfds.builder_from_directory(str(out_dir))
    ds = builder.as_dataset(split="train")
    actions, n_episodes = [], 0
    for ep in ds:
        n_episodes += 1
        for step in ep["steps"]:
            actions.append(step["action"].numpy())
    arr = np.stack(actions, axis=0)
    action_dim = int(arr.shape[1])
    stats = {
        "action": {
            "mean":  arr.mean(axis=0).tolist(),
            "std":   arr.std(axis=0).tolist(),
            "min":   arr.min(axis=0).tolist(),
            "max":   arr.max(axis=0).tolist(),
            "p01":   np.percentile(arr, 1, axis=0).tolist(),
            "p99":   np.percentile(arr, 99, axis=0).tolist(),
            "mask": [True] * (action_dim - 1) + [False],  # don't normalize gripper dim
        },
        "num_transitions": int(arr.shape[0]),
        "num_trajectories": int(n_episodes),
    }
    out_path = out_dir / "dataset_statistics.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nwrote {out_path}  ({arr.shape[0]} transitions, {n_episodes} episodes)")


# ---------- main ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default=str(REPO_ROOT / "vla_data"),
                        help="Root containing <task>/<renderer>/seed_*/ episode dirs.")
    parser.add_argument("--renderer", default="rasterizer",
                        help="Subdirectory under each task to read from (rasterizer|raytracer).")
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "openvla" / "data"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--include-failures", action="store_true",
                        help="Include episodes with meta.success==false (default: success only).")
    parser.add_argument("--primary-cam", default="head_camera")
    parser.add_argument("--wrist-cam", default="right_wrist")
    parser.add_argument("--control-mode", choices=["ee", "qpos", "qpos_abs", "qpos_target_abs"], default="ee",
                        help="State/action schema to emit. ee keeps Bridge-style "
                             "7D EE deltas; qpos emits [7 qpos, gripper_open] "
                             "state and [7 qpos_delta, gripper_open] actions; "
                             "qpos_abs emits [7 qpos, gripper_open] state and "
                             "[7 next measured qpos, gripper_open] actions; "
                             "qpos_target_abs emits [7 qpos, gripper_open] state "
                             "and [7 commanded target qpos, gripper_open] actions.")
    parser.add_argument("--skip-stats", action="store_true",
                        help="Skip computing dataset_statistics.json after build.")
    args = parser.parse_args()

    _BUILDER_CFG.update({
        "raw_root": args.raw_root,
        "renderer": args.renderer,
        "image_size": args.image_size,
        "include_failures": args.include_failures,
        "primary_cam": args.primary_cam,
        "wrist_cam": args.wrist_cam,
        "control_mode": args.control_mode,
    })

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"raw_root        = {args.raw_root}")
    print(f"renderer        = {args.renderer}")
    print(f"data_dir        = {data_dir}")
    print(f"image_size      = {args.image_size}")
    print(f"primary/wrist   = {args.primary_cam} / {args.wrist_cam}")
    print(f"control_mode    = {args.control_mode}")
    print(f"include_failures= {args.include_failures}")
    print()

    builder = GenesisHrBench(data_dir=str(data_dir))
    builder.download_and_prepare()
    print()
    print(f"built TFDS dataset at: {data_dir / 'genesis_hr_bench' / '1.0.0'}")

    if not args.skip_stats:
        compute_action_stats(data_dir / "genesis_hr_bench" / "1.0.0")


if __name__ == "__main__":
    main()
