"""Build a DP3 zarr replay buffer from genesis_hr_bench rollouts.

Reads the streaming-recorder output from `scripts/vla_data/collect_vla.py`
(`vla_data/<task>/<renderer>/seed_<N>/{steps.h5, meta.json}`) and writes a
DP3-flavoured zarr (matching the AdroitDataset / RealDexDataset layout) to
`runs/dp3/<task>/<task>.zarr/`.

Layout:
  data/point_cloud : (N_total, n_points, 3 or 6) float32  world-frame XYZ[+RGB]
  data/state       : (N_total, 8) float32                 [qpos (7), gripper (1)]
  data/action      : (N_total, 8) float32                 [qpos_delta (7), gripper_action (1)]
  data/img         : (N_total, H, W, 3) uint8             primary RGB (kept for visualisation; AdroitDataset reads but doesn't use)
  meta/episode_ends: (n_episodes,) int64

Point clouds are derived from `depth/<primary_cam>` in the recorded h5,
lifted with the same `depth_to_world_points` helper the inference client uses
(`scripts/vla_client.py`). The camera extrinsic is auto-resolved from
the robot's embodiment YAML (`assets/embodiments/<name>/config.yml`) — most
robots (franka, piper, arx_x5, ur5_wsg) override the head_camera position
there, and using the BaseTask default would put the cloud in the wrong frame.
Pass `--cam-pos` / `--cam-forward` to override; pass `--robot NAME` to skip
the task→robot lookup.

Run from the dp3 venv:
    /scratch4/workspace/qinhongzhou_umass_edu-simple/yz/env/dp3/bin/python \
        scripts/build_dp3_zarr.py --task pour_water --n-points 1024
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml
import zarr
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "baseline" / "3d_diffusion_policy" / "3D-Diffusion-Policy"))

from scripts.vla_client import (  # noqa: E402
    depth_to_world_points, crop_bbox, voxel_downsample, farthest_point_sample,
)
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer  # noqa: E402


def intrinsic_from_fov(fovy_deg: float, w: int, h: int) -> np.ndarray:
    fy = (h / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    fx = fy
    return np.array([[fx, 0, w / 2.0], [0, fy, h / 2.0], [0, 0, 1]], dtype=np.float64)


def extrinsic_from_pos_forward(pos, forward, up=(0, 0, 1)) -> np.ndarray:
    """Same convention as `envs/camera.py:_extrinsic_from_pos_lookat` (OpenGL:
    +Y up, -Z forward in cam frame).
    """
    pos = np.asarray(pos, dtype=np.float64)
    forward = np.asarray(forward, dtype=np.float64)
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    up = np.asarray(up, dtype=np.float64)
    left = np.cross(forward, up)
    n = np.linalg.norm(left)
    left = left / n if n > 1e-9 else np.array([1, 0, 0.0])
    up = np.cross(left, forward)
    up /= np.linalg.norm(up)
    R = np.eye(4)
    R[:3, :3] = np.stack([left, up, -forward], axis=1)
    R[:3, 3] = pos
    return R


def _decode_jpeg_resize(blob, size):
    if isinstance(blob, np.ndarray):
        if blob.size == 0:
            return np.zeros((size, size, 3), dtype=np.uint8)
        raw = blob.tobytes()
    else:
        if not blob:
            return np.zeros((size, size, 3), dtype=np.uint8)
        raw = bytes(blob)
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    if pil.size != (size, size):
        pil = pil.resize((size, size), Image.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def lift_one_step(depth, intrinsic, extrinsic, rgb=None, n_points=1024,
                  use_color=False, voxel_size=0.005):
    """Single-step pipeline matching the inference client."""
    pts = depth_to_world_points(
        depth.astype(np.float32), intrinsic, extrinsic,
        rgb=rgb if use_color else None,
    )
    pts = crop_bbox(pts)
    if pts.shape[0] == 0:
        D = 6 if use_color else 3
        return np.zeros((n_points, D), dtype=np.float32)
    pts = voxel_downsample(pts, voxel_size=voxel_size)
    return farthest_point_sample(pts, n_points)


def _episode_arrays(h5_path, image_size, primary_cam, intrinsic, extrinsic,
                    n_points, use_color):
    with h5py.File(h5_path, "r") as f:
        n = int(f["t"].shape[0])
        if n < 2:
            return None
        if primary_cam not in f["images"]:
            print(f"  skip {h5_path} (missing image cam {primary_cam!r})")
            return None
        if "depth" not in f or primary_cam not in f["depth"]:
            print(f"  skip {h5_path} (missing depth cam {primary_cam!r})")
            return None

        qpos = f["qpos"][:].astype(np.float32)
        qpos_delta = f["qpos_delta_action"][:].astype(np.float32)
        grip_meas = f["gripper_meas"][:].astype(np.float32)
        grip_act = f["gripper_action"][:].astype(np.float32)

        depth_arr = f["depth"][primary_cam][:].astype(np.float32)  # (n, H, W)
        img_blobs = list(f["images"][primary_cam][:])

    state = np.concatenate([qpos, grip_meas[:, None]], axis=1)
    action = np.concatenate([qpos_delta, grip_act[:, None]], axis=1)

    point_clouds = np.empty((n, n_points, 6 if use_color else 3), dtype=np.float32)
    img_resized = np.empty((n, image_size, image_size, 3), dtype=np.uint8)
    for i in range(n):
        rgb = _decode_jpeg_resize(img_blobs[i], image_size)
        img_resized[i] = rgb
        # Use the original-resolution RGB to colour the cloud; resize to
        # depth's H×W if shapes differ.
        rgb_full = np.asarray(Image.open(io.BytesIO(bytes(img_blobs[i]))).convert("RGB"))
        if rgb_full.shape[:2] != depth_arr[i].shape[:2]:
            rgb_full = np.asarray(Image.fromarray(rgb_full).resize(
                (depth_arr[i].shape[1], depth_arr[i].shape[0]), Image.BILINEAR))
        point_clouds[i] = lift_one_step(
            depth_arr[i], intrinsic, extrinsic,
            rgb=rgb_full if use_color else None,
            n_points=n_points, use_color=use_color,
        )
    return point_clouds, state, action, img_resized


# Maps `robot_type` (as used by BaseTask._load_robot) to the embodiment dir
# under assets/embodiments/. None = no embodiment YAML; use BaseTask defaults.
ROBOT_TO_EMBODIMENT = {
    "franka": "franka-panda",
    "piper": "piper",
    "arx_x5": "ARX-X5", "arx-x5": "ARX-X5", "arxx5": "ARX-X5",
    "ur5_wsg": "ur5-wsg", "ur5-wsg": "ur5-wsg", "ur5wsg": "ur5-wsg",
    "xarm7": None,
    "stretch": None,
}

# Default head_camera if no embodiment YAML overrides it. Matches
# envs/base_task.py:_load_cameras() fallback.
BASETASK_DEFAULT_CAM = {
    "position": [0.0, 0.2, 2.0],
    "forward": [0.0, -0.4, -0.9],
}


def detect_robot_type(task_name: str) -> str:
    """Static-source lookup of `setdefault("robot_type", ...)` for a task.

    Doesn't import the task class (which would pull Genesis). Walks the file
    that exports the task and returns the first match; falls back to "piper"
    (BaseTask default) if not found.
    """
    tasks_dir = REPO_ROOT / "envs" / "tasks"
    init_text = (tasks_dir / "__init__.py").read_text()
    m = re.search(rf'["\']{re.escape(task_name)}["\']\s*:\s*(\w+)', init_text)
    if not m:
        return "piper"
    cls = m.group(1)
    m2 = re.search(rf'from \.(\w+) import [^\n]*\b{re.escape(cls)}\b', init_text)
    if not m2:
        return "piper"
    src = (tasks_dir / f"{m2.group(1)}.py").read_text()
    m3 = re.search(
        r'setdefault\(\s*["\']robot_type["\']\s*,\s*["\']([\w-]+)["\']', src
    )
    return m3.group(1).lower() if m3 else "piper"


def resolve_camera(robot_type: str, primary_cam: str) -> tuple[list, list, str]:
    """Return (position, forward, source_str) for `primary_cam` given a robot.

    Reads `static_camera_list` from the embodiment YAML when the robot has
    one; otherwise returns the BaseTask default.
    """
    embodiment = ROBOT_TO_EMBODIMENT.get(robot_type)
    if embodiment is None:
        return (list(BASETASK_DEFAULT_CAM["position"]),
                list(BASETASK_DEFAULT_CAM["forward"]),
                f"BaseTask default (robot {robot_type!r} has no embodiment YAML)")
    yaml_path = REPO_ROOT / "assets" / "embodiments" / embodiment / "config.yml"
    if not yaml_path.exists():
        return (list(BASETASK_DEFAULT_CAM["position"]),
                list(BASETASK_DEFAULT_CAM["forward"]),
                f"BaseTask default (no {yaml_path})")
    cfg = yaml.safe_load(yaml_path.read_text()) or {}
    cams = cfg.get("static_camera_list") or []
    for cam in cams:
        if cam.get("name") == primary_cam:
            return (list(cam["position"]), list(cam["forward"]),
                    f"{yaml_path.relative_to(REPO_ROOT)} → static_camera_list[{primary_cam}]")
    return (list(BASETASK_DEFAULT_CAM["position"]),
            list(BASETASK_DEFAULT_CAM["forward"]),
            f"BaseTask default ({primary_cam!r} not found in {yaml_path.name})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--raw-root", default=str(REPO_ROOT / "vla_data"))
    parser.add_argument("--renderer", default="rasterizer")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--primary-cam", default="head_camera")
    parser.add_argument("--image-size", type=int, default=84,
                        help="Resize for the kept RGB visualisation array (DP3 ignores it).")
    parser.add_argument("--n-points", type=int, default=1024)
    parser.add_argument("--use-color", action="store_true",
                        help="Store XYZRGB clouds (n_points, 6) instead of XYZ.")
    parser.add_argument("--cam-w", type=int, default=640)
    parser.add_argument("--cam-h", type=int, default=480)
    parser.add_argument("--cam-fovy", type=float, default=60.0)
    parser.add_argument("--robot", default=None,
                        choices=sorted(ROBOT_TO_EMBODIMENT.keys()),
                        help="robot_type for camera lookup. Default: auto-detect from task source.")
    parser.add_argument("--cam-pos", nargs=3, type=float, default=None,
                        help="Override camera position. Default: read from embodiment YAML.")
    parser.add_argument("--cam-forward", nargs=3, type=float, default=None,
                        help="Override camera forward. Default: read from embodiment YAML.")
    parser.add_argument("--include-failures", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    task_root = raw_root / args.task / args.renderer
    if not task_root.exists():
        raise SystemExit(f"no episodes found under {task_root}")

    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / "runs" / "dp3" / args.task / f"{args.task}.zarr"
    )
    if out_dir.exists():
        raise SystemExit(f"refuse to overwrite existing {out_dir} — delete it first.")
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    robot_type = args.robot or detect_robot_type(args.task)
    yaml_pos, yaml_forward, cam_source = resolve_camera(robot_type, args.primary_cam)
    cam_pos = args.cam_pos if args.cam_pos is not None else yaml_pos
    cam_forward = args.cam_forward if args.cam_forward is not None else yaml_forward

    intrinsic = intrinsic_from_fov(args.cam_fovy, args.cam_w, args.cam_h)
    extrinsic = extrinsic_from_pos_forward(cam_pos, cam_forward)

    print(f"task         = {args.task}")
    print(f"raw          = {task_root}")
    print(f"out          = {out_dir}")
    print(f"n_points     = {args.n_points}")
    print(f"use_color    = {args.use_color}")
    print(f"primary_cam  = {args.primary_cam}")
    print(f"robot_type   = {robot_type}{' (auto)' if args.robot is None else ''}")
    print(f"cam_pos      = {cam_pos}")
    print(f"cam_forward  = {cam_forward}")
    print(f"cam_source   = {cam_source}"
          + ("" if (args.cam_pos is None and args.cam_forward is None)
             else "  [overridden by CLI]"))
    print()

    store = zarr.DirectoryStore(str(out_dir))
    rb = ReplayBuffer.create_empty_zarr(storage=store)

    n_total, n_skipped, total_steps = 0, 0, 0
    seed_dirs = sorted(p for p in task_root.iterdir()
                       if p.is_dir() and p.name.startswith("seed_"))
    for seed_dir in seed_dirs:
        meta_path = seed_dir / "meta.json"
        h5_path = seed_dir / "steps.h5"
        if not meta_path.exists() or not h5_path.exists():
            n_skipped += 1
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if not args.include_failures and not bool(meta.get("success", False)):
            n_skipped += 1
            continue

        ep = _episode_arrays(
            h5_path, args.image_size, args.primary_cam,
            intrinsic, extrinsic, args.n_points, args.use_color,
        )
        if ep is None:
            n_skipped += 1
            continue
        pc, state, action, img = ep
        rb.add_episode({
            "point_cloud": pc,
            "state": state,
            "action": action,
            "img": img,
        })
        total_steps += int(state.shape[0])
        n_total += 1
        if n_total % 10 == 0:
            print(f"  [{n_total}] {seed_dir.name} ({state.shape[0]} steps)")

    if n_total == 0:
        raise SystemExit("no episodes written — check --raw-root / --renderer / success filter.")

    print(f"\nwrote {n_total} episodes ({total_steps} steps); skipped {n_skipped}")
    print(f"zarr at: {out_dir}")
    print(f"  data/point_cloud {rb.data['point_cloud'].shape} {rb.data['point_cloud'].dtype}")
    print(f"  data/state       {rb.data['state'].shape} {rb.data['state'].dtype}")
    print(f"  data/action      {rb.data['action'].shape} {rb.data['action'].dtype}")
    print(f"  meta/episode_ends {rb.episode_ends.shape}  last={int(rb.episode_ends[-1])}")


if __name__ == "__main__":
    main()
