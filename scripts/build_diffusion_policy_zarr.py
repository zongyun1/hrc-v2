"""Build a Diffusion Policy zarr replay buffer from genesis_hr_bench rollouts.

Reads the streaming-recorder output written by ``scripts/vla_data/collect_vla.py``
(``vla_data/<task>/<renderer>/seed_<N>/{steps.h5, meta.json}``) and writes a
pusht-style zarr replay buffer to ``runs/diffusion_policy/<task>/<task>.zarr/``.

Layout matches what ``baseline/diffusion_policy/diffusion_policy/dataset/
pusht_image_dataset.py:PushTImageDataset`` reads — ``img``, ``state``,
``action`` arrays under ``data/`` plus ``meta/episode_ends``:

  data/img      : (N_total, H, W, 3) uint8   primary RGB (head_camera, resized)
  data/state    : (N_total, 8) float32       [qpos (7), gripper (1)] — agent_pos
  data/action   : (N_total, 8) float32       [qpos_delta (7), gripper_action (1)]
  meta/episode_ends : (n_episodes,) int64    cumulative end indices

Action / proprio convention is wired to match BaseTask.take_action(action_type=
"qpos") at inference (first 7 dims are joint deltas, dim 7 is gripper command;
gripper polarity 1=open follows the recorder/repo convention end-to-end).

Run:
    python scripts/build_diffusion_policy_zarr.py --task pour_water --image-size 96

The ``diffusion_policy`` env at ``yz/env/diffusion_policy`` already has zarr,
h5py, and pillow — invoke its python directly.
"""
import argparse
import io
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import zarr
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "baseline" / "diffusion_policy"))

from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402


def _decode_jpeg_resize(blob, size):
    """h5 vlen uint8 (or raw bytes) → uint8 (size, size, 3)."""
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


def _episode_arrays(h5_path: Path, image_size: int, primary_cam: str):
    """Return (img, state, action) for one episode, or None if too short / missing cam."""
    with h5py.File(h5_path, "r") as f:
        n = int(f["t"].shape[0])
        if n < 2:
            return None
        if primary_cam not in f["images"]:
            print(f"  skip {h5_path} (missing {primary_cam!r}; have {list(f['images'].keys())})")
            return None

        qpos = f["qpos"][:].astype(np.float32)                 # (n, 7)
        qpos_delta = f["qpos_delta_action"][:].astype(np.float32)  # (n, 7)
        grip_meas = f["gripper_meas"][:].astype(np.float32)    # (n,)
        grip_act = f["gripper_action"][:].astype(np.float32)   # (n,)
        img_blobs = list(f["images"][primary_cam][:])

    state = np.concatenate([qpos, grip_meas[:, None]], axis=1)        # (n, 8)
    action = np.concatenate([qpos_delta, grip_act[:, None]], axis=1)  # (n, 8)
    img = np.stack(
        [_decode_jpeg_resize(b, image_size) for b in img_blobs], axis=0
    )  # (n, H, W, 3) uint8
    return img, state, action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True,
                        help="Task name; reads vla_data/<task>/<renderer>/seed_*/.")
    parser.add_argument("--raw-root", default=str(REPO_ROOT / "vla_data"))
    parser.add_argument("--renderer", default="rasterizer")
    parser.add_argument("--out-dir", default=None,
                        help="Output dir (default: runs/diffusion_policy/<task>/<task>.zarr).")
    parser.add_argument("--image-size", type=int, default=96,
                        help="Square resize for the primary camera (default 96 — pusht-style).")
    parser.add_argument("--primary-cam", default="head_camera")
    parser.add_argument("--include-failures", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    task_root = raw_root / args.task / args.renderer
    if not task_root.exists():
        raise SystemExit(f"no episodes found under {task_root}")

    out_dir = Path(args.out_dir) if args.out_dir else (
        REPO_ROOT / "runs" / "diffusion_policy" / args.task / f"{args.task}.zarr"
    )
    if out_dir.exists():
        raise SystemExit(f"refuse to overwrite existing {out_dir} — delete it first.")
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"task        = {args.task}")
    print(f"raw         = {task_root}")
    print(f"out         = {out_dir}")
    print(f"image_size  = {args.image_size}")
    print()

    store = zarr.DirectoryStore(str(out_dir))
    rb = ReplayBuffer.create_empty_zarr(storage=store)

    n_total, n_skipped, total_steps = 0, 0, 0
    seed_dirs = sorted(p for p in task_root.iterdir() if p.is_dir() and p.name.startswith("seed_"))
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

        ep = _episode_arrays(h5_path, args.image_size, args.primary_cam)
        if ep is None:
            n_skipped += 1
            continue
        img, state, action = ep
        rb.add_episode({"img": img, "state": state, "action": action})
        total_steps += int(state.shape[0])
        n_total += 1
        if n_total % 10 == 0:
            print(f"  [{n_total}] {seed_dir.name}  ({state.shape[0]} steps)")

    if n_total == 0:
        raise SystemExit("no episodes written — check --raw-root / --renderer / success filter.")

    print(f"\nwrote {n_total} episodes ({total_steps} steps); skipped {n_skipped}")
    print(f"zarr at: {out_dir}")
    print(f"  data/img    {rb.data['img'].shape} {rb.data['img'].dtype}")
    print(f"  data/state  {rb.data['state'].shape} {rb.data['state'].dtype}")
    print(f"  data/action {rb.data['action'].shape} {rb.data['action'].dtype}")
    print(f"  meta/episode_ends {rb.episode_ends.shape}  last={int(rb.episode_ends[-1])}")


if __name__ == "__main__":
    main()
