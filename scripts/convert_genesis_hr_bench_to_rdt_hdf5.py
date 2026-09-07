#!/usr/bin/env python
"""Convert genesis-hr-bench raw episodes to RDT-1B finetune HDF5.

RDT's finetune dataloader (`baseline/rdt/data/hdf5_vla_dataset.py`, adapted for
genesis-hr-bench) expects one HDF5 file per episode:

    observations/qpos                    (T, 8)  float32  [7 Franka joints, gripper]
    observations/images/cam_high         (T,)    vlen uint8  JPEG bytes (head cam)
    observations/images/cam_right_wrist  (T,)    vlen uint8  JPEG bytes (right wrist)
    action                               (T, 8)  float32  [7 joint deltas, gripper cmd]

plus a sibling ``instruction.json``:  {"instruction": "<task language>"}.

Source: the raw collection h5 at ``vla_data/<task>/<renderer>/seed_*/steps.h5``,
which already stores ``qpos`` (T,7), ``qpos_delta_action`` (T,7),
``gripper_meas`` / ``gripper_action`` (T,), and JPEG-encoded
``images/{head_camera,right_wrist}`` (T,) object arrays. The per-episode
instruction + success flag come from the sibling ``meta.json``.

Joint-space (not EE-space) is used deliberately: the raw h5 carries joint qpos
directly, and RDT's pretrained-eval server is joint-space — so a joint-space
finetune stays consistent with eval.

Output layout (one dir per episode so each carries its own instruction.json):

    <out-dir>/<task>/<task>_seed_<n>/episode.hdf5
    <out-dir>/<task>/<task>_seed_<n>/instruction.json

Run from the MAWM venv (needs h5py + numpy):

    .../yz/env/MAWM/bin/python scripts/convert_genesis_hr_bench_to_rdt_hdf5.py \
        --vla-data vla_data --renderer rasterizer \
        --out-dir runs/rdt/data/genesis_hr_bench

Then point RDT at it:  export RDT_HDF5_DIR=runs/rdt/data/genesis_hr_bench
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np


def convert_episode(steps_h5: Path, out_h5: Path) -> int:
    """Read one raw steps.h5, write one RDT-schema episode HDF5. Returns #steps."""
    with h5py.File(steps_h5, "r") as f:
        qpos = np.asarray(f["qpos"][:], dtype=np.float32)                    # (T, 7)
        qpos_delta = np.asarray(f["qpos_delta_action"][:], dtype=np.float32)  # (T, 7)
        grip_meas = np.asarray(f["gripper_meas"][:], dtype=np.float32)        # (T,)
        grip_act = np.asarray(f["gripper_action"][:], dtype=np.float32)       # (T,)
        head = [np.asarray(x, dtype=np.uint8) for x in f["images/head_camera"][:]]
        wrist = None
        if "images/right_wrist" in f:
            wrist = [np.asarray(x, dtype=np.uint8) for x in f["images/right_wrist"][:]]

    # Trim every stream to the common length (defensive — recorder writes equal
    # lengths, but a truncated episode must not desync state vs. images).
    t = min(len(qpos), len(qpos_delta), len(grip_meas), len(grip_act), len(head))
    if wrist is not None:
        t = min(t, len(wrist))

    state = np.concatenate([qpos[:t], grip_meas[:t, None]], axis=1)          # (T, 8)
    action = np.concatenate([qpos_delta[:t], grip_act[:t, None]], axis=1)    # (T, 8)

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    vlen = h5py.vlen_dtype(np.uint8)
    with h5py.File(out_h5, "w") as g:
        obs = g.create_group("observations")
        obs.create_dataset("qpos", data=state)
        imgs = obs.create_group("images")
        d_high = imgs.create_dataset("cam_high", (t,), dtype=vlen)
        for i in range(t):
            d_high[i] = head[i]
        if wrist is not None:
            d_wrist = imgs.create_dataset("cam_right_wrist", (t,), dtype=vlen)
            for i in range(t):
                d_wrist[i] = wrist[i]
        g.create_dataset("action", data=action)
    return t


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vla-data", default="vla_data",
                    help="Root holding <task>/<renderer>/seed_*/ (default: vla_data).")
    ap.add_argument("--renderer", default="rasterizer",
                    help="Renderer subdir to read (default: rasterizer).")
    ap.add_argument("--out-dir", default="runs/rdt/data/genesis_hr_bench",
                    help="Output dir for the RDT HDF5 tree.")
    ap.add_argument("--task", action="append", default=None,
                    help="Restrict to these task names (repeatable; default: all).")
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="Cap episodes converted (useful for smoke runs).")
    ap.add_argument("--include-failures", action="store_true",
                    help="Also convert episodes whose meta.success is false.")
    ap.add_argument("--min-steps", type=int, default=128,
                    help="Skip episodes shorter than this (RDT drops <128 anyway).")
    args = ap.parse_args()

    src = Path(args.vla_data)
    out = Path(args.out_dir)
    if not src.is_dir():
        raise SystemExit(f"--vla-data not found: {src}")

    tasks = sorted(p.name for p in src.iterdir() if p.is_dir())
    if args.task:
        want = set(args.task)
        present = [t for t in tasks if t in want]
        for missing in sorted(want - set(present)):
            print(f"[convert] WARNING: requested task not present: {missing}", file=sys.stderr)
        tasks = present
    if not tasks:
        raise SystemExit("no tasks to convert")

    n_ep = n_skip = n_steps = 0
    for task in tasks:
        rdir = src / task / args.renderer
        if not rdir.is_dir():
            continue
        for seed_dir in sorted(rdir.glob("seed_*")):
            steps_h5 = seed_dir / "steps.h5"
            meta_json = seed_dir / "meta.json"
            if not steps_h5.is_file() or not meta_json.is_file():
                continue
            try:
                meta = json.loads(meta_json.read_text())
            except json.JSONDecodeError as e:
                print(f"[convert] skip {seed_dir} (bad meta.json: {e})", file=sys.stderr)
                n_skip += 1
                continue
            if not args.include_failures and not bool(meta.get("success", False)):
                n_skip += 1
                continue
            if int(meta.get("n_steps", 0)) < args.min_steps:
                n_skip += 1
                continue

            ep_dir = out / task / f"{task}_{seed_dir.name}"
            try:
                t = convert_episode(steps_h5, ep_dir / "episode.hdf5")
            except Exception as e:  # noqa: BLE001
                print(f"[convert] FAILED {steps_h5}: {e}", file=sys.stderr)
                n_skip += 1
                continue
            instr = str(meta.get("instruction") or "").strip()
            (ep_dir / "instruction.json").write_text(
                json.dumps({"instruction": instr}, indent=2))
            n_ep += 1
            n_steps += t
            if n_ep % 25 == 0:
                print(f"[convert] {n_ep} episodes ({n_steps} steps)", flush=True)
            if args.max_episodes is not None and n_ep >= args.max_episodes:
                print(f"[convert] hit --max-episodes={args.max_episodes}", flush=True)
                _summary(out, n_ep, n_skip, n_steps)
                return
    _summary(out, n_ep, n_skip, n_steps)


def _summary(out: Path, n_ep: int, n_skip: int, n_steps: int) -> None:
    print(f"[convert] done: {n_ep} episodes, {n_skip} skipped, {n_steps} steps", flush=True)
    print(f"[convert] output: {out}", flush=True)
    if n_ep:
        print(f"[convert] point RDT at it via:  export RDT_HDF5_DIR={out}", flush=True)


if __name__ == "__main__":
    main()
