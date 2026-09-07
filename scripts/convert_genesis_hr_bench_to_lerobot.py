"""Convert the genesis_hr_bench RLDS dataset to a LeRobot dataset for openpi.

The openpi finetune pipeline reads LeRobot datasets (the only RLDS loader
upstream ships is hard-wired to DROID). genesis_hr_bench is built (by
`scripts/build_genesis_hr_bench_rlds.py`) as TFDS/RLDS at
`openvla/data/genesis_hr_bench/1.0.0/`. This script reads that TFDS source and
writes a LeRobot dataset whose schema matches openpi's libero expected layout
(image, wrist_image, state, actions, prompt).

The output dataset is written under $HF_LEROBOT_HOME (default
``~/.cache/huggingface/lerobot``) at ``--repo-id``. openpi loads it via that
repo id (the configs in `openpi.training.config` use
``repo_id="genesis-hr-bench/genesis_hr_bench"``).

Run from the pi0 venv (it ships lerobot + torch). tensorflow + tensorflow_datasets
must be installed in the same env (``uv pip install tensorflow tensorflow_datasets``).

    /scratch4/.../yz/env/pi0/bin/python \
        scripts/convert_genesis_hr_bench_to_lerobot.py \
        --tfds-data-dir openvla/data \
        --repo-id genesis-hr-bench/genesis_hr_bench

Pass ``--max-episodes N`` for a small smoke conversion. Pass ``--overwrite`` to
delete an existing LeRobot dataset at the target path.

Schema (per frame, EE mode):
    image         : (256, 256, 3) uint8     head_camera RGB
    wrist_image   : (256, 256, 3) uint8     right_wrist RGB
    state         : (7,) float32            [x, y, z, roll, pitch, yaw, gripper]
    actions       : (7,) float32            [dx, dy, dz, droll, dpitch, dyaw, gripper]
    task          : str                     copied from RLDS step.language_instruction

After conversion, run:
    uv run scripts/compute_norm_stats.py --config-name pi05_genesis_hr_bench
    uv run scripts/train.py pi05_genesis_hr_bench --exp-name=<name>
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tfds-data-dir",
        default="openvla/data",
        help="Parent dir holding the tfds `genesis_hr_bench/<version>` tree.",
    )
    parser.add_argument(
        "--tfds-name",
        default="genesis_hr_bench",
        help="TFDS dataset name (matches the directory under --tfds-data-dir).",
    )
    parser.add_argument(
        "--repo-id",
        default="genesis-hr-bench/genesis_hr_bench",
        help="LeRobot repo id (also the path under $HF_LEROBOT_HOME).",
    )
    parser.add_argument(
        "--robot-type",
        default="franka",
        help="Free-form robot type label stored in the LeRobot dataset metadata.",
    )
    parser.add_argument("--fps", type=int, default=20, help="Control frequency.")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Cap the number of episodes copied (default: all). Useful for smoke runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing LeRobot dataset at the target path before writing.",
    )
    parser.add_argument(
        "--lerobot-naming",
        action="store_true",
        help="Use LeRobot-native feature keys (observation.image, "
             "observation.wrist_image, observation.state, action) instead of "
             "the openpi/LIBERO convention (image, wrist_image, state, "
             "actions). Required for native LeRobot policies (ACT, "
             "DiffusionPolicy, VQBeT) that key off the 'action' / "
             "'observation.*' prefix.",
    )
    parser.add_argument(
        "--single-image",
        action="store_true",
        help="Drop the wrist camera from the output dataset. VQ-BeT's "
             "validate_features() requires exactly one image input, and "
             "lerobot's factory.make_policy rebuilds cfg.input_features from "
             "dataset metadata so a CLI override can't dodge the constraint. "
             "Use this to produce a sibling dataset (suggested repo_id: "
             "genesis-hr-bench/genesis_hr_bench_lerobot_singlecam).",
    )
    parser.add_argument(
        "--control-mode",
        choices=["ee", "qpos", "qpos_abs", "qpos_target_abs"],
        default="ee",
        help="Expected state/action schema from the RLDS source. qpos uses "
             "8D [7 qpos, gripper_open] state and delta actions; qpos_abs "
             "uses the same state and 8D [7 next measured qpos, gripper_open] "
             "actions; qpos_target_abs uses 8D commanded target qpos actions.",
    )
    args = parser.parse_args()
    if args.lerobot_naming:
        img_key, wrist_key, state_key, action_key = (
            "observation.image",
            "observation.wrist_image",
            "observation.state",
            "action",
        )
    else:
        img_key, wrist_key, state_key, action_key = (
            "image",
            "wrist_image",
            "state",
            "actions",
        )

    output_path = Path(HF_LEROBOT_HOME) / args.repo_id
    if output_path.exists():
        if not args.overwrite:
            raise SystemExit(
                f"refusing to overwrite existing dataset at {output_path}; "
                "pass --overwrite to delete it first."
            )
        shutil.rmtree(output_path)

    print(f"[convert] reading tfds {args.tfds_name} from {args.tfds_data_dir}", flush=True)
    raw = tfds.load(args.tfds_name, data_dir=args.tfds_data_dir, split="train")

    print(f"[convert] writing LeRobot dataset to {output_path}"
          + (" (single-image)" if args.single_image else ""), flush=True)
    lowdim_shape = (8,) if args.control_mode in {"qpos", "qpos_abs", "qpos_target_abs"} else (7,)
    features = {
        img_key: {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        state_key: {
            "dtype": "float32",
            "shape": lowdim_shape,
            "names": ["state"],
        },
        action_key: {
            "dtype": "float32",
            "shape": lowdim_shape,
            "names": ["actions"],
        },
    }
    if not args.single_image:
        features[wrist_key] = {
            "dtype": "image",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        fps=args.fps,
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    n_eps = 0
    n_steps = 0
    for episode in raw:
        for step in episode["steps"].as_numpy_iterator():
            obs = step["observation"]
            frame = {
                img_key: np.asarray(obs["image"], dtype=np.uint8),
                state_key: np.asarray(obs["state"], dtype=np.float32),
                action_key: np.asarray(step["action"], dtype=np.float32),
                "task": step["language_instruction"].decode(),
            }
            if not args.single_image:
                frame[wrist_key] = np.asarray(obs["wrist_image"], dtype=np.uint8)
            dataset.add_frame(frame)
            n_steps += 1
        dataset.save_episode()
        n_eps += 1
        if n_eps % 10 == 0:
            print(f"[convert] episode {n_eps} written ({n_steps} frames so far)", flush=True)
        if args.max_episodes is not None and n_eps >= args.max_episodes:
            break

    print(f"[convert] done. {n_eps} episodes, {n_steps} frames -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
