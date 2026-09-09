"""RDT-1B finetune dataset — ADAPTED FOR genesis-hr-bench (Franka single-arm).

RDT ships this file as a `[Modify]`-it-yourself template (the upstream version
targets the dual-arm `agilex` embodiment). This copy is rewritten for the
genesis-hr-bench benchmark: a single 7-DOF Franka arm + 1 gripper.

It consumes the per-episode HDF5 produced by
`scripts/convert_genesis_hr_bench_to_rdt_hdf5.py`:

    observations/qpos                    (T, 8)  [7 Franka joints, gripper]
    observations/images/cam_high         (T,) vlen uint8  JPEG bytes
    observations/images/cam_right_wrist  (T,) vlen uint8  JPEG bytes (optional)
    action                               (T, 8)  [7 joint deltas, gripper cmd]

plus a sibling `instruction.json`: {"instruction": "..."}.

The dataset root is read from $RDT_HDF5_DIR (set by scripts/finetune_rdt.sh).
The upstream agilex version is recoverable from this submodule's git history.
"""
import os
import fnmatch
import json

import h5py
import yaml
import cv2
import numpy as np

from configs.state_vec import STATE_VEC_IDX_MAPPING

# Genesis-hr-bench is too-short-episode-safe (episodes are ~400 steps), but keep
# RDT's drop threshold so a stray truncated episode can't poison a batch.
MIN_EPISODE_STEPS = 48


class HDF5VLADataset:
    """Sample episodes from the genesis-hr-bench RDT HDF5 dataset."""

    # 7 Franka arm joints + 1 gripper -> slots in RDT's 128-D unified vector.
    UNI_STATE_INDICES = [
        STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(7)
    ] + [STATE_VEC_IDX_MAPPING["right_gripper_open"]]

    def __init__(self) -> None:
        hdf5_dir = os.environ.get("RDT_HDF5_DIR", "data/datasets/genesis_hr_bench")
        self.DATASET_NAME = "genesis_hr_bench"

        self.file_paths = []
        # followlinks=True so a symlinked hdf5_dir (or any symlinked subdir,
        # e.g. when datasets live on a separate lustre filesystem) is walked.
        for root, _, files in os.walk(hdf5_dir, followlinks=True):
            for filename in fnmatch.filter(files, "*.hdf5"):
                self.file_paths.append(os.path.join(root, filename))
        self.file_paths.sort()
        if not self.file_paths:
            raise FileNotFoundError(
                f"no *.hdf5 episodes under RDT_HDF5_DIR={hdf5_dir!r} — build them "
                "with scripts/convert_genesis_hr_bench_to_rdt_hdf5.py"
            )

        with open("configs/base.yaml", "r") as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config["common"]["action_chunk_size"]
        self.IMG_HISORY_SIZE = config["common"]["img_history_size"]
        self.STATE_DIM = config["common"]["state_dim"]

        # Per-episode length -> sampling weights (longer episodes sampled more).
        episode_lens = []
        for file_path in self.file_paths:
            valid, res = self.parse_hdf5_file_state_only(file_path)
            episode_lens.append(res["state"].shape[0] if valid else 0)
        total = float(np.sum(episode_lens))
        if total <= 0:
            raise RuntimeError(
                f"every episode under {hdf5_dir!r} is shorter than "
                f"{MIN_EPISODE_STEPS} steps — nothing to train on"
            )
        self.episode_sample_weights = np.array(episode_lens) / total

    def __len__(self):
        return len(self.file_paths)

    def get_dataset_name(self):
        return self.DATASET_NAME

    def get_item(self, index: int = None, state_only=False):
        """Get a training sample at a random timestep (see upstream docstring)."""
        while True:
            if index is None:
                file_path = np.random.choice(self.file_paths, p=self.episode_sample_weights)
            else:
                file_path = self.file_paths[index]
            valid, sample = (
                self.parse_hdf5_file_state_only(file_path) if state_only
                else self.parse_hdf5_file(file_path)
            )
            if valid:
                return sample
            index = np.random.randint(0, len(self.file_paths))

    def _fill_in_state(self, values):
        """Scatter a genesis [7 joints, gripper] vector into the 128-D space."""
        uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
        uni_vec[..., self.UNI_STATE_INDICES] = values
        return uni_vec

    @staticmethod
    def _load_instruction(file_path):
        instr_path = os.path.join(os.path.dirname(file_path), "instruction.json")
        with open(instr_path, "r") as f:
            return json.load(f)["instruction"]

    @staticmethod
    def _first_moving_idx(qpos):
        """Index of the first step whose qpos has moved off the start pose."""
        qpos_delta = np.abs(qpos - qpos[0:1])
        indices = np.where(np.any(qpos_delta > 1e-2, axis=1))[0]
        return int(indices[0]) if len(indices) > 0 else 1

    def parse_hdf5_file(self, file_path):
        """Parse one episode HDF5 into a training sample at a random timestep."""
        with h5py.File(file_path, "r") as f:
            qpos = f["observations"]["qpos"][:]
            num_steps = qpos.shape[0]
            if num_steps < MIN_EPISODE_STEPS:
                return False, None

            first_idx = self._first_moving_idx(qpos)
            step_id = np.random.randint(first_idx - 1, num_steps)

            meta = {
                "dataset_name": self.DATASET_NAME,
                "#steps": num_steps,
                "step_id": step_id,
                "instruction": self._load_instruction(file_path),
            }

            # State at t; action chunk [t : t+CHUNK_SIZE], last-action padded.
            state = qpos[step_id:step_id + 1]
            state_std = np.std(qpos, axis=0)
            state_mean = np.mean(qpos, axis=0)
            state_norm = np.sqrt(np.mean(qpos ** 2, axis=0))

            actions = f["action"][step_id:step_id + self.CHUNK_SIZE]
            if actions.shape[0] < self.CHUNK_SIZE:
                actions = np.concatenate([
                    actions,
                    np.tile(actions[-1:], (self.CHUNK_SIZE - actions.shape[0], 1)),
                ], axis=0)

            state = self._fill_in_state(state)
            state_indicator = self._fill_in_state(np.ones_like(state_std))
            state_std = self._fill_in_state(state_std)
            state_mean = self._fill_in_state(state_mean)
            state_norm = self._fill_in_state(state_norm)
            actions = self._fill_in_state(actions)

            def parse_img(key):
                grp = f["observations"]["images"]
                if key not in grp:
                    return np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0))
                imgs = []
                for i in range(max(step_id - self.IMG_HISORY_SIZE + 1, 0), step_id + 1):
                    raw = np.asarray(grp[key][i], dtype=np.uint8)
                    imgs.append(cv2.imdecode(raw, cv2.IMREAD_COLOR))
                imgs = np.stack(imgs)
                if imgs.shape[0] < self.IMG_HISORY_SIZE:
                    imgs = np.concatenate([
                        np.tile(imgs[:1], (self.IMG_HISORY_SIZE - imgs.shape[0], 1, 1, 1)),
                        imgs,
                    ], axis=0)
                return imgs

            cam_high = parse_img("cam_high")
            valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISORY_SIZE)
            cam_high_mask = np.array(
                [False] * (self.IMG_HISORY_SIZE - valid_len) + [True] * valid_len)
            cam_right_wrist = parse_img("cam_right_wrist")
            if cam_right_wrist.shape[1] > 0:
                cam_right_wrist_mask = cam_high_mask.copy()
            else:
                cam_right_wrist_mask = np.zeros(self.IMG_HISORY_SIZE, dtype=bool)

            return True, {
                "meta": meta,
                "state": state,
                "state_std": state_std,
                "state_mean": state_mean,
                "state_norm": state_norm,
                "actions": actions,
                "state_indicator": state_indicator,
                "cam_high": cam_high,
                "cam_high_mask": cam_high_mask,
                # genesis-hr-bench has no left wrist camera.
                "cam_left_wrist": np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0)),
                "cam_left_wrist_mask": np.zeros(self.IMG_HISORY_SIZE, dtype=bool),
                "cam_right_wrist": cam_right_wrist,
                "cam_right_wrist_mask": cam_right_wrist_mask,
            }

    def parse_hdf5_file_state_only(self, file_path):
        """Parse one episode HDF5 into a full (state, action) trajectory."""
        with h5py.File(file_path, "r") as f:
            qpos = f["observations"]["qpos"][:]
            if qpos.shape[0] < MIN_EPISODE_STEPS:
                return False, None
            first_idx = self._first_moving_idx(qpos)
            state = self._fill_in_state(qpos[first_idx - 1:])
            action = self._fill_in_state(f["action"][first_idx - 1:])
            return True, {"state": state, "action": action}


if __name__ == "__main__":
    ds = HDF5VLADataset()
    print(f"{len(ds)} episodes")
    for i in range(min(len(ds), 3)):
        ds.get_item(i)
        print(f"episode {i} OK")
