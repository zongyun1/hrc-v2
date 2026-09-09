"""Train an STS-GCN avatar-motion forecaster with the ManiCast recipe.

ManiCast (Kedia et al., CoRL 2023) fine-tunes a forecaster to be accurate
where it matters for the robot: (a) upsample "transition" windows where the
human moves toward the robot workspace, (b) upweight wrist joints in the
loss. We train on avatar skeleton trajectories logged by
scripts/hri/collect_avatar_skeleton.py.

Windows: history T_in frames -> future T_out frames at 20 Hz policy cadence.
Positions are expressed relative to the last-history-frame Hips position
(translation invariance); the runtime detector adds the offset back.

Transition labeling (observation-derived, no task GT): a frame is a
"transition" if either avatar wrist comes within --transition-dist of the
robot EE position (logged proprio) at any point within the next T_out frames.

Usage:
  python scripts/hri/train_manicast.py \
      --data-root data/hri_manicast/skeleton_data \
      --output runs/hri_manicast/forecaster/stsgcn_v1.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs.manicast.stsgcn import STSGCN

WRIST_BONES = ("LeftHand", "RightHand")
ROOT_BONE = "Hips"


def load_episodes(data_root: Path, max_per_task: int | None = None):
    """Yield dicts {joints (T,J,3), ee (T,7), bones, task, path} per episode."""
    episodes = []
    for task_dir in sorted(data_root.iterdir()):
        if not task_dir.is_dir():
            continue
        paths = sorted(task_dir.glob("seed_*/steps.h5"))
        if max_per_task:
            paths = paths[:max_per_task]
        for p in paths:
            try:
                with h5py.File(p, "r") as f:
                    joints = f["avatar/joints"][:]
                    present = f["avatar/present"][:].astype(bool)
                    ee = f["ee"][:]
                    bones = json.loads(f.attrs["bones"])
            except Exception as e:
                print(f"skip {p}: {e}", flush=True)
                continue
            if joints.ndim != 3 or len(joints) < 10 or not present.any():
                continue
            # Keep the contiguous present prefix/suffix; NaN rows break windows.
            valid = present & np.isfinite(joints).all(axis=(1, 2))
            episodes.append({
                "joints": joints, "valid": valid, "ee": ee,
                "bones": bones, "task": task_dir.name, "path": str(p),
            })
    return episodes


def build_windows(episodes, t_in, t_out, transition_dist):
    """Return arrays X (N,T_in,J,3), Y (N,T_out,J,3), trans (N,) bool."""
    xs, ys, trans = [], [], []
    for ep in episodes:
        joints, valid, ee = ep["joints"], ep["valid"], ep["ee"]
        bones = ep["bones"]
        wrist_idx = [bones.index(b) for b in WRIST_BONES if b in bones]
        T = len(joints)
        win = t_in + t_out
        for s in range(0, T - win + 1):
            sl = slice(s, s + win)
            if not valid[sl].all():
                continue
            seq = joints[sl]
            x, y = seq[:t_in], seq[t_in:]
            # transition: wrist near robot EE within the future horizon
            fut_wrists = y[:, wrist_idx, :]                  # (T_out, W, 3)
            ee_pos = ee[s + t_in:s + win, None, :3]          # (T_out, 1, 3)
            d = np.linalg.norm(fut_wrists - ee_pos, axis=-1)
            trans.append(bool((d < transition_dist).any()))
            xs.append(x)
            ys.append(y)
    return (np.asarray(xs, np.float32), np.asarray(ys, np.float32),
            np.asarray(trans, bool))


class WindowDataset(Dataset):
    def __init__(self, X, Y, idx, root_idx, input_noise=0.0):
        self.X, self.Y, self.idx, self.root_idx = X, Y, idx, root_idx
        self.input_noise = float(input_noise)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = self.X[j].copy()
        y = self.Y[j].copy()
        if self.input_noise > 0:
            # eval-time histories come from a learned pose estimator, not GT
            x += np.random.normal(0.0, self.input_noise,
                                  size=x.shape).astype(np.float32)
        center = x[-1, self.root_idx, :].copy()
        x -= center
        y -= center
        # (T,J,3) -> model input (3,T,J)
        return (torch.from_numpy(x.transpose(2, 0, 1)),
                torch.from_numpy(y),
                torch.from_numpy(center))


def make_sampler_indices(trans, n_samples, upsample_frac, rng):
    """ManiCast transition upsampling: ~upsample_frac of draws from
    transition windows, rest uniform."""
    t_pool = np.flatnonzero(trans)
    all_pool = np.arange(len(trans))
    if len(t_pool) == 0:
        return rng.choice(all_pool, size=n_samples)
    n_t = int(n_samples * upsample_frac)
    picks = np.concatenate([
        rng.choice(t_pool, size=n_t),
        rng.choice(all_pool, size=n_samples - n_t),
    ])
    rng.shuffle(picks)
    return picks


def mpjpe(pred, target, joint_w=None):
    # pred/target: (N, T, J, 3)
    err = torch.norm(pred - target, dim=-1)  # (N,T,J)
    if joint_w is not None:
        err = err * joint_w
    return err.mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--t-in", type=int, default=8)
    p.add_argument("--t-out", type=int, default=20)
    p.add_argument("--transition-dist", type=float, default=0.45)
    p.add_argument("--upsample-frac", type=float, default=0.5)
    p.add_argument("--wrist-weight", type=float, default=5.0)
    p.add_argument("--input-noise", type=float, default=0.02,
                   help="train-time Gaussian noise (m) on input joints, "
                        "matching pose-estimator error at eval")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--max-episodes-per-task", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    episodes = load_episodes(Path(args.data_root), args.max_episodes_per_task)
    if not episodes:
        raise SystemExit("no episodes found")
    bones = episodes[0]["bones"]
    root_idx = bones.index(ROOT_BONE)
    wrist_idx = [bones.index(b) for b in WRIST_BONES if b in bones]
    print(f"episodes={len(episodes)} bones={bones}")

    # Split by episode so val windows are from unseen rollouts.
    rng.shuffle(episodes)
    n_val = max(1, int(len(episodes) * args.val_frac))
    val_eps, train_eps = episodes[:n_val], episodes[n_val:]
    Xtr, Ytr, Ttr = build_windows(train_eps, args.t_in, args.t_out,
                                  args.transition_dist)
    Xva, Yva, Tva = build_windows(val_eps, args.t_in, args.t_out,
                                  args.transition_dist)
    print(f"train windows={len(Xtr)} ({Ttr.mean():.1%} transition) "
          f"val windows={len(Xva)} ({Tva.mean():.1%} transition)")

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    model = STSGCN(input_time_frame=args.t_in, output_time_frame=args.t_out,
                   joints_to_consider=len(bones)).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"model params: {n_params/1e3:.1f}k device={device}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    joint_w = torch.ones(len(bones), device=device)
    for i in wrist_idx:
        joint_w[i] = args.wrist_weight
    joint_w = joint_w / joint_w.mean()

    val_ds = WindowDataset(Xva, Yva, np.arange(len(Xva)), root_idx)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        idx = make_sampler_indices(
            Ttr, args.steps_per_epoch * args.batch_size,
            args.upsample_frac, rng)
        loader = DataLoader(WindowDataset(Xtr, Ytr, idx, root_idx,
                                          input_noise=args.input_noise),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=2)
        total, nb = 0.0, 0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).permute(0, 1, 3, 2)  # (N,T_out,J,3)
            loss = mpjpe(pred, y, joint_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss)
            nb += 1

        model.eval()
        with torch.no_grad():
            v_all, v_wrist, v_trans, nb_v = 0.0, 0.0, [], 0
            for bi, (x, y, _) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                pred = model(x).permute(0, 1, 3, 2)
                v_all += float(mpjpe(pred, y))
                werr = torch.norm(
                    pred[:, :, wrist_idx] - y[:, :, wrist_idx], dim=-1)
                v_wrist += float(werr.mean())
                nb_v += 1
            v_all /= max(1, nb_v)
            v_wrist /= max(1, nb_v)
        print(f"epoch={epoch} train={total/max(1,nb):.4f} "
              f"val_mpjpe={v_all:.4f} val_wrist={v_wrist:.4f}", flush=True)
        if v_wrist <= best_val:
            best_val = v_wrist
            torch.save({
                "model": model.state_dict(),
                "bones": bones,
                "root_bone": ROOT_BONE,
                "wrist_bones": list(WRIST_BONES),
                "t_in": args.t_in,
                "t_out": args.t_out,
                "val_mpjpe": v_all,
                "val_wrist_mpjpe": v_wrist,
                "transition_dist": args.transition_dist,
                "wrist_weight": args.wrist_weight,
                "n_train_windows": int(len(Xtr)),
            }, out)
    print(f"saved {out} (best val wrist MPJPE {best_val:.4f} m)")


if __name__ == "__main__":
    main()
