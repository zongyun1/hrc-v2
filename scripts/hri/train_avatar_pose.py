"""Train AvatarPoseNet (head-camera RGB -> avatar joints, camera frame).

Data: skeleton episodes from scripts/hri/collect_avatar_skeleton.py.
Labels: GT world joints -> camera frame via the episode's recorded
cam-to-world extrinsic (training-time privilege only).

Usage:
  python scripts/hri/train_avatar_pose.py \
      --data-root data/hri_manicast/skeleton_data \
      --output runs/hri_manicast/pose/avatar_pose_v1.pt
"""

from __future__ import annotations

import argparse
import io
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

from envs.manicast.avatar_pose_net import (
    AvatarPoseNet, joint_visibility, joints_world_to_cam, preprocess_rgb,
)

# Joints whose in-frame visibility defines "the avatar is present in view".
PRESENCE_BONES = ("Head", "LeftHand", "RightHand")
# Loss weight for joints outside the frustum on present frames: the net must
# still predict them (the forecaster consumes a full skeleton), but they are
# inferred from the visible body parts rather than directly observed.
HIDDEN_JOINT_W = 0.3


def _decode_jpeg(buf) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(bytes(buf))).convert("RGB"))


def gather_samples(data_root: Path, camera: str, frame_stride: int,
                   max_eps_per_task: int | None):
    """Returns list of (h5_path, frame_idx, extrinsic, valid) and bones.

    Stores references, not decoded images (decode lazily in the Dataset).
    Split must be done by episode — return per-episode groups.
    """
    groups = []
    bones = None
    for task_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        paths = sorted(task_dir.glob("seed_*/steps.h5"))
        if max_eps_per_task:
            paths = paths[:max_eps_per_task]
        for p in paths:
            try:
                with h5py.File(p, "r") as f:
                    if camera not in f.get("images", {}):
                        continue
                    n = len(f[f"images/{camera}"])
                    joints = f["avatar/joints"][:]
                    present = f["avatar/present"][:].astype(bool)
                    cam_meta = json.loads(f.attrs["cam_meta"])
                    ep_bones = json.loads(f.attrs["bones"])
            except Exception as e:
                print(f"skip {p}: {e}", flush=True)
                continue
            if camera not in cam_meta:
                continue
            if bones is None:
                bones = ep_bones
            ext = np.asarray(cam_meta[camera]["extrinsic"], np.float64)
            intr = np.asarray(cam_meta[camera]["intrinsic"], np.float64)
            res_wh = cam_meta[camera].get("resolution", [640, 480])
            valid = present & np.isfinite(joints).all(axis=(1, 2))
            jc = joints_world_to_cam(joints, ext)            # (T,J,3)
            vis = joint_visibility(jc, intr, res_wh)         # (T,J) bool
            pres_idx = [ep_bones.index(b) for b in PRESENCE_BONES
                        if b in ep_bones]
            frame_present = vis[:, pres_idx].any(axis=1) & valid
            idxs = [i for i in range(0, n, frame_stride)]
            groups.append({
                "path": str(p), "ext": ext, "idxs": idxs,
                "joints_cam": jc.astype(np.float32),
                "vis": vis,
                "present": frame_present,
                "valid": valid,
            })
    return groups, bones


class PoseDataset(Dataset):
    def __init__(self, groups, camera, size_hw, augment=False):
        self.camera = camera
        self.size_hw = size_hw
        self.augment = augment
        self.index = []  # (group_i, frame_idx)
        self.groups = groups
        for gi, g in enumerate(groups):
            for i in g["idxs"]:
                if i < len(g["valid"]):
                    self.index.append((gi, i))
        self._h5 = {}

    def __len__(self):
        return len(self.index)

    def _file(self, path):
        f = self._h5.get(path)
        if f is None:
            f = h5py.File(path, "r")
            self._h5[path] = f
        return f

    def __getitem__(self, k):
        gi, i = self.index[k]
        g = self.groups[gi]
        f = self._file(g["path"])
        rgb = _decode_jpeg(f[f"images/{self.camera}"][i])
        if self.augment:
            arr = rgb.astype(np.float32)
            arr *= np.random.uniform(0.85, 1.15)
            arr += np.random.uniform(-12, 12, size=(1, 1, 3))
            rgb = np.clip(arr, 0, 255).astype(np.uint8)
        x = preprocess_rgb(rgb, self.size_hw)
        present = bool(g["present"][i])
        jc = g["joints_cam"][i] if present else np.zeros_like(g["joints_cam"][i])
        # per-joint loss weight: 1 for visible, HIDDEN_JOINT_W for
        # out-of-frame joints on present frames, 0 on absent frames
        if present:
            w = np.where(g["vis"][i], 1.0, HIDDEN_JOINT_W).astype(np.float32)
        else:
            w = np.zeros(g["joints_cam"].shape[1], dtype=np.float32)
        return (x, torch.from_numpy(np.ascontiguousarray(jc)),
                torch.from_numpy(w), torch.tensor(float(present)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--camera", default="head_camera")
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--size", type=int, nargs=2, default=(192, 256),
                   metavar=("H", "W"))
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--max-eps-per-task", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    groups, bones = gather_samples(Path(args.data_root), args.camera,
                                   args.frame_stride, args.max_eps_per_task)
    if not groups:
        raise SystemExit("no samples found")
    random.shuffle(groups)
    n_val = max(1, int(len(groups) * args.val_frac))
    val_g, train_g = groups[:n_val], groups[n_val:]
    train_ds = PoseDataset(train_g, args.camera, tuple(args.size), augment=True)
    val_ds = PoseDataset(val_g, args.camera, tuple(args.size))
    print(f"episodes train={len(train_g)} val={len(val_g)} "
          f"frames train={len(train_ds)} val={len(val_ds)} bones={bones}")

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    model = AvatarPoseNet(len(bones), width=args.width).to(device)
    print(f"params {sum(q.numel() for q in model.parameters())/1e6:.2f}M "
          f"device={device}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=args.workers)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    wrist_idx = [bones.index(b) for b in ("LeftHand", "RightHand")
                 if b in bones]
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for x, y, w, v in train_loader:
            x, y = x.to(device), y.to(device)
            w, v = w.to(device), v.to(device)
            pred, logit = model(x)
            per_joint = F.smooth_l1_loss(pred, y, reduction="none",
                                         beta=0.05).mean(dim=-1)  # (N,J)
            l_pose = (per_joint * w).sum() / w.sum().clamp(min=1)
            l_pres = F.binary_cross_entropy_with_logits(logit, v)
            loss = l_pose + 0.1 * l_pres
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        model.eval()
        with torch.no_grad():
            errs, werrs, pres_acc = [], [], []
            for x, y, w, v in val_loader:
                x, y = x.to(device), y.to(device)
                w = w.to(device)
                pred, logit = model(x)
                pres_acc.append(
                    (((torch.sigmoid(logit) > 0.5).float().cpu() == v)
                     .float().mean().item()))
                m = v.to(device) > 0.5
                if m.any():
                    e = torch.norm(pred[m] - y[m], dim=-1)      # (n, J)
                    vis = (w[m] >= 0.99).float()
                    errs.append(((e * vis).sum()
                                 / vis.sum().clamp(min=1)).item())
                    wvis = vis[:, wrist_idx]
                    ew = e[:, wrist_idx]
                    if wvis.sum() > 0:
                        werrs.append(((ew * wvis).sum() / wvis.sum()).item())
            v_all = float(np.mean(errs)) if errs else float("nan")
            v_wrist = float(np.mean(werrs)) if werrs else float("nan")
            v_pres = float(np.mean(pres_acc)) if pres_acc else float("nan")
        print(f"epoch={epoch} train={tot/max(1,nb):.4f} "
              f"val_joint_err={v_all:.4f}m val_wrist_err={v_wrist:.4f}m "
              f"val_presence_acc={v_pres:.3f}", flush=True)
        if v_wrist <= best:
            best = v_wrist
            torch.save({
                "model": model.state_dict(),
                "bones": bones,
                "size_hw": list(args.size),
                "width": args.width,
                "camera": args.camera,
                "val_joint_err_m": v_all,
                "val_wrist_err_m": v_wrist,
            }, out)
    print(f"saved {out} (best val wrist err {best:.4f} m)")


if __name__ == "__main__":
    main()
