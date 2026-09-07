#!/usr/bin/env python3
"""Train a small unified visual PACE progress detector from VLA HDF5 data."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs.pace.visual_progress import VisualProgressNet, preprocess_pair


DEFAULT_TASKS = (
    "stack_bowls_three",
    "stack_bowls_three_interrupt",
    "dump_bin",
    "dump_bin_interrupt",
    "place_dual_shoes",
    "place_dual_shoes_interrupt",
    "place_food_in_skillet",
    "place_food_in_skillet_interrupt",
    "place_bread_in_basket",
    "place_bread_in_basket_interrupt",
    "place_burger_fries",
    "place_burger_fries_interrupt",
    "put_object_cabinet",
    "put_object_cabinet_interrupt",
)


def _decode_jpeg(buf) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("failed to decode JPEG frame")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _motion_scores(frames: list[np.ndarray], crop_top_frac: float) -> np.ndarray:
    scores = np.zeros(len(frames), dtype=np.float32)
    for i in range(1, len(frames)):
        h = frames[i].shape[0]
        y1 = max(1, int(round(h * crop_top_frac)))
        a = cv2.cvtColor(frames[i - 1][:y1], cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(frames[i][:y1], cv2.COLOR_RGB2GRAY)
        scores[i] = float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)
    return scores


def _episode_samples(
    path: Path,
    positive_task: bool,
    stride: int,
    crop_top_frac: float,
    max_pairs: int,
):
    with h5py.File(path, "r") as f:
        ds = f["images/head_camera"]
        n = len(ds)
        if n < 2:
            return []
        indices = list(range(0, n, max(1, stride)))
        if indices[-1] != n - 1:
            indices.append(n - 1)
        frames = [_decode_jpeg(ds[i]) for i in indices]
    motion = _motion_scores(frames, crop_top_frac)
    nonzero = motion[motion > 0]
    threshold = float(np.quantile(nonzero, 0.70)) if len(nonzero) else 1.0
    threshold = max(threshold, 0.010)
    active_mask = (motion >= threshold) if positive_task else np.zeros_like(motion, dtype=bool)
    active_positions = np.flatnonzero(active_mask)
    progress = np.zeros(len(frames), dtype=np.float32)
    if len(active_positions):
        start = int(active_positions[0])
        end = max(start + 1, int(active_positions[-1]))
        for j in range(start, len(frames)):
            progress[j] = np.clip((j - start) / max(1, end - start), 0.0, 1.0)
    samples = []
    pair_indices = list(range(1, len(frames)))
    if max_pairs > 0 and len(pair_indices) > max_pairs:
        keep = np.linspace(0, len(pair_indices) - 1, max_pairs).round().astype(int)
        pair_indices = [pair_indices[i] for i in keep]
    for j in pair_indices:
        present = 1.0 if positive_task else 0.0
        # Presence is the reliable supervised signal available uniformly
        # across the VLA data. Runtime still gates triggering on observed
        # pair-frame motion, so static human appearance alone is not enough.
        active = 1.0 if positive_task else 0.0
        target_progress = float(j / max(1, len(frames) - 1)) if positive_task else 0.0
        samples.append((frames[j - 1], frames[j], present, active, target_progress))
    return samples


class PacePairDataset(Dataset):
    def __init__(self, samples, image_size: int, crop_top_frac: float):
        self.samples = samples
        self.image_size = image_size
        self.crop_top_frac = crop_top_frac

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prev, curr, present, active, progress = self.samples[idx]
        x = preprocess_pair(prev, curr, self.image_size, self.crop_top_frac)
        y = torch.tensor([present, active, progress], dtype=torch.float32)
        return x, y


def _collect_samples(args):
    root = Path(args.data_root)
    samples = []
    rng = random.Random(args.seed)
    for task in args.tasks:
        task_root = root / task / "rasterizer"
        if not task_root.exists():
            continue
        paths = sorted(task_root.glob("seed_*/steps.h5"))
        rng.shuffle(paths)
        paths = paths[: args.episodes_per_task]
        positive = task.endswith("_interrupt") or task.endswith("_assist") or task.endswith("_neutral")
        for path in paths:
            try:
                samples.extend(_episode_samples(
                    path,
                    positive,
                    args.frame_stride,
                    args.crop_top_frac,
                    args.max_pairs_per_episode,
                ))
            except Exception as exc:
                print(f"skip unreadable episode {path}: {exc}", flush=True)
    rng.shuffle(samples)
    return samples


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    samples = _collect_samples(args)
    if not samples:
        raise RuntimeError("no training samples found")
    split = int(len(samples) * 0.9)
    train_ds = PacePairDataset(samples[:split], args.image_size, args.crop_top_frac)
    val_ds = PacePairDataset(samples[split:], args.image_size, args.crop_top_frac)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = VisualProgressNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    pos_weight = torch.tensor([1.0, args.active_pos_weight], device=device)
    best_val = float("inf")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            cls_loss = F.binary_cross_entropy_with_logits(
                logits[:, :2],
                y[:, :2],
                pos_weight=pos_weight,
            )
            pred_progress = torch.sigmoid(logits[:, 2])
            weight = 0.25 + 0.75 * y[:, 0]
            prog_loss = torch.mean(weight * torch.abs(pred_progress - y[:, 2]))
            loss = cls_loss + prog_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item())
            n_batches += 1
        val_loss = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch} train_loss={total / max(1, n_batches):.4f} "
            f"val_loss={val_loss:.4f} samples={len(samples)}",
            flush=True,
        )
        if val_loss <= best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "image_size": args.image_size,
                    "crop_top_frac": args.crop_top_frac,
                    "tasks": list(args.tasks),
                    "samples": len(samples),
                    "val_loss": best_val,
                },
                out,
            )
    print(f"saved {out}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    n_batches = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        cls_loss = F.binary_cross_entropy_with_logits(logits[:, :2], y[:, :2])
        pred_progress = torch.sigmoid(logits[:, 2])
        prog_loss = torch.mean(torch.abs(pred_progress - y[:, 2]))
        total += float((cls_loss + prog_loss).item())
        n_batches += 1
    return total / max(1, n_batches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="vla_data_qpos_target_20260524")
    parser.add_argument("--output", default="runs/pace/visual_progress_unified.pt")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--episodes-per-task", type=int, default=60)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--max-pairs-per-episode", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--crop-top-frac", type=float, default=0.48)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--active-pos-weight", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
