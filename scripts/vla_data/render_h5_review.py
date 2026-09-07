#!/usr/bin/env python3
"""Render review MP4s from JPEG camera streams in a VLA steps.h5 file."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--cameras", default="side,head_camera,right_wrist")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-width", type=int, default=1280)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cameras = [name.strip() for name in args.cameras.split(",") if name.strip()]
    stride = max(1, int(args.stride))

    with h5py.File(args.input, "r") as h5:
        image_group = h5["images"]
        for camera in cameras:
            if camera not in image_group:
                print(f"skip missing camera: {camera}")
                continue
            dataset = image_group[camera]
            output = args.output_dir / f"suitcase_seed0_{camera}.mp4"
            with imageio.get_writer(
                output,
                fps=float(args.fps),
                codec="libx264",
                quality=8,
                macro_block_size=2,
            ) as writer:
                for frame_idx in range(0, len(dataset), stride):
                    frame = Image.open(io.BytesIO(bytes(dataset[frame_idx]))).convert("RGB")
                    if args.max_width > 0 and frame.width > args.max_width:
                        height = round(frame.height * args.max_width / frame.width)
                        frame = frame.resize((args.max_width, height), Image.Resampling.LANCZOS)
                    draw = ImageDraw.Draw(frame)
                    label = f"{camera}  frame {frame_idx}/{len(dataset) - 1}"
                    draw.rectangle((8, 8, 8 + 8 * len(label), 32), fill=(0, 0, 0))
                    draw.text((12, 12), label, fill=(255, 255, 255))
                    writer.append_data(np.asarray(frame))
            frame_count = (len(dataset) + stride - 1) // stride
            print(f"wrote {output} ({frame_count} frames)")


if __name__ == "__main__":
    main()
