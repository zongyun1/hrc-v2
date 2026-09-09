"""Render multi-angle still images of the shared factory work-cell scene.

Builds the FactorySceneMixin cell (bench + rack + floor tape + pallet)
with the Franka mounted on the bench and the avatar worker standing at
it, settles physics briefly, then renders a set of static review cameras
to PNG.

Usage:
    python scripts/preview_factory_scene.py [--out-dir data/factory_scene_preview]
                                            [--seed 0] [--no-avatar]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import imageio.v2 as imageio

from envs.base_task import BaseTask
from envs.scenes.factory import FactorySceneMixin


# Review cameras: name → (position, lookat). All frame the bench center.
PREVIEW_CAMERAS = {
    "front":               ([0.0, -2.30, 1.55], [0.0, 0.30, 0.90]),
    "three_quarter_left":  ([-2.40, -1.10, 1.95], [0.0, 0.20, 0.90]),
    "three_quarter_right": ([1.90, -1.55, 1.75], [0.0, 0.20, 0.90]),
    "side_right":          ([2.40, 0.10, 1.25], [0.0, 0.10, 0.90]),
    "top_down":            ([0.05, -0.45, 3.10], [0.0, 0.00, 0.76]),
    "bench_close":         ([0.95, -1.25, 1.40], [-0.25, 0.10, 0.82]),
}


class FactoryScenePreview(FactorySceneMixin, BaseTask):
    """Empty factory cell — furniture only, no task objects."""

    use_avatar = True

    def __init__(self, config=None):
        cfg = dict(config or {})
        cfg.setdefault("robot_type", "franka")
        cfg.setdefault("robot_single_arm", True)
        super().__init__(cfg)

    def play_once(self) -> bool:
        return True

    def check_success(self) -> bool:
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/factory_scene_preview")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-avatar", action="store_true",
                        help="Render the cell without the human worker.")
    args = parser.parse_args()

    config = {
        "static_camera_list": [
            {"name": name, "position": pos, "lookat": lookat}
            for name, (pos, lookat) in PREVIEW_CAMERAS.items()
        ],
        "camera_config": {"default": {"w": 960, "h": 720, "fovy": 60}},
        "settle_steps": 80,
        "skip_reset_obs": True,
        "side_video": True,
    }

    if args.no_avatar:
        FactoryScenePreview.use_avatar = False
    task = FactoryScenePreview(config)
    task.reset(seed=args.seed)

    # Pose the avatar naturally: drive the idle clip through the motion
    # system for a stretch (a bare reset leaves the avatar solver unposed).
    if task.avatar is not None:
        task.avatar.play_animation("idle")
        for _ in range(150):
            task.step_sim()

    # Render twice with a sim step in between: the rasterizer's first
    # render builds its buffers from the import-pose mesh and only applies
    # avatar skinning updates on a later render after the sim clock
    # advances (videos never hit this — they render every frame).
    task.cameras.render_all()
    task.step_sim()
    task.cameras.render_all()

    os.makedirs(args.out_dir, exist_ok=True)
    saved = []
    for name, rgb in task.cameras.get_all_rgb().items():
        if name in ("left_wrist", "right_wrist"):
            continue
        path = os.path.join(args.out_dir, f"{name}.png")
        imageio.imwrite(path, rgb)
        saved.append(path)
        print(f"[preview] saved {path}")
    print(f"[preview] done — {len(saved)} images under {args.out_dir}")


if __name__ == "__main__":
    main()
