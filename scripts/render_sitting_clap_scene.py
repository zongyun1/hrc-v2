"""Render a seated avatar clapping beside a table.

This is a review-only renderer for the Mixamo ``sitting_clap`` conversion.
It keeps the chair/table staging out of benchmark task code.
"""
import argparse
import os
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
os.environ.setdefault("GENESIS_BACKEND", "cpu")

import cv2
import genesis as gs
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
ASSETS_PATH = os.path.join(REPO, "assets")

from envs.utils import Pose, create_primitive, load_mesh

CHAIR_GLB = os.path.join(ASSETS_PATH, "objects", "sapien-chair-179", "visual", "base0.glb")
# Same measured constants used by scripts/render_writing.py. Chair 179 has
# wheel bottom at raw z=-0.832 and seat-pan top at raw z=-0.046 after the
# URDF's y-up fix, so seat top height = 0.786 * scale when the wheels sit on
# the ground.
CHAIR_RAW_BOTTOM_Z = 0.832048
CHAIR_RAW_SEAT_H = 0.786

TABLE_GLB = os.path.join(ASSETS_PATH, "objects", "sapien-table-21467", "visual", "base0.glb")
TABLE_BAKED_EXTENTS = np.array([1.2, 0.7, 0.765])


def _yaw_matrix(deg: float) -> np.ndarray:
    rad = np.deg2rad(float(deg))
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _render_rgb(cam) -> np.ndarray:
    out = cam.render(rgb=True, depth=False)
    rgb = out[0] if isinstance(out, (tuple, list)) else out
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu().numpy()
    rgb = np.asarray(rgb)
    if rgb.dtype in (np.float32, np.float64):
        rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        rgb = rgb.astype(np.uint8)
    return rgb


def _save_video(frames: list[np.ndarray], path: str, fps: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(w), int(h)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _box(scene, pos, half_size, color, *, collision=False):
    return create_primitive(
        scene,
        "box",
        Pose(pos),
        size={"half_size": half_size},
        color=color,
        is_static=True,
        collision=collision,
    )


def _cylinder(scene, pos, radius, half_length, color, *, collision=False):
    return create_primitive(
        scene,
        "cylinder",
        Pose(pos),
        size={"radius": radius, "half_length": half_length},
        color=color,
        is_static=True,
        collision=collision,
    )


def _add_table(scene, top_z: float = 0.74):
    load_mesh(
        scene,
        TABLE_GLB,
        Pose([0.72, 0.0, 0.0]),
        scale=(1.03, 1.25, top_z / TABLE_BAKED_EXTENTS[2]),
        is_static=True,
    )
    # Small tabletop props give the table scale and make the side seating clear.
    _box(scene, (0.64, -0.22, top_z + 0.018), (0.18, 0.12, 0.018), (0.08, 0.10, 0.12, 1.0))
    _cylinder(scene, (0.86, 0.22, top_z + 0.055), 0.045, 0.055, (0.85, 0.88, 0.82, 1.0))


def _add_chair_asset(scene, xy, *, seat_height: float, z_offset: float, yaw_deg: float):
    scale = float(seat_height) / CHAIR_RAW_SEAT_H
    yaw = np.deg2rad(float(yaw_deg))
    quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64)
    chair_z = CHAIR_RAW_BOTTOM_Z * scale + float(z_offset)
    chair = load_mesh(
        scene,
        CHAIR_GLB,
        Pose([float(xy[0]), float(xy[1]), chair_z], quat),
        scale=(scale, scale, scale),
        is_static=True,
        collision=False,
    )
    print(
        f"[chair] asset=sapien-chair-179 visual={os.path.relpath(CHAIR_GLB, REPO)} scale={scale:.3f} "
        f"seat_top={seat_height + z_offset:.3f} z={chair_z:.3f} yaw={yaw_deg:.1f}"
    )
    return chair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", default="sitting_clap")
    parser.add_argument(
        "--motion-pkl",
        default=os.path.join(REPO, "retarget", "output_motion_claping", "motion.pkl"),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=os.path.join(REPO, "data", "claping", "sitting_clap_vico.mp4"),
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--res", type=int, nargs=2, default=[960, 720])
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--avatar-x", type=float, default=-0.12)
    parser.add_argument("--avatar-y", type=float, default=0.0)
    parser.add_argument("--avatar-z", type=float, default=-0.1)
    parser.add_argument("--avatar-yaw", type=float, default=0.0)
    parser.add_argument("--chair-xy", type=float, nargs=2, default=[-0.12, 0.0])
    parser.add_argument("--chair-yaw", type=float, default=270.0)
    parser.add_argument("--chair-seat-height", type=float, default=0.46)
    parser.add_argument("--chair-z-offset", type=float, default=-0.1)
    parser.add_argument(
        "--avatar-glb",
        default=os.path.join(ASSETS_PATH, "avatars", "models", "custom_Adrian_Keller.glb"),
    )
    args = parser.parse_args()

    if not os.path.isfile(args.motion_pkl):
        raise FileNotFoundError(args.motion_pkl)

    backend = gs.gpu if os.environ.get("GENESIS_BACKEND") == "gpu" else gs.cpu
    gs.init(backend=backend, logging_level="warning")

    scene = gs.Scene(
        show_viewer=False,
        renderer=gs.renderers.Rasterizer(),
        sim_options=gs.options.SimOptions(dt=0.01),
        vis_options=gs.options.VisOptions(ambient_light=(0.42, 0.42, 0.42)),
    )
    scene.add_entity(gs.morphs.Plane())
    _add_table(scene)
    chair = _add_chair_asset(
        scene,
        args.chair_xy,
        seat_height=args.chair_seat_height,
        z_offset=args.chair_z_offset,
        yaw_deg=args.chair_yaw,
    )

    from envs.avatar.controller import AvatarController

    scene._skip_avatar_walk_modules = True
    avatar = AvatarController(
        scene=scene,
        motion_data_path=os.path.join(ASSETS_PATH, "avatars", "motions", "motion.pkl"),
        skin_options={
            "glb_path": args.avatar_glb,
            "euler": (-90, 0, 90),
            "pos": (0.0, 0.0, -0.959008030),
        },
        frame_ratio=1.0,
        name="sitting_clap_review",
        assets_dir=ASSETS_PATH,
        generated_motion_path=os.path.abspath(args.motion_pkl),
    )
    if args.motion not in avatar.motion_data:
        raise KeyError(f"{args.motion!r} not found. Available generated keys include: {list(avatar.motion_data.keys())[:20]}")

    cam = scene.add_camera(
        pos=(2.35, -1.85, 1.25),
        lookat=(0.25, 0.0, 0.72),
        res=tuple(args.res),
        fov=42,
        GUI=False,
    )
    scene.build()

    avatar.reset(
        np.array([args.avatar_x, args.avatar_y, args.avatar_z], dtype=np.float64),
        _yaw_matrix(args.avatar_yaw),
    )
    avatar.play_animation(args.motion)

    frames = []
    steps = 0
    while not avatar.spare() and steps < int(args.max_steps):
        avatar.step()
        scene.step()
        frames.append(_render_rgb(cam))
        steps += 1

    if not frames:
        raise RuntimeError("No frames rendered")
    _save_video(frames, args.output, args.fps)

    # Save a still frame for quick visual inspection from the agent side.
    preview = os.path.splitext(args.output)[0] + "_preview.jpg"
    cv2.imwrite(preview, cv2.cvtColor(frames[min(len(frames) // 2, len(frames) - 1)], cv2.COLOR_RGB2BGR))
    print(f"Saved {len(frames)} frames to {args.output}")
    print(f"Saved preview frame to {preview}")


if __name__ == "__main__":
    main()
