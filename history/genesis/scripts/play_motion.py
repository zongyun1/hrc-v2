"""Play a generated avatar motion on the avatar and save a video.

Usage:
    python scripts/play_motion.py --motion Shove_Reaction -o output.mp4
"""
import argparse
import os
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
os.environ.setdefault("GENESIS_BACKEND", "gpu")

import numpy as np
import genesis as gs
import cv2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
ASSETS_PATH = os.path.join(REPO, "assets")

from envs.utils import ASSETS_PATH, Pose, create_primitive, load_mesh

SPONGE_FACE_DOWN_QUAT = np.array(
    [0.7071067811865476, 0.7071067811865475, 0.0, 0.0],
    dtype=np.float64,
)


def render_rgb(cam):
    out = cam.render(rgb=True, depth=False)
    rgb = out[0] if isinstance(out, (list, tuple)) else out
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu().numpy()
    rgb = np.asarray(rgb)
    if rgb.dtype in (np.float32, np.float64):
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    else:
        rgb = rgb.astype(np.uint8)
    return rgb


def draw_label(frame, text, duplicate=True):
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.6
    thickness = 4
    pad = 12
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    def stamp(x, y):
        cv2.rectangle(
            bgr,
            (x - pad, y - th - pad),
            (x + tw + pad, y + baseline + pad),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            bgr,
            text,
            (x, y),
            font,
            scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    x = 12
    stamp(x, th + 24)
    if duplicate:
        stamp(x, max(th + pad, bgr.shape[0] - 16))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_video(frames, path, fps=30):
    import imageio

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    w = imageio.get_writer(path, fps=fps)
    for f in frames:
        w.append_data(f)
    w.close()


def set_debug_sponge_pose(sponge, avatar, hand_id, down_offset, fixed_z=None):
    """Place the sponge under the palm without using wrist-frame attachment."""
    palm = np.asarray(avatar.robot.get_palm_center(int(hand_id)), dtype=np.float64).ravel()[:3]
    pos = palm.copy()
    if fixed_z is None:
        pos[2] = float(palm[2] - down_offset)
    else:
        pos[2] = float(fixed_z)
    sponge.set_pos(pos.astype(float))
    sponge.set_quat(SPONGE_FACE_DOWN_QUAT.astype(float))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True, help="Motion name (key in motion.pkl)")
    parser.add_argument(
        "--motion-pkl",
        default=os.path.join(ASSETS_PATH, "avatars", "motions", "generated_motions.pkl"),
        help="Path to generated motion pkl",
    )
    parser.add_argument("-o", "--output", default=None, help="Output mp4 path")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--res", type=int, nargs=2, default=[640, 480])
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--show-frame-number", action="store_true")
    parser.add_argument(
        "--frame-number-only",
        action="store_true",
        help="When showing frame numbers, draw only the zero-padded number.",
    )
    parser.add_argument(
        "--source-start-frame",
        type=int,
        default=None,
        help="First source motion frame to render. Uses source indexing in the input pkl.",
    )
    parser.add_argument(
        "--source-end-frame",
        type=int,
        default=None,
        help="Last source motion frame to render, inclusive.",
    )
    parser.add_argument(
        "--frame-number-offset",
        type=int,
        default=None,
        help="Displayed frame number for the first rendered frame.",
    )
    parser.add_argument("--attach-red-cube", action="store_true")
    parser.add_argument("--attach-sponge", action="store_true")
    parser.add_argument("--attach-frame", type=int, default=None, help="Source frame where the cube attaches.")
    parser.add_argument("--detach-frame", type=int, default=None, help="Source frame where the cube detaches.")
    parser.add_argument("--hand-id", type=int, default=1, choices=[0, 1])
    parser.add_argument("--cube-half-size", type=float, default=0.035)
    parser.add_argument(
        "--sponge-palm-down-offset",
        type=float,
        default=0.045,
        help="Debug sponge vertical offset below palm center while attached.",
    )
    parser.add_argument(
        "--sponge-fixed-z",
        type=float,
        default=None,
        help="Optional fixed world z for debug sponge follow, matching table-surface follow.",
    )
    parser.add_argument(
        "--avatar-glb",
        default=os.path.join(ASSETS_PATH, "avatars", "models", "custom_Adrian_Keller.glb"),
        help="Avatar GLB path. Use a workspace-local copy if shared assets are not readable on compute nodes.",
    )
    args = parser.parse_args()

    motion_pkl = os.path.abspath(args.motion_pkl)
    if not os.path.isfile(motion_pkl):
        print(f"Error: motion pkl not found: {motion_pkl}")
        sys.exit(1)

    backend = gs.gpu if os.environ.get("GENESIS_BACKEND") == "gpu" else gs.cpu
    gs.init(backend=backend, logging_level="warning")

    scene = gs.Scene(
        show_viewer=False,
        renderer=gs.renderers.Rasterizer(),
        sim_options=gs.options.SimOptions(dt=0.01),
        vis_options=gs.options.VisOptions(ambient_light=(0.3, 0.3, 0.3)),
    )
    scene.add_entity(gs.morphs.Plane())

    from envs.avatar.controller import AvatarController

    avatar = AvatarController(
        scene=scene,
        motion_data_path=os.path.join(ASSETS_PATH, "avatars", "motions", "motion.pkl"),
        skin_options={
            "glb_path": args.avatar_glb,
            "euler": (-90, 0, 90),
            "pos": (0.0, 0.0, -0.959008030),
        },
        frame_ratio=1.0,
        name="human",
        assets_dir=ASSETS_PATH,
        generated_motion_path=motion_pkl,
    )

    if args.motion not in avatar.motion_data:
        print(f"Error: motion '{args.motion}' not in motion_data. Keys: {list(avatar.motion_data.keys())[:20]} ...")
        sys.exit(1)

    motion_len = int(avatar.motion_data[args.motion]["trans"].shape[0])
    source_start = 0 if args.source_start_frame is None else int(args.source_start_frame)
    source_end = motion_len - 1 if args.source_end_frame is None else int(args.source_end_frame)
    if source_start < 0 or source_end < source_start or source_end >= motion_len:
        print(
            f"Error: invalid source frame range {source_start}..{source_end} "
            f"for motion length {motion_len}"
        )
        sys.exit(1)

    # PlayAnimationMotion renders after its first step, so include one hidden
    # source frame before source_start. The visible labels still start at
    # source_start.
    hidden_start = max(0, source_start - 1)
    if args.source_start_frame is not None or args.source_end_frame is not None:
        idx = slice(hidden_start, source_end + 1)
        avatar.motion_data[args.motion] = {
            k: v[idx].copy()
            for k, v in avatar.motion_data[args.motion].items()
        }
    frame_number_offset = (
        int(args.frame_number_offset)
        if args.frame_number_offset is not None
        else source_start
    )

    cam = scene.add_camera(
        pos=(3.0, -2.5, 1.4),
        lookat=(0.0, 0.0, 0.8),
        res=tuple(args.res),
        fov=45,
        GUI=False,
    )

    attach_obj = None
    sponge_obj = None
    if args.attach_red_cube:
        attach_obj = create_primitive(
            scene,
            "box",
            Pose([0.0, 0.0, 0.9]),
            size={"half_size": (args.cube_half_size,) * 3},
            color=(1.0, 0.0, 0.0, 1.0),
            is_static=True,
            collision=False,
        )
    elif args.attach_sponge:
        sponge_obj = load_mesh(
            scene,
            ASSETS_PATH / "objects" / "cc0_sponge_3" / "visual" / "base0.glb",
            Pose([0.0, 0.0, 0.9], SPONGE_FACE_DOWN_QUAT),
            scale=1.0,
            is_static=True,
            collision=False,
        )

    scene.build()

    attach_frame = None
    detach_frame = None
    if attach_obj is not None or sponge_obj is not None:
        if args.attach_frame is None or args.detach_frame is None:
            print("Error: --attach-red-cube/--attach-sponge requires --attach-frame and --detach-frame")
            sys.exit(1)
        attach_frame = int(args.attach_frame) - hidden_start
        detach_frame = int(args.detach_frame) - hidden_start
        if attach_frame < 0 or detach_frame < attach_frame:
            print(
                f"Error: attach/detach frames {args.attach_frame}/{args.detach_frame} "
                f"do not fit source range {source_start}..{source_end}"
            )
            sys.exit(1)

        avatar.reset(np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64))
        avatar.play_animation(args.motion)
        for _ in range(min(attach_frame, int(avatar.motion_data[args.motion]["trans"].shape[0]) - 1)):
            avatar.step()
            scene.step()
        if attach_obj is not None:
            hand_pos, hand_rot = avatar.robot._get_hand_frame(args.hand_id)
            attach_obj.set_pos(np.asarray(hand_pos, dtype=float))
            try:
                import genesis.utils.geom as geom_utils
                attach_obj.set_quat(geom_utils.R_to_quat(hand_rot).astype(float))
            except Exception:
                pass
        if sponge_obj is not None:
            set_debug_sponge_pose(
                sponge_obj,
                avatar,
                args.hand_id,
                args.sponge_palm_down_offset,
                fixed_z=args.sponge_fixed_z,
            )

    avatar.reset(np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64))
    avatar.play_animation(
        args.motion,
        attach_obj=attach_obj,
        hand_id=args.hand_id,
        attach_frame=attach_frame,
        detach_frame=detach_frame,
    )

    frames = []
    steps = 0
    while not avatar.spare() and steps < args.max_steps:
        avatar.step()
        if sponge_obj is not None and attach_frame is not None and attach_frame <= steps <= detach_frame:
            set_debug_sponge_pose(
                sponge_obj,
                avatar,
                args.hand_id,
                args.sponge_palm_down_offset,
                fixed_z=args.sponge_fixed_z,
            )
        scene.step()
        frame = render_rgb(cam)
        if args.show_frame_number:
            frame_no = frame_number_offset + steps
            text = f"{frame_no:04d}" if args.frame_number_only else f"FRAME {frame_no:04d}"
            frame = draw_label(frame, text, duplicate=not args.frame_number_only)
        frames.append(frame)
        steps += 1

    if not frames:
        print("Error: no frames produced (animation may have ended immediately)")
        sys.exit(1)

    out_path = args.output or os.path.join(REPO, f"{args.motion.replace(' ', '_')}.mp4")
    save_video(frames, out_path, fps=args.fps)
    print(f"Saved {len(frames)} frames to {out_path}")


if __name__ == "__main__":
    main()
