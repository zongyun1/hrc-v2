"""Render the `talking` avatar motion: standing alongside a table, talking,
holding a phone in the left hand.

Two modes (run probe first, then render):
  probe  — play the motion headless, record per-frame left/right-hand frame +
           root to JSON and print stats (standing height, hand zones).
  render — place a table beside the avatar, attach the 077_phone to the left
           hand (orientation/offset tunable), replay the motion, save an MP4.

Usage:
  python scripts/render_talking.py --mode probe  --probe-json data/talking/probe.json
  python scripts/render_talking.py --mode render --probe-json data/talking/probe.json \
      -o data/talking/talking.mp4
"""
import argparse
import json
import os
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
ASSETS_PATH = os.path.join(REPO, "assets")

import genesis as gs

from envs.utils import Pose, load_mesh, load_object

TABLE_GLB = os.path.join(ASSETS_PATH, "objects", "sapien-table-21467", "visual", "base0.glb")
TABLE_BAKED_EXTENTS = np.array([1.2, 0.7, 0.765])  # baked to top z=0.765 at scale 1

PHONE_ID = "077_phone"
# 077_phone (model_data0): scale 0.078 -> scaled extents ~ x0.075(width)
# y0.016(thin/screen-normal) z0.151(length). Long axis local +Z.


def render_rgb(cam):
    out = cam.render(rgb=True, depth=False)
    rgb = out[0] if isinstance(out, (list, tuple)) else out
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu().numpy()
    rgb = np.asarray(rgb)
    if rgb.dtype in (np.float32, np.float64):
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return rgb.astype(np.uint8)


def save_video(frames, path, fps=30):
    import imageio

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    w = imageio.get_writer(path, fps=fps)
    for f in frames:
        w.append_data(f)
    w.close()


def quat_from_R(R):
    import genesis.utils.geom as geom_utils

    return np.asarray(geom_utils.R_to_quat(np.asarray(R, dtype=np.float64)))


def rot_about(axis, deg):
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    x, y, z = axis
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


def build_avatar(scene, args):
    from envs.avatar.controller import AvatarController

    avatar = AvatarController(
        scene=scene,
        motion_data_path=os.path.join(ASSETS_PATH, "avatars", "motions", "motion.pkl"),
        skin_options={
            "glb_path": os.path.join(ASSETS_PATH, "avatars", "models", "custom_Adrian_Keller.glb"),
            "euler": (-90, 0, 90),
            "pos": (0.0, 0.0, -0.959008030),
        },
        frame_ratio=1.0,
        name="human",
        assets_dir=ASSETS_PATH,
        generated_motion_path=os.path.abspath(args.motion_pkl),
    )
    if args.motion not in avatar.motion_data:
        print(f"Error: motion '{args.motion}' missing. Keys: {list(avatar.motion_data.keys())[:10]}")
        sys.exit(1)
    return avatar


def init_scene():
    backend = gs.cpu if os.environ.get("GENESIS_BACKEND", "gpu") == "cpu" else gs.gpu
    gs.init(backend=backend, logging_level="warning")
    scene = gs.Scene(
        show_viewer=False,
        renderer=gs.renderers.Rasterizer(),
        sim_options=gs.options.SimOptions(dt=0.01),
        vis_options=gs.options.VisOptions(ambient_light=(0.4, 0.4, 0.4)),
    )
    scene.add_entity(gs.morphs.Plane())
    return scene


def probe(args):
    scene = init_scene()
    avatar = build_avatar(scene, args)
    scene.build()
    avatar.reset(np.zeros(3), np.eye(3))
    avatar.play_animation(args.motion)

    rec = {"left_hand": [], "left_hand_R": [], "right_hand": [], "root": []}
    steps = 0
    motion = avatar.motion_data[args.motion]
    while not avatar.spare() and steps < args.max_steps:
        avatar.step()
        scene.step()
        lp, lR = avatar.robot._get_hand_frame(0)
        rp, _ = avatar.robot._get_hand_frame(1)
        root = np.asarray(motion["trans"][min(steps, len(motion["trans"]) - 1)], dtype=float)
        rec["left_hand"].append(np.asarray(lp, dtype=float).tolist())
        rec["left_hand_R"].append(np.asarray(lR, dtype=float).reshape(-1).tolist())
        rec["right_hand"].append(np.asarray(rp, dtype=float).tolist())
        rec["root"].append(root.tolist())
        steps += 1

    os.makedirs(os.path.dirname(args.probe_json) or ".", exist_ok=True)
    json.dump(rec, open(args.probe_json, "w"))

    lh = np.array(rec["left_hand"])
    n = len(lh)
    mid = slice(int(0.15 * n), int(0.9 * n))
    print(f"frames={n}")
    print(f"left hand mid mean: {lh[mid].mean(axis=0)}")
    print(f"left hand z range: {lh[:, 2].min():.3f} .. {lh[:, 2].max():.3f}")
    print(f"left hand xy spread (mid): {lh[mid, :2].std(axis=0)}")
    lR0 = np.array(rec["left_hand_R"][args.attach_step]).reshape(3, 3)
    print(f"left hand frame @{args.attach_step}: e1={lR0[:,0]} e2={lR0[:,1]} e3={lR0[:,2]}")
    rt = np.array(rec["root"])
    print(f"root first: {rt[0]}  root mid mean: {rt[mid].mean(axis=0)}")
    print(f"saved {args.probe_json}")


def render(args):
    scene = init_scene()

    # --- table beside the avatar (sapien table glb, baked top z=0.765) ------
    # Avatar resets at origin facing roughly +X.  Put the table to one side.
    tscale = (
        args.table_size[0] / TABLE_BAKED_EXTENTS[0],
        args.table_size[1] / TABLE_BAKED_EXTENTS[1],
        args.table_top / TABLE_BAKED_EXTENTS[2],
    )
    tyaw = np.deg2rad(args.table_yaw_deg)
    tquat = np.array([np.cos(tyaw / 2), 0, 0, np.sin(tyaw / 2)])
    print(f"[table] center={args.table_xy} yaw={args.table_yaw_deg} scale={tscale}")
    load_mesh(
        scene, TABLE_GLB, Pose([args.table_xy[0], args.table_xy[1], 0.0], tquat),
        scale=tscale, is_static=True,
    )

    # --- phone (visual; kinematically attached to the left hand) -----------
    phone = load_object(
        scene, Pose([0.0, 0.0, -1.0]),  # parked below floor until attach
        PHONE_ID, model_id=0, convex=True,
    )

    avatar = build_avatar(scene, args)

    # --- cameras: several angles so we can pick the best + see the phone ----
    look = tuple(np.array(args.cam_lookat).tolist())
    views = {
        "front":      (2.4, 0.0, 1.45),
        "front_left": (2.1, 1.5, 1.45),   # avatar's left (phone/ear side)
        "front_right":(2.1, -1.5, 1.45),
    }
    if args.single_view:
        views = {"main": tuple(args.cam_pos)}
    cams = {
        name: scene.add_camera(pos=pos, lookat=look, res=tuple(args.res),
                               fov=45, GUI=False)
        for name, pos in views.items()
    }

    scene.build()

    avatar.reset(np.zeros(3), np.eye(3))

    # --- warm-up pass: roll to a stable frame, attach the phone there -------
    # First frames interpolate from T-pose, so the hand frame is garbage early.
    # Attach mid-clip where the hand pose is genuine; the hand-local offset
    # persists across the restart.
    avatar.play_animation(args.motion)
    for _ in range(args.attach_step):
        avatar.step()
        scene.step()
    hand_pos, R_hand = avatar.robot._get_hand_frame(0)  # left hand (wrist bone)
    e1, e2, e3 = R_hand[:, 0], R_hand[:, 1], R_hand[:, 2]
    palm = np.asarray(avatar.robot.get_palm_center(0), dtype=float)
    head = np.asarray(
        avatar.robot.skin.get_global_translation("Head")[0], dtype=float
    ).ravel()[:3]
    # Held phone in the palm during a call: the screen lies against the
    # cheek/ear (faces the head), the back faces out, and the long axis runs
    # along the fingers.  Build the frame from real geometry so it's robust to
    # the hand-frame axis signs.
    #   y (phone local +Y, screen normal) -> toward the head (palm->head)
    #   z (phone local +Z, length)        -> finger direction, in-plane
    #   x (phone local +X, width)         -> y x z   (right-handed)
    y = head - palm
    y = y / (np.linalg.norm(y) + 1e-8)
    if args.phone_face_out:
        y = -y
    z = e2 - np.dot(e2, y) * y
    if np.linalg.norm(z) < 1e-6:
        z = np.array([0.0, 0.0, 1.0]) - np.dot([0, 0, 1], y) * y
    z = z / (np.linalg.norm(z) + 1e-8)
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x) + 1e-8)
    z = np.cross(x, y)
    R_phone = np.column_stack([x, y, z])
    # tunable extra rotations: roll about the length axis, tilt about width
    R_phone = rot_about(z, args.phone_roll_deg) @ R_phone
    R_phone = rot_about(x, args.phone_tilt_deg) @ R_phone
    # Seat the phone in the palm: palm center, nudged toward the head along y
    # (so the screen meets the cheek) and along the finger/width axes if needed.
    pos = (palm
           + y * args.phone_off_palm
           + z * args.phone_off_finger
           + x * args.phone_off_side
           + np.array([0.0, 0.0, -args.phone_down]))
    phone.entity.set_pos(pos)
    phone.entity.set_quat(quat_from_R(R_phone))
    avatar.attach_object_to_hand(phone.entity, hand_id=0)
    print(f"[phone] palm={palm} head={head} wrist={hand_pos}")
    print(f"[phone] pos={pos} y(screen->head)={y} z(len)={z}")

    # --- recorded pass ------------------------------------------------------
    # NOTE: do NOT reset() here — reset clears the hand attachment.  Restart
    # the clip directly; the phone's hand-local offset persists.
    avatar.play_animation(args.motion)
    frames = {name: [] for name in cams}
    steps = 0
    while not avatar.spare() and steps < args.max_steps:
        avatar.step()
        scene.step()
        if steps >= args.skip_record_steps:
            for name, cam in cams.items():
                frames[name].append(render_rgb(cam))
        steps += 1

    if not any(frames.values()):
        print("Error: no frames produced")
        sys.exit(1)
    base, ext = os.path.splitext(args.output)
    for name, fr in frames.items():
        out = args.output if len(frames) == 1 else f"{base}_{name}{ext}"
        save_video(fr, out, fps=args.fps)
        print(f"Saved {len(fr)} frames to {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["probe", "render"], required=True)
    p.add_argument("--motion", default="talking")
    p.add_argument("--motion-pkl",
                   default=os.path.join(REPO, "retarget", "output_motion_talking", "motion.pkl"))
    p.add_argument("--probe-json", default=os.path.join(REPO, "data", "talking", "probe.json"))
    p.add_argument("-o", "--output", default=os.path.join(REPO, "data", "talking", "talking.mp4"))
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--res", type=int, nargs=2, default=[960, 720])
    p.add_argument("--max-steps", type=int, default=2000)
    # table
    p.add_argument("--table-size", type=float, nargs=2, default=[1.2, 0.7],
                   help="width, depth (m)")
    p.add_argument("--table-top", type=float, default=0.74)
    p.add_argument("--table-xy", type=float, nargs=2, default=[0.35, -0.75])
    p.add_argument("--table-yaw-deg", type=float, default=0.0)
    # phone attach (left hand)
    p.add_argument("--attach-step", type=int, default=40,
                   help="warm-up steps before capturing the hand frame for attach")
    p.add_argument("--phone-face-out", action="store_true",
                   help="flip the screen to face out (default: screen toward head)")
    p.add_argument("--phone-off-palm", type=float, default=0.0,
                   help="offset from palm center toward the head along screen normal (m)")
    p.add_argument("--phone-off-finger", type=float, default=0.0,
                   help="offset along the phone length / finger direction (m)")
    p.add_argument("--phone-off-side", type=float, default=0.0,
                   help="offset across the palm / phone width (m)")
    p.add_argument("--phone-down", type=float, default=0.04,
                   help="lower the phone in world -z by this much (m)")
    p.add_argument("--phone-roll-deg", type=float, default=0.0,
                   help="roll the phone about its length axis")
    p.add_argument("--phone-tilt-deg", type=float, default=0.0,
                   help="tilt the phone about its width axis")
    # camera
    p.add_argument("--skip-record-steps", type=int, default=6)
    p.add_argument("--single-view", action="store_true",
                   help="render only --cam-pos instead of the 3-view set")
    p.add_argument("--cam-pos", type=float, nargs=3, default=[2.4, 0.0, 1.45])
    p.add_argument("--cam-lookat", type=float, nargs=3, default=[0.05, 0.1, 1.25])
    args = p.parse_args()

    if args.mode == "probe":
        probe(args)
    else:
        render(args)


if __name__ == "__main__":
    main()
