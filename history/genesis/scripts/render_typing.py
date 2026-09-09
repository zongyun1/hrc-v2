"""Render the `Typing` avatar motion: seated at a desk, typing on a laptop.

A hybrid of render_writing.py (seated avatar + chair) and render_fax_laptop.py
(SAPIEN laptop placed under the typing hands). Empty background (ground plane
only) so the avatar / chair / desk / laptop alignment can be verified before
the motion is wired into a neutral task.

Two modes (run probe first, then render):
  probe  — play the motion headless, record per-frame hand / hip / root to JSON.
  render — place chair under the avatar, desk slab + laptop under the hands
           (keyboard centered below the mean hand position, display facing the
           avatar), replay the motion, save MP4(s).

Usage:
  python scripts/render_typing.py --mode probe  --probe-json data/typing/probe.json
  python scripts/render_typing.py --mode render --probe-json data/typing/probe.json \
      -o data/typing/typing.mp4 --side-output data/typing/typing_side.mp4
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

from envs.utils import Pose, create_primitive, load_object, load_urdf


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])

CHAIR_URDF = os.path.join(ASSETS_PATH, "objects", "sapien-chair-179", "mobility.urdf")
# Raw (pre-scale) geometry of chair 179 after the URDF's y-up fix: wheels bottom
# z=-0.832, seat-pan top z=-0.046, backrest on the +X side (chair faces -X).
CHAIR_RAW_BOTTOM_Z = 0.832048
CHAIR_RAW_SEAT_H = 0.786  # wheel bottoms -> seat pan top

LAPTOP_URDF = os.path.join(ASSETS_PATH, "objects", "015_laptop", "9960", "mobility.urdf")
# Pre-scale geometry of model 9960 in its URDF world frame at qpos=0 (lid open).
# Asset axes: keyboard slab spans x in [-0.713, +0.218], top z=-0.118, bottom
# z=-0.223; screen rises at +x and its display faces -x. So asset -x is "toward
# the user".
LAPTOP_RAW_BOTTOM_Z = 0.223
LAPTOP_RAW_KB_TOP_Z = 0.118        # keyboard surface this far BELOW the origin
LAPTOP_RAW_FRONT_X = -0.713        # user-side front edge
LAPTOP_RAW_KB_CENTER_X = -0.36     # center of the typing area (front edge .. hinge)


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

    rec = {"left_hand": [], "right_hand": [], "hips": [], "root": []}
    steps = 0
    motion = avatar.motion_data[args.motion]
    while not avatar.spare() and steps < args.max_steps:
        avatar.step()
        scene.step()
        lp, _ = avatar.robot._get_hand_frame(0)
        rp, _ = avatar.robot._get_hand_frame(1)
        hips = np.asarray(
            avatar.robot.skin.get_global_translation("Hips")[0], dtype=float
        ).ravel()[:3]
        root = np.asarray(motion["trans"][min(steps, len(motion["trans"]) - 1)], dtype=float)
        rec["left_hand"].append(np.asarray(lp, dtype=float).tolist())
        rec["right_hand"].append(np.asarray(rp, dtype=float).tolist())
        rec["hips"].append(hips.tolist())
        rec["root"].append(root.tolist())
        steps += 1

    os.makedirs(os.path.dirname(args.probe_json) or ".", exist_ok=True)
    json.dump(rec, open(args.probe_json, "w"))

    lh = np.array(rec["left_hand"]); rh = np.array(rec["right_hand"])
    hips_all = np.array(rec["hips"])
    n = len(lh)
    mid = slice(int(0.2 * n), int(0.9 * n))
    hands_mid = (lh[mid] + rh[mid]) / 2.0
    print(f"frames={n}")
    print(f"hand mid-window mean: {hands_mid.mean(axis=0)}")
    print(f"hand mid-window z range: {hands_mid[:, 2].min():.3f} .. {hands_mid[:, 2].max():.3f}")
    print(f"left  hand z range: {lh[:, 2].min():.3f} .. {lh[:, 2].max():.3f}")
    print(f"right hand z range: {rh[:, 2].min():.3f} .. {rh[:, 2].max():.3f}")
    print(f"hips mid mean: {hips_all[mid].mean(axis=0)}  (z={hips_all[mid,2].mean():.3f})")
    d = hands_mid.mean(axis=0)[:2] - hips_all[mid].mean(axis=0)[:2]
    print(f"facing (hips->hands) d={d / (np.linalg.norm(d) + 1e-8)}")
    print(f"saved {args.probe_json}")


def placement(args):
    """Laptop / desk / chair placement from the probe operation window."""
    rec = json.load(open(args.probe_json))
    lh = np.array(rec["left_hand"]); rh = np.array(rec["right_hand"])
    hips_all = np.array(rec["hips"])
    n = len(lh)
    mid = slice(int(0.2 * n), int(0.9 * n))
    hand_mean = ((lh[mid] + rh[mid]) / 2.0).mean(axis=0)
    hips = hips_all[mid].mean(axis=0)

    d = hand_mean[:2] - hips[:2]
    d = d / (np.linalg.norm(d) + 1e-8)

    # Rightmost reach of the right hand (where the mouse lives): max projection
    # onto the avatar's right direction, over the operation window.
    right_dir = np.array([d[1], -d[0]])
    proj_r = (rh[mid][:, :2] - hand_mean[:2]) @ right_dir
    rh_mid = rh[mid]
    mouse_xy = rh_mid[int(np.argmax(proj_r))][:2]
    return hand_mean, hips, d, mouse_xy


def render(args):
    hand_mean, hips, d, mouse_xy = placement(args)
    print(f"[place] hand_mean={hand_mean} hips={hips} d={d} mouse_xy={mouse_xy}")

    scene = init_scene()
    s = args.laptop_scale

    # --- laptop: keyboard typing-area center directly below the mean hand -----
    yaw = float(np.arctan2(d[1], d[0]) + np.deg2rad(args.laptop_yaw_offset_deg))
    fwd = np.array([np.cos(yaw), np.sin(yaw)])           # laptop +x in world
    kb_center = hand_mean[:2] - args.laptop_back_offset * d
    lap_xy = kb_center - LAPTOP_RAW_KB_CENTER_X * s * fwd
    lap_origin_z = hand_mean[2] - args.keyboard_clearance + LAPTOP_RAW_KB_TOP_Z * s
    if args.laptop_xy is not None:
        lap_xy = np.array(args.laptop_xy, dtype=float)
    if args.laptop_z is not None:
        lap_origin_z = float(args.laptop_z)
    desk_top = lap_origin_z - LAPTOP_RAW_BOTTOM_Z * s
    lap_quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    laptop = load_urdf(
        scene, LAPTOP_URDF, Pose([lap_xy[0], lap_xy[1], lap_origin_z], lap_quat),
        scale=s, fix_root=True,
    )
    print(f"[laptop] xy={lap_xy} origin_z={lap_origin_z:.3f} yaw={np.rad2deg(yaw):.1f} desk_top={desk_top:.3f}")

    # --- desk slab: near edge between the avatar body and the hands -----------
    desk = None
    if args.desk and desk_top > 0.05:
        # laptop footprint corners projected on d to find the user-side edge
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        corners = np.array([
            [LAPTOP_RAW_FRONT_X, -0.582], [LAPTOP_RAW_FRONT_X, 0.571],
            [0.218, -0.582], [0.218, 0.571],
        ]) * s
        corner_proj = ((R @ corners.T).T + lap_xy) @ d
        edge = float(corner_proj.min()) - 0.02
        body_front = float(hips[:2] @ d) + args.body_half_depth
        hands_proj = float(hand_mean[:2] @ d)
        if not (body_front < edge < hands_proj):
            edge = min(max(edge, body_front + 0.02), hands_proj - 0.02)
        desk_yaw = float(np.arctan2(d[1], d[0]))
        desk_quat = np.array([np.cos(desk_yaw / 2), 0, 0, np.sin(desk_yaw / 2)])
        lateral = lap_xy - (lap_xy @ d) * d
        desk_center = lateral + (edge + args.desk_half_x) * d
        desk = create_primitive(
            scene, "box",
            Pose([desk_center[0], desk_center[1], desk_top / 2.0], desk_quat),
            size={"half_size": (args.desk_half_x, args.desk_half_y, desk_top / 2.0)},
            color=(0.55, 0.42, 0.30),
            is_static=True,
        )
        print(f"[desk] center={desk_center} edge={edge:.3f} top={desk_top:.3f}")

    # --- chair under the avatar ----------------------------------------------
    chair = None
    if args.chair:
        cs = args.chair_scale if args.chair_scale else args.seat_height / CHAIR_RAW_SEAT_H
        chair_z = CHAIR_RAW_BOTTOM_Z * cs  # wheels on the floor
        # chair faces -X in asset frame; rotate so its front (-X) points along
        # +d (avatar forward) => backrest ends up behind the avatar's back.
        chair_yaw = (float(np.rad2deg(np.arctan2(d[1], d[0]))) + 180.0
                     if args.chair_yaw_deg is None else args.chair_yaw_deg)
        cyaw = np.deg2rad(chair_yaw)
        cquat = np.array([np.cos(cyaw / 2), 0, 0, np.sin(cyaw / 2)])
        chair_xy = (hips[:2] + np.array(args.chair_offset)
                    if args.chair_xy is None else np.array(args.chair_xy))
        chair = load_urdf(
            scene, CHAIR_URDF,
            Pose([chair_xy[0], chair_xy[1], chair_z], cquat),
            scale=cs, fix_root=True,
        )
        print(f"[chair] xy={chair_xy} scale={cs:.3f} yaw={chair_yaw:.1f} z={chair_z:.3f}")

    # --- mouse: on the desk, just below the right hand's rightmost reach ------
    if args.mouse:
        m_xy = np.array(args.mouse_xy, dtype=float) if args.mouse_xy is not None else mouse_xy
        m_xy = m_xy + args.mouse_fwd_offset * d  # shift toward the table (facing dir)
        m_z = desk_top + args.mouse_z_offset
        upright = np.array([0.7071067811865476, 0.7071067811865475, 0.0, 0.0])  # Rx90
        myaw = np.deg2rad(float(np.rad2deg(np.arctan2(d[1], d[0])) + args.mouse_yaw_offset_deg))
        yaw_quat = np.array([np.cos(myaw / 2), 0, 0, np.sin(myaw / 2)])
        m_quat = quat_mul(yaw_quat, upright)
        load_object(
            scene, Pose([float(m_xy[0]), float(m_xy[1]), m_z], m_quat),
            "047_mouse", model_id=0, convex=True, is_static=True,
        )
        print(f"[mouse] xy={m_xy} z={m_z:.3f} yaw_offset={args.mouse_yaw_offset_deg}")

    avatar = build_avatar(scene, args)

    # --- cameras --------------------------------------------------------------
    side = np.array([-d[1], d[0]])
    look = np.array([(hips[0] + lap_xy[0]) / 2, (hips[1] + lap_xy[1]) / 2, desk_top + 0.05])
    cam_xy = look[:2] + d * args.cam_fwd + side * args.cam_side
    cam = scene.add_camera(
        pos=(float(cam_xy[0]), float(cam_xy[1]), args.cam_height),
        lookat=tuple(look.tolist()),
        res=tuple(args.res), fov=45, GUI=False,
    )
    cam_side = None
    if args.side_output:
        # avatar's right side (where the mouse is), so it stays in foreground
        right = -side
        sxy = look[:2] + right * args.cam_fwd
        cam_side = scene.add_camera(
            pos=(float(sxy[0]), float(sxy[1]), args.cam_height),
            lookat=tuple(look.tolist()),
            res=tuple(args.res), fov=45, GUI=False,
        )
    cam_top = None
    if args.topdown_output:
        # straight down over the laptop+mouse area to verify hand/mouse alignment
        tc = (lap_xy + mouse_xy) / 2.0
        cam_top = scene.add_camera(
            pos=(float(tc[0] + 0.01), float(tc[1]), lap_origin_z + 0.9),
            lookat=(float(tc[0]), float(tc[1]), desk_top),
            res=tuple(args.res), fov=50, GUI=False,
        )

    scene.build()

    if chair is not None and chair.n_qs > 0:
        chair_hold = np.zeros((chair.n_qs,), dtype=np.float64)
        chair.set_qpos(chair_hold)
    else:
        chair_hold = None
    if laptop is not None and laptop.n_qs > 0:
        lap_hold = np.full((laptop.n_qs,), args.lid_qpos, dtype=np.float64)
        laptop.set_qpos(lap_hold)
    else:
        lap_hold = None

    avatar.reset(np.zeros(3), np.eye(3))
    avatar.play_animation(args.motion)

    frames, frames_side, frames_top = [], [], []
    steps = 0
    while not avatar.spare() and steps < args.max_steps:
        avatar.step()
        scene.step()
        if chair_hold is not None:
            chair.set_qpos(chair_hold)
        if lap_hold is not None:
            laptop.set_qpos(lap_hold)
        if steps >= args.skip_frames:
            frames.append(render_rgb(cam))
            if cam_side is not None:
                frames_side.append(render_rgb(cam_side))
            if cam_top is not None:
                frames_top.append(render_rgb(cam_top))
        steps += 1

    if not frames:
        print("Error: no frames produced")
        sys.exit(1)
    save_video(frames, args.output, fps=args.fps)
    print(f"Saved {len(frames)} frames to {args.output}")
    if frames_side:
        save_video(frames_side, args.side_output, fps=args.fps)
        print(f"Saved {len(frames_side)} frames to {args.side_output}")
    if frames_top:
        save_video(frames_top, args.topdown_output, fps=args.fps)
        print(f"Saved {len(frames_top)} frames to {args.topdown_output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["probe", "render"], required=True)
    p.add_argument("--motion", default="Typing")
    p.add_argument("--motion-pkl", default=os.path.join(REPO, "retarget", "output_motion_typing", "motion.pkl"))
    p.add_argument("--probe-json", default=os.path.join(REPO, "data", "typing", "probe.json"))
    p.add_argument("-o", "--output", default=os.path.join(REPO, "data", "typing", "typing.mp4"))
    p.add_argument("--side-output", default=None, help="optional second video from a side camera")
    p.add_argument("--topdown-output", default=None, help="optional top-down video over laptop+mouse")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--res", type=int, nargs=2, default=[960, 720])
    p.add_argument("--max-steps", type=int, default=2000)
    # laptop
    p.add_argument("--laptop-scale", type=float, default=0.28)
    p.add_argument("--laptop-xy", type=float, nargs=2, default=None)
    p.add_argument("--laptop-z", type=float, default=None)
    p.add_argument("--laptop-yaw-offset-deg", type=float, default=0.0)
    p.add_argument("--laptop-back-offset", type=float, default=0.0,
                   help="pull the laptop toward the avatar along facing dir (m)")
    p.add_argument("--lid-qpos", type=float, default=0.0)
    p.add_argument("--keyboard-clearance", type=float, default=0.04,
                   help="hand height above the keyboard surface (m)")
    # desk
    p.add_argument("--desk", action="store_true", default=True)
    p.add_argument("--no-desk", dest="desk", action="store_false")
    p.add_argument("--desk-half-x", type=float, default=0.30, help="desk half-depth along facing")
    p.add_argument("--desk-half-y", type=float, default=0.40, help="desk half-width lateral")
    p.add_argument("--body-half-depth", type=float, default=0.12)
    # chair
    p.add_argument("--chair", action="store_true", default=True)
    p.add_argument("--no-chair", dest="chair", action="store_false")
    p.add_argument("--chair-scale", type=float, default=None, help="default: from --seat-height")
    p.add_argument("--seat-height", type=float, default=0.46)
    p.add_argument("--chair-xy", type=float, nargs=2, default=None, help="default: under the hips")
    p.add_argument("--chair-offset", type=float, nargs=2, default=[0.0, 0.0],
                   help="extra xy offset from the hips for the chair center")
    p.add_argument("--chair-yaw-deg", type=float, default=None, help="default: backrest behind avatar")
    # mouse
    p.add_argument("--mouse", action="store_true", default=True)
    p.add_argument("--no-mouse", dest="mouse", action="store_false")
    p.add_argument("--mouse-xy", type=float, nargs=2, default=None,
                   help="default: right hand's rightmost-reach xy")
    p.add_argument("--mouse-z-offset", type=float, default=0.012,
                   help="mouse origin above the desk top (m)")
    p.add_argument("--mouse-fwd-offset", type=float, default=0.1,
                   help="shift the mouse toward the table along facing dir (m)")
    p.add_argument("--mouse-yaw-offset-deg", type=float, default=-90.0,
                   help="mouse yaw relative to avatar facing (matches use_mouse)")
    # camera
    p.add_argument("--skip-frames", type=int, default=12)
    p.add_argument("--cam-fwd", type=float, default=2.4)
    p.add_argument("--cam-side", type=float, default=1.5)
    p.add_argument("--cam-height", type=float, default=1.5)
    args = p.parse_args()

    if args.mode == "probe":
        probe(args)
    else:
        render(args)


if __name__ == "__main__":
    main()
