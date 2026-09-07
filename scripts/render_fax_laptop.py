"""Render the using_a_fax avatar motion with a SAPIEN laptop at the operating spot.

Two modes (run probe first, then render):
  probe  — play the motion headless, record per-frame hand/root positions to JSON.
  render — place the laptop (auto from probe JSON, or via explicit overrides),
           replay the motion, save an MP4.

Usage:
  python scripts/render_fax_laptop.py --mode probe  --probe-json data/laptop_fax/probe.json
  python scripts/render_fax_laptop.py --mode render --probe-json data/laptop_fax/probe.json \
      -o data/laptop_fax/using_a_fax.mp4
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

from envs.utils import Pose, create_primitive, load_urdf

LAPTOP_URDF = os.path.join(ASSETS_PATH, "objects", "015_laptop", "9960", "mobility.urdf")
# Pre-scale geometry of model 9960 in its URDF world frame at qpos=0 (lid open),
# measured from the per-link visual meshes. Asset axes: keyboard slab spans
# x in [-0.713, +0.218], top surface z = -0.118, bottom z = -0.223; the screen
# rises at the +x side and its display faces -x. So the asset's -x axis is
# "toward the user".
LAPTOP_RAW_BOTTOM_Z = 0.223
LAPTOP_RAW_KB_TOP_Z = 0.118        # keyboard surface is this far BELOW the origin
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


def build_scene(args, with_laptop, laptop_pose=None, laptop_scale=0.15):
    backend = gs.cpu if os.environ.get("GENESIS_BACKEND", "gpu") == "cpu" else gs.gpu
    gs.init(backend=backend, logging_level="warning")
    scene = gs.Scene(
        show_viewer=False,
        renderer=gs.renderers.Rasterizer(),
        sim_options=gs.options.SimOptions(dt=0.01),
        vis_options=gs.options.VisOptions(ambient_light=(0.4, 0.4, 0.4)),
    )
    scene.add_entity(gs.morphs.Plane())

    laptop = None
    if with_laptop:
        laptop = load_urdf(scene, LAPTOP_URDF, laptop_pose, scale=laptop_scale, fix_root=True)

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
    return scene, avatar, laptop


def probe(args):
    scene, avatar, _ = build_scene(args, with_laptop=False)
    scene.build()
    avatar.reset(np.zeros(3), np.eye(3))
    avatar.play_animation(args.motion)

    rec = {"left_hand": [], "right_hand": [], "hips": []}
    steps = 0
    while not avatar.spare() and steps < args.max_steps:
        avatar.step()
        scene.step()
        lp, _ = avatar.robot._get_hand_frame(0)
        rp, _ = avatar.robot._get_hand_frame(1)
        hips = np.asarray(
            avatar.robot.skin.get_global_translation("Hips")[0], dtype=float
        ).ravel()[:3]
        rec["left_hand"].append(np.asarray(lp, dtype=float).tolist())
        rec["right_hand"].append(np.asarray(rp, dtype=float).tolist())
        rec["hips"].append(hips.tolist())
        steps += 1

    os.makedirs(os.path.dirname(args.probe_json) or ".", exist_ok=True)
    json.dump(rec, open(args.probe_json, "w"))

    lh = np.array(rec["left_hand"]); rh = np.array(rec["right_hand"])
    n = len(lh)
    mid = slice(int(0.25 * n), int(0.85 * n))
    hands_mid = (lh[mid] + rh[mid]) / 2.0
    print(f"frames={n}")
    print(f"hand mid-window mean: {hands_mid.mean(axis=0)}")
    print(f"hand mid-window z range: {hands_mid[:, 2].min():.3f} .. {hands_mid[:, 2].max():.3f}")
    print(f"left  hand overall z range: {lh[:, 2].min():.3f} .. {lh[:, 2].max():.3f}")
    print(f"right hand overall z range: {rh[:, 2].min():.3f} .. {rh[:, 2].max():.3f}")
    print(f"saved {args.probe_json}")


def auto_placement(args):
    """Compute laptop pose + desk slab from the probed operation window.

    All measurements use the operation window (middle of the clip), not the
    init/blend-in frames. `d` is the facing direction hips -> hands in XY.
    Constraints implemented:
      - keyboard typing-area center directly below the mean hand position;
      - display faces the avatar (asset -x axis points back along d);
      - desk near edge between the avatar's body boundary and the hands.
    """
    rec = json.load(open(args.probe_json))
    lh = np.array(rec["left_hand"]); rh = np.array(rec["right_hand"])
    hips_all = np.array(rec["hips"])
    n = len(lh)
    mid = slice(int(0.25 * n), int(0.85 * n))
    hand_mean = ((lh[mid] + rh[mid]) / 2.0).mean(axis=0)
    hips = hips_all[mid].mean(axis=0)

    d = hand_mean[:2] - hips[:2]
    d = d / (np.linalg.norm(d) + 1e-8)
    s = args.laptop_scale

    # Optional rigid transform of the AVATAR (not the laptop): rotate by
    # avatar-yaw-deg (+ = leftward/CCW) about the operation-window hand
    # center — so the typing hands stay put — then step back along the new
    # forward direction. The back offset moves ONLY the avatar; the laptop
    # and desk are placed from the un-shifted hand position, so it directly
    # increases the avatar-to-table standoff.
    av_yaw = np.deg2rad(args.avatar_yaw_deg)
    R2 = np.array([[np.cos(av_yaw), -np.sin(av_yaw)], [np.sin(av_yaw), np.cos(av_yaw)]])
    pivot = hand_mean[:2].copy()
    fwd_av = R2 @ d                                    # avatar forward after yaw
    left_av = np.array([-fwd_av[1], fwd_av[0]])        # avatar's left
    shift = -args.avatar_back_offset * fwd_av + args.avatar_left_offset * left_av
    hips = np.array([*(R2 @ (hips[:2] - pivot) + pivot + shift), hips[2]])
    avatar_transform = {
        "rot": R2,
        "trans": pivot - R2 @ pivot + shift,
    }

    # Asset +x must point AWAY from the avatar (display faces -x).
    # yaw-offset rotates the laptop about its keyboard center, leftward
    # (CCW from above) for positive values, relative to the avatar facing.
    yaw = float(np.arctan2(d[1], d[0]) + np.deg2rad(args.laptop_yaw_offset_deg))
    fwd = np.array([np.cos(yaw), np.sin(yaw)])  # laptop +x in world
    # Keyboard center stays under the hands; back-offset pulls the whole
    # laptop toward the avatar along its facing direction.
    kb_center = hand_mean[:2] - args.laptop_back_offset * d
    xy = kb_center - LAPTOP_RAW_KB_CENTER_X * s * fwd
    origin_z = hand_mean[2] - args.keyboard_clearance + LAPTOP_RAW_KB_TOP_Z * s

    # Desk near edge along d: just under the laptop's nearest base corner,
    # and it must fall between the body boundary and the hands.
    hands_proj = float(hand_mean[:2] @ d)
    body_front = float(hips[:2] @ d) + args.body_half_depth
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    corners = np.array([
        [LAPTOP_RAW_FRONT_X, -0.582], [LAPTOP_RAW_FRONT_X, 0.571],
        [0.218, -0.582], [0.218, 0.571],
    ]) * s
    corner_proj = ((R @ corners.T).T + xy) @ d
    laptop_front = float(corner_proj.min())
    edge = laptop_front - 0.01
    if not (body_front < edge < hands_proj):
        print(f"[place][WARN] desk edge {edge:.3f} not in (body {body_front:.3f}, hands {hands_proj:.3f}); clamping")
        edge = min(max(edge, body_front + 0.02), hands_proj - 0.02)
    print(f"[place] d={d} body_front={body_front:.3f} edge={edge:.3f} "
          f"laptop_front={laptop_front:.3f} hands_proj={hands_proj:.3f}")
    return xy, origin_z, yaw, d, edge, hips, avatar_transform


def render(args):
    xy, origin_z, yaw, d, edge, hips, av_tf = auto_placement(args)
    if args.laptop_xy is not None:
        xy = np.array(args.laptop_xy, dtype=float)
    if args.laptop_z is not None:
        origin_z = float(args.laptop_z)
    if args.laptop_yaw_deg is not None:
        yaw = float(np.deg2rad(args.laptop_yaw_deg))

    scale = args.laptop_scale
    quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    laptop_pose = Pose([xy[0], xy[1], origin_z], quat)
    desk_top = origin_z - LAPTOP_RAW_BOTTOM_Z * scale
    print(f"[place] laptop xy={xy} origin_z={origin_z:.3f} yaw={np.rad2deg(yaw):.1f} deg desk_top={desk_top:.3f}")

    scene, avatar, laptop = build_scene(args, with_laptop=True, laptop_pose=laptop_pose, laptop_scale=scale)

    if args.desk and desk_top > 0.03:
        # Desk slab yawed to the avatar facing d (not the laptop yaw); its
        # near face sits exactly at `edge`, extending away from the avatar.
        desk_yaw = float(np.arctan2(d[1], d[0]))
        desk_quat = np.array([np.cos(desk_yaw / 2), 0, 0, np.sin(desk_yaw / 2)])
        lateral = xy - (xy @ d) * d
        desk_center = lateral + (edge + args.desk_half_x) * d
        create_primitive(
            scene, "box",
            Pose([desk_center[0], desk_center[1], desk_top / 2.0], desk_quat),
            size={"half_size": (args.desk_half_x, args.desk_half_y, desk_top / 2.0)},
            color=(0.55, 0.42, 0.30),
            is_static=True,
        )

    # Camera: front-side three-quarter view (in front of the avatar, off-axis).
    side = np.array([-d[1], d[0]])
    look = np.array([(hips[0] + xy[0]) / 2, (hips[1] + xy[1]) / 2, 0.95])
    cam_xy = look[:2] + d * 2.6 + side * 1.6
    cam = scene.add_camera(
        pos=(float(cam_xy[0]), float(cam_xy[1]), 1.7),
        lookat=tuple(look.tolist()),
        res=tuple(args.res), fov=45, GUI=False,
    )
    cam_top = None
    if args.topdown_output:
        # Straight-down view over the keyboard (small offset avoids a
        # degenerate up vector) to verify hands-over-keyboard alignment.
        cam_top = scene.add_camera(
            pos=(float(xy[0] + 0.10 * d[0]), float(xy[1] + 0.10 * d[1]), origin_z + 1.0),
            lookat=(float(xy[0]), float(xy[1]), float(origin_z)),
            res=tuple(args.res), fov=45, GUI=False,
        )

    scene.build()
    if laptop is not None and laptop.n_qs > 0:
        qpos_hold = np.full((laptop.n_qs,), args.lid_qpos, dtype=np.float64)
        laptop.set_qpos(qpos_hold)

    R3 = np.eye(3)
    R3[:2, :2] = av_tf["rot"]
    t3 = np.array([av_tf["trans"][0], av_tf["trans"][1], 0.0])
    avatar.reset(t3, R3)
    avatar.play_animation(args.motion)

    frames = []
    frames_top = []
    steps = 0
    while not avatar.spare() and steps < args.max_steps:
        avatar.step()
        scene.step()
        if laptop is not None and laptop.n_qs > 0:
            laptop.set_qpos(qpos_hold)  # hold the lid against gravity
        if steps >= args.skip_frames:  # drop the initial T-pose blend-in
            frames.append(render_rgb(cam))
            if cam_top is not None:
                frames_top.append(render_rgb(cam_top))
        steps += 1

    if not frames:
        print("Error: no frames produced")
        sys.exit(1)
    save_video(frames, args.output, fps=args.fps)
    print(f"Saved {len(frames)} frames to {args.output}")
    if frames_top:
        save_video(frames_top, args.topdown_output, fps=args.fps)
        print(f"Saved {len(frames_top)} frames to {args.topdown_output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["probe", "render"], required=True)
    p.add_argument("--motion", default="using_a_fax")
    p.add_argument("--motion-pkl", default=os.path.join(REPO, "retarget", "output_motion_fax", "motion.pkl"))
    p.add_argument("--probe-json", default=os.path.join(REPO, "data", "laptop_fax", "probe.json"))
    p.add_argument("-o", "--output", default=os.path.join(REPO, "data", "laptop_fax", "using_a_fax.mp4"))
    p.add_argument("--topdown-output", default=None,
                   help="optional second video from a straight-down camera over the keyboard")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--res", type=int, nargs=2, default=[960, 720])
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--laptop-scale", type=float, default=0.15)
    p.add_argument("--laptop-xy", type=float, nargs=2, default=None)
    p.add_argument("--laptop-z", type=float, default=None, help="override laptop origin z")
    p.add_argument("--laptop-yaw-deg", type=float, default=None)
    p.add_argument("--laptop-yaw-offset-deg", type=float, default=0.0,
                   help="extra laptop yaw relative to avatar facing; + = leftward/CCW")
    p.add_argument("--laptop-back-offset", type=float, default=0.0,
                   help="pull the laptop toward the avatar by this many meters")
    p.add_argument("--avatar-yaw-deg", type=float, default=0.0,
                   help="rotate the avatar (+ = leftward/CCW) about the typing-hand center")
    p.add_argument("--avatar-back-offset", type=float, default=0.0,
                   help="step ONLY the avatar backward (laptop/desk stay) — increases avatar-to-table standoff")
    p.add_argument("--avatar-left-offset", type=float, default=0.0,
                   help="step ONLY the avatar to its left (laptop/desk stay) by this many meters")
    p.add_argument("--lid-qpos", type=float, default=0.0)
    p.add_argument("--keyboard-clearance", type=float, default=0.04)
    p.add_argument("--body-half-depth", type=float, default=0.12,
                   help="pelvis-to-front body boundary used for the desk edge constraint")
    p.add_argument("--desk", action="store_true", default=True)
    p.add_argument("--no-desk", dest="desk", action="store_false")
    p.add_argument("--desk-half-x", type=float, default=0.30, help="desk half-depth along facing dir")
    p.add_argument("--desk-half-y", type=float, default=0.30, help="desk half-width lateral")
    p.add_argument("--skip-frames", type=int, default=12)
    args = p.parse_args()

    if args.mode == "probe":
        probe(args)
    else:
        render(args)


if __name__ == "__main__":
    main()
