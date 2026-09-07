"""Render the `writing` avatar motion: seated at a desk, writing on a notebook with a pen.

Two modes (run probe first, then render):
  probe  — play the motion headless, record per-frame right-hand frame + root to JSON.
  render — place chair / table / notebook from the probe stats, attach a pen to the
           right hand (near-vertical), replay the motion, save an MP4.

Usage:
  python scripts/render_writing.py --mode probe  --probe-json data/writing/probe.json
  python scripts/render_writing.py --mode render --probe-json data/writing/probe.json \
      -o data/writing/writing.mp4
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

from envs.utils import Pose, load_mesh, load_urdf

CHAIR_URDF = os.path.join(ASSETS_PATH, "objects", "sapien-chair-179", "mobility.urdf")
# Raw (pre-scale) geometry of chair 179 after the URDF's y-up fix: wheels bottom
# z=-0.832, seat-pan top z=-0.046, backrest on the +X side (chair faces -X).
CHAIR_RAW_BOTTOM_Z = 0.832048
CHAIR_RAW_SEAT_H = 0.786  # wheel bottoms -> seat pan top

TABLE_GLB = os.path.join(ASSETS_PATH, "objects", "sapien-table-21467", "visual", "base0.glb")
TABLE_BAKED_EXTENTS = np.array([1.2, 0.7, 0.765])  # baked to top z=0.765 at scale 1

NOTEBOOK_GLB = os.path.join(ASSETS_PATH, "objects", "092_notebook", "visual", "base0.glb")
NOTEBOOK_SCALE = 0.1          # trimesh-verified: 0.190 x 0.125 m, 0.017 thick,
                              # mesh origin on the bottom face (local y=0)

PEN_GLB = os.path.join(ASSETS_PATH, "objects", "058_markpen", "visual", "base0.glb")
PEN_SCALE = 0.1
PEN_LEN = 0.190               # trimesh-verified; long axis local +Y, origin at y=0 end


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


def rot_x(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


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

    rec = {"right_hand": [], "right_hand_R": [], "left_hand": [], "root": []}
    steps = 0
    motion = avatar.motion_data[args.motion]
    while not avatar.spare() and steps < args.max_steps:
        avatar.step()
        scene.step()
        lp, _ = avatar.robot._get_hand_frame(0)
        rp, rR = avatar.robot._get_hand_frame(1)
        root = np.asarray(motion["trans"][min(steps, len(motion["trans"]) - 1)], dtype=float)
        rec["left_hand"].append(np.asarray(lp, dtype=float).tolist())
        rec["right_hand"].append(np.asarray(rp, dtype=float).tolist())
        rec["right_hand_R"].append(np.asarray(rR, dtype=float).reshape(-1).tolist())
        rec["root"].append(root.tolist())
        steps += 1

    os.makedirs(os.path.dirname(args.probe_json) or ".", exist_ok=True)
    json.dump(rec, open(args.probe_json, "w"))

    rh = np.array(rec["right_hand"])
    n = len(rh)
    mid = slice(int(0.15 * n), int(0.9 * n))
    print(f"frames={n}")
    print(f"right hand mid mean: {rh[mid].mean(axis=0)}")
    print(f"right hand z range: {rh[:, 2].min():.3f} .. {rh[:, 2].max():.3f}")
    print(f"right hand mid z mean: {rh[mid, 2].mean():.3f}")
    print(f"right hand xy spread (mid): {rh[mid, :2].std(axis=0)}")
    rR0 = np.array(rec["right_hand_R"][0]).reshape(3, 3)
    print(f"hand frame @0: e1={rR0[:,0]} e2={rR0[:,1]} e3={rR0[:,2]}")
    print(f"root first: {rec['root'][0]}")
    print(f"saved {args.probe_json}")


def placement(args):
    """Compute chair/table/notebook/pen placement from probe stats + CLI overrides."""
    rec = json.load(open(args.probe_json))
    rh = np.array(rec["right_hand"])
    n = len(rh)
    mid = slice(int(0.15 * n), int(0.9 * n))
    hand_mean = rh[mid].mean(axis=0)

    # Avatar pelvis (motion root is y-up: [x, height, y_fwd] -> controller maps to world).
    # Use world-frame seat anchor = avatar origin (reset at 0,0) — probe roots tell facing.
    write_xy = hand_mean[:2] if args.write_xy is None else np.array(args.write_xy, dtype=float)
    table_top = (hand_mean[2] - args.grip_clearance) if args.table_top is None else args.table_top
    return write_xy, float(table_top), hand_mean


def render(args):
    write_xy, table_top, hand_mean = placement(args)
    print(f"[place] write_xy={write_xy} table_top={table_top:.3f} hand_mean={hand_mean}")

    scene = init_scene()

    # --- chair under the avatar -------------------------------------------
    chair = None
    if args.chair:
        cs = args.chair_scale if args.chair_scale else args.seat_height / CHAIR_RAW_SEAT_H
        chair_z = CHAIR_RAW_BOTTOM_Z * cs  # wheels on the floor
        if args.chair_yaw_deg is None:
            # chair faces -X in asset frame; face it toward the writing zone
            facing = np.arctan2(write_xy[1], write_xy[0])
            args.chair_yaw_deg = float(np.rad2deg(facing)) + 180.0
        cyaw = np.deg2rad(args.chair_yaw_deg)
        print(f"[chair] scale={cs:.3f} yaw={args.chair_yaw_deg:.1f} deg")
        cquat = np.array([np.cos(cyaw / 2), 0, 0, np.sin(cyaw / 2)])
        chair = load_urdf(
            scene, CHAIR_URDF,
            Pose([args.chair_xy[0], args.chair_xy[1], chair_z], cquat),
            scale=cs, fix_root=True,
        )

    # --- desk (sapien table glb, baked top z=0.765 at scale 1) -------------
    # Local x = width (across the writer), local y = depth (along the facing
    # direction); place the near edge between the avatar body and the notebook.
    fwd = write_xy / (np.linalg.norm(write_xy) + 1e-8)
    tscale = (
        args.table_size[0] / TABLE_BAKED_EXTENTS[0],
        args.table_size[1] / TABLE_BAKED_EXTENTS[1],
        table_top / TABLE_BAKED_EXTENTS[2],
    )
    table_xy = write_xy + fwd * args.table_fwd + np.array(args.table_offset)
    if args.table_yaw_deg is None:
        # rotate local +y onto the facing direction so depth runs toward the writer
        args.table_yaw_deg = float(np.rad2deg(np.arctan2(write_xy[1], write_xy[0]))) - 90.0
    tyaw = np.deg2rad(args.table_yaw_deg)
    tquat = np.array([np.cos(tyaw / 2), 0, 0, np.sin(tyaw / 2)])
    near_edge = table_xy - fwd * (args.table_size[1] / 2.0)
    print(f"[table] center={table_xy} yaw={args.table_yaw_deg:.1f} near_edge={near_edge}")
    load_mesh(
        scene, TABLE_GLB, Pose([table_xy[0], table_xy[1], 0.0], tquat),
        scale=tscale, is_static=True,
    )

    # --- notebook flat on the desk under the writing hand ------------------
    # GLB long axes are local X/Z, flat axis local Y; rotate +Y -> +Z (Rx 90).
    facing_deg = float(np.rad2deg(np.arctan2(write_xy[1], write_xy[0])))
    nb_yaw = (facing_deg + 90.0) if args.notebook_yaw_deg is None else args.notebook_yaw_deg
    nb_R = rot_z(nb_yaw) @ rot_x(90)
    nb_quat = quat_from_R(nb_R)
    nb_origin_z = table_top + 0.001  # mesh origin is on the bottom face
    nb_xy = write_xy + np.array(args.notebook_offset)
    notebook = load_mesh(
        scene, NOTEBOOK_GLB, Pose([nb_xy[0], nb_xy[1], nb_origin_z], nb_quat),
        scale=args.notebook_scale, is_static=True, collision=False,
    )

    # --- pen (visual only; kinematically attached to the hand) -------------
    pen = load_mesh(
        scene, PEN_GLB, Pose([0, 0, -1.0]),  # parked below floor until attach
        scale=PEN_SCALE, is_static=False, collision=False,
    )

    avatar = build_avatar(scene, args)

    # --- camera: front-side three-quarter view -----------------------------
    look = np.array([write_xy[0] * 0.5, write_xy[1] * 0.5, table_top + 0.05])
    fwd = write_xy / (np.linalg.norm(write_xy) + 1e-8)  # avatar(0,0) -> desk
    side = np.array([-fwd[1], fwd[0]])
    cam_xy = look[:2] + fwd * args.cam_fwd + side * args.cam_side
    cam = scene.add_camera(
        pos=(float(cam_xy[0]), float(cam_xy[1]), args.cam_height),
        lookat=tuple(look.tolist()),
        res=tuple(args.res), fov=45, GUI=False,
    )

    scene.build()

    if chair is not None and chair.n_qs > 0:
        chair_hold = np.zeros((chair.n_qs,), dtype=np.float64)
        chair.set_qpos(chair_hold)
    else:
        chair_hold = None

    avatar.reset(np.zeros(3), np.eye(3))

    # --- warm-up pass: roll to a stable writing frame, attach the pen there ---
    # The first frames of a freshly reset skin interpolate from T-pose, so the
    # hand frame is garbage early on.  Attach mid-clip where the hand pose is
    # genuine, then restart the clip; the hand-local offset persists.
    avatar.play_animation(args.motion)
    for k in range(args.attach_step):
        avatar.step()
        scene.step()
    hand_pos, R_hand = avatar.robot._get_hand_frame(1)
    e2 = R_hand[:, 1].copy()  # finger direction
    e2[2] = 0.0
    e2 /= np.linalg.norm(e2) + 1e-8
    tip = np.array([
        hand_pos[0] + e2[0] * args.pen_fwd,
        hand_pos[1] + e2[1] * args.pen_fwd,
        table_top + 0.002,
    ])
    # Local +Y (pen long axis) -> world up, tilted toward the writer.
    tilt_axis = np.array([-e2[1], e2[0], 0.0])
    pen_R = rot_about(tilt_axis, args.pen_tilt_deg) @ rot_x(90)
    pos = tip.copy()
    if args.pen_flip:
        # writing tip is at the far (local y=PEN_LEN) end: point it down
        pen_R = pen_R @ rot_x(180)
        pos = tip - pen_R @ np.array([0.0, PEN_LEN, 0.0])
    pen.set_pos(pos)
    pen.set_quat(quat_from_R(pen_R))
    avatar.attach_object_to_hand(pen, hand_id=1)
    print(f"[pen] attached @warmup step {args.attach_step} tip={tip} hand={hand_pos}")

    # --- recorded pass ------------------------------------------------------
    avatar.play_animation(args.motion)
    frames = []
    steps = 0
    while not avatar.spare() and steps < args.max_steps:
        avatar.step()
        scene.step()
        if chair_hold is not None:
            chair.set_qpos(chair_hold)
        if steps >= args.skip_record_steps:
            frames.append(render_rgb(cam))
        steps += 1

    if not frames:
        print("Error: no frames produced")
        sys.exit(1)
    save_video(frames, args.output, fps=args.fps)
    print(f"Saved {len(frames)} frames to {args.output}")


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["probe", "render"], required=True)
    p.add_argument("--motion", default="writing")
    p.add_argument("--motion-pkl", default=os.path.join(REPO, "retarget", "output_motion_writing", "motion.pkl"))
    p.add_argument("--probe-json", default=os.path.join(REPO, "data", "writing", "probe.json"))
    p.add_argument("-o", "--output", default=os.path.join(REPO, "data", "writing", "writing.mp4"))
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--res", type=int, nargs=2, default=[960, 720])
    p.add_argument("--max-steps", type=int, default=2000)
    # placement
    p.add_argument("--write-xy", type=float, nargs=2, default=None)
    p.add_argument("--table-top", type=float, default=None)
    p.add_argument("--grip-clearance", type=float, default=0.07,
                   help="hand height above the table top while writing")
    p.add_argument("--table-size", type=float, nargs=2, default=[1.0, 0.6],
                   help="width (across writer), depth (along facing)")
    p.add_argument("--table-fwd", type=float, default=0.08,
                   help="shift table center along the facing direction")
    p.add_argument("--table-offset", type=float, nargs=2, default=[0.0, 0.0],
                   help="extra raw-xy table center offset")
    p.add_argument("--table-yaw-deg", type=float, default=None,
                   help="default: depth axis along the facing direction")
    p.add_argument("--notebook-offset", type=float, nargs=2, default=[0.0, 0.0])
    p.add_argument("--notebook-yaw-deg", type=float, default=None,
                   help="default: long edge across the writing direction")
    p.add_argument("--notebook-scale", type=float, default=0.18,
                   help="0.18 -> ~34x22 cm A4-ish writing pad")
    # chair
    p.add_argument("--chair", action="store_true", default=True)
    p.add_argument("--no-chair", dest="chair", action="store_false")
    p.add_argument("--chair-scale", type=float, default=None,
                   help="default: derived from --seat-height")
    p.add_argument("--seat-height", type=float, default=0.46)
    p.add_argument("--chair-xy", type=float, nargs=2, default=[0.0, 0.0])
    p.add_argument("--chair-yaw-deg", type=float, default=None,
                   help="default: auto-face the writing zone")
    # pen
    p.add_argument("--attach-step", type=int, default=45,
                   help="warm-up steps before capturing the hand frame for attach")
    p.add_argument("--pen-fwd", type=float, default=0.01,
                   help="pen tip offset from palm center along finger direction (m)")
    p.add_argument("--pen-tilt-deg", type=float, default=18.0,
                   help="tilt from vertical, leaning back toward the writer")
    p.add_argument("--pen-flip", action="store_true",
                   help="use if the writing tip turns out to be the far mesh end")
    # camera
    p.add_argument("--skip-record-steps", type=int, default=8,
                   help="drop the first clip frames (settle-in artifacts)")
    p.add_argument("--cam-fwd", type=float, default=2.1)
    p.add_argument("--cam-side", type=float, default=1.3)
    p.add_argument("--cam-height", type=float, default=1.55)
    args = p.parse_args()

    if args.mode == "probe":
        probe(args)
    else:
        render(args)


if __name__ == "__main__":
    main()
