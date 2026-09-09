"""Render the `using_a_filing_cabinet` motion with a real filing cabinet scene.

Scene story (frame indices from the converted Mixamo clip, 359 frames @30fps):
  f0-37    avatar walks up to the cabinet
  f42-46   right hand hooks the top-drawer handle (fingertips x~0.87-0.895, z~1.24)
  f46-64   pulls the drawer open ~0.19 m (drawer qpos tracks the fingertip x)
  f55-182  searches through the hanging files in the open drawer
  f182     grips the top of the red target file with index+middle fingertips
           -> file attaches to the right hand (kinematic avatar attach)
  f230-290 raises the file and looks at it
  f300+    walks away carrying the file

Cabinet: assets/objects/036_cabinet/46653 (3-drawer filing cabinet, prismatic
drawers sliding along world -X under identity quat). Scaled 0.806 so the top
drawer handle sits at the motion's grab point (x=0.895, z~1.24).

The drawer + decor files are kinematically driven (non-robot articulation /
props in a render-only avatar scene); the target file rides the drawer until
the avatar attach takes over at f182.

Usage (GPU node for EGL, CPU genesis backend):
    python scripts/cabinet_filing/render_filing_cabinet_scene.py \
        -o data/cabinet/scene_v1
"""
import argparse
import json
import os
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
os.environ.setdefault("GENESIS_BACKEND", "cpu")

import numpy as np
import genesis as gs

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
ASSETS_PATH = os.path.join(REPO, "assets")
from envs.genesis_compat import set_dofs_kp_kv_compat  # noqa: E402

from envs.utils import Pose, create_primitive  # noqa: E402

MOTION_NAME = "using_a_filing_cabinet"
MOTION_PKL = os.path.join(REPO, "retarget", "output_motion_cabinet", "motion.pkl")
TRAJ_JSON = os.path.join(REPO, "data", "cabinet", "hand_traj.json")

# ---- table (SAPIEN visual mesh, baked to 1.2 x 0.7 x 0.765 at scale 1) --
TABLE_MESH = os.path.join(ASSETS_PATH, "objects", "sapien-table-20279", "visual", "base0.glb")
TABLE_TOP_Z = 0.765
# long axis rotated to world Y: footprint x half 0.35, y half 0.6; the front
# edge at x=0.69 stays clear of the avatar's lean (body front ~0.62).
TABLE_XY = (1.04, 0.12)
TABLE_QUAT = (0.70710678, 0.0, 0.0, 0.70710678)   # Rz(90)

# ---- avatar ----
AVATAR_Z = 0.10                     # raise the avatar (user request) -> all hand z +0.10

# ---- cabinet placement (desktop size, standing on the table) ------------
# URDF local spans at scale 1 (probed): body x[-0.385,+0.433] z[-0.803,+0.811],
# closed drawer front (incl. handle) x=-0.465; handle bar ~1.538 above bottom.
# Scale chosen so the table-standing cabinet's top-drawer handle lands at the
# raised motion's hook point z~1.34: 0.765 + 1.538*s = 1.34 -> s = 0.374.
CAB_SCALE = 0.374
CAB_X = 1.069                       # closed handle at x = CAB_X - 0.465*s = 0.895
CAB_Y = 0.12
CAB_Z = TABLE_TOP_Z + 0.803 * CAB_SCALE
HANDLE_CLOSED_X = CAB_X - 0.465 * CAB_SCALE   # ~0.895
BODY_FRONT_X = CAB_X - 0.385 * CAB_SCALE      # ~0.925

# ---- drawer opening profile --------------------------------------------
PULL_START, PULL_END = 44, 64       # frames where the hand drags the handle
Q_MAX = 0.195                       # joint limit at this scale is 0.247
SUBSTEPS = 4                        # sim substeps/frame so friction can carry the file

# ---- files in the top drawer -------------------------------------------
# Open-drawer x positions (after the pull completes). The target file sits at
# the pick fingertip x (~0.905). Decor folders fill the exposed cavity
# (open tub interior x ~[0.73, 0.95], y ~[-0.02, 0.26], floor ~1.175).
TUB_FLOOR_Z = 1.175
FOLDER_HALF_Y = 0.10
FOLDER_Y = 0.12                     # folder centerline (drawer is at CAB_Y)
DECOR_OPEN_X = [0.755, 0.785, 0.815, 0.845, 0.875]
DECOR_TOP_Z = [1.29, 1.26, 1.30, 1.27, 1.285]
TARGET_OPEN_X = 0.905               # pick fingertips close at (0.90, 0.02, ~1.31)
TARGET_Y = 0.085
TARGET_HALF = (0.0125, 0.08, 0.065)  # 2.5 x 16 x 13 cm document, dynamic
ATTACH_FRAME = 182

MANILA = (0.83, 0.71, 0.46, 1.0)
MANILA2 = (0.78, 0.66, 0.42, 1.0)
RED = (0.78, 0.13, 0.10, 1.0)


def render_rgb(cam):
    out = cam.render(rgb=True, depth=False)
    rgb = out[0] if isinstance(out, (list, tuple)) else out
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu().numpy()
    rgb = np.asarray(rgb)
    if rgb.dtype in (np.float32, np.float64):
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return rgb.astype(np.uint8)


def stamp_frame_no(frame, n):
    import cv2

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.putText(bgr, f"{n:04d}", (10, 34), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_video(frames, path, fps=30):
    import imageio

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    w = imageio.get_writer(path, fps=fps)
    for f in frames:
        w.append_data(f)
    w.close()
    print(f"saved {len(frames)} frames -> {path}")


def drawer_open_profile(n_frames):
    """qpos per frame: track the right-index fingertip x during the pull."""
    with open(TRAJ_JSON) as f:
        traj = json.load(f)
    q = np.zeros(n_frames)
    for f in range(PULL_START, min(PULL_END + 1, n_frames)):
        tip = traj[f].get("RightHandIndex3")
        if tip is None:
            continue
        q[f] = np.clip(HANDLE_CLOSED_X - tip[0], 0.0, Q_MAX)
    # monotonic while pulling, hold after
    q[: PULL_END + 1] = np.maximum.accumulate(q[: PULL_END + 1])
    if PULL_END + 1 < n_frames:
        q[PULL_END + 1:] = q[PULL_END]
    return q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", default=os.path.join(REPO, "data", "cabinet", "scene_v1"))
    parser.add_argument("--res", type=int, nargs=2, default=[960, 720])
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-render", action="store_true",
                        help="physics/placement probe only (no EGL needed)")
    args = parser.parse_args()

    backend = gs.gpu if os.environ.get("GENESIS_BACKEND") == "gpu" else gs.cpu
    gs.init(backend=backend, logging_level="warning")

    scene = gs.Scene(
        show_viewer=False,
        renderer=gs.renderers.Rasterizer(),
        sim_options=gs.options.SimOptions(dt=0.01),
        vis_options=gs.options.VisOptions(ambient_light=(0.35, 0.35, 0.35)),
    )
    scene.add_entity(gs.morphs.Plane())

    # ---- table ----
    from envs.utils import load_mesh

    load_mesh(
        scene, TABLE_MESH,
        Pose([TABLE_XY[0], TABLE_XY[1], 0.0], TABLE_QUAT),
        scale=1.0,
        is_static=True, convex=False, collision=False, visual=True,
    )

    # ---- cabinet ----
    urdf = os.path.join(ASSETS_PATH, "objects", "036_cabinet", "46653", "mobility.urdf")
    cabinet = scene.add_entity(
        gs.morphs.URDF(
            file=urdf,
            pos=(CAB_X, CAB_Y, CAB_Z),
            quat=(1.0, 0.0, 0.0, 0.0),
            scale=CAB_SCALE,
            fixed=True,
        )
    )

    # ---- avatar ----
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
        generated_motion_path=MOTION_PKL,
    )
    assert MOTION_NAME in avatar.motion_data, list(avatar.motion_data.keys())[:10]
    n_frames = int(avatar.motion_data[MOTION_NAME]["trans"].shape[0])

    q_profile = drawer_open_profile(n_frames)
    q_final = float(q_profile[-1])
    print(f"n_frames={n_frames} drawer opens to {q_final:.3f} m")

    # ---- files (closed-drawer positions = open + q_final) ----
    decor = []
    for i, (ox, top) in enumerate(zip(DECOR_OPEN_X, DECOR_TOP_Z)):
        half_h = (top - TUB_FLOOR_Z) / 2.0
        ent = create_primitive(
            scene, "box",
            Pose([ox + q_final, FOLDER_Y, (top + TUB_FLOOR_Z) / 2.0]),
            size={"half_size": (0.006, FOLDER_HALF_Y, half_h)},
            color=MANILA if i % 2 == 0 else MANILA2,
            is_static=True, collision=False,
        )
        decor.append((ent, ox + q_final, (top + TUB_FLOOR_Z) / 2.0))

    # Target document: a real dynamic body resting on the tub floor. It rides
    # the drawer pull through contact friction (no scripted motion) and is
    # only touched again by the standard hand attach at ATTACH_FRAME.
    target_closed_x = TARGET_OPEN_X + q_final
    target = scene.add_entity(
        gs.morphs.Box(
            size=tuple(2 * h for h in TARGET_HALF),
            pos=(target_closed_x, TARGET_Y, TUB_FLOOR_Z + TARGET_HALF[2] + 0.003),
            quat=(1.0, 0.0, 0.0, 0.0),
            fixed=False, collision=True, visualization=True,
        ),
        material=gs.materials.Rigid(friction=4.0),
        surface=gs.surfaces.Default(color=RED),
    )

    # ---- cameras (front quadrant: cabinet front faces -X) ----
    cam_main = scene.add_camera(
        pos=(0.0, -2.4, 1.9), lookat=(0.85, 0.05, 1.0),
        res=tuple(args.res), fov=45, GUI=False,
    )
    cam_close = scene.add_camera(
        pos=(0.25, -1.30, 1.95), lookat=(0.88, 0.10, 1.12),
        res=tuple(args.res), fov=40, GUI=False,
    )

    scene.build()

    # top drawer dof (joint_3 / link_3 — highest z)
    dof_idx = None
    for j in cabinet.joints:
        if str(j.name) == "joint_3":
            dof_idx = int(getattr(j, "dof_idx_local", None) if getattr(j, "dof_idx_local", None) is not None else j.dof_start)
            break
    assert dof_idx is not None, [str(j.name) for j in cabinet.joints]
    # PD-drive the drawer joints (real joint velocity -> contact friction can
    # transport the document; a set_qpos teleport has zero velocity and
    # leaves contents behind).
    all_dofs = list(range(cabinet.n_dofs))
    set_dofs_kp_kv_compat(
        cabinet,
        kp=np.full(cabinet.n_dofs, 2000.0),
        kv=np.full(cabinet.n_dofs, 200.0),
        dofs_idx_local=all_dofs,
    )
    qvec = np.zeros(cabinet.n_dofs)

    # Let the document settle onto the tub floor before the motion starts.
    for _ in range(60):
        scene.step()

    avatar.reset(np.array([0.0, 0.0, AVATAR_Z]), np.eye(3, dtype=np.float64))
    avatar.play_animation(
        MOTION_NAME,
        attach_obj=target,
        hand_id=1,
        attach_frame=ATTACH_FRAME,
    )

    frames_main, frames_close = [], []
    steps = 0
    q_prev = 0.0
    while not avatar.spare() and steps < n_frames + 50:
        f = min(steps, n_frames - 1)
        dq = float(q_profile[f])
        for ent, cx, cz in decor:
            ent.set_pos(np.array([cx - dq, FOLDER_Y, cz]))

        avatar.step()
        # PD the drawer toward the hand-tracked target with sub-steps so the
        # document rides the moving drawer through real contact friction.
        # After the attach the hand machinery set_pos-es the document once per
        # frame, so run a single step to avoid gravity sag between updates.
        n_sub = SUBSTEPS if steps < ATTACH_FRAME else 1
        for s in range(n_sub):
            qvec[dof_idx] = q_prev + (dq - q_prev) * (s + 1) / n_sub
            cabinet.control_dofs_position(qvec, all_dofs)
            scene.step()
        q_prev = dq

        if steps in (0, PULL_START, PULL_END, 80, 120, 160, ATTACH_FRAME - 1, ATTACH_FRAME):
            tp = target.get_pos()
            tp = np.asarray(tp.cpu() if hasattr(tp, "cpu") else tp).ravel()
            tq = target.get_quat()
            tq = np.asarray(tq.cpu() if hasattr(tq, "cpu") else tq).ravel()
            print(f"f{steps}: target pos {np.round(tp, 3)} quat {np.round(tq, 3)}")

        if not args.no_render:
            frames_main.append(stamp_frame_no(render_rgb(cam_main), steps))
            frames_close.append(stamp_frame_no(render_rgb(cam_close), steps))
        steps += 1

    if not args.no_render:
        out = args.output_dir
        save_video(frames_main, os.path.join(out, "filing_cabinet_main.mp4"), fps=args.fps)
        save_video(frames_close, os.path.join(out, "filing_cabinet_close.mp4"), fps=args.fps)


if __name__ == "__main__":
    main()
