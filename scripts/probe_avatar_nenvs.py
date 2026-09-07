"""Feasibility probe: can the kinematic avatar skin diverge PER-ENV under
Genesis n_envs > 0?

The HRI-batching question reduces to one mechanism: the avatar skin is drawn
by rewriting a custom visual-vertex buffer every step
(KinematicAvatarSkin.update_mesh -> entity.set_vverts).  Genesis 1.2.0
documents an ``envs_idx`` arg on set_vverts/get_vverts, which would mean the
buffer is per-env and N envs can each show a different avatar pose.  This
probe verifies that end-to-end with the benchmark's real avatar stack
(AvatarController + KinematicAvatarSkin + box proxy):

  V1  scene containing the avatar builds with n_envs=2
  V2  set_vverts(pose_A, envs_idx=[0]) / set_vverts(pose_B, envs_idx=[1])
      round-trips: get_vverts differs across envs and matches what was written
  V3  per-env cameras render visibly different avatar poses
  V4  the per-env buffers survive scene.step() (physics doesn't clobber them)
  V5  (info) today's update_mesh() with no envs_idx broadcasts one pose to
      all envs — the integration change a batched avatar would need

Run on a GPU node (gs.gpu backend + rasterizer EGL):
  sbatch ... scripts/probe_avatar_nenvs.py
Outputs: data/parallel/avatar_nenvs_probe/{env0,env1}*.png + result.json
Sentinel: PROBE_AVATAR_NENVS_PASS / PROBE_AVATAR_NENVS_FAIL
"""

import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUT_DIR = os.path.join(
    "data", "parallel",
    "avatar_nenvs_probe" + os.environ.get("PROBE_OUT_SUFFIX", ""),
)
N_ENVS = 2
POSE_B_SHIFT = np.array([0.6, 0.0, 0.0])
POSE_B_YAW_DEG = 90.0

results = {"checks": {}, "notes": []}


def record(name, ok, detail):
    results["checks"][name] = {"ok": bool(ok), "detail": detail}
    print(f"[probe] {name}: {'PASS' if ok else 'FAIL'} — {detail}", flush=True)


def tnp(x):
    """Torch tensor (any device) or array-like -> numpy."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def save_rgb(rgb, path):
    from PIL import Image

    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr[..., :3]).save(path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    import genesis as gs
    from scipy.spatial.transform import Rotation as R

    from envs.genesis_compat import supported_kwargs
    from envs.utils import ASSETS_PATH

    gs.init(backend=gs.gpu, logging_level="warning")

    vis_kwargs = supported_kwargs(
        gs.options.VisOptions,
        {"env_separate_rigid": True, "rendered_envs_idx": list(range(N_ENVS))},
    )
    results["notes"].append(f"VisOptions kwargs accepted: {sorted(vis_kwargs)}")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.002),
        vis_options=gs.options.VisOptions(**vis_kwargs),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    # Rigid marker cube: sanity-check that ordinary rigid state diverges
    # per-env too (and gives the render diff an unambiguous landmark).
    marker = scene.add_entity(
        gs.morphs.Box(size=(0.15, 0.15, 0.15), pos=(0.0, 0.9, 0.6)),
        surface=gs.surfaces.Default(color=(0.9, 0.1, 0.1)),
    )

    from envs.avatar import AvatarController

    avatar = AvatarController(
        scene=scene,
        motion_data_path="avatars/motions/motion.pkl",
        skin_options={
            "glb_path": "avatars/models/custom_Adrian_Keller.glb",
            "euler": (-90, 0, 90),
            "pos": (0.0, 0.0, -0.959008030),
        },
        frame_ratio=1.0,
        name="human",
        assets_dir=str(ASSETS_PATH),
        generated_motion_path="avatars/motions/generated_motions.pkl",
    )

    # ONE batched camera (env_idx=None) rendering all rendered_envs_idx.
    # Genesis 1.2.0 bug: cameras bound to a specific env_idx in a batched
    # env_separate_rigid scene allocate unbatched pose buffers but set_pose
    # expands pos to (n_rendered_envs, 3) -> crash at camera.build().
    # The batched-camera path is consistent and returns stacked frames.
    cam = scene.add_camera(
        res=(640, 480), pos=(2.6, -2.6, 2.2), lookat=(0.3, 0.0, 1.0),
        fov=45, GUI=False,
    )

    # ---- V1: build with n_envs ----
    # env_spacing is visual-only (physics poses unchanged). It is REQUIRED for
    # per-env avatar rendering: the rasterizer keeps every env's custom-vverts
    # skin node visible in the one shared pyrender scene, each drawn at
    # vverts + envs_offset[env]. With zero spacing all envs' skins overlap and
    # every camera sees all of them; with real spacing each env camera
    # (offset the same way) sees only its own env's skin.
    spacing = float(os.environ.get("PROBE_ENV_SPACING", "8.0"))
    results["notes"].append(f"env_spacing={spacing}")
    try:
        scene.build(n_envs=N_ENVS, env_spacing=(spacing, spacing))
        record("V1_build_n_envs", True, f"scene.build(n_envs={N_ENVS}) with avatar + skin OK")
    except Exception as e:
        record("V1_build_n_envs", False, f"{type(e).__name__}: {e}")
        raise

    robot = avatar.robot
    skin = robot.skin
    ent = getattr(skin, "_entity", skin)

    # Pose A: stop pose at origin (update() broadcasts it to all envs).
    robot.reset(np.zeros(3), np.eye(3))
    pose_a = skin._latest_render_vverts.copy()

    # Pose B: translated + yawed stop pose.
    rot_b = R.from_euler("z", POSE_B_YAW_DEG, degrees=True).as_matrix()
    robot.reset(POSE_B_SHIFT.copy(), rot_b)
    pose_b = skin._latest_render_vverts.copy()

    expected_disp = float(np.linalg.norm(pose_a - pose_b, axis=1).mean())
    results["notes"].append(f"expected mean vert displacement A vs B: {expected_disp:.3f} m")

    # ---- V2: per-env write + round-trip ----
    try:
        ent.set_vverts(pose_a, envs_idx=[0])
        ent.set_vverts(pose_b, envs_idx=[1])
        va0 = tnp(ent.get_vverts(envs_idx=[0])).reshape(-1, 3)
        va1 = tnp(ent.get_vverts(envs_idx=[1])).reshape(-1, 3)
        cross = float(np.linalg.norm(va0 - va1, axis=1).mean())
        rt0 = float(np.abs(va0 - pose_a).max())
        rt1 = float(np.abs(va1 - pose_b).max())
        ok = cross > 0.5 * expected_disp and rt0 < 0.01 and rt1 < 0.01
        record(
            "V2_per_env_vverts",
            ok,
            f"cross-env mean disp {cross:.3f} m (expected {expected_disp:.3f}), "
            f"round-trip max err env0 {rt0:.2e} / env1 {rt1:.2e}",
        )
    except TypeError as e:
        record("V2_per_env_vverts", False, f"set_vverts rejected envs_idx: {e}")
        raise

    # Box proxy + marker cube per env (rigid path).
    try:
        robot.box.set_pos(np.array([[0.0, 0.0, 0.959]]), envs_idx=[0])
        robot.box.set_pos(np.array([[*(POSE_B_SHIFT + [0, 0, 0.959])]]), envs_idx=[1])
        marker.set_pos(np.array([[0.0, 0.9, 0.6]]), envs_idx=[0])
        marker.set_pos(np.array([[0.0, -0.9, 0.6]]), envs_idx=[1])
        record("V2b_rigid_per_env", True, "box + marker set_pos(envs_idx) accepted")
    except Exception as e:
        record("V2b_rigid_per_env", False, f"{type(e).__name__}: {e}")

    # ---- V3: per-env renders differ ----
    def render_frames(tag):
        rgb = tnp(cam.render()[0])
        results["notes"].append(f"render[{tag}] output shape: {rgb.shape}")
        if rgb.ndim == 4:
            out = [rgb[i] for i in range(min(N_ENVS, rgb.shape[0]))]
        else:
            out = [rgb]  # camera not batched: only one env visible
        for i, fr in enumerate(out):
            save_rgb(fr, os.path.join(OUT_DIR, f"env{i}{tag}.png"))
        return out

    def dump_nodes(tag):
        """Ground truth: what vverts does each per-env pyrender node hold?"""
        try:
            ctx = getattr(scene.visualizer, "context", None) or scene.visualizer._context
            offs = tnp(scene.envs_offset)
            note = [f"envs_offset={offs.tolist()}",
                    f"A_mean={pose_a.mean(0).round(3).tolist()}",
                    f"B_mean={pose_b.mean(0).round(3).tolist()}"]
            for (i_b, uid), node in sorted(ctx.vverts_nodes.items(), key=str):
                m = np.asarray(node.mesh.primitives[0].positions).mean(axis=0)
                note.append(f"node(env={i_b},{str(uid)[:6]}) mean={m.round(3).tolist()}")
            results["notes"].append(f"nodes[{tag}]: " + " | ".join(note))
            print(f"[probe] nodes[{tag}]: " + " | ".join(note), flush=True)
        except Exception as e:
            results["notes"].append(f"nodes[{tag}] dump failed: {type(e).__name__}: {e}")

    frames = render_frames("")
    dump_nodes("first_render")
    if len(frames) >= 2:
        pix = float(np.abs(frames[0].astype(np.float64) - frames[1].astype(np.float64)).mean())
        record("V3_render_divergence", pix > 1.0, f"mean abs pixel diff env0 vs env1: {pix:.2f}")
    else:
        record("V3_render_divergence", False,
               "camera returned a single unbatched frame; cannot compare envs")

    # Re-write per-env poses now that GL buffers exist, then render again:
    # isolates the node-UPDATE path from the node-SEED path.
    ent.set_vverts(pose_a, envs_idx=[0])
    ent.set_vverts(pose_b, envs_idx=[1])
    frames2 = render_frames("_rewrite")
    dump_nodes("after_rewrite")
    if len(frames2) >= 2:
        pix2 = float(np.abs(frames2[0].astype(np.float64) - frames2[1].astype(np.float64)).mean())
        record("V3b_render_after_rewrite", pix2 > 1.0,
               f"mean abs pixel diff after re-write+re-render: {pix2:.2f}")

    # ---- V3z: node census + step-refresh + broadcast-render recipe ----
    try:
        ctx = getattr(scene.visualizer, "context", None) or scene.visualizer._context
        pysc = ctx._scene
        results["notes"].append(f"pyrender scene type: {type(pysc).__module__}.{type(pysc).__name__}")
        known = {id(n) for n in ctx.vverts_nodes.values()}
        census = []
        for node in list(getattr(pysc, "nodes", [])):
            mesh = getattr(node, "mesh", None)
            if mesh is None:
                continue
            try:
                pts = np.asarray(mesh.primitives[0].positions)
                if pts.ndim != 2 or len(pts) < 3000:
                    continue
                census.append(
                    f"n_verts={len(pts)} mean={pts.mean(0).round(2).tolist()} "
                    f"vverts_reg={id(node) in known} visible={mesh.is_visible} "
                    f"matrix_ndim={np.asarray(node.matrix).ndim}"
                )
            except Exception as e:
                census.append(f"census err {type(e).__name__}")
        results["notes"].append("dense nodes: " + " || ".join(census))
        print("[probe] node census: " + " || ".join(census), flush=True)

        # Does a scene.step() make per-env vverts updates reach the GL buffers?
        ent.set_vverts(pose_a, envs_idx=[0])
        ent.set_vverts(pose_b, envs_idx=[1])
        scene.step()
        f_step = render_frames("_step_refresh")
        if len(f_step) >= 2:
            pix_s = float(np.abs(f_step[0].astype(np.float64) - f_step[1].astype(np.float64)).mean())
            record("V3z_step_refresh", pix_s > 1.0,
                   f"pixel diff after per-env write + scene.step + render: {pix_s:.2f}")

        # Fallback recipe: broadcast env i's pose to ALL envs, step, render, keep
        # frame i. Serial rendering, batched physics; works iff the visible skin
        # tracks broadcast set_vverts.
        seq = []
        for i, pose in enumerate((pose_a, pose_b)):
            ent.set_vverts(pose)  # broadcast
            scene.step()
            fr = render_frames(f"_seq{i}")
            seq.append(fr[min(i, len(fr) - 1)])
        pix_q = float(np.abs(seq[0].astype(np.float64) - seq[1].astype(np.float64)).mean())
        record("V3z_seq_broadcast", pix_q > 1.0,
               f"sequential broadcast recipe: frame0-vs-frame1 pixel diff {pix_q:.2f}")
    except Exception as e:
        record("V3z_diag", False, f"{type(e).__name__}: {e}")

    # ---- V3c: env-local compensation + stale-node removal ----
    # Batched rasterizer passes render in ENV-LOCAL coords (camera transform
    # gets no env offset when env_separate_rigid=True), but Genesis places
    # per-env vverts nodes at vverts + envs_offset -> off-camera. Fix 1: write
    # vverts pre-shifted by -envs_offset. Fix 2: the skin's build-time node
    # (init pose) survives in the shared pyrender scene -> identical figure in
    # every frame; find nodes NOT in vverts_nodes that look like the skin and
    # hide them.
    try:
        ctx = getattr(scene.visualizer, "context", None) or scene.visualizer._context
        offs = tnp(scene.envs_offset)
        known = {id(n) for n in ctx.vverts_nodes.values()}
        hidden = 0
        for node in list(getattr(ctx._scene, "mesh_nodes", [])):
            if id(node) in known or node.mesh is None:
                continue
            try:
                pts = np.asarray(node.mesh.primitives[0].positions)
            except Exception:
                continue
            if pts is None or pts.ndim != 2 or len(pts) < 5000:
                continue  # skin groups are dense; skip floor/cube/etc.
            m = pts.mean(axis=0)
            if np.linalg.norm(m[:2]) < 2.0 and 0.0 < m[2] < 2.0:  # near env-local origin, human height
                node.mesh.is_visible = False
                hidden += 1
        results["notes"].append(f"stale-node hunt: hid {hidden} dense unregistered mesh nodes")

        ent.set_vverts(pose_a - offs[0], envs_idx=[0])
        ent.set_vverts(pose_b - offs[1], envs_idx=[1])
        frames3 = render_frames("_envlocal")
        if len(frames3) >= 2:
            pix3 = float(np.abs(frames3[0].astype(np.float64) - frames3[1].astype(np.float64)).mean())
            record("V3c_envlocal_fix", pix3 > 1.0,
                   f"hidden stale nodes={hidden}; pixel diff with -envs_offset compensation: {pix3:.2f}")
        # restore uncompensated state for V4
        ent.set_vverts(pose_a, envs_idx=[0])
        ent.set_vverts(pose_b, envs_idx=[1])
        va0 = tnp(ent.get_vverts(envs_idx=[0])).reshape(-1, 3)
        va1 = tnp(ent.get_vverts(envs_idx=[1])).reshape(-1, 3)
    except Exception as e:
        record("V3c_envlocal_fix", False, f"{type(e).__name__}: {e}")

    # ---- V4: buffers survive physics steps ----
    for _ in range(5):
        scene.step()
    vb0 = tnp(ent.get_vverts(envs_idx=[0])).reshape(-1, 3)
    vb1 = tnp(ent.get_vverts(envs_idx=[1])).reshape(-1, 3)
    drift0 = float(np.abs(vb0 - va0).max())
    drift1 = float(np.abs(vb1 - va1).max())
    render_frames("_after_step")
    record(
        "V4_survives_step",
        drift0 < 1e-4 and drift1 < 1e-4,
        f"max vvert drift after 5 scene.step(): env0 {drift0:.2e}, env1 {drift1:.2e}",
    )

    # ---- V5 (info): today's broadcast path clobbers all envs ----
    robot.reset(np.zeros(3), np.eye(3))  # update_mesh with no envs_idx
    vc0 = tnp(ent.get_vverts(envs_idx=[0])).reshape(-1, 3)
    vc1 = tnp(ent.get_vverts(envs_idx=[1])).reshape(-1, 3)
    same = float(np.linalg.norm(vc0 - vc1, axis=1).mean())
    record(
        "V5_broadcast_info",
        True,
        f"update_mesh() without envs_idx: cross-env disp now {same:.4f} m "
        f"({'broadcasts to all envs as expected' if same < 0.01 else 'UNEXPECTED: envs still differ'})",
    )

    core = ["V1_build_n_envs", "V2_per_env_vverts", "V4_survives_step"]
    render_ok = any(
        results["checks"].get(k, {}).get("ok")
        for k in ("V3_render_divergence", "V3b_render_after_rewrite", "V3c_envlocal_fix",
                  "V3z_step_refresh", "V3z_seq_broadcast")
    )
    all_ok = all(results["checks"][k]["ok"] for k in core) and render_ok
    results["verdict"] = "PASS" if all_ok else "FAIL"
    with open(os.path.join(OUT_DIR, "result.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"PROBE_AVATAR_NENVS_{results['verdict']}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        try:
            os.makedirs(OUT_DIR, exist_ok=True)
            results["verdict"] = "ERROR"
            with open(os.path.join(OUT_DIR, "result.json"), "w") as f:
                json.dump(results, f, indent=2)
        except Exception:
            pass
        print("PROBE_AVATAR_NENVS_FAIL", flush=True)
        sys.exit(1)
