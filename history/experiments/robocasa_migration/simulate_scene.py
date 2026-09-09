"""Load a converted task into Isaac Lab and record passive/joint-actuation probes.

The joint probe is an asset diagnostic, not robot task completion.
"""
import argparse
import faulthandler
import json
import math
import traceback
from pathlib import Path
faulthandler.enable()
faulthandler.dump_traceback_later(180, repeat=True)
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--steps", type=int, default=360)
parser.add_argument("--hold_steps", type=int, default=0)
parser.add_argument("--output", default="outputs/robocasa_migration/probes")
parser.add_argument("--migration_root", type=Path, default=Path("outputs/robocasa_migration"))
parser.add_argument("--demonstration", type=Path, help="Verified native demonstration result for joint trajectory tracking")
parser.add_argument("--demonstration_feedback", action="store_true", help="Retarget the placement phase from live cup/saucer positions")
parser.add_argument("--source_contact_friction", action="store_true", help="Restore gripper-pad static friction and max pair combination from source sliding friction")
parser.add_argument("--joint_probe", action="store_true")
parser.add_argument("--robot_attempt", action="store_true")
parser.add_argument("--serve_tea_contact_probe", action="store_true",
                    help="Diagnostic only: release cup above saucer after one second")
parser.add_argument("--no_camera", action="store_true")
parser.add_argument("--sample_every", type=int, default=8)
parser.add_argument("--mobile_friction_scale", type=float, default=1., help="Diagnostic scale; 0 disables the explicit friction approximation")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1 or args.hold_steps < 0:
    parser.error("steps must be positive and hold_steps nonnegative")
if args.robot_attempt and args.joint_probe:
    parser.error("Robot contact attempts and direct joint probes must be separate runs")
if args.demonstration and (args.task != "ServeTea" or not args.robot_attempt):
    parser.error("Demonstration tracking requires --task ServeTea --robot_attempt")
if args.demonstration_feedback and not args.demonstration:
    parser.error("Placement feedback requires --demonstration")
if args.serve_tea_contact_probe and (args.task != "ServeTea" or args.robot_attempt or args.joint_probe or args.steps < 240):
    parser.error("ServeTea contact probe requires ServeTea, >=240 steps, and no robot/joint probe")
if args.sample_every < 1 or (args.sample_every != 8 and not args.no_camera):
    parser.error("sample_every must be positive; custom sampling requires --no_camera")
if not math.isfinite(args.mobile_friction_scale) or args.mobile_friction_scale < 0:
    parser.error("mobile_friction_scale must be finite and nonnegative")
launcher = AppLauncher(args)
app = launcher.app

import imageio.v2 as imageio
import numpy as np
import torch
if not args.no_camera:
    import omni.replicator.core as rep
from pxr import Usd, UsdGeom, UsdPhysics, Gf
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import Camera, CameraCfg
from task_semantics import evaluate
from geometry_rules import apply_geometry_rules, repair_material_textures


def tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch


def main():
    root = args.migration_root
    manifest = json.loads((root / "source" / args.task / "manifest.json").read_text())
    navigation = manifest["success_spec"]["kind"] == "navigation"
    serve_tea = manifest["success_spec"]["kind"] == "serve_tea"
    mobile_scene = navigation or serve_tea
    if serve_tea and args.joint_probe:
        raise ValueError("ServeTea has no direct joint-effort asset probe")
    conversion = json.loads((root / "usd" / args.task / "conversion.json").read_text())
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=1/120, device=args.device,
        render=sim_utils.RenderCfg(antialiasing_mode="TAA", samples_per_pixel=32,
                                  enable_dlssg=False, enable_dl_denoiser=False)))
    cfg = sim_utils.UsdFileCfg(usd_path=conversion["usd_path"])
    cfg.func("/World/Kitchen", cfg)
    stage = sim.stage
    kitchen = stage.GetPrimAtPath("/World/Kitchen")
    while instances := [p for p in Usd.PrimRange(kitchen) if p.IsInstance()]:
        for p in instances:
            p.SetInstanceable(False)
    mesh_count = sum(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(kitchen))
    geometry_report = apply_geometry_rules(kitchen, root / "source" / args.task / "scene.xml")
    texture_report = repair_material_textures(kitchen, root / "source" / args.task / "scene.xml",
                                              Path(conversion["usd_path"]).parent / "repaired_textures")
    print("GEOMETRY_RULES", geometry_report, flush=True)
    mobile_physics_report = None
    if mobile_scene:
        from mobile_physics import apply_mobile_physics
        mobile_physics_report = apply_mobile_physics(kitchen, root / "source" / args.task / "mobile_dynamics.json",
                                                   restore_gripper_friction=bool(args.demonstration),
                                                   source_contact_friction=args.source_contact_friction)
        print("MOBILE_PHYSICS", mobile_physics_report, flush=True)
    # Hide only room-enclosing walls for the inspection camera, retain collisions.
    for p in Usd.PrimRange(kitchen):
        if p.GetName().startswith(("wall_front", "ceiling")) and p.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(p).MakeInvisible()
        if p.GetName().endswith("_eef_target"):
            p.SetActive(False)
        # The importer leaves massless hinge helper links with invalid automatic
        # mass. A small explicit mass avoids PhysX's undocumented fallback.
        # Geometry-bearing child links retain their imported mass and inertia.
        if p.HasAPI(UsdPhysics.RigidBodyAPI) and "hinge" in p.GetName():
            mass = UsdPhysics.MassAPI.Apply(p)
            value = mass.GetMassAttr().Get()
            if value is None or value <= 0:
                mass.CreateMassAttr(0.001)
                mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-6, 1e-6, 1e-6))
    lights = sim_utils.DomeLightCfg(intensity=1500)
    lights.func("/World/InspectionLight", lights)
    target = manifest["fixtures"][manifest["target_fixture"]]
    center = np.array(target["pos"], dtype=float)
    center[2] = max(center[2], 0.8)
    eye = center + np.array([1.6, -2.0, 1.0])
    if navigation:
        start_pos = np.asarray(manifest["bodies"]["mobilebase0_base"]["pos"])
        center = (start_pos + np.asarray(manifest["success_spec"]["target_pos"])) / 2
        center[1] = -.55
        center[2] = .95
        eye = center + np.array([1.0, -3.2, 1.5])
    if serve_tea:
        cup = np.asarray(manifest["objects"]["teacup"]["pos"])
        saucer = np.asarray(manifest["objects"]["saucer"]["pos"])
        center = (cup + saucer) / 2
        eye = center + np.array([2., -4., 2.5])
    sim.set_camera_view(eye.tolist(), center.tolist())
    camera = None if args.no_camera else Camera(CameraCfg(prim_path="/World/InspectionCamera", height=540, width=720,
                              data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(
                                  focal_length=18., horizontal_aperture=24., clipping_range=(0.03, 100.))))
    roots = [p for p in Usd.PrimRange(kitchen) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    print("ARTICULATION_ROOTS", [str(p.GetPath()) for p in roots], flush=True)
    articulations = []
    mobile_roots = []
    for p in roots:
        path = str(p.GetPath())
        if (navigation or (serve_tea and args.robot_attempt)) and path == mobile_physics_report["reference_anchor"]["articulation_root"]:
            mobile_roots.append(path)
            continue
        robot_root = serve_tea and path == mobile_physics_report["reference_anchor"]["articulation_root"]
        reset_state = ArticulationCfg.InitialStateCfg(joint_pos={
            name: spec["qpos"][0] for name, spec in manifest["joints"].items()
            if spec["type"] >= 2 and name.startswith(("robot0_", "mobilebase0_", "gripper0_"))
        }) if robot_root else ArticulationCfg.InitialStateCfg()
        actuators = {"passive": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0., damping=0.)}
        if robot_root:
            actuators = {
                "base": ImplicitActuatorCfg(joint_names_expr=["mobilebase0_joint_mobile_.*"],
                    stiffness=10000., damping=1500., effort_limit_sim=600.),
                "torso": ImplicitActuatorCfg(joint_names_expr=["mobilebase0_joint_torso_height"],
                    stiffness=20000., damping=2000., effort_limit_sim=100000.),
                "arm": ImplicitActuatorCfg(joint_names_expr=["robot0_joint.*"],
                    stiffness=1000., damping=100., effort_limit_sim=300.),
                "gripper": ImplicitActuatorCfg(joint_names_expr=["gripper0_.*"],
                    stiffness=1000., damping=100., effort_limit_sim=100.)}
        articulation = Articulation(ArticulationCfg(prim_path=path, spawn=None,
            init_state=reset_state, actuators=actuators))
        articulations.append(articulation)
    tea_scene = None
    if serve_tea:
        from serve_tea_scene import ServeTeaScene
        tea_scene = ServeTeaScene(manifest, kitchen, grasp_contacts=args.robot_attempt)
    attempt = None
    passive_object = None
    if args.robot_attempt:
        if serve_tea:
            from serve_tea_policy import ServeTeaPolicy
            if len(mobile_roots) != 1:
                raise RuntimeError(f"Expected one PandaOmron articulation, found {mobile_roots}")
            if args.demonstration:
                from demonstration_policy import DemonstrationPolicy
                attempt = DemonstrationPolicy(manifest, mobile_roots[0], sim.device, tea_scene, args.demonstration,
                                              placement_feedback=args.demonstration_feedback)
            else:
                attempt = ServeTeaPolicy(manifest, mobile_roots[0], sim.device, tea_scene)
        elif navigation:
            from mobile_navigation import MobileNavigation
            if len(mobile_roots) != 1:
                raise RuntimeError(f"Expected one original PandaOmron articulation, found {mobile_roots}")
            attempt = MobileNavigation(manifest, mobile_roots[0], sim.device, friction_scale=args.mobile_friction_scale)
        else:
            from robot_attempt import RobotAttempt
            attempt = RobotAttempt(manifest, stage, sim.device)
    elif args.task == "PickPlaceCounterToSink":
        passive_object = RigidObject(RigidObjectCfg(prim_path="/World/Kitchen/Geometry/obj_main", spawn=None))
    sim.reset()
    if camera:
        camera.set_world_poses_from_view(torch.tensor([eye.tolist()], device=sim.device),
                                        torch.tensor([center.tolist()], device=sim.device))
        rep.settings.set_render_pathtraced(samples_per_pixel=32)
    initial = {}
    for a in articulations:
        print("ARTICULATION", a.cfg.prim_path, a.joint_names, a.body_names, flush=True)
        q = tensor(a.data.joint_pos).clone()
        for i, name in enumerate(a.joint_names):
            if name in manifest["joints"] and manifest["joints"][name]["type"] >= 2:
                q[:, i] = manifest["joints"][name]["qpos"][0]
        a.write_joint_state_to_sim(q, torch.zeros_like(q))
        initial[a.cfg.prim_path] = q.clone()
        a.set_joint_position_target(q)
        a.write_data_to_sim()
    if attempt:
        attempt.reset()
    if passive_object:
        passive_object.reset()
    if tea_scene:
        tea_scene.reset()
    output = Path(args.output) / args.task
    output.mkdir(parents=True, exist_ok=True)
    trace = []
    constructed_state = None
    writer = imageio.get_writer(str(output / "preview.mp4"), fps=15) if camera else None
    try:
        for step in range(args.steps + args.hold_steps):
            if args.serve_tea_contact_probe and step == 120:
                constructed_state = tea_scene.construct_contact_probe()
            positions = {}
            if attempt:
                attempt.command(step, args.steps)
            for a in ([] if serve_tea else articulations):
                effort = torch.zeros_like(tensor(a.data.joint_pos))
                if args.joint_probe and step > 60:
                    for i, name in enumerate(a.joint_names):
                        if name in target["door_joints"]:
                            lo, hi = manifest["joints"][name]["range"]
                            goal = lo + .95*(hi-lo) if lo >= 0 else hi - .95*(hi-lo)
                            effort[:, i] = torch.clamp(20*(goal-tensor(a.data.joint_pos)[:, i])
                                                     - 4*tensor(a.data.joint_vel)[:, i], -8, 8)
                a.set_joint_effort_target(effort)
                a.write_data_to_sim()
            sim.step(render=not args.no_camera)
            if attempt:
                attempt.update(sim.get_physics_dt())
            if passive_object:
                passive_object.update(sim.get_physics_dt())
            if tea_scene:
                tea_scene.update(sim.get_physics_dt())
            for a in articulations:
                a.update(sim.get_physics_dt())
                if serve_tea and step % args.sample_every != 0:
                    continue
                q = tensor(a.data.joint_pos)
                if not bool(torch.isfinite(q).all()):
                    raise RuntimeError("Non-finite joint state")
                positions.update({name: float(q[0, i]) for i, name in enumerate(a.joint_names)})
            if camera:
                camera.update(sim.get_physics_dt())
            if step % args.sample_every == 0:
                if camera:
                    rgb = tensor(camera.data.output["rgb"])[0, ..., :3].cpu().numpy()
                    writer.append_data(rgb)
                    if step == 0:
                        imageio.imwrite(output / "start.png", rgb)
                body_positions = {}
                for a in ([] if serve_tea else articulations):
                    bp = tensor(a.data.body_pos_w)[0]
                    body_positions.update({name: bp[i].cpu().tolist() for i, name in enumerate(a.body_names)})
                success = None
                if not navigation and target["door_joints"] and all(n in positions for n in target["door_joints"]):
                    success = evaluate(manifest, body_positions, positions)
                tea_state = tea_scene.state(articulations + ([attempt.robot] if attempt else []), sim.get_physics_dt()) if tea_scene else None
                if tea_state:
                    success = tea_state["success"]
                robot_state = attempt.state() if attempt else None
                if serve_tea and robot_state and step % 240 == 0:
                    print("SERVE_TEA", step, robot_state, flush=True)
                    (output / "serve_tea_live.json").write_text(json.dumps({"step": step, "robot": robot_state, "task": tea_state}, indent=2))
                if navigation and robot_state:
                    success = robot_state["navigation_success"]
                    if step % 240 == 0:
                        print("NAVIGATION", step, robot_state["phase"], robot_state["base_pos"],
                              "distance", robot_state["target_distance_m"], "success", success,
                              "command", robot_state["velocity_command"],
                              "qvel", robot_state["base_joint_velocities"], flush=True)
                        (output / "navigation_live.json").write_text(json.dumps({"step": step, "robot": robot_state}, indent=2))
                if attempt and attempt.pick:
                    body_positions[manifest["objects"]["obj"]["body"]] = robot_state["object_pos"]
                    success = evaluate(manifest, body_positions, positions, robot_state["tcp"])
                trace.append({"step": step, "door_positions": {n: positions.get(n) for n in target["door_joints"]},
                              "predicate": success, "robot": robot_state, "serve_tea": tea_state,
                              "passive_object_position": tensor(passive_object.data.root_pos_w)[0].cpu().tolist() if passive_object else None})
        if camera:
            imageio.imwrite(output / "preview.png", rgb)
            if np.std(rgb.astype(float)) < 1.0:
                raise RuntimeError("Inspection image is empty or uniform")
    finally:
        if writer:
            writer.close()
    hold_samples = math.ceil(1 / (args.sample_every * sim.get_physics_dt()))
    report = {"status": "simulated", "task": args.task,
              "mode": "original_pandaomron_serve_tea" if attempt and serve_tea else "original_pandaomron_navigation" if attempt and navigation else "fixed_franka_contact_attempt" if attempt else "direct_joint_effort_asset_probe" if args.joint_probe else "passive_scene_probe",
              "robot_task_success": bool(attempt and (not serve_tea or attempt.phase == "done") and len(trace) >= hold_samples and all(t["predicate"] for t in trace[-hold_samples:])),
              "physics_dt": sim.get_physics_dt(), "sample_every": args.sample_every,
              "mobile_friction_scale": args.mobile_friction_scale if navigation else None,
              "source_predicate_at_end": bool(trace and trace[-1]["predicate"]),
              "steps": args.steps + args.hold_steps, "trajectory_steps": args.steps,
              "hold_steps": args.hold_steps, "trace": trace,
              "expanded_mesh_count": mesh_count,
              "geometry_rules": geometry_report,
              "repaired_material_count": len(texture_report),
              "mobile_physics": mobile_physics_report,
              "camera_enabled": bool(camera),
              "articulations": [{"path": a.cfg.prim_path, "joints": a.joint_names} for a in articulations]}
    if args.serve_tea_contact_probe:
        report["mode"] = "constructed_contact_predicate_probe"
        report["constructed_state"] = constructed_state
        report["contact_predicate_probe_passed"] = any(t["predicate"] for t in trace if t["step"] >= 120)
        if not report["contact_predicate_probe_passed"]:
            report["status"] = "failed"
    if args.demonstration:
        report["upright_task_success"] = bool(report["robot_task_success"] and all(
            t["serve_tea"]["upright_success"] for t in trace[-hold_samples:]))
        report["upright_tilt_limit_degrees"] = 10.
        report["mode"] = ("verified_native_demonstration_with_placement_feedback" if args.demonstration_feedback
                          else "verified_native_demonstration_joint_tracking")
        report["demonstration"] = str(args.demonstration)
        report["demonstration_feedback"] = args.demonstration_feedback
    (output / "result.json").write_text(json.dumps(report, indent=2))
    if report["status"] != "simulated":
        raise RuntimeError("Constructed ServeTea contact probe never satisfied the task predicate")


try:
    main()
except Exception:
    output = Path(args.output) / args.task
    output.mkdir(parents=True, exist_ok=True)
    error = traceback.format_exc()
    print(error, flush=True)
    (output / "result.json").write_text(json.dumps({"status": "failed", "task": args.task, "error": error}, indent=2))
    raise
finally:
    faulthandler.cancel_dump_traceback_later()
    app.close()
