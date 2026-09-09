"""Run pinned, unconverted RoboCasa through its original env.step/controllers."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import traceback
import xml.etree.ElementTree as ET


def main(args):
    # Must be selected before importing MuJoCo / robosuite.
    os.environ["MUJOCO_GL"] = "egl" if args.video else "disable"
    if args.video:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    else:
        os.environ.pop("PYOPENGL_PLATFORM", None)
    import numpy as np
    import robosuite
    import robocasa  # Registers original tasks.
    from robosuite.controllers import load_composite_controller_config
    from task_semantics import evaluate, navigation_success, native_serve_tea_contacts

    source = args.source / args.task
    manifest = json.loads((source / "manifest.json").read_text())
    np.random.seed(manifest["seed"])
    navigation = manifest["success_spec"]["kind"] == "navigation"
    original = (source / "original.xml").read_text()
    output = args.output / args.task
    output.mkdir(parents=True, exist_ok=True)
    env = robosuite.make(
        env_name=args.task, robots="PandaOmron",
        controller_configs=load_composite_controller_config(robot="PandaOmron"),
        has_renderer=False, has_offscreen_renderer=args.video, use_camera_obs=False,
        layout_and_style_ids=[[manifest["layout"], manifest["style"]]], seed=manifest["seed"],
        generative_textures=None, randomize_cameras=False, control_freq=20,
        ignore_done=True, horizon=args.steps,
    )
    writer = None
    try:
        env.reset()
        target_fixture = env.dining_table if args.task == "ServeTea" else env.target_fixture if navigation else env.sink if args.task == "PickPlaceCounterToSink" else env.fxtr
        if target_fixture.name != manifest["target_fixture"]:
            raise ValueError("Sampled fixture differs from export; refusing an unmatched comparison")
        if navigation and (not np.allclose(env.target_pos, manifest["success_spec"]["target_pos"])
                           or not np.allclose(env.target_ori[2], manifest["success_spec"]["target_ori"][2])):
            raise ValueError("Native task target differs from exported target")

        xml = ET.fromstring(original)
        if args.video:
            # Same inspection viewpoint as the Isaac navigation runner. Camera
            # addition and hiding room walls change rendering only.
            start = np.array(manifest["bodies"]["mobilebase0_base"]["pos"])
            target = np.array(manifest["success_spec"]["target_pos"] if navigation else target_fixture.pos)
            center = (start + target) / 2
            center[1:3] = [-.55, .95]
            eye = center + [1., -3.2, 1.5]
            forward = center - eye
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, [0., 0., 1.])
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            ET.SubElement(xml.find("worldbody"), "camera", name="migration_inspection",
                          pos=" ".join(map(str, eye)), xyaxes=" ".join(map(str, [*right, *up])), fovy="53.130102")
        env.reset_from_xml_string(ET.tostring(xml, encoding="unicode"))
        reset = np.load(source / "mujoco_reset.npz")
        # Clear solver warm-start, controls and activation state from the
        # constructor/reset before restoring the shared physical snapshot.
        env.sim.reset()
        env.sim.data.qpos[:] = reset["qpos"]
        env.sim.data.qvel[:] = reset["qvel"]
        env.sim.forward()
        initial_body_error = max(float(np.linalg.norm(
            env.sim.data.body_xpos[env.sim.model.body_name2id(name)] - spec["pos"]))
            for name, spec in manifest["bodies"].items())
        # MuJoCo's exported XML rounds transforms (the current snapshot differs
        # by ~10 micrometres); allow 0.1 mm, while retaining the measured error.
        if initial_body_error > 1e-4:
            raise ValueError(f"Restored source body poses differ by {initial_body_error} m")
        robot = env.robots[0]
        composite = robot.composite_controller
        composite.update_state()
        for controller in composite.part_controllers.values():
            controller.update(force=True)
        composite.reset()
        # reset_goal() in the pinned OSC stores a world-frame pose. Before
        # HYBRID_MOBILE_BASE switches to "desired", seed its base-frame goal
        # via the original controller's achieved-pose action path.
        controller_reset_checks = {}
        for arm in composite.arms:
            controller = composite.part_controllers[arm]
            before = controller.origin_pos + controller.origin_ori @ controller.goal_pos
            controller.update_initial_joints(controller.joint_pos.copy())
            controller.set_goal_update_mode("achieved")
            controller.set_goal(np.zeros(controller.control_dim))
            after = controller.origin_pos + controller.origin_ori @ controller.goal_pos
            orientation = controller.origin_ori @ controller.goal_ori
            check = {"before_position_error_m": float(np.linalg.norm(before-controller.ref_pos)),
                     "after_position_error_m": float(np.linalg.norm(after-controller.ref_pos)),
                     "after_orientation_max_error": float(np.max(np.abs(orientation-controller.ref_ori_mat)))}
            if check["after_position_error_m"] > 1e-6 or check["after_orientation_max_error"] > 1e-6:
                raise ValueError(f"OSC reset target is inconsistent: {check}")
            controller_reset_checks[arm] = check
        print("CONTROLLER_RESET_CHECKS", controller_reset_checks, flush=True)
        replay = None
        if args.replay is not None:
            if args.task != "ServeTea":
                raise ValueError("EEF replay is only supported for ServeTea")
            from serve_tea_native_replay import NativeEEFReplay
            replay = NativeEEFReplay(args.replay, env)
        if args.video:
            import imageio.v2 as imageio
            for i in range(env.sim.model.ngeom):
                name = env.sim.model.geom_id2name(i) or ""
                if name.startswith(("wall_front", "ceiling")):
                    env.sim.model.geom_rgba[i, 3] = 0
            writer = imageio.get_writer(str(output / "preview.mp4"), fps=20)
        base_id = env.sim.model.body_name2id("mobilebase0_base")
        root_id = env.sim.model.body_name2id("robot0_base")
        names = [env.sim.model.joint_id2name(i) for i in range(env.sim.model.njnt)]
        initial_error = float(np.max(np.abs(env.sim.data.qpos - reset["qpos"])))
        base_controller = composite.part_controllers["base"]
        init_yaw = math.atan2(base_controller.init_ori[1, 0], base_controller.init_ori[0, 0])
        root_matrix = env.sim.data.body_xmat[root_id].reshape(3, 3).copy()
        target = np.array(manifest["success_spec"].get("target_pos", [0, 0, 0]))
        target_yaw = manifest["success_spec"].get("target_ori", [0, 0, 0])[2]
        start = env.sim.data.body_xpos[base_id].copy()
        aisle_y = min(start[1], target[1]) - .5
        waypoints = [np.array([start[0], aisle_y]), np.array([target[0], aisle_y]), target[:2]]
        waypoint = 0
        traces, actions, qpos, qvel = [], [], [], []
        initial_time = float(env.sim.data.time)
        for step in range(args.steps):
            phase = "zero_action_smoke"
            base_action = np.zeros(3)
            if navigation and args.policy == "navigate" and step >= 10:
                pos = env.sim.data.body_xpos[base_id]
                matrix = env.sim.data.body_xmat[base_id].reshape(3, 3)
                yaw = math.atan2(matrix[1, 0], matrix[0, 0])
                if waypoint < 2 and np.linalg.norm(waypoints[waypoint] - pos[:2]) < args.waypoint_tolerance:
                    waypoint += 1
                phase = ("retreat_to_aisle", "traverse_aisle", "approach_target")[waypoint]
                world_velocity = 4 * (waypoints[waypoint] - pos[:2])
                speed = np.linalg.norm(world_velocity)
                if speed > args.speed_cap:
                    world_velocity *= args.speed_cap / speed
                joint_velocity = root_matrix[:2, :2].T @ world_velocity
                _, current_ori = base_controller.get_base_pose()
                theta = math.atan2(current_ori[1, 0], current_ori[0, 0]) - init_yaw
                # Invert the pinned controller's current->initial-base transform
                # and actuator-range scaling, without replacing its controller.
                c, s = math.cos(theta), math.sin(theta)
                limits = np.stack([base_controller.actuator_min, base_controller.actuator_max], axis=-1)
                bias, scale = limits.mean(axis=1), (limits[:, 1] - limits[:, 0]) / 2
                desired = np.array([*joint_velocity, np.clip(2 * math.atan2(math.sin(target_yaw-yaw), math.cos(target_yaw-yaw)), -.4, .4)])
                normalized = (desired - bias) / scale
                base_action[:2] = np.array([[c, -s], [s, c]]) @ normalized[:2]
                base_action[2] = normalized[2]
            action = robot.create_action_vector({"base": np.clip(base_action, -1, 1), "base_mode": 1})
            if replay is not None:
                action, phase = replay.action(step / 20.)
            submitted_action = action.copy()  # The upstream controller can mutate its action view.
            obs, reward, done, info = env.step(action)
            if not np.isfinite(env.sim.data.qpos).all() or not np.isfinite(env.sim.data.qvel).all():
                raise RuntimeError("Non-finite native state")
            pos = env.sim.data.body_xpos[base_id].copy()
            matrix = env.sim.data.body_xmat[base_id].reshape(3, 3)
            yaw = math.atan2(matrix[1, 0], matrix[0, 0])
            source_success = bool(env._check_success())
            positions = {name: env.sim.data.body_xpos[env.sim.model.body_name2id(name)].tolist()
                         for name in manifest["bodies"]}
            joints = {name: float(env.sim.data.get_joint_qpos(name)) for name in names
                      if name in manifest["joints"] and manifest["joints"][name]["type"] >= 2}
            adapted = navigation_success(pos, yaw, target, target_yaw) if navigation else evaluate(
                manifest, positions, joints, env.sim.data.site_xpos[robot.eef_site_id["right"]],
                native_serve_tea_contacts(env) if args.task == "ServeTea" else None)
            if source_success != adapted:
                raise AssertionError("Original and migrated predicates disagree on native trajectory")
            trace = {"step": step, "time_s": float(env.sim.data.time) - initial_time,
                     "phase": phase, "source_success": source_success, "adapted_success": adapted,
                     "base_pos": pos.tolist(), "base_yaw": yaw,
                     "reference_root_pos": env.sim.data.body_xpos[root_id].tolist(),
                     "joint_positions": joints, "reward": float(reward)}
            if navigation:
                trace.update(target_distance_m=float(np.linalg.norm(pos[:2]-target[:2])),
                             orientation_cos=math.cos(target_yaw-yaw))
            if args.task == "ServeTea":
                from serve_tea_native_replay import observe
                trace.update(observe(env))
            traces.append(trace)
            actions.append(submitted_action)
            qpos.append(env.sim.data.qpos.copy())
            qvel.append(env.sim.data.qvel.copy())
            if writer:
                frame = env.sim.render(width=720, height=540, camera_name="migration_inspection")[::-1]
                writer.append_data(frame)
                if step == 0:
                    imageio.imwrite(output / "start.png", frame)
            if step % 100 == 0:
                print("NATIVE", args.task, step, phase, trace.get("target_distance_m"), source_success, flush=True)
        if writer:
            imageio.imwrite(output / "preview.png", frame)
        np.savez_compressed(output / "rollout.npz", actions=actions, qpos=qpos, qvel=qvel,
                            joint_names=np.array(names), qpos_addresses=env.sim.model.jnt_qposadr,
                            qvel_addresses=env.sim.model.jnt_dofadr, joint_types=env.sim.model.jnt_type,
                            time_s=[t["time_s"] for t in traces])
        report = {"status": "simulated", "backend": "original_robocasa_mujoco", "task": args.task,
                  "policy": "isaac_achieved_eef_replay" if replay else args.policy if navigation else "zero_action_smoke",
                  "replay": replay.metadata if replay else None,
                  "controller": "original HYBRID_MOBILE_BASE / OSC_POSE / JOINT_VELOCITY",
                  "versions": {n: importlib.metadata.version(n) for n in ("robocasa", "robosuite", "mujoco", "numpy")},
                  "source_commits": {n: subprocess.check_output(["git", "-C", f"external/{n}", "rev-parse", "HEAD"], text=True).strip()
                                     for n in ("robocasa", "robosuite")},
                  "source_xml_sha256": hashlib.sha256(original.encode()).hexdigest(),
                  "layout": manifest["layout"], "style": manifest["style"], "seed": manifest["seed"],
                  "target_fixture": manifest["target_fixture"], "success_spec": manifest["success_spec"],
                  "initial_qpos_max_error": initial_error, "control_hz": 20,
                  "initial_body_position_max_error_m": initial_body_error,
                  "controller_reset_checks": controller_reset_checks,
                  "navigation_parameters": {"speed_cap": args.speed_cap, "waypoint_tolerance": args.waypoint_tolerance},
                  "physics_dt": float(env.sim.model.opt.timestep), "action_dim": len(actions[0]),
                  "camera_enabled": args.video, "steps": args.steps,
                  "predicate_agreement": True, "source_predicate_at_end": traces[-1]["source_success"],
                  "robot_task_success": bool(navigation and args.policy == "navigate" and len(traces) >= 20
                                             and all(t["source_success"] for t in traces[-20:])),
                  "trace": traces,
                  "comparison_note": "Same exported source/reset/target; native controller and physics timestep differ from Isaac. Navigation parameters are recorded separately. Not identical-action dynamics equivalence."}
        (output / "result.json").write_text(json.dumps(report, indent=2))
    finally:
        if writer:
            writer.close()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="NavigateKitchen", choices=["NavigateKitchen", "PickPlaceCounterToSink", "OpenCabinet", "OpenMicrowave", "ServeTea"])
    parser.add_argument("--source", type=Path, default=Path("outputs/robocasa_migration/source"))
    parser.add_argument("--output", type=Path, default=Path("outputs/robocasa_migration/native"))
    parser.add_argument("--steps", type=int, default=840)
    parser.add_argument("--policy", choices=["navigate", "zero"], default="navigate")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--replay", type=Path, help="ServeTea Isaac result: track achieved EEF poses with original OSC")
    parser.add_argument("--speed-cap", type=float, default=.6)
    parser.add_argument("--waypoint-tolerance", type=float, default=.12)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    if not all(math.isfinite(v) and v > 0 for v in (args.speed_cap, args.waypoint_tolerance)):
        parser.error("speed cap and waypoint tolerance must be positive and finite")
    try:
        main(args)
    except Exception:
        output = args.output / args.task
        output.mkdir(parents=True, exist_ok=True)
        (output / "result.json").write_text(json.dumps({"status": "failed", "task": args.task, "error": traceback.format_exc()}, indent=2))
        raise
