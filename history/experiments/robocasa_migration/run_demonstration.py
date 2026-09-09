"""Validate official ServeTea actions in native dynamics and export its initial scene.

Reads the pinned RoboCasa LeRobot extras and action modality metadata directly,
without installing a training stack. State writes occur only during reset/export.
"""
import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def load_episode(dataset, episode):
    import numpy as np
    import pyarrow.parquet as pq
    ep = dataset / "extras" / f"episode_{episode:06d}"
    states = np.load(ep / "states.npz")["states"]
    meta = json.loads((ep / "ep_meta.json").read_text())
    xml = gzip.decompress((ep / "model.xml.gz").read_bytes()).decode()
    parquet = next(dataset.glob(f"data/*/episode_{episode:06d}.parquet"))
    raw = np.asarray(pq.read_table(parquet, columns=["action"])["action"].to_pylist())
    modality = json.loads((dataset / "meta/modality.json").read_text())["action"]
    order = {"end_effector_position": (0, 3), "end_effector_rotation": (3, 6),
             "gripper_close": (6, 7), "base_motion": (7, 11), "control_mode": (11, 12)}
    actions = np.zeros_like(raw)
    for key, (start, end) in order.items():
        spec = modality[key]
        actions[:, start:end] = raw[:, spec["start"]:spec["end"]]
    if len(states) != len(actions) or actions.shape[1] != 12:
        raise ValueError("Unexpected official states/actions shape")
    return states, meta, xml, actions


def restore(env, meta, xml, state):
    import numpy as np
    env.set_ep_meta(meta)
    env.reset()
    env.reset_from_xml_string(env.edit_model_xml(xml))
    env.sim.reset()
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    if hasattr(env, "update_state"):
        env.update_state()
    robot = env.robots[0]
    composite = robot.composite_controller
    composite.update_state()
    for controller in composite.part_controllers.values():
        controller.update(force=True)
    composite.reset()
    for arm in composite.arms:
        controller = composite.part_controllers[arm]
        controller.update_initial_joints(controller.joint_pos.copy())
        controller.set_goal_update_mode("achieved")
        controller.set_goal(np.zeros(controller.control_dim))
    return float(np.max(np.abs(env.sim.get_state().flatten()-state)))


def main(args):
    os.environ["MUJOCO_GL"] = "egl" if args.video else "disable"
    if args.video:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    import numpy as np
    import robosuite
    import robocasa
    from scipy.spatial.transform import Rotation
    from serve_tea_native_replay import observe
    from task_semantics import native_serve_tea_contacts
    env_meta = json.loads((args.dataset / "extras/dataset_meta.json").read_text())["env_args"]
    states, meta, xml, actions = load_episode(args.dataset, args.episode)
    kwargs = dict(env_meta["env_kwargs"])
    kwargs.update(has_renderer=False, has_offscreen_renderer=args.video, use_camera_obs=False,
                  ignore_done=True, horizon=len(actions)+40)
    env = robosuite.make(**kwargs)
    output = args.output / f"episode_{args.episode:06d}"
    output.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        initial_error = restore(env, meta, xml, states[0])
        if initial_error > 1e-8:
            raise ValueError(f"Reset mismatch: {initial_error}")
        model, data = env.sim.model, env.sim.data
        names = [model.joint_id2name(i) for i in range(model.njnt)]
        scalar_names = [n for i,n in enumerate(names) if n and model.jnt_type[i] >= 2]
        if args.video:
            import imageio.v2 as imageio
            writer = imageio.get_writer(str(output / "preview.mp4"), fps=20)
        traces, qpos, qvel, targets = [], [], [], []
        initial_time = float(data.time)
        for step, action in enumerate(actions):
            env.step(action.copy())
            if not np.isfinite(data.qpos).all():
                raise ValueError("Non-finite state")
            robot = env.robots[0]
            controller = robot.composite_controller.part_controllers["right"]
            if controller.input_ref_frame == "base":
                pos = controller.origin_pos + controller.origin_ori @ controller.goal_pos
                rot = controller.origin_ori @ controller.goal_ori
            else:
                pos, rot = controller.goal_pos, controller.goal_ori
            targets.append([*pos, *Rotation.from_matrix(rot).as_quat()])
            trace = {"step": step, "time_s": float(data.time)-initial_time,
                     "source_success": bool(env._check_success()),
                     "joint_positions": {n: float(data.get_joint_qpos(n)) for n in scalar_names},
                     "contacts": native_serve_tea_contacts(env), **observe(env)}
            trace["gripper_position_targets"] = {
                model.joint_id2name(int(model.actuator_trnid[i, 0])): float(data.ctrl[i])
                for i in range(model.nu) if (model.actuator_id2name(i) or "").startswith("gripper")}
            from task_semantics import upright_tilt_degrees
            w, x, y, z = data.get_body_xquat(env.objects['teacup'].root_body)
            trace['object_quat_xyzw'] = [float(x), float(y), float(z), float(w)]
            trace['cup_tilt_degrees'] = upright_tilt_degrees(trace['object_quat_xyzw'])
            if step < len(states)-1:
                trace["recorded_state_error"] = float(np.linalg.norm(env.sim.get_state().flatten()-states[step+1]))
            traces.append(trace)
            qpos.append(data.qpos.copy())
            qvel.append(data.qvel.copy())
            if writer:
                frame = env.sim.render(width=720, height=544, camera_name="robot0_agentview_center")[::-1]
                writer.append_data(frame)
                if step == 0:
                    imageio.imwrite(output / "start.png", frame)
            if step % 100 == 0:
                print("DEMO", args.episode, step, trace["source_success"], trace.get("recorded_state_error"), flush=True)
        success = bool(traces[-1]["source_success"])
        report = {"status": "simulated", "task": "ServeTea", "backend": "original_robocasa_mujoco",
                  "mode": "official_demonstration_action_replay", "episode": args.episode,
                  "dataset": str(args.dataset), "dataset_env_metadata": env_meta,
                  "runtime_versions": {n: importlib.metadata.version(n) for n in ("robocasa", "robosuite", "mujoco")},
                  "episode_metadata": meta, "initial_state_max_error": initial_error,
                  "steps": len(actions), "source_predicate_at_end": success,
                  "robot_task_success": success, "source_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
                  "runtime_object_pose_writes": 0, "trace": traces}
        (output / "result.json").write_text(json.dumps(report, indent=2))
        np.savez_compressed(output / "rollout.npz", actions=actions, qpos=qpos, qvel=qvel,
                            eef_targets_xyzw=targets, time_s=[r["time_s"] for r in traces],
                            joint_names=np.array(names), qpos_addresses=model.jnt_qposadr,
                            qvel_addresses=model.jnt_dofadr, joint_types=model.jnt_type)
        if writer:
            imageio.imwrite(output / "preview.png", frame)
            writer.close()
            writer = None
        print("DEMO_RESULT", args.episode, success, flush=True)
        if success and args.export:
            restore(env, meta, xml, states[0])
            from export_tasks import export
            export("ServeTea", args.export / "source/ServeTea", meta["layout_id"], meta["style_id"], 0, env=env)
            manifest_path = args.export / "source/ServeTea/manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["demonstration"] = {"episode": args.episode, "native_result": str(output / "result.json"),
                                         "native_rollout": str(output / "rollout.npz"), "seed_note": "Restored dataset state, not a newly sampled seed"}
            manifest_path.write_text(json.dumps(manifest, indent=2))
    finally:
        if writer:
            writer.close()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("outputs/robocasa_migration/demonstrations/ServeTea/lerobot"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/robocasa_migration/demonstrations/native"))
    parser.add_argument("--export", type=Path, help="Isolated migration root, used only after native success")
    parser.add_argument("--video", action="store_true")
    main(parser.parse_args())
