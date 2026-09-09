"""Sample original RoboCasa tasks and export MJCF plus simulator-independent metadata."""
import argparse
import json
import os
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

# Export is CPU-only and never creates a render context.
os.environ["MUJOCO_GL"] = "disable"
os.environ.pop("PYOPENGL_PLATFORM", None)
import numpy as np
import mujoco
import robosuite
import robocasa
from robosuite.controllers import load_composite_controller_config
from task_semantics import evaluate, navigation_success, native_serve_tea_contacts

TASKS = ("PickPlaceCounterToSink", "OpenCabinet", "OpenMicrowave")
SUPPORTED_TASKS = (*TASKS, "NavigateKitchen", "ServeTea")


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def export(task, output, layout, style, seed, env=None):
    if task == "ServeTea" and layout in robocasa.ServeTea.EXCLUDE_LAYOUTS:
        raise ValueError(f"ServeTea excludes layout {layout}; use a dining-counter layout such as 2")
    output.mkdir(parents=True, exist_ok=True)
    supplied_env = env is not None
    if not supplied_env:
        env = robosuite.make(
            env_name=task, robots="PandaOmron",
            controller_configs=load_composite_controller_config(robot="PandaOmron"),
            has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
            layout_and_style_ids=[[layout, style]], seed=seed,
            generative_textures=None, randomize_cameras=False, control_freq=20,
        )
    try:
        if not supplied_env:
            env.reset()
        model, data = env.sim.model, env.sim.data
        xml = ET.fromstring(model.get_xml())
        (output / "original.xml").write_text(ET.tostring(xml, encoding="unicode"))
        bodies = {}
        joints = {}
        for i in range(model.nbody):
            name = model.body_id2name(i)
            if name:
                bodies[name] = {"pos": data.body_xpos[i], "quat_wxyz": data.body_xquat[i]}
        for i in range(model.njnt):
            name = model.joint_id2name(i)
            adr = int(model.jnt_qposadr[i])
            kind = int(model.jnt_type[i])
            n = 7 if kind == 0 else 4 if kind == 1 else 1
            joints[name] = {"type": kind, "qpos": data.qpos[adr:adr+n],
                            "range": model.jnt_range[i], "body": model.body_id2name(int(model.jnt_bodyid[i])),
                            "axis": model.jnt_axis[i], "anchor_world": data.xanchor[i]}
        fixtures = {}
        for name, fixture in env.fixtures.items():
            fixtures[name] = {"class": type(fixture).__name__, "pos": fixture.pos,
                              "rot": getattr(fixture, "rot", None),
                              "door_joints": getattr(fixture, "door_joint_names", []),
                              "internal_regions": (fixture.get_int_sites(relative=False)
                                                   if hasattr(fixture, "get_int_sites") else {})}
        objects = {}
        for name, obj in env.objects.items():
            objects[name] = {"body": model.body_id2name(env.obj_body_id[name]),
                             "pos": data.body_xpos[env.obj_body_id[name]],
                             "quat_wxyz": data.body_xquat[env.obj_body_id[name]],
                             "horizontal_radius": float(obj.horizontal_radius),
                             "contact_geoms": obj.contact_geoms}
        navigation = task == "NavigateKitchen"
        serve_tea = task == "ServeTea"
        keep_robot = navigation or serve_tea
        target = env.dining_table if serve_tea else env.target_fixture if navigation else env.sink if task == "PickPlaceCounterToSink" else env.fxtr
        metadata = {"task": task, "layout": layout, "style": style, "seed": seed,
                    "robot": "PandaOmron", "physics": "MuJoCo 3.3.1", "control_hz": 20,
                    "source_success": bool(env._check_success()), "episode": env.get_ep_meta(),
                    "target_fixture": target.name, "fixtures": fixtures, "objects": objects,
                    "bodies": bodies, "joints": joints,
                    "success_spec": ({"kind": "inside_and_gripper_far", "partial_check": True,
                                      "gripper_distance_m": 0.25} if task == TASKS[0]
                                     else {"kind": "all_doors_open", "normalized_threshold": 0.90})}
        if serve_tea:
            metadata["robot_in_scene"] = True
            metadata["source_fixture"] = env.microwave.name
            eef_id = env.robots[0].eef_site_id["right"]
            metadata["gripper_site"] = {
                "body": model.body_id2name(int(model.site_bodyid[eef_id])),
                "local_pos": model.site_pos[eef_id],
            }
            metadata["success_spec"] = {"kind": "serve_tea", "xy_radius_scale": 0.7,
                                        "gripper_distance_m": 0.25,
                                        "required_contacts": ["teacup_saucer", "saucer_table"]}
            metadata["fixtures"][target.name]["contact_geoms"] = target.contact_geoms
        if navigation:
            metadata["robot_in_scene"] = True
            metadata["success_spec"] = {"kind": "navigation", "distance_m": 0.20, "orientation_cos_min": 0.98,
                                        "target_pos": env.target_pos, "target_ori": env.target_ori}
            metadata["source_fixture"] = env.src_fixture.name
        (output / "manifest.json").write_text(json.dumps(metadata, default=serial, indent=2))
        np.savez(output / "mujoco_reset.npz", qpos=data.qpos.copy(), qvel=data.qvel.copy())

        # Bake free-object reset poses into MJCF body transforms; scalar joint
        # reset positions remain explicit in the manifest for the Isaac runtime.
        for body in xml.findall(".//body"):
            joint = body.find("joint")
            free = body.find("freejoint")
            if free is not None or (joint is not None and joint.get("type") == "free"):
                pose = bodies.get(body.get("name"))
                if pose:
                    body.set("pos", " ".join(map(str, pose["pos"])))
                    body.set("quat", " ".join(map(str, pose["quat_wxyz"])))
        # Explicit actuator/state semantics are retained in original.xml/manifest.
        # Scene import omits the robot, replaced by a separately configured Lab robot.
        world = xml.find("worldbody")
        for child in list(world):
            if not keep_robot and child.tag == "body" and child.get("name", "").startswith(("robot", "gripper")):
                world.remove(child)
        for tag in ("actuator", "sensor", "tendon", "equality", "contact", "keyframe"):
            for node in xml.findall(tag):
                xml.remove(node)
        # Validate the emitted scene through the source parser before conversion.
        scene_path = output / "scene.xml"
        scene_path.write_text(ET.tostring(xml, encoding="unicode"))
        if keep_robot:
            # Refresh the pre-adaptation source when exporting another seed.
            (output / "scene_unsplit.xml").write_text(scene_path.read_text())
        check = mujoco.MjModel.from_xml_path(str(scene_path))
        # Verify the migrated predicates against the original implementation in
        # constructed positive/negative source states. These are unit probes,
        # not robot demonstrations or rollout success claims.
        cases = []
        saved = data.qpos.copy()
        for positive in ((False,) if serve_tea else (False, True)):
            data.qpos[:] = saved
            if navigation:
                yaw_joint = "mobilebase0_joint_mobile_yaw"
                xy_joints = ["mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side"]
                base_id = model.body_name2id("mobilebase0_base")
                env.sim.forward()
                matrix = data.body_xmat[base_id].reshape(3, 3)
                yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
                data.set_joint_qpos(yaw_joint, float(data.get_joint_qpos(yaw_joint)) + env.target_ori[2] - yaw)
                env.sim.forward()
                origin = data.body_xpos[base_id][:2].copy()
                columns = []
                for name in xy_joints:
                    q0 = float(data.get_joint_qpos(name))
                    data.set_joint_qpos(name, q0 + .01)
                    env.sim.forward()
                    columns.append((data.body_xpos[base_id][:2].copy() - origin) / .01)
                    data.set_joint_qpos(name, q0)
                    env.sim.forward()
                delta = np.asarray(env.target_pos[:2]) - origin + (0 if positive else np.array([1., 1.]))
                for name, dq in zip(xy_joints, np.linalg.solve(np.array(columns).T, delta)):
                    data.set_joint_qpos(name, float(data.get_joint_qpos(name)) + dq)
            elif serve_tea:
                pass  # The sampled initial state is a real negative contact case.
            elif task == TASKS[0]:
                region = next(iter(target.get_int_sites(relative=False).values()))
                p0, px, py, pz = map(np.asarray, region)
                center = p0 + ((px-p0) + (py-p0) + (pz-p0)) * 0.5
                qpos = data.get_joint_qpos(env.objects["obj"].joints[0]).copy()
                qpos[:3] = center if positive else center + np.array([4., 4., 4.])
                data.set_joint_qpos(env.objects["obj"].joints[0], qpos)
            elif positive:
                target.open_door(env, min=0.95, max=0.95)
            else:
                target.close_door(env)
            env.sim.forward()
            actual = bool(env._check_success())
            body_pos = {name: data.body_xpos[model.body_name2id(name)].tolist() for name in bodies}
            scalar_q = {name: float(data.get_joint_qpos(name)) for name, spec in joints.items() if spec["type"] >= 2}
            gripper = data.site_xpos[env.robots[0].eef_site_id["right"]].tolist()
            if navigation:
                matrix = data.body_xmat[base_id].reshape(3, 3)
                adapted = navigation_success(body_pos["mobilebase0_base"], np.arctan2(matrix[1, 0], matrix[0, 0]),
                                             env.target_pos, env.target_ori[2])
            else:
                adapted = evaluate(metadata, body_pos, scalar_q, gripper,
                                   native_serve_tea_contacts(env) if serve_tea else None)
            cases.append({"constructed_positive": positive, "source": actual, "adapted": adapted})
            if actual != adapted:
                raise AssertionError(f"Source predicate mismatch: {cases[-1]}")
            if actual != positive:
                raise AssertionError(f"Constructed source case did not exercise expected label: {cases[-1]}")
        if serve_tea:
            # Search a contacting cup-on-saucer state using the actual source
            # collision meshes. No fabricated contact booleans or rollout claim.
            cup_joint = env.objects["teacup"].joints[0]
            saucer_joint = env.objects["saucer"].joints[0]
            saucer_pos = data.body_xpos[env.obj_body_id["saucer"]].copy()
            cup_q = data.get_joint_qpos(cup_joint).copy()
            cup_q[3:] = [1., 0., 0., 0.]
            for height in np.linspace(.30, -.05, 701):
                cup_q[:3] = saucer_pos + [0., 0., height]
                data.set_joint_qpos(cup_joint, cup_q)
                env.sim.forward()
                if env._check_success():
                    break
            else:
                raise AssertionError("Could not construct an actual cup/saucer/table contact positive")
            contact_positive = data.qpos.copy()
            for label in ("on_saucer", "cup_above_saucer", "saucer_off_table"):
                data.qpos[:] = contact_positive
                if label == "cup_above_saucer":
                    q = data.get_joint_qpos(cup_joint).copy()
                    q[2] += 1.
                    data.set_joint_qpos(cup_joint, q)
                elif label == "saucer_off_table":
                    for name in (cup_joint, saucer_joint):
                        q = data.get_joint_qpos(name).copy()
                        q[2] += 1.
                        data.set_joint_qpos(name, q)
                env.sim.forward()
                actual = bool(env._check_success())
                body_pos = {spec["body"]: data.body_xpos[env.obj_body_id[name]].tolist()
                            for name, spec in objects.items()}
                contacts = native_serve_tea_contacts(env)
                adapted = evaluate(metadata, body_pos, {},
                                   data.site_xpos[env.robots[0].eef_site_id["right"]], contacts)
                cases.append({"case": label, "constructed_positive": label == "on_saucer",
                              "source": actual, "adapted": adapted, "contacts": contacts})
                if actual != adapted or actual != (label == "on_saucer"):
                    raise AssertionError(f"ServeTea source contact mismatch: {cases[-1]}")
        data.qpos[:] = saved
        env.sim.forward()
        (output / "predicate_checks.json").write_text(json.dumps(cases, indent=2))
        if serve_tea:
            from mobile_structure import prepare
            prepare(output)
        print("EXPORTED", task, "bodies", check.nbody, "joints", check.njnt, "geoms", check.ngeom, flush=True)
        return {"task": task, "status": "exported", "bodies": check.nbody, "joints": check.njnt}
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/robocasa_migration/source"))
    parser.add_argument("--tasks", nargs="+", default=TASKS, choices=SUPPORTED_TASKS)
    parser.add_argument("--layout", type=int, default=1)
    parser.add_argument("--style", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    results = []
    for task in args.tasks:
        try:
            results.append(export(task, args.output / task, args.layout, args.style, args.seed))
        except Exception:
            error = traceback.format_exc()
            print(error, flush=True)
            results.append({"task": task, "status": "failed", "error": error})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "export_results.json").write_text(json.dumps(results, indent=2))
    if any(r["status"] != "exported" for r in results):
        raise SystemExit(1)
