"""Represent RoboCasa's three planar DOFs as serial single-DOF joints for PhysX."""
import xml.etree.ElementTree as ET


def split_planar_base(xml):
    body = xml.find('.//body[@name="mobilebase0_base"]')
    parent = next(p for p in xml.iter() if body in list(p))
    joints = [body.find(f'joint[@name="mobilebase0_joint_mobile_{n}"]')
              for n in ("forward", "side", "yaw")]
    if any(j is None for j in joints):
        raise ValueError("Expected the original three planar base joints")
    if set(body.attrib) != {"name"}:
        raise ValueError("Only the pinned zero-transform mobilebase0_base is supported")
    anchor = [float(x) for x in joints[2].get("pos", "0 0 0").split()]
    parent.remove(body)
    current = parent
    for index, (name, joint) in enumerate(zip(("forward", "side", "yaw"), joints)):
        link = ET.SubElement(current, "body", name=f"migration_base_{name}")
        if index == 2:
            link.set("pos", " ".join(map(str, anchor)))
        ET.SubElement(link, "inertial", pos="0 0 0", mass="0.001", diaginertia="1e-6 1e-6 1e-6")
        body.remove(joint)
        joint.set("pos", "0 0 0")
        link.append(joint)
        current = link
    body.set("pos", " ".join(str(-x) for x in anchor))
    current.append(body)


def prepare(directory):
    import json
    import numpy as np
    import mujoco
    from pathlib import Path
    directory = Path(directory)
    original = directory / "scene_unsplit.xml"
    if not original.exists():
        original.write_text((directory / "scene.xml").read_text())
    xml = ET.parse(original).getroot()
    split_planar_base(xml)
    target = directory / "scene.xml"
    target.write_text(ET.tostring(xml, encoding="unicode"))
    models = [mujoco.MjModel.from_xml_path(str(p)) for p in (original, target)]
    datas = [mujoco.MjData(m) for m in models]
    manifest = json.loads((directory / "manifest.json").read_text())
    cases = []
    for offset in ([0, 0, 0], [.5, -.5, .3], [-.4, .2, -.7], [1., 1., 1.5]):
        for model, data in zip(models, datas):
            for name, spec in manifest["joints"].items():
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid >= 0:
                    adr = model.jnt_qposadr[jid]
                    data.qpos[adr:adr+len(spec["qpos"])] = spec["qpos"]
            for name, value in zip(("forward", "side", "yaw"), offset):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mobilebase0_joint_mobile_"+name)
                data.qpos[model.jnt_qposadr[jid]] += value
            mujoco.mj_forward(model, data)
        errors = []
        for name in manifest["bodies"]:
            if not name.startswith(("robot", "mobilebase", "gripper")):
                continue
            ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name) for m in models]
            if min(ids) >= 0:
                errors.append(float(np.max(np.abs(datas[0].xpos[ids[0]]-datas[1].xpos[ids[1]]))))
                errors.append(float(np.max(np.abs(datas[0].xmat[ids[0]]-datas[1].xmat[ids[1]]))))
        cases.append({"base_joint_offset": offset, "max_pose_component_error": max(errors)})
    assert all(c["max_pose_component_error"] < 1e-8 for c in cases), cases
    report = {"status": "passed", "checks": cases, "virtual_link_mass_kg": .001,
              "note": "Kinematics checked in MuJoCo; added link inertia means dynamics are not identical."}
    (directory / "mobile_structure_checks.json").write_text(json.dumps(report, indent=2))
    model = models[0]
    dynamics = {}
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name.startswith(("robot", "mobilebase", "gripper", "manipulator")):
            dynamics[name] = {"mass": float(model.body_mass[i]), "inertia": model.body_inertia[i].tolist(),
                              "com": model.body_ipos[i].tolist(), "inertia_quat_wxyz": model.body_iquat[i].tolist()}
    (directory / "mobile_dynamics.json").write_text(json.dumps(dynamics, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    import sys
    prepare(sys.argv[1])
