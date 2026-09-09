"""Search source collision geometry for bilateral pad contact before GPU trials."""
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

root = Path("outputs/robocasa_migration")
model = mujoco.MjModel.from_xml_path(str(root / "source/ServeTea/original.xml"))
data = mujoco.MjData(model)
reset = np.load(root / "source/ServeTea/mujoco_reset.npz")["qpos"]
manifest = json.loads((root / "source/ServeTea/manifest.json").read_text())
def jid(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
arm = [model.jnt_qposadr[jid(f"robot0_joint{i}")] for i in range(1, 8)]
limits = np.array([model.jnt_range[jid(f"robot0_joint{i}")] for i in range(1, 8)])
fingers = [model.jnt_qposadr[jid(f"gripper0_right_finger_joint{i}")] for i in (1, 2)]
cup_addr = model.jnt_qposadr[jid("teacup_joint0")]
eef = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper0_right_eef")
cup = np.array(manifest["objects"]["teacup"]["pos"])
w, x, y, z = manifest["objects"]["teacup"]["quat_wxyz"]
cup_rot = Rotation.from_quat([x, y, z, w]).as_matrix()
yaw = Rotation.from_matrix(cup_rot).as_euler("xyz")[2]
rz = Rotation.from_euler("z", yaw)
results = []
for label, orientation in (
    ("down_across", rz * Rotation.from_quat([0, 1, 0, 0])),
    ("down_radial", rz * Rotation.from_quat([np.sqrt(.5), np.sqrt(.5), 0, 0])),
    ("tilted_across", rz * Rotation.from_quat([0, np.cos(np.pi/8), np.sin(np.pi/8), 0])),
):
    data.qpos[:] = reset
    data.qpos[model.jnt_qposadr[jid("mobilebase0_joint_torso_height")]] = .18
    target = cup + cup_rot @ [0, -.05, .025]
    def residual(q):
        data.qpos[arm] = q
        mujoco.mj_forward(model, data)
        return np.r_[data.xpos[eef]-target,
                     Rotation.from_matrix(orientation.as_matrix() @ data.xmat[eef].reshape(3, 3).T).as_rotvec()*.2]
    fit = least_squares(residual, reset[arm], bounds=(limits[:, 0]+1e-5, limits[:, 1]-1e-5), max_nfev=300)
    error = float(np.linalg.norm(residual(fit.x)))
    if error > .003:
        print(label, "unreachable", error, flush=True)
        continue
    actual = data.xpos[eef].copy()
    for cy in np.linspace(-.065, -.033, 17):
        for cz in np.linspace(.008, .045, 16):
            desired = cup + cup_rot @ [0, cy, cz]
            data.qpos[cup_addr:cup_addr+3] = cup + actual - desired
            for width in (.003, .005, .007, .009, .012, .015):
                data.qpos[fingers] = [width, -width]
                mujoco.mj_forward(model, data)
                pads = set()
                unwanted = []
                depths = []
                for contact in data.contact:
                    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "" for g in (contact.geom1, contact.geom2)]
                    if not any(n.startswith("teacup") for n in names):
                        continue
                    for name in names:
                        if name.startswith("gripper"):
                            if "pad_collision" in name:
                                pads.add(name)
                                depths.append(float(contact.dist))
                            elif contact.dist < -.001:
                                unwanted.append(name)
                if len(pads) == 2:
                    results.append({"orientation": label, "eef_quat_xyzw": orientation.as_quat().tolist(),
                                    "cup_local_offset": [0, float(cy), float(cz)], "width": width,
                                    "unwanted": sorted(set(unwanted)), "max_penetration": -min(depths),
                                    "eef_target": desired.tolist()})
    print(label, "candidates", len(results), flush=True)
results.sort(key=lambda r: (len(r["unwanted"]), r["max_penetration"]))
out = root / "serve_tea_grasp_candidates.json"
out.write_text(json.dumps(results[:40], indent=2))
print(out.read_text())
