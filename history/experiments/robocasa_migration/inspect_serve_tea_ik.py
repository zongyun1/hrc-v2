"""CPU source-model IK diagnostic; never counted as a robot rollout."""
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

source = Path("outputs/robocasa_migration/source/ServeTea")
model = mujoco.MjModel.from_xml_path(str(source / "original.xml"))
data = mujoco.MjData(model)
reset = np.load(source / "mujoco_reset.npz")
data.qpos[:] = reset["qpos"]
ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{i}") for i in range(1, 8)]
addresses = model.jnt_qposadr[ids]
torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mobilebase0_joint_torso_height")
data.qpos[model.jnt_qposadr[torso]] = .18
eef = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper0_right_eef")
for index, width in ((1, .015), (2, -.015)):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"gripper0_right_finger_joint{index}")
    data.qpos[model.jnt_qposadr[jid]] = width
yaw = .721
c, s = np.cos(yaw/2), np.sin(yaw/2)
target_rot = Rotation.from_quat([np.sqrt(.5)*(c-s), np.sqrt(.5)*(c+s), 0, 0]).as_matrix()
rng = np.random.default_rng(0)
results = []
for name, target in (("pregrasp", [.87765, -.80, 1.52032]), ("reach", [.87765, -.55740, 1.41032])):
    def residual(q):
        data.qpos[addresses] = q
        mujoco.mj_forward(model, data)
        return np.r_[data.xpos[eef] - target,
                     Rotation.from_matrix(target_rot @ data.xmat[eef].reshape(3, 3).T).as_rotvec() * .2]
    solutions = []
    for attempt in range(3):
        start = reset["qpos"][addresses] if attempt == 0 else rng.uniform(model.jnt_range[ids, 0], model.jnt_range[ids, 1])
        fit = least_squares(residual, start, bounds=(model.jnt_range[ids, 0]+1e-5, model.jnt_range[ids, 1]-1e-5), max_nfev=200)
        error = float(np.linalg.norm(residual(fit.x)))
        contacts = []
        for contact in data.contact:
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "" for g in (contact.geom1, contact.geom2)]
            robot = [n.startswith(("robot", "gripper")) for n in names]
            if any(robot) and not all(robot):
                contacts.append({"geoms": names, "distance": float(contact.dist)})
        solutions.append({"error": error, "q": fit.x.tolist(), "contacts": contacts})
    results.append({"waypoint": name, "solutions": sorted(solutions, key=lambda s: (s["error"] > .001, len(s["contacts"]), s["error"]))[:3]})
best = results[-1]["solutions"][0]
data.qpos[addresses] = best["q"]
cup_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "teacup_main")
sweep = []
for width in (.015, .012, .009, .006, .003, 0.):
    for index, sign in ((1, 1), (2, -1)):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"gripper0_right_finger_joint{index}")
        data.qpos[model.jnt_qposadr[jid]] = sign*width
    mujoco.mj_forward(model, data)
    contacts = []
    for contact in data.contact:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "" for g in (contact.geom1, contact.geom2)]
        if any(n.startswith("gripper") for n in names) and any(n.startswith("teacup") for n in names):
            point = data.xmat[cup_id].reshape(3, 3).T @ (contact.pos-data.xpos[cup_id])
            contacts.append({"geoms": names, "cup_local_point": point.tolist(), "distance": float(contact.dist)})
    sweep.append({"width": width, "contacts": contacts})
results[-1]["closure_sweep"] = sweep
out = Path("outputs/robocasa_migration/serve_tea_ik.json")
out.write_text(json.dumps(results, indent=2))
print(out.read_text())
