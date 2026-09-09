"""Diagnostic EEF tracking in original RoboCasa; not an expert ServeTea policy.

The source trajectory is Isaac's achieved pose, not its action array. Original
OSC/actuators/contact geometry remain active. No runtime state writes are used.
"""
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def observe(env):
    sim = env.sim
    contacts = []
    for contact in sim.data.contact:
        names = [sim.model.geom_id2name(int(g)) or "" for g in (contact.geom1, contact.geom2)]
        if any(n.startswith("teacup") for n in names) and any(n.startswith("gripper") for n in names):
            contacts.append({"geoms": names, "distance": float(contact.dist)})
    eef = sim.model.body_name2id("gripper0_right_eef")
    cup = sim.model.body_name2id("teacup_main")
    return {"tcp": sim.data.body_xpos[eef].tolist(),
            "eef_quat_xyzw": Rotation.from_matrix(sim.data.body_xmat[eef].reshape(3, 3)).as_quat().tolist(),
            "object_pos": sim.data.body_xpos[cup].tolist(), "gripper_cup_contacts": contacts}


class NativeEEFReplay:
    def __init__(self, path, env):
        self.env = env
        raw = path.read_bytes()
        result = json.loads(raw)
        if result["task"] != "ServeTea" or result["mode"] != "original_pandaomron_serve_tea":
            raise ValueError("Expected an Isaac ServeTea robot trajectory")
        self.rows = result["trace"]
        self.times = np.array([r["step"] * result["physics_dt"] for r in self.rows])
        self.positions = np.array([r["robot"]["tcp"] for r in self.rows])
        self.rotations = Slerp(self.times, Rotation.from_quat([r["robot"]["eef_quat"] for r in self.rows]))
        robot = env.robots[0]
        robot.composite_controller.part_controllers["right"].input_type = "absolute"
        robot.composite_controller.part_controllers["right"].input_ref_frame = "world"
        robot.composite_controller.part_controllers["torso"].input_type = "absolute"
        self.metadata = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                         "mode": "achieved_world_eef_pose_tracking_original_OSC",
                         "limitations": "Different closed-loop controller and gripper dynamics; not identical-action engine equivalence or a native expert demo."}

    def action(self, time):
        time = float(np.clip(time, self.times[0], self.times[-1]))
        row = self.rows[max(0, np.searchsorted(self.times, time, side="right")-1)]["robot"]
        pos = [np.interp(time, self.times, self.positions[:, i]) for i in range(3)]
        rotation = self.rotations(time).as_rotvec()
        joints = row["joint_positions"]
        # Match measured jaw aperture through the original incremental gripper
        # interface, including its speed limit. Do not overwrite finger states.
        robot = self.env.robots[0]
        width = np.clip((joints["gripper0_right_finger_joint1"]-joints["gripper0_right_finger_joint2"])/2, 0, .04)
        desired = 2 * width / .04 - 1
        current = robot.gripper["right"].current_action[0]
        grip = 0. if abs(current-desired) < .1 else float(np.sign(current-desired))
        action = robot.create_action_vector({"right": np.r_[pos, rotation], "right_gripper": [grip],
                                            "torso": [joints["mobilebase0_joint_torso_height"]],
                                            "base": [0., 0., 0.], "base_mode": -1})
        return action, row["phase"]
