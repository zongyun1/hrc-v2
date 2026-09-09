"""Original PandaOmron navigation using driven planar joints, never root teleports."""
import math
import numpy as np
import torch
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from task_semantics import navigation_success


def tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch


def yaw_xyzw(quat):
    x, y, z, w = quat
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


class MobileNavigation:
    pick = False

    def __init__(self, manifest, prim_path, device, friction_scale=1.):
        self.m, self.device = manifest, device
        self.friction_scale = friction_scale
        reset_joints = {name: spec["qpos"][0] for name, spec in manifest["joints"].items()
                        if name.startswith(("robot0_", "mobilebase0_", "gripper0_"))}
        self.robot = Articulation(ArticulationCfg(prim_path=prim_path, spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(joint_pos=reset_joints), actuators={
            "base": ImplicitActuatorCfg(joint_names_expr=["mobilebase0_joint_mobile_.*"],
                                        stiffness=0., damping=1500., effort_limit_sim=600.),
            "torso": ImplicitActuatorCfg(joint_names_expr=["mobilebase0_joint_torso_height"],
                                         stiffness=20000., damping=2000., effort_limit_sim=100000.),
            "arm": ImplicitActuatorCfg(joint_names_expr=["robot0_joint.*"],
                                       stiffness=1000., damping=100., effort_limit_sim=300.),
            "gripper": ImplicitActuatorCfg(joint_names_expr=["gripper0_.*"],
                                           stiffness=1000., damping=100., effort_limit_sim=100.)}))
        self.target = np.asarray(manifest["success_spec"]["target_pos"])
        self.target_yaw = manifest["success_spec"]["target_ori"][2]
        self.phase = "settle"
        self.waypoint = 0

    def reset(self):
        self.robot.reset()
        if not self.robot.is_fixed_base:
            raise RuntimeError("PandaOmron reference root must be fixed; chassis motion uses planar joints")
        self.phase, self.waypoint = "settle", 0
        self.base_id = self.robot.body_names.index("mobilebase0_base")
        self.base_joints = [self.robot.joint_names.index("mobilebase0_joint_mobile_" + n)
                            for n in ("forward", "side", "yaw")]
        self.hold_joints = [i for i in range(self.robot.num_joints) if i not in self.base_joints]
        self.q0 = tensor(self.robot.data.joint_pos).clone()
        for i, name in enumerate(self.robot.joint_names):
            self.q0[:, i] = self.m["joints"][name]["qpos"][0]
        self.robot.write_joint_state_to_sim(self.q0, torch.zeros_like(self.q0))
        # Translation axes are anchored in robot0_base, not the rotating chassis.
        w, x, y, z = self.m["bodies"]["robot0_base"]["quat_wxyz"]
        root_yaw = yaw_xyzw([x, y, z, w])
        c, s = math.cos(root_yaw), math.sin(root_yaw)
        self.world_to_joint = np.array([[c, s], [-s, c]])
        start = np.asarray(self.m["bodies"]["mobilebase0_base"]["pos"])
        # Fixed-layout aisle route: retreat from cabinets, translate, then approach.
        # This is not a general navigation planner.
        aisle_y = min(start[1], self.target[1]) - .50
        self.waypoints = [np.array([start[0], aisle_y]),
                          np.array([self.target[0], aisle_y]), self.target[:2]]
        self.velocity_command = [0., 0., 0.]

    def command(self, step, total):
        pose = tensor(self.robot.data.body_pose_w)[0, self.base_id].cpu().numpy()
        pos, yaw = pose[:3], yaw_xyzw(pose[3:7])
        velocity = np.zeros(3)
        if step >= 60:
            if self.waypoint < len(self.waypoints)-1 and np.linalg.norm(self.waypoints[self.waypoint]-pos[:2]) < .08:
                self.waypoint += 1
            self.phase = ("retreat_to_aisle", "traverse_aisle", "approach_target")[self.waypoint]
            world_velocity = 4.0*(self.waypoints[self.waypoint]-pos[:2])
            speed = np.linalg.norm(world_velocity)
            if speed > .30:
                world_velocity *= .30/speed
            velocity[:2] = self.world_to_joint @ world_velocity
            error = math.atan2(math.sin(self.target_yaw-yaw), math.cos(self.target_yaw-yaw))
            velocity[2] = np.clip(2*error, -.4, .4)
            if navigation_success(pos, yaw, self.target, self.target_yaw):
                self.phase = "target_reached"
        self.velocity_command = velocity.tolist()
        self.robot.set_joint_velocity_target(torch.tensor([velocity.tolist()], device=self.device), joint_ids=self.base_joints)
        self.robot.set_joint_position_target(self.q0[:, self.hold_joints], joint_ids=self.hold_joints)
        # MuJoCo frictionloss has force/torque units; it is not a PhysX coefficient.
        # Use a smooth dissipative approximation, explicitly not exact static friction.
        qvel = tensor(self.robot.data.joint_vel)
        resistance = torch.zeros_like(qvel)
        resistance[:, self.base_joints] = -250.*torch.tanh(qvel[:, self.base_joints]/.01)
        torso = self.robot.joint_names.index("mobilebase0_joint_torso_height")
        resistance[:, torso] = -1000.*torch.tanh(qvel[:, torso]/.01)
        self.robot.set_joint_effort_target(self.friction_scale * resistance)
        self.robot.write_data_to_sim()

    def update(self, dt):
        self.robot.update(dt)

    def state(self):
        pose = tensor(self.robot.data.body_pose_w)[0, self.base_id].cpu().tolist()
        yaw = yaw_xyzw(pose[3:7])
        return {"phase": self.phase, "base_pos": pose[:3], "base_yaw": yaw,
                "reference_root_pose": tensor(self.robot.data.root_pose_w)[0].cpu().tolist(),
                "target_distance_m": math.dist(pose[:2], self.target[:2]),
                "orientation_cos": math.cos(self.target_yaw-yaw),
                "navigation_success": navigation_success(pose[:3], yaw, self.target, self.target_yaw),
                "velocity_command": self.velocity_command,
                "base_joint_velocities": tensor(self.robot.data.joint_vel)[0, self.base_joints].cpu().tolist(),
                "joint_velocities": dict(zip(self.robot.joint_names, tensor(self.robot.data.joint_vel)[0].cpu().tolist())),
                "joint_positions": dict(zip(self.robot.joint_names, tensor(self.robot.data.joint_pos)[0].cpu().tolist()))}
