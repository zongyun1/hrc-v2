"""Fixed-base Franka scripted attempts on the migrated fixtures, using contact only."""
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import torch
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_apply, quat_mul
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG


def tensor(x):
    return x if isinstance(x, torch.Tensor) else x.torch


class RobotAttempt:
    def __init__(self, manifest, stage, device):
        self.m = manifest
        self.device = device
        self.pick = manifest["task"] == "PickPlaceCounterToSink"
        cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/Franka")
        cfg.spawn.usd_path = ("https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
                              "Assets/Isaac/5.1/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd")
        # Match the original PandaOmron arm's +Y-facing mount (Lab 3 uses XYZW).
        cfg.init_state.rot = (0., 0., math.sqrt(.5), math.sqrt(.5))
        target = manifest["fixtures"][manifest["target_fixture"]]
        if self.pick:
            obj = np.array(manifest["objects"]["obj"]["pos"])
            centers = [np.array(p0) + .5*(np.array(px)+np.array(py)+np.array(pz)-3*np.array(p0))
                       for p0, px, py, pz in target["internal_regions"].values()]
            self.destination = min(centers, key=lambda c: np.linalg.norm(c-obj))
            cfg.init_state.pos = tuple(manifest["bodies"]["robot0_link0"]["pos"])
            self.object = RigidObject(RigidObjectCfg(prim_path="/World/Kitchen/Geometry/obj_main", spawn=None))
        else:
            source_base = manifest["bodies"]["robot0_link0"]["pos"]
            cfg.init_state.pos = (source_base[0], -1.03, 1.02)
            self.object = None
            joint = manifest["joints"][target["door_joints"][0]]
            self.anchor = np.array(joint["anchor_world"])
            # Leave enough margin above the original 90%-open success threshold
            # for contact tracking error while respecting the imported joint limit.
            fraction = .995 if manifest["task"] == "OpenMicrowave" else .95
            self.angle = -fraction * (joint["range"][1]-joint["range"][0])
            if manifest["task"] == "OpenCabinet":
                self.handle = np.array(manifest["bodies"][manifest["target_fixture"]+"_door_handle_main"]["pos"])
                self.handle[1] -= .0152716
            else:
                xml = ET.parse(Path("outputs/robocasa_migration/source") / manifest["task"] / "scene.xml")
                geom = xml.find(f'.//geom[@name="{manifest["target_fixture"]}_door_handle_main"]')
                self.handle = np.array(manifest["bodies"][joint["body"]]["pos"]) + np.fromstring(geom.get("pos"), sep=" ")
        self.robot = Articulation(cfg)
        self.phase = "reset"
        self.ik_error = None
        self.grasp_ready = False

    def reset(self):
        self.robot.reset()
        self.arm = [self.robot.joint_names.index(f"panda_joint{i}") for i in range(1, 8)]
        self.fingers = [self.robot.joint_names.index(f"panda_finger_joint{i}") for i in (1, 2)]
        self.hand_id = self.robot.body_names.index("panda_hand")
        q = tensor(self.robot.data.default_joint_pos).clone()
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        self.ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False,
                                                                      ik_method="dls"), num_envs=1, device=self.device)
        self.root = tensor(self.robot.data.root_pose_w).clone()
        self.start = tensor(self.robot.data.body_pose_w)[0, self.hand_id, :3].cpu().numpy()
        self.offset = torch.tensor([[0., 0., .107]], device=self.device)
        if self.object:
            self.object.reset()
            self.obj_start = tensor(self.object.data.root_pos_w)[0].cpu().numpy().copy()
        self.initialized = False

    def command(self, step, total):
        # Start from measured poses after a short settling interval.
        if step < 60:
            self.robot.set_joint_position_target(tensor(self.robot.data.default_joint_pos))
            self.robot.write_data_to_sim()
            return
        if not self.initialized:
            self.start = tensor(self.robot.data.body_pose_w)[0, self.hand_id, :3].cpu().numpy().copy()
            if self.object:
                self.obj_start = tensor(self.object.data.root_pos_w)[0].cpu().numpy().copy()
            self.initialized = True
        t = min(1., (step-60)/max(1, total-60))
        finger = .04
        if self.pick:
            grasp = self.obj_start + np.array([0., 0., .107])
            if .20 <= t < .46:
                measured = tensor(self.object.data.root_pos_w)[0].cpu().numpy()
                grasp = measured + [0., 0., .107]
            above = grasp + [0., 0., .20]
            lift = grasp + [0., 0., .12]
            place = self.destination.copy()
            # Release above the basin rim, then let gravity and collisions act.
            place[2] = self.obj_start[2] + .07 + .107
            out = lift.copy()
            out[1] = -.65
            across = place.copy()
            across[1] = -.65
            goals = [(0.20, self.start, above, "approach", .04),
                     (0.34, above, grasp, "descend", .04),
                     (0.46, grasp, grasp, "close", 0.),
                     (0.60, grasp, lift, "lift", 0.),
                     (0.68, lift, out, "clear_counter", 0.),
                     (0.78, out, across, "transport", 0.),
                     (0.86, across, place, "over_basin", 0.),
                     (0.90, place, place, "release", .04),
                     (1.01, place, place + [0., -.20, .25], "withdraw", .04)]
            begin = 0.
            for end, a, b, phase, finger in goals:
                if t < end:
                    u = np.clip((t-begin)/(end-begin), 0, 1)
                    u = u*u*(3-2*u)
                    desired = (1-u)*np.array(a) + u*np.array(b)
                    self.phase = phase
                    break
                begin = end
            quat = torch.tensor([[math.sqrt(.5), math.sqrt(.5), 0., 0.]], device=self.device)
        else:
            quat0 = torch.tensor([[-.5, -.5, -.5, .5]], device=self.device)
            if t < .28:
                self.phase = "approach_handle"
                u = t/.28
                desired = (1-u)*self.start + u*(self.handle + [0., -.22, 0.])
                angle = 0.
            elif t < .42:
                self.phase = "reach_handle"
                u = (t-.28)/.14
                desired = self.handle + [0., -.22 + .113*u, 0.]
                angle = 0.
            elif t < .54:
                self.phase = "grasp_handle"
                desired = self.handle + [0., -.107, 0.]
                angle = 0.
                finger = 0.
            else:
                self.phase = "pull_door"
                angle = self.angle * min(1., (t-.54)/.40)
                rot = np.array([[math.cos(angle), -math.sin(angle), 0],
                                [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
                desired = self.anchor + rot @ (self.handle-self.anchor) - rot @ np.array([0., .107, 0.])
                finger = 0.
            qz = torch.tensor([[0., 0., math.sin(angle/2), math.cos(angle/2)]], device=self.device)
            quat = quat_mul(qz, quat0)
        pose = tensor(self.robot.data.body_pose_w)[:, self.hand_id]
        pos_b, quat_b = subtract_frame_transforms(self.root[:, :3], self.root[:, 3:7], pose[:, :3], pose[:, 3:7])
        goal_pos, goal_quat = subtract_frame_transforms(self.root[:, :3], self.root[:, 3:7],
            torch.tensor([desired.tolist()], device=self.device), quat)
        self.ik.set_command(torch.cat([goal_pos, goal_quat], dim=-1))
        jac = tensor(self.robot.data.body_link_jacobian_w)[:, self.hand_id-1, :, self.arm]
        # The Lab Jacobian is world-aligned; the IK poses above are root-aligned.
        world_to_root = torch.tensor([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]], device=self.device)
        jac = torch.cat([world_to_root @ jac[:, :3], world_to_root @ jac[:, 3:]], dim=1)
        q_goal = self.ik.compute(pos_b, quat_b, jac, tensor(self.robot.data.joint_pos)[:, self.arm])
        current = tensor(self.robot.data.joint_pos)[:, self.arm]
        q_goal = current + torch.clamp(q_goal-current, -.10, .10)
        limits = tensor(self.robot.data.soft_joint_pos_limits)[:, self.arm]
        q_goal = torch.maximum(torch.minimum(q_goal, limits[..., 1]), limits[..., 0])
        self.ik_error = float(torch.linalg.vector_norm(goal_pos-pos_b))
        if self.phase in ("close", "grasp_handle") and self.ik_error < (.020 if self.pick else .012):
            self.grasp_ready = True
        if finger == 0. and not self.grasp_ready:
            finger = .04
        self.robot.set_joint_position_target(q_goal, joint_ids=self.arm)
        self.robot.set_joint_position_target(torch.full((1, 2), finger, device=self.device), joint_ids=self.fingers)
        self.robot.write_data_to_sim()

    def update(self, dt):
        self.robot.update(dt)
        if self.object:
            self.object.update(dt)

    def state(self):
        pose = tensor(self.robot.data.body_pose_w)[:, self.hand_id]
        tcp = pose[:, :3] + quat_apply(pose[:, 3:7], self.offset)
        return {"phase": self.phase, "ik_error": self.ik_error, "tcp": tcp[0].cpu().tolist(),
                "grasp_alignment_reached": self.grasp_ready,
                "object_pos": tensor(self.object.data.root_pos_w)[0].cpu().tolist() if self.object else None,
                "fingers": tensor(self.robot.data.joint_pos)[0, self.fingers].cpu().tolist()}
