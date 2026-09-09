"""Contact-only PandaOmron ServeTea controller for the exported layout-2 instance.

All commands are joint position/velocity/effort targets. No object pose writes,
attachments, collision disabling, or success predicate changes are used.
"""
import math
import numpy as np
import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from mobile_navigation import MobileNavigation, tensor, yaw_xyzw


class ServeTeaPolicy(MobileNavigation):
    pick = False

    def __init__(self, manifest, prim_path, device, scene):
        if (manifest["layout"], manifest["style"], manifest["seed"]) != (2, 1, 0):
            raise ValueError("ServeTea scripted route is validated only for layout=2/style=1/seed=0")
        adapted = dict(manifest)
        adapted["success_spec"] = {"target_pos": [3.416, -3.85, 0.], "target_ori": [0, 0, math.pi/2]}
        super().__init__(adapted, prim_path, device)
        self.robot.cfg.actuators["torso"].stiffness = 100000.
        self.robot.cfg.actuators["torso"].damping = 3000.
        self.robot.cfg.actuators["arm"].stiffness = 3000.
        self.robot.cfg.actuators["arm"].damping = 150.
        self.robot.cfg.actuators["base"].stiffness = 10000.
        self.m, self.scene = manifest, scene
        self.phase = "settle"
        self.failure = None
        self.phase_start = 0
        self.events = []

    def reset(self):
        super().reset()
        self.arm = [self.robot.joint_names.index(f"robot0_joint{i}") for i in range(1, 8)]
        self.fingers = [self.robot.joint_names.index(f"gripper0_right_finger_joint{i}") for i in (1, 2)]
        self.torso = self.robot.joint_names.index("mobilebase0_joint_torso_height")
        self.eef = self.robot.body_names.index(self.m["gripper_site"]["body"])
        self.ik = DifferentialIKController(DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"), 1, self.device)
        self.phase, self.phase_start = "settle", 0
        self.q_hold = self.q0.clone()
        self.error = None
        self.cup_start = self.cup().copy()
        self.grasp_offset = None
        self.closed = False
        self.target_xyz = None
        self.stage_pose = None
        self.events = []
        self.failure = None
        self.nav_index = 0
        self.base_hold = tensor(self.robot.data.body_pose_w)[0, self.base_id, :2].cpu().numpy().copy()
        self.base_initial = self.base_hold.copy()
        self.route = [[.65, -1.50], [.65, -3.95], [3.416, -3.95], [3.416, -3.85]]

    def cup(self):
        return tensor(self.scene.objects["teacup"].data.root_pos_w)[0].cpu().numpy()

    def saucer(self):
        return tensor(self.scene.objects["saucer"].data.root_pos_w)[0].cpu().numpy()

    def pose(self):
        return tensor(self.robot.data.body_pose_w)[0, self.eef].cpu().numpy()

    def transition(self, phase, step):
        self.events.append({"step": step, "from": self.phase, "to": phase,
                            "cup": self.cup().tolist(), "eef": self.pose()[:3].tolist()})
        if phase == "over_saucer":
            self.base_hold = tensor(self.robot.data.body_pose_w)[0, self.base_id, :2].cpu().numpy().copy()
        self.phase, self.phase_start = phase, step
        self.stage_pose = self.pose().copy()
        self.q_hold = tensor(self.robot.data.joint_pos).clone()
        print("SERVE_TEA_PHASE", step, phase, "cup", self.cup().tolist(), flush=True)

    def fail(self, reason, step):
        self.failure = reason
        self.transition("failed", step)

    def command(self, step, total):
        if self.stage_pose is None:
            self.stage_pose = self.pose().copy()
        elapsed = step - self.phase_start
        phase = self.phase
        current = tensor(self.robot.data.joint_pos)
        q_target = self.q_hold.clone()
        base_pose = tensor(self.robot.data.body_pose_w)[0, self.base_id].cpu().numpy()
        velocity = np.zeros(3)
        velocity[:2] = self.world_to_joint @ np.clip(5*(self.base_hold-base_pose[:2]), -.15, .15)
        velocity[2] = np.clip(3*(math.pi/2-yaw_xyzw(base_pose[3:7])), -.25, .25)
        goal = None
        # Approach the handle from above/front, with the jaws closing across
        # its plane. The vessel itself is wider than the maximum jaw opening.
        w, x, y, z = self.m["objects"]["teacup"]["quat_wxyz"]
        cup_yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        c, s = math.cos(cup_yaw/2), math.sin(cup_yaw/2)
        orientation = np.array([-s*math.cos(math.pi/8), c*math.cos(math.pi/8),
                                c*math.sin(math.pi/8), -s*math.sin(math.pi/8)])
        duration = 300
        next_phase = None
        q_target[:, self.torso] = .18
        if phase == "settle":
            q_target = self.q0.clone()
            if elapsed >= 60:
                self.cup_start = self.cup().copy()
                self.transition("raise_torso", step)
        elif phase == "raise_torso":
            q_target[:, self.torso] = .18 * min(1., elapsed / 240)
            if elapsed >= 300:
                self.transition("pregrasp", step)
        elif phase in ("pregrasp", "reach", "close"):
            # Grasp the handle outside the vessel; the vessel exceeds the jaw opening.
            w, x, y, z = self.m["objects"]["teacup"]["quat_wxyz"]
            yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
            offset = np.array([.057*math.sin(yaw), -.057*math.cos(yaw), .038])
            grasp = self.cup_start + offset
            goal = grasp.copy()
            if phase == "pregrasp":
                goal[1] = -.80
                goal[2] = self.cup_start[2] + .08
            duration = 420 if phase == "pregrasp" else 300
            next_phase = {"pregrasp": "reach", "reach": "close", "close": "lift"}[phase]
            if phase == "close":
                self.closed = True
                duration = 180
        elif phase == "lift":
            goal = self.stage_pose[:3] + [0., 0., .065]
            duration, next_phase = 240, "extract"
        elif phase == "extract":
            goal = self.stage_pose[:3].copy()
            goal[1] = -1.01
            duration, next_phase = 360, "carry"
        elif phase == "carry":
            goal = self.stage_pose[:3].copy()
            goal[2] = 1.20
            duration, next_phase = 360, "navigate"
        elif phase == "navigate":
            base_pose = tensor(self.robot.data.body_pose_w)[0, self.base_id].cpu().numpy()
            delta = np.array(self.route[self.nav_index]) - base_pose[:2]
            if np.linalg.norm(delta) < .045:
                if self.nav_index < len(self.route)-1:
                    self.nav_index += 1
                else:
                    self.transition("over_saucer", step)
            speed = 2.0 * delta
            speed *= min(1., .20 / max(np.linalg.norm(speed), 1e-9))
            velocity[:2] = self.world_to_joint @ speed
            yaw = yaw_xyzw(base_pose[3:7])
            velocity[2] = np.clip(2*(math.pi/2-yaw), -.25, .25)
            if elapsed > 7200:
                self.fail("Navigation timeout", step)
        elif phase in ("over_saucer", "lower", "release"):
            goal = self.saucer() + self.grasp_offset + [0., 0., .15 if phase == "over_saucer" else .059]
            duration = 480 if phase == "over_saucer" else 300
            next_phase = {"over_saucer": "lower", "lower": "release", "release": "withdraw"}[phase]
            if phase == "release":
                self.closed = False
                duration = 180
        elif phase == "withdraw":
            goal = self.stage_pose[:3] + [0., -.35, .10]
            duration, next_phase = 360, "done"
        elif phase in ("done", "failed"):
            q_target = self.q_hold.clone()

        if goal is not None:
            self.target_xyz = np.asarray(goal).copy()
            u = min(1., elapsed / duration)
            u = u*u*(3-2*u)
            desired = (1-u)*self.stage_pose[:3] + u*np.asarray(goal)
            start_q = self.stage_pose[3:7]
            if np.dot(start_q, orientation) < 0:
                orientation = -orientation
            quat = (1-u)*start_q + u*orientation
            quat /= np.linalg.norm(quat)
            pose = tensor(self.robot.data.body_pose_w)[:, self.eef]
            command = torch.tensor([[*desired, *quat]], dtype=torch.float32, device=self.device)
            self.ik.set_command(command)
            jac = tensor(self.robot.data.body_link_jacobian_w)[:, self.eef-1, :, self.arm]
            q_goal = self.ik.compute(pose[:, :3], pose[:, 3:7], jac, current[:, self.arm])
            q_goal = current[:, self.arm] + torch.clamp(q_goal-current[:, self.arm], -.04, .04)
            limits = tensor(self.robot.data.soft_joint_pos_limits)[:, self.arm]
            q_target[:, self.arm] = torch.maximum(torch.minimum(q_goal, limits[..., 1]), limits[..., 0])
            self.error = float(np.linalg.norm(self.pose()[:3] - goal))
            if elapsed >= duration and self.error < .018:
                if phase == "close":
                    self.grasp_offset = self.pose()[:3] - self.cup()
                if phase == "lift" and self.cup()[2] < self.cup_start[2]+.035:
                    self.fail("Cup did not lift with the fingers", step)
                else:
                    self.transition(next_phase, step)
            elif elapsed > duration+600:
                self.fail(f"{phase} IK timeout (position error {self.error:.4f} m)", step)

        width = 0. if self.closed else (.015 if phase in ("pregrasp", "reach") else .04)
        if phase == "close":
            width = .015 * max(0., 1-elapsed/120)
        q_target[:, self.fingers] = torch.tensor([[width, -width]], device=self.device)
        self.velocity_command = velocity.tolist()
        self.robot.set_joint_position_target(q_target[:, self.hold_joints], joint_ids=self.hold_joints)
        base_q = self.q0[:, self.base_joints].clone()
        shift = self.world_to_joint @ (self.base_hold-self.base_initial)
        base_q[:, :2] += torch.tensor([shift.tolist()], device=self.device)
        if phase == "navigate":
            base_q = current[:, self.base_joints].clone()
        self.robot.set_joint_position_target(base_q, joint_ids=self.base_joints)
        self.robot.set_joint_velocity_target(torch.tensor([velocity.tolist()], device=self.device), joint_ids=self.base_joints)
        resistance = torch.zeros_like(current)
        v = tensor(self.robot.data.joint_vel)
        resistance[:, self.base_joints] = -250.*torch.tanh(v[:, self.base_joints]/.01)
        resistance[:, self.torso] = -1000.*torch.tanh(v[:, self.torso]/.01)
        self.robot.set_joint_effort_target(resistance)
        self.robot.write_data_to_sim()

    def state(self):
        base = tensor(self.robot.data.body_pose_w)[0, self.base_id].cpu().tolist()
        return {"phase": self.phase, "failure": self.failure, "ik_error": self.error,
                "tcp": self.pose()[:3].tolist(), "object_pos": self.cup().tolist(),
                "base_pos": base[:3], "base_yaw": yaw_xyzw(base[3:7]),
                "fingers": tensor(self.robot.data.joint_pos)[0, self.fingers].cpu().tolist(),
                "velocity_command": self.velocity_command, "target_xyz": None if self.target_xyz is None else self.target_xyz.tolist(),
                "joint_positions": dict(zip(self.robot.joint_names, tensor(self.robot.data.joint_pos)[0].cpu().tolist())),
                "eef_quat": self.pose()[3:7].tolist(), "phase_events": self.events, "direct_object_pose_writes": 0}
