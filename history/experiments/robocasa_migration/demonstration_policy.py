"""Track verified native joint motion through Isaac actuators and real contacts.

Object states are observed only. Positions are written only by reset inherited
from MobileNavigation; rollout uses joint position/velocity targets.
"""
import json
import numpy as np
import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from mobile_navigation import MobileNavigation, tensor, yaw_xyzw
from pose_feedback import held_object_target, blend_quat, multiply, inverse


class DemonstrationPolicy(MobileNavigation):
    def __init__(self, manifest, prim_path, device, scene, result_path, placement_feedback=False):
        report = json.loads(result_path.read_text())
        if report["mode"] != "official_demonstration_action_replay" or not report["robot_task_success"]:
            raise ValueError("Refusing a demonstration without verified native action success")
        if manifest.get("demonstration", {}).get("episode") != report["episode"]:
            raise ValueError("Scene is not the selected demonstration")
        adapted = dict(manifest)
        adapted["success_spec"] = {"target_pos": [0, 0, 0], "target_ori": [0, 0, 0]}
        super().__init__(adapted, prim_path, device)
        self.m, self.scene, self.rows = manifest, scene, report["trace"]
        self.placement_feedback = placement_feedback
        self.times = np.array([0., *[r["time_s"] for r in self.rows]])
        self.robot.cfg.actuators["base"].stiffness = 30000.
        self.robot.cfg.actuators["torso"].stiffness = 100000.
        self.robot.cfg.actuators["torso"].damping = 3000.
        self.robot.cfg.actuators["arm"].stiffness = 10000.
        self.robot.cfg.actuators["arm"].damping = 300.
        self.robot.cfg.actuators["gripper"].effort_limit_sim = 20.

    def reset(self):
        super().reset()
        self.eef = self.robot.body_names.index(self.m["gripper_site"]["body"])
        self.fingers = [self.robot.joint_names.index(f"gripper0_right_finger_joint{i}") for i in (1, 2)]
        self.arm = [self.robot.joint_names.index(f"robot0_joint{i}") for i in range(1, 8)]
        self.reference = np.array([[self.m["joints"][n]["qpos"][0] for n in self.robot.joint_names],
                                   *[[r["gripper_position_targets"].get(n, r["joint_positions"][n])
                                      for n in self.robot.joint_names] for r in self.rows]])
        self.phase = "demonstration"
        self.failure = None
        self.target_q = self.q0.clone()
        initial_eef = self.m["bodies"][self.m["gripper_site"]["body"]]
        w, x, y, z = initial_eef["quat_wxyz"]
        self.eef_reference = np.array([initial_eef["pos"], *[r["tcp"] for r in self.rows]])
        self.cup_reference = np.array([self.m["objects"]["teacup"]["pos"], *[r["object_pos"] for r in self.rows]])
        self.quat_reference = np.array([[x, y, z, w], *[r["eef_quat_xyzw"] for r in self.rows]])
        for i in range(1, len(self.quat_reference)):
            if np.dot(self.quat_reference[i-1], self.quat_reference[i]) < 0:
                self.quat_reference[i] *= -1
        self.feedback_start = None
        self.placement_offset = np.zeros(3)
        self.feedback_released = False
        self.contact_release_start = None
        self.orientation_offset = np.array([0., 0., 0., 1.])
        self.cup_quat_reference = None
        if self.placement_feedback and all('object_quat_xyzw' in row for row in self.rows):
            w, x, y, z = self.m['objects']['teacup']['quat_wxyz']
            self.cup_quat_reference = np.array([[x, y, z, w], *[r['object_quat_xyzw'] for r in self.rows]])
            for i in range(1, len(self.cup_quat_reference)):
                if np.dot(self.cup_quat_reference[i-1], self.cup_quat_reference[i]) < 0:
                    self.cup_quat_reference[i] *= -1
        self.ik = DifferentialIKController(DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"), 1, self.device)

    def command(self, step, total):
        observed = getattr(self.scene, 'latest_state', None)
        if (self.cup_quat_reference is not None and self.feedback_start is not None
                and self.contact_release_start is None and observed
                and observed['contacts']['teacup_saucer'] and observed['cup_tilt_degrees'] <= 10.):
            positions = observed['object_positions']
            cup = np.array(positions[self.m['objects']['teacup']['body']])
            saucer = np.array(positions[self.m['objects']['saucer']['body']])
            if np.linalg.norm(cup[:2]-saucer[:2]) < .7*self.m['objects']['saucer']['horizontal_radius']:
                self.contact_release_start = step/120.
                self.contact_pose = tensor(self.robot.data.body_pose_w)[:, self.eef].clone()
                self.contact_joints = tensor(self.robot.data.joint_pos).clone()
        if self.contact_release_start is not None:
            elapsed = step/120.-self.contact_release_start
            self.phase = 'contact_release' if elapsed < .8 else 'withdraw' if elapsed < 2.3 else 'done'
            goal = self.contact_pose.clone()
            goal[:, 2] += .32*min(1., max(0., (elapsed-.8)/1.5))
            current = tensor(self.robot.data.joint_pos)
            pose = tensor(self.robot.data.body_pose_w)[:, self.eef]
            jac = tensor(self.robot.data.body_link_jacobian_w)[:, self.eef-1, :, self.arm]
            self.ik.set_command(goal)
            arm_goal = self.ik.compute(pose[:, :3], pose[:, 3:7], jac, current[:, self.arm])
            limits = tensor(self.robot.data.soft_joint_pos_limits)[:, self.arm]
            arm_goal = current[:, self.arm]+torch.clamp(arm_goal-current[:, self.arm], -.04, .04)
            self.target_q = self.contact_joints.clone()
            self.target_q[:, self.arm] = torch.maximum(torch.minimum(arm_goal, limits[..., 1]), limits[..., 0])
            fraction = min(1., elapsed/.4)
            opening = torch.tensor([[.04, -.04]], device=self.device)
            self.target_q[:, self.fingers] = (1-fraction)*self.contact_joints[:, self.fingers]+fraction*opening
            self.robot.set_joint_position_target(self.target_q)
            self.robot.set_joint_velocity_target(torch.zeros_like(self.target_q))
            resistance = torch.zeros_like(self.target_q)
            resistance[:, self.fingers] = -torch.tanh(tensor(self.robot.data.joint_vel)[:, self.fingers]/.001)
            self.robot.set_joint_effort_target(resistance)
            self.robot.write_data_to_sim()
            return
        time = min(step / 120., self.times[-1])
        self.phase = "done" if step / 120. >= self.times[-1] else "demonstration"
        q = np.array([np.interp(time, self.times, self.reference[:, i]) for i in range(self.reference.shape[1])])
        index = max(0, min(len(self.times)-2, np.searchsorted(self.times, time, side="right")-1))
        velocity = (self.reference[index+1]-self.reference[index])/(self.times[index+1]-self.times[index])
        if self.phase == "done":
            velocity[:] = 0
        self.target_q = torch.tensor(q[None], dtype=torch.float32, device=self.device)
        cup_ref = np.array([np.interp(time, self.times, self.cup_reference[:, i]) for i in range(3)])
        saucer_initial = np.array(self.m["objects"]["saucer"]["pos"])
        if self.placement_feedback and self.feedback_start is None and np.linalg.norm(cup_ref-saucer_initial) < .75:
            self.feedback_start = time
        if self.feedback_start is not None:
            pose = tensor(self.robot.data.body_pose_w)[:, self.eef]
            eef_ref = np.array([np.interp(time, self.times, self.eef_reference[:, i]) for i in range(3)])
            quat = np.array([np.interp(time, self.times, self.quat_reference[:, i]) for i in range(4)])
            quat /= np.linalg.norm(quat)
            if not self.feedback_released and max(abs(q[self.fingers])) > .002:
                self.feedback_released = True
            if not self.feedback_released:
                cup = tensor(self.scene.objects["teacup"].data.root_pos_w)[0].cpu().numpy()
                saucer = tensor(self.scene.objects["saucer"].data.root_pos_w)[0].cpu().numpy()
                desired_cup = cup_ref + saucer - saucer_initial
                held_cup_offset = cup - pose[0, :3].cpu().numpy()
                blend = min(1., (time-self.feedback_start)/1.)
                self.placement_offset = blend*(desired_cup-held_cup_offset-eef_ref)
                if self.cup_quat_reference is not None:
                    cup_quat = tensor(self.scene.objects['teacup'].data.root_pose_w)[0, 3:7].cpu().numpy()
                    target_quat = np.array([np.interp(time, self.times, self.cup_quat_reference[:, i]) for i in range(4)])
                    target_quat /= np.linalg.norm(target_quat)
                    target_pos, target_eef_quat = held_object_target(
                        pose[0, :3].cpu().numpy(), pose[0, 3:7].cpu().numpy(),
                        cup, cup_quat, desired_cup, target_quat)
                    self.placement_offset = blend*(target_pos-eef_ref)
                    self.orientation_offset = multiply(blend_quat(quat, target_eef_quat, blend), inverse(quat))
            quat = multiply(self.orientation_offset, quat)
            goal = torch.tensor([[*(eef_ref+self.placement_offset), *quat]], dtype=torch.float32, device=self.device)
            self.ik.set_command(goal)
            current = tensor(self.robot.data.joint_pos)
            jac = tensor(self.robot.data.body_link_jacobian_w)[:, self.eef-1, :, self.arm]
            arm_goal = self.ik.compute(pose[:, :3], pose[:, 3:7], jac, current[:, self.arm])
            arm_goal = current[:, self.arm] + torch.clamp(arm_goal-current[:, self.arm], -.04, .04)
            limits = tensor(self.robot.data.soft_joint_pos_limits)[:, self.arm]
            self.target_q[:, self.arm] = torch.maximum(torch.minimum(arm_goal, limits[..., 1]), limits[..., 0])
            velocity[self.arm] = 0
        self.robot.set_joint_position_target(self.target_q)
        self.robot.set_joint_velocity_target(torch.tensor(velocity[None], dtype=torch.float32, device=self.device))
        # MuJoCo frictionloss is a force, not PhysX's dimensionless joint
        # friction coefficient. The loader clears the mismapped coefficient.
        resistance = torch.zeros_like(self.target_q)
        speed = tensor(self.robot.data.joint_vel)[:, self.fingers]
        resistance[:, self.fingers] = -torch.tanh(speed / .001)  # 1 N, source Panda fingers
        self.robot.set_joint_effort_target(resistance)
        self.robot.write_data_to_sim()

    def state(self):
        pose = tensor(self.robot.data.body_pose_w)[0, self.eef].cpu().tolist()
        base = tensor(self.robot.data.body_pose_w)[0, self.base_id].cpu().tolist()
        actual = tensor(self.robot.data.joint_pos)[0]
        return {"phase": self.phase, "failure": self.failure, "tcp": pose[:3], "eef_quat": pose[3:],
                "object_pos": tensor(self.scene.objects["teacup"].data.root_pos_w)[0].cpu().tolist(),
                "base_pos": base[:3], "base_yaw": yaw_xyzw(base[3:]),
                "joint_positions": dict(zip(self.robot.joint_names, actual.cpu().tolist())),
                "joint_tracking_max_error": float(torch.max(torch.abs(actual-self.target_q[0]))),
                "placement_feedback_start_s": self.feedback_start,
                "placement_offset_m": self.placement_offset.tolist(),
                "orientation_feedback": self.cup_quat_reference is not None,
                "orientation_offset_xyzw": self.orientation_offset.tolist(),
                "contact_release_start_s": self.contact_release_start,
                "direct_object_pose_writes": 0}
