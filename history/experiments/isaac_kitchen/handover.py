"""Proximity-gated compliant human grasp, applied as forces to a dynamic block."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation


class HumanGrasp:
    def __init__(self, mass=0.1):
        self.mass = mass
        self.captured = False
        self.capture_step = None
        self.capture_offset = None
        self.capture_position = None
        self.capture_jump = None
        self.max_follow_error = 0.0
        self.max_force = 0.0

    def capture(self, cube, avatar, step):
        position = cube.data.root_pos_w[0].detach().cpu().numpy()
        palm = avatar.get_hand_pos()
        offset = position - palm
        if np.linalg.norm(offset[:2]) > 0.015 or not 0.027 < offset[2] < 0.055:
            return False
        self.captured = True
        self.capture_step = step
        self.capture_offset = offset.copy()
        self.capture_position = position.copy()
        self.capture_rotation = Rotation.from_quat(cube.data.root_quat_w[0].detach().cpu().numpy())
        self.previous_palm = palm.copy()
        self.force_tensor = torch.zeros((1, 1, 3), device=cube.device)
        self.torque_tensor = torch.zeros_like(self.force_tensor)
        print('HUMAN_GRASP_CAPTURE', step, offset.tolist(), flush=True)
        return True

    def apply(self, cube, avatar, dt):
        if not self.captured:
            return
        palm = avatar.get_hand_pos()
        target_velocity = (palm - self.previous_palm) / dt
        self.previous_palm = palm.copy()
        position = cube.data.root_pos_w[0].detach().cpu().numpy()
        velocity = cube.data.root_lin_vel_w[0].detach().cpu().numpy()
        angular_velocity = cube.data.root_ang_vel_w[0].detach().cpu().numpy()
        rotation = Rotation.from_quat(cube.data.root_quat_w[0].detach().cpu().numpy())
        # A damped 6-DoF spring approximates the human's grip. Gains are stable
        # for this 100 g block at 60 Hz. Upward grip force balances its weight;
        # gravity, contact and the block's rigid-body dynamics remain enabled.
        force = 80.0 * (palm + self.capture_offset - position) + 5.6 * (target_velocity - velocity)
        force[2] += self.mass * 9.81
        torque = 0.006 * (self.capture_rotation * rotation.inv()).as_rotvec() - 0.0011 * angular_velocity
        force *= min(1.0, 5.0 / max(np.linalg.norm(force), 1e-8))
        torque *= min(1.0, 0.015 / max(np.linalg.norm(torque), 1e-8))
        self.max_force = max(self.max_force, float(np.linalg.norm(force)))
        self.force_tensor.copy_(torch.as_tensor(force, dtype=torch.float32, device=cube.device).reshape(1, 1, 3))
        self.torque_tensor.copy_(torch.as_tensor(torque, dtype=torch.float32, device=cube.device).reshape(1, 1, 3))
        # Keep GPU inputs alive and synchronize Torch/Warp/PhysX streams.
        if str(cube.device).startswith('cuda'):
            torch.cuda.synchronize()
        cube.permanent_wrench_composer.set_forces_and_torques_index(
            self.force_tensor, self.torque_tensor, is_global=True)
        cube.write_data_to_sim()
        if str(cube.device).startswith('cuda'):
            torch.cuda.synchronize()

    def update(self, cube, avatar):
        if not self.captured:
            return
        position = cube.data.root_pos_w[0].detach().cpu().numpy()
        if self.capture_jump is None:
            self.capture_jump = float(np.linalg.norm(position - self.capture_position))
            print('HUMAN_CAPTURE_JUMP_M', self.capture_jump, flush=True)
        error = np.linalg.norm(position - avatar.get_hand_pos() - self.capture_offset)
        self.max_follow_error = max(self.max_follow_error, float(error))
