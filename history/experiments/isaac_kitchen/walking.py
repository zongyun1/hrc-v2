"""Kinematic walking with alternating planted feet and two-bone leg IK.

This animates the existing skinned avatar; it is not a dynamic balance controller.
"""
import math
import numpy as np


def smoothstep(u):
    return u * u * (3.0 - 2.0 * u)


class WalkMotion:
    def __init__(self, rig, position, destination, duration=6.0):
        self.start = np.asarray(position, float).copy()
        self.destination = np.asarray(destination, float).copy()
        if self.destination.shape != (3,) or not np.isfinite(self.destination).all() or duration <= 0:
            raise ValueError('Walking needs a finite XYZ destination and positive duration')
        if abs(self.destination[2] - self.start[2]) > 1e-6:
            raise ValueError('Walking currently supports level ground')
        self.duration, self.elapsed = float(duration), 0.0
        self.count = max(2, 2 * math.ceil(np.linalg.norm(self.destination - self.start) / 0.30))
        self.segment = -1
        self.finished = False
        self.max_foot_error = 0.0
        self.feet = {side: rig.position(side + 'Foot') for side in ('Left', 'Right')}
        self.offsets = {side: foot - self.start for side, foot in self.feet.items()}
        self.rotations = {side: rig.world[rig.names[side + 'Foot'], :3, :3].copy() for side in self.feet}

    def advance(self, dt):
        self.elapsed = min(self.elapsed + dt, self.duration)
        progress = self.elapsed / self.duration
        segment = min(int(progress * self.count), self.count - 1)
        swing = 'Left' if segment % 2 == 0 else 'Right'
        if segment != self.segment:
            self.segment = segment
            self.swing_start = self.feet[swing].copy()
            landing_u = 1.0 if segment >= self.count - 2 else (segment + 1) / self.count
            self.swing_end = self.start + smoothstep(landing_u) * (self.destination - self.start) + self.offsets[swing]
        u = min(progress * self.count - segment, 1.0)
        self.feet[swing] = self.swing_start + smoothstep(u) * (self.swing_end - self.swing_start)
        self.feet[swing][2] += 0.055 * math.sin(math.pi * u) ** 2
        position = self.start + smoothstep(progress) * (self.destination - self.start)
        # A small knee bend provides reach for planted feet during the stride.
        position[2] -= 0.05 * math.sin(math.pi * progress)
        self.finished = self.elapsed >= self.duration
        return position

    def pose(self, rig):
        for side, foot_target in self.feet.items():
            hip, knee, ankle = [rig.position(side + name) for name in ('UpLeg', 'Leg', 'Foot')]
            upper, lower = np.linalg.norm(knee - hip), np.linalg.norm(ankle - knee)
            direction = foot_target - hip
            distance = np.linalg.norm(direction)
            direction /= max(distance, 1e-8)
            distance = np.clip(distance, abs(upper - lower) + 1e-5, upper + lower - 1e-5)
            ankle_target = hip + direction * distance
            # Legacy avatar faces local -Y. Knees bend toward the walking direction.
            pole = rig.base[:3, :3] @ np.array([0., -1., 0.])
            pole -= np.dot(pole, direction) * direction
            pole /= max(np.linalg.norm(pole), 1e-8)
            along = (upper ** 2 - lower ** 2 + distance ** 2) / (2 * distance)
            knee_target = hip + along * direction + math.sqrt(max(upper ** 2 - along ** 2, 0)) * pole
            rig.aim(side + 'UpLeg', side + 'Leg', knee_target)
            rig.aim(side + 'Leg', side + 'Foot', ankle_target)
            foot_id = rig.names[side + 'Foot']
            parent_rotation = rig.world[rig.parents[foot_id], :3, :3]
            rig.local[foot_id, :3, :3] = np.linalg.solve(parent_rotation, self.rotations[side])
            rig.fk()
            self.max_foot_error = max(self.max_foot_error, float(np.linalg.norm(rig.position(side + 'Foot') - foot_target)))
