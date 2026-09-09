"""Existing textured Genesis GLB with procedural walking, separate from SMPL-X."""
import math
from pathlib import Path
import sys
import numpy as np


class LegacyWalk:
    def __init__(self, stage, args, device):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'isaac_kitchen'))
        from avatar import AvatarController
        start, end = np.asarray(args.walk_start), np.asarray(args.walk_end)
        if not np.isfinite([start,end]).all() or abs(start[2]-end[2]) > 1e-6:
            raise ValueError('Legacy walking requires finite endpoints on level ground')
        delta = end-start
        if np.linalg.norm(delta[:2]) < .1:
            raise ValueError('Walking path must be at least 10 cm')
        yaw = math.atan2(delta[1],delta[0])+math.pi/2
        self.avatar = AvatarController(stage, args.legacy_avatar_glb,
            args.output_dir/'avatar_textures', position=start, yaw=yaw, device=device)
        self.avatar.walk_to(end,duration=args.walk_duration)
        self.time = 0.

    @property
    def max_tracking_error(self):
        return self.avatar.max_proxy_error

    def set_time(self, time, render=True):
        if time < self.time:
            raise ValueError('Legacy procedural playback is forward-only')
        self.avatar.step(time-self.time,render_skin=render)
        self.time = time

    def update(self, dt):
        self.avatar.update(dt)
        return None  # This older controller has no robot-filtered contact sensors.

    def clearance(self, point, time):
        return None  # Do not claim SMPL capsule distances for a different body.
