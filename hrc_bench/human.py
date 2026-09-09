"""TRUMANS replay interface; no simulator import until attach()."""
from dataclasses import asdict
import math
from .config import HumanMotionConfig


class TrumanHumanMotion:
    """One prescribed human, simulation-time playback, hold at the last frame.

    reset/advance/observe work with NumPy only. attach/before_physics/after_physics
    connect the same timeline to Isaac Lab's physical capsules and visual skin.
    """
    def __init__(self, config: HumanMotionConfig):
        self.config = config
        self.player = None
        self.avatar = None
        self.elapsed_s = 0.
        self.peak_contact_N = None
        self._pending = False
        self.last_episode = None
        self.provenance = {'config': asdict(config)}
        if config.enabled:
            from isaac_human.motion import MotionPlayer, sha256
            self.player = MotionPlayer.load(config.motion, config.skeleton, config.skin,
                                            yaw=config.yaw, translation=config.translation)
            meta = self.player.sequence.meta
            if meta.get('backend') != 'trumans' or meta.get('synthetic_diagnostic'):
                raise ValueError('Human interface requires a real TRUMANS export')
            if self.player.sequence.n_envs != 1:
                raise ValueError('Human interface requires one motion sequence')
            self.provenance.update(source_meta=meta, sha256={name: sha256(getattr(config, name))
                for name in ('motion', 'skeleton', 'skin') if getattr(config, name)})

    @property
    def source_time_s(self):
        if self.player is None:
            return 0.
        return min(self.player.end_time, max(0., self.elapsed_s-self.config.start_delay_s))

    def reset(self):
        if self.elapsed_s > 0:
            self.last_episode = self.observe()
        self.elapsed_s = 0.
        self.peak_contact_N = 0. if self.avatar is not None else None
        self._pending = False
        if self.avatar is not None:
            self.avatar.max_tracking_error = 0.
            for sensor in self.avatar.sensors:
                sensor.reset()
            self.avatar.set_time(0., render=True)

    def advance(self, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive and finite')
        self.elapsed_s += dt

    def observe(self):
        """Current state only. Does not expose the recorded future motion."""
        result = {'enabled': self.config.enabled, 'elapsed_s': self.elapsed_s,
                  'source_time_s': self.source_time_s, 'peak_robot_contact_N': self.peak_contact_N,
                  'finished': bool(self.player is not None and self.elapsed_s >=
                                   self.config.start_delay_s+self.player.end_time)}
        if self.player is not None:
            joints, _ = self.player.pose(self.source_time_s)
            result['joints_w'] = joints[0].tolist()
        if self.avatar is not None:
            result['max_tracking_error_m'] = self.avatar.max_tracking_error
        return result

    def attach(self, stage, device, robot_path='/World/envs/env_0/Robot'):
        if not self.config.enabled:
            return
        if self.avatar is not None:
            raise RuntimeError('Human is already attached')
        from isaac_human.adapter import IsaacHuman
        self.avatar = IsaacHuman(stage, self.player, device,
            prim_path='/World/envs/env_0/Human', robot_path=robot_path)
        self.peak_contact_N = 0.

    def before_physics(self, dt, render=False):
        self.after_physics(dt)
        self.advance(dt)
        if self.avatar is not None:
            self.avatar.set_time(self.source_time_s, render=render)
            self._pending = True

    def after_physics(self, dt):
        if self.avatar is not None and self._pending:
            force = self.avatar.update(dt)
            self.peak_contact_N = max(self.peak_contact_N, force)
            self._pending = False
