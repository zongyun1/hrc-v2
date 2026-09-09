"""Small, simulator-independent benchmark configuration."""
from dataclasses import dataclass, field
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class HumanMotionConfig:
    enabled: bool = False
    motion: str | None = None
    skeleton: str | None = None
    skin: str | None = None
    yaw: float = 0.
    translation: tuple[float, float, float] = (0., 0., 0.)
    start_delay_s: float = 0.

    def __post_init__(self):
        if len(self.translation) != 3 or not all(math.isfinite(x) for x in (*self.translation, self.yaw, self.start_delay_s)):
            raise ValueError('Human transform and delay must be finite')
        if self.start_delay_s < 0:
            raise ValueError('Human start delay cannot be negative')
        if self.enabled and (not self.motion or not self.skeleton):
            raise ValueError('Enabled TRUMANS motion requires motion and skeleton paths')


@dataclass(frozen=True)
class BenchmarkConfig:
    backend: str = 'lightwheel'
    task: str = 'NavigateKitchen'
    scene: str = 'robocasakitchen-4-2'
    robot: str = 'PandaOmron-Rel'
    seed: int = 0
    num_envs: int = 1
    episode_length_s: float = 12.
    lightwheel_root: str = 'external/LW-BenchHub'
    human: HumanMotionConfig = field(default_factory=HumanMotionConfig)

    def __post_init__(self):
        if self.backend != 'lightwheel':
            raise ValueError('The active benchmark backend is lightwheel')
        if self.num_envs != 1:
            raise ValueError('The minimal benchmark currently supports num_envs=1')
        if not math.isfinite(self.episode_length_s) or self.episode_length_s <= 0:
            raise ValueError('episode_length_s must be positive and finite')

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        human = data.pop('human', {})
        for key in ('motion', 'skeleton', 'skin'):
            if human.get(key):
                human[key] = str((path.parent / human[key]).resolve())
        if 'translation' in human:
            human['translation'] = tuple(human['translation'])
        if data.get('lightwheel_root'):
            data['lightwheel_root'] = str((path.parent / data['lightwheel_root']).resolve())
        return cls(**data, human=HumanMotionConfig(**human))
