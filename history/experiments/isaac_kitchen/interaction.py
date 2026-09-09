"""C1-inspired event receiver; no simulator, motion-network or robot-phase input.

Behavior-layer experiment only. Capture is supplied by the existing spring-grip
proxy and must not be interpreted as measured finger contact.
"""
from dataclasses import dataclass, field


@dataclass
class HandoverReceiver:
    confirmation_steps: int = 12
    state: str = 'reach'
    stable_steps: int = 0
    events: list = field(default_factory=list)

    def step(self, step, *, palm_ready, captured, fingers_open, robot_distance,
             carry_finished):
        previous = self.state
        if self.state == 'reach' and palm_ready:
            self.state = 'wait'
        elif self.state == 'wait' and captured:
            self.state = 'receive'
        elif self.state == 'receive':
            clear = captured and fingers_open and robot_distance > 0.15
            self.stable_steps = self.stable_steps + 1 if clear else 0
            if self.stable_steps >= self.confirmation_steps:
                self.state = 'carry'
        elif self.state == 'carry' and carry_finished:
            self.state = 'complete'
        if previous != self.state:
            self.events.append({'step': step, 'from': previous, 'to': self.state})
        return self.state

    @property
    def can_carry(self):
        return self.state in ('carry', 'complete')
