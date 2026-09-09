"""Exercise event gates against absent capture and transient clearance."""
from interaction import HandoverReceiver

r = HandoverReceiver()
def advance(step, **kw):
    observed = dict(palm_ready=True, captured=False, fingers_open=False,
                    robot_distance=0.0, carry_finished=False)
    observed.update(kw)
    return r.step(step, **observed)
assert advance(0) == 'wait'
for i in range(1, 601):
    assert advance(i, fingers_open=True, robot_distance=1.) == 'wait'
assert advance(601, captured=True) == 'receive'
for i in range(602, 613):
    assert advance(i, captured=True, fingers_open=True, robot_distance=.2) == 'receive'
assert advance(613, captured=True, robot_distance=.2) == 'receive'
for i in range(614, 625):
    assert advance(i, captured=True, fingers_open=True, robot_distance=.2) == 'receive'
assert advance(625, captured=True, fingers_open=True, robot_distance=.2) == 'carry'
assert advance(626, captured=True, carry_finished=True) == 'complete'
assert [e['step'] for e in r.events] == [0, 601, 625, 626]
assert HandoverReceiver().state == 'reach'
print('PASS: waits without capture, rejects transient clearance, confirms 12 consecutive steps, completes, fresh reset')
