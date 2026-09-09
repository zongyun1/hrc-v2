"""Offline check of C1 event gates using a saved rollout; does not run physics.

The 15 Hz observations are held for four 60 Hz controller ticks. This checks
compatibility with recorded geometry, not closed-loop success or new motion.
"""
import argparse
import json
import math
from pathlib import Path
from interaction import HandoverReceiver

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('rollout', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
data = json.loads((args.rollout / 'result.json').read_text())
avatar = json.loads((args.rollout / 'avatar_result.json').read_text())
by_step = {row['step']: row for row in avatar['trace']}
receiver = HandoverReceiver()
final_palm = data['trace'][-1]['human_palm']
for row in data['trace']:
    human = by_step[row['step']]
    for tick in range(4):
        receiver.step(row['step'] + tick,
                      palm_ready=human['motion_finished'] and human['target_error_m'] < .025,
                      captured=row['human_grasp_active'],
                      fingers_open=min(row['finger_positions_m']) > .035,
                      robot_distance=row['tcp_cube_distance_m'],
                      carry_finished=human['motion_finished']
                      and math.dist(row['human_palm'], final_palm) < .025
                      and math.dist(row['human_palm'], data['trace'][0]['human_palm']) > .15)
summary = dict(mode='offline_recorded_observation_replay_not_new_physics',
               source=str(args.rollout), observation_hz=15, controller_hz=60,
               interpolation='zero_order_hold', state=receiver.state,
               events=receiver.events,
               recorded_capture_step=data['handover']['capture_step'],
               recorded_carry_step=data['handover']['human_carry_step'])
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
if receiver.state != 'complete':
    raise SystemExit('Recorded observation gate check did not complete')
