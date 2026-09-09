"""Derive native cup orientations from saved actual rollout free-joint qpos."""
import argparse
import json
from pathlib import Path
import numpy as np
from task_semantics import upright_tilt_degrees

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--result', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
report = json.loads(args.result.read_text())
data = np.load(args.result.parent / 'rollout.npz')
index = list(data['joint_names']).index('teacup_joint0')
assert data['joint_types'][index] == 0
address = int(data['qpos_addresses'][index])
assert len(data['qpos']) == len(report['trace'])
for row, qpos in zip(report['trace'], data['qpos']):
    w, x, y, z = qpos[address+3:address+7]
    row['object_quat_xyzw'] = [float(x), float(y), float(z), float(w)]
    row['cup_tilt_degrees'] = upright_tilt_degrees(row['object_quat_xyzw'])
report['orientation_derived_from'] = str(args.result.parent / 'rollout.npz')
args.output.write_text(json.dumps(report, indent=2))
print([(r['time_s'], r['cup_tilt_degrees']) for r in report['trace'][::100]])
print('final tilt', report['trace'][-1]['cup_tilt_degrees'])
