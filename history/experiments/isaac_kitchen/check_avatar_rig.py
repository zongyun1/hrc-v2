"""CPU validation of the actual GLB bind pose, skinning and two-hand IK."""
import argparse
import json

import numpy as np
from scipy.spatial.transform import Rotation

from avatar import AvatarRig

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('glb')
args = parser.parse_args()
rig = AvatarRig(args.glb)
bind_errors = []
for part in rig.parts:
    points, normals = rig.deform(part)
    # This legacy GLB bakes Z-up positions into the bind mesh; inverse-bind
    # matrices cancel the skeleton root transform. Skinned nodes must not
    # have the mesh node transform applied a second time.
    expected = part['positions'][:, :3]
    bind_errors.append(float(np.max(np.linalg.norm(points - expected, axis=1))))
    assert np.allclose(part['weights'].sum(1), 1, atol=1e-4)
    assert np.isfinite(points).all() and np.isfinite(normals).all()
assert max(bind_errors) < 1e-4, bind_errors
rig.base[:3, :3] = Rotation.from_euler('z', -np.pi / 2).as_matrix()
rig.base[:3, 3] = [1.25, -0.95, -rig.floor_z]
rig.fk()
errors = []
for side in ('Left', 'Right'):
    for delta in ((0, 0, 0), (0.04, 0, 0.04), (0, 0.04, 0)):
        rig.local = rig.rest_local.copy()
        rig.fk()
        target = np.array([0.83, -0.90 if side == 'Right' else -1.15, 1.10]) + delta
        lengths = [np.linalg.norm(rig.position(side + b) - rig.position(side + a))
                   for a, b in [('Arm', 'ForeArm'), ('ForeArm', 'Hand')]]
        rig.solve_hand(side, target)
        error = float(np.linalg.norm(rig.palm(side) - target))
        errors.append(error)
        assert error < 0.025, (side, target, rig.palm(side), error)
        new_lengths = [np.linalg.norm(rig.position(side + b) - rig.position(side + a))
                       for a, b in [('Arm', 'ForeArm'), ('ForeArm', 'Hand')]]
        assert np.allclose(new_lengths, lengths, atol=1e-5), (lengths, new_lengths)
        for part in rig.parts:
            assert np.isfinite(rig.deform(part)[0]).all()
rig.base[:3, 3] = [1.10, -0.95, -rig.floor_z]
receiving_errors = []
for target in ([0.74, -0.9, 1.08], [0.94, -0.9, 1.15]):
    rig.local = rig.rest_local.copy()
    rig.fk()
    rig.solve_hand('Right', target, palm_up=True)
    receiving_errors.append(float(np.linalg.norm(rig.palm('Right') - target)))
    wrist = rig.position('RightHand')
    normal = np.cross(rig.position('RightHandIndex1') - wrist, rig.position('RightHandPinky1') - wrist)
    assert normal[2] / np.linalg.norm(normal) > 0.99
    palm = rig.palm('Right').copy()
    rig.curl_fingers('Right', 1.0)
    assert np.linalg.norm(rig.palm('Right') - palm) < 1e-6
    for part in rig.parts:
        assert np.isfinite(rig.deform(part)[0]).all()
assert max(receiving_errors) < 0.005, receiving_errors
print(json.dumps({'status': 'passed', 'height_m': rig.height,
                  'max_bind_error_m': max(bind_errors), 'max_palm_ik_error_m': max(errors),
                  'max_receiving_palm_error_m': max(receiving_errors),
                  'ik_targets_checked': len(errors)}))
