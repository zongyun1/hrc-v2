"""CPU validation of the warehouse avatar's receiving and carry trajectory."""
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from avatar import AvatarRig

rig = AvatarRig(sys.argv[1])
rig.base[:3, :3] = Rotation.from_euler('z', -np.pi / 2).as_matrix()
rig.base[:3, 3] = [0.90, 1.65, -rig.floor_z]
rig.fk()
receive = np.array([0.68, 1.70, 0.97])
carry = np.array([0.74, 1.70, 1.15])
errors = []
lengths = [np.linalg.norm(rig.position('Right' + b) - rig.position('Right' + a))
           for a, b in [('Arm', 'ForeArm'), ('ForeArm', 'Hand')]]
for u in np.linspace(0, 1, 31):
    target = receive + u * (carry - receive)
    for _ in range(4):
        rig.solve_hand('Right', target, palm_up=True)
    error = float(np.linalg.norm(rig.palm('Right') - target))
    errors.append(error)
    assert error < 0.005, (target, error)
    new_lengths = [np.linalg.norm(rig.position('Right' + b) - rig.position('Right' + a))
                   for a, b in [('Arm', 'ForeArm'), ('ForeArm', 'Hand')]]
    assert np.allclose(lengths, new_lengths, atol=1e-5)
    wrist = rig.position('RightHand')
    normal = np.cross(rig.position('RightHandIndex1') - wrist,
                      rig.position('RightHandPinky1') - wrist)
    assert normal[2] / np.linalg.norm(normal) > 0.99
    palm = rig.palm('Right').copy()
    rig.curl_fingers('Right', 1.0)
    assert np.linalg.norm(rig.palm('Right') - palm) < 1e-6
print(json.dumps({'status': 'passed', 'targets_checked': len(errors),
                  'max_palm_error_m': max(errors), 'carry_distance_m': float(np.linalg.norm(carry - receive))}))
