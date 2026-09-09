"""Check walking on the actual avatar rig without starting Isaac Sim."""
import sys
import json
import numpy as np
from scipy.spatial.transform import Rotation
from avatar import AvatarRig
from walking import WalkMotion

rig = AvatarRig(sys.argv[1])
rig.base[:3, :3] = Rotation.from_euler('z', -np.pi / 2).as_matrix()
start, end = np.array([2.10, 1.65, 0.]), np.array([0.90, 1.65, 0.])
rig.base[:3, 3] = start - [0, 0, rig.floor_z]
rig.fk()
walk = WalkMotion(rig, start, end)
lengths = {side: [np.linalg.norm(rig.position(side+b)-rig.position(side+a))
                 for a,b in [('UpLeg','Leg'),('Leg','Foot')]] for side in ('Left','Right')}
feet = []
for step in range(361):
    pos = walk.advance(1/60)
    rig.base[:3, 3] = pos - [0, 0, rig.floor_z]
    rig.local = rig.rest_local.copy()
    rig.fk()
    walk.pose(rig)
    feet.append([rig.position(side+'Foot').tolist() for side in ('Left','Right')])
    for side in lengths:
        current = [np.linalg.norm(rig.position(side+b)-rig.position(side+a))
                   for a,b in [('UpLeg','Leg'),('Leg','Foot')]]
        assert np.allclose(current, lengths[side], atol=1e-5)
assert walk.finished and np.linalg.norm(pos-end) < 1e-6
assert walk.max_foot_error < 0.025, walk.max_foot_error
feet = np.asarray(feet)
assert np.max(np.abs(np.diff(feet,axis=0))) < 0.04
assert np.min(feet[:,:,2]) >= min(v[2] for v in walk.offsets.values()) - 0.025
print(json.dumps({'status':'passed', 'steps':361, 'max_foot_ik_error_m':walk.max_foot_error,
                  'max_foot_step_m':float(np.max(np.linalg.norm(np.diff(feet,axis=0),axis=2)))}))
