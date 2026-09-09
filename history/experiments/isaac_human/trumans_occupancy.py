"""Build TRUMANS occupancy from the existing RoboCasa MuJoCo export, no Genesis."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco

parser = argparse.ArgumentParser()
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--transform',type=Path,required=True)
args = parser.parse_args()
model = mujoco.MjModel.from_xml_path(str(args.source/'scene.xml'))
data = mujoco.MjData(model)
manifest = json.loads((args.source/'manifest.json').read_text())
for name, spec in manifest['joints'].items():
    j = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
    if j >= 0:
        address = model.jnt_qposadr[j]
        data.qpos[address:address+len(spec['qpos'])] = spec['qpos']
mujoco.mj_forward(model,data)
grid = np.zeros((300,100,400),dtype=bool)
grid[:,0,:] = True
center = np.array([2.75,-1.5,0.])
count = 0
for i in range(model.ngeom):
    body = mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,model.geom_bodyid[i]) or ''
    if body.startswith(('robot','mobilebase','gripper','manipulator','migration_')):
        continue
    if not (model.geom_contype[i] or model.geom_conaffinity[i]):
        continue
    rot = data.geom_xmat[i].reshape(3,3)
    pos = data.geom_xpos[i]+rot@model.geom_aabb[i,:3]
    half = abs(rot)@model.geom_aabb[i,3:]
    if not (-1 < pos[0] < 6 and -4.5 < pos[1] < 1.5 and -.1 < pos[2] < 3):
        continue
    low = ((pos-half-center)[[1,2,0]]-[-3,0,-4])/.02
    high = ((pos+half-center)[[1,2,0]]-[-3,0,-4])/.02
    lo = np.clip(np.floor(low).astype(int),0,[300,100,400])
    hi = np.clip(np.ceil(high).astype(int)+1,0,[300,100,400])
    grid[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]] = True
    count += 1
args.output.parent.mkdir(parents=True,exist_ok=True)
np.save(args.output,grid)
args.transform.parent.mkdir(parents=True,exist_ok=True)
args.transform.write_text(json.dumps({'xcen':center[0],'ycen':center[1],
    'source':'RoboCasa collision geom world AABBs from MuJoCo',
    'geometry_count':count,'occupied_fraction':float(grid.mean())},indent=2))
print(args.transform.read_text())
