"""CPU regression on the real GLB: neutral skin and root displacement invariants."""
import argparse
from pathlib import Path
import sys
import tempfile
import numpy as np
from scipy.spatial.transform import Rotation
from .diagnostic_fixture import write_fixture
from .motion import MotionPlayer
from .retarget import Retargeter, MODEL_TO_WORLD

parser = argparse.ArgumentParser()
parser.add_argument('glb')
args = parser.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'isaac_kitchen'))
from avatar import AvatarRig
with tempfile.TemporaryDirectory() as directory:
    motion,skeleton = write_fixture(directory)
    player = MotionPlayer.load(motion,skeleton)
    rig = AvatarRig(args.glb)
    retarget = Retargeter(rig,player)
    expected = rig.world.copy()
    expected_skin = [rig.deform(part)[0] for part in rig.parts]
    pelvis = rig.position('Hips')/[1,1,retarget.height_scale]
    seq = player.sequence
    seq.transl[:] = pelvis-player.skeleton.rest_pelvis
    seq.global_orient[:] = Rotation.from_matrix(MODEL_TO_WORLD).as_rotvec()
    seq.body_pose[:] = 0.
    player = MotionPlayer(seq,player.skeleton)
    retarget = Retargeter(rig,player)
    retarget.set_time(0.)
    np.testing.assert_allclose(rig.world,expected,atol=2e-6)
    for part,vertices in zip(rig.parts,expected_skin):
        np.testing.assert_allclose(rig.deform(part)[0],vertices,atol=2e-6)
    before = rig.world[:,:3,3].copy()
    player.sequence.transl[:,:,0] += 1.
    retarget.set_time(1.)
    mapped = [target for _,target,_ in retarget.mapping]
    np.testing.assert_allclose(rig.world[mapped,:3,3]-before[mapped],
                              np.broadcast_to([1.,0.,0.],(len(mapped),3)),atol=2e-6)
    assert retarget.max_rotation_error < 1e-5
    print('PASS: real GLB neutral skin, mapped joint orientations, unchanged source root XY')
