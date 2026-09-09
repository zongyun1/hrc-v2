"""Synthetic capsule fixture for integration tests ONLY, not generated human motion."""
import argparse
import json
from pathlib import Path
import numpy as np
from .vendor.schema import empty, BODY_JOINT_NAMES, BODY_JOINT_PARENTS


def write_fixture(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    offsets = np.array([
        [0, 1, 0], [-.1, -.08, 0], [.1, -.08, 0], [0, .12, 0],
        [0, -.4, 0], [0, -.4, 0], [0, .12, 0],
        [0, -.4, 0], [0, -.4, 0], [0, .12, 0],
        [0, -.04, .12], [0, -.04, .12], [0, .13, 0],
        [-.09, .08, 0], [.09, .08, 0], [0, .13, 0],
        [-.1, 0, 0], [.1, 0, 0], [-.24, 0, 0], [.24, 0, 0],
        [-.22, 0, 0], [.22, 0, 0]], dtype=float)
    joints = offsets.copy()
    bones = []
    for j in range(1, 22):
        p = BODY_JOINT_PARENTS[j]
        joints[j] += joints[p]
        bones.append({'name': BODY_JOINT_NAMES[j], 'body': BODY_JOINT_NAMES[p],
                      'from': [0., 0., 0.], 'to': offsets[j].tolist(),
                      'radius': .055 if j >= 16 else .085})
    skeleton = {'joint_names': BODY_JOINT_NAMES, 'parents': BODY_JOINT_PARENTS,
                'rest_offsets': offsets.tolist(), 'rest_joints': joints.tolist(),
                'bones': bones, 'gender': 'male', 'synthetic_diagnostic': True}
    (directory/'skeleton.json').write_text(json.dumps(skeleton, indent=2))
    seq = empty(1, 241, fps=30., gender='male')
    seq.global_orient[..., 0] = np.pi/2
    seq.transl[0, :, 1] = np.linspace(-1.8, 1.8, 241)-1
    seq.transl[0, :, 2] = 1.
    # Lower both arms; this fixture is not a learned or mocap gait.
    seq.body_pose[0, :, 15, 2] = 1.25
    seq.body_pose[0, :, 16, 2] = -1.25
    seq.meta = {'synthetic_diagnostic': True, 'source': 'hand-authored mathematical integration fixture'}
    seq.save(directory/'motion.npz')
    return directory/'motion.npz', directory/'skeleton.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir')
    args = parser.parse_args()
    print(write_fixture(args.output_dir))
