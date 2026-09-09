"""Rest-frame rotation retargeting from SMPL-X to the legacy Mixamo GLB rig.

Preserves source timing, root XY and joint rotations through per-joint bind-frame
corrections. Root height is scaled for the avatar's pelvis height. No procedural
gait or leg IK is applied; different proportions can cause foot sliding.
"""
import numpy as np
from .vendor.skin import quats_to_R

JOINT_MAP = dict(zip(
    ('pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2',
     'left_ankle','right_ankle','spine3','left_foot','right_foot','neck',
     'left_collar','right_collar','head','left_shoulder','right_shoulder',
     'left_elbow','right_elbow','left_wrist','right_wrist'),
    ('Hips','LeftUpLeg','RightUpLeg','Spine','LeftLeg','RightLeg','Spine1',
     'LeftFoot','RightFoot','Spine2','LeftToeBase','RightToeBase','Neck',
     'LeftShoulder','RightShoulder','Head','LeftArm','RightArm',
     'LeftForeArm','RightForeArm','LeftHand','RightHand')))
MODEL_TO_WORLD = np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])


class Retargeter:
    def __init__(self, rig, player):
        self.rig, self.player = rig, player
        # glTF avatar faces -Y; neutral SMPL-X mapped into RoboCasa faces +X.
        rig.base = np.eye(4)
        rig.base[:3,:3] = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        rig.base[2,3] = -rig.floor_z
        rig.local = rig.rest_local.copy()
        rig.fk()
        self.mapping = []
        for index, name in enumerate(player.skeleton.joint_names):
            target = rig.names[JOINT_MAP[name]]
            rest_rotation = rig.world[target,:3,:3].copy()
            if not np.allclose(rest_rotation.T@rest_rotation,np.eye(3),atol=1e-4):
                raise ValueError('Retargeting requires a unit-scale avatar joint hierarchy')
            self.mapping.append((index,target,MODEL_TO_WORLD.T@rest_rotation))
        # SMPL-X's model origin is NOT the floor (its rest pelvis can be negative).
        # Use the matching skin's sole height; the joint-only fallback is for CPU
        # neutral-frame tests, while the real retarget entry requires source skin.
        source_floor = (player.skin.rest_verts[:,1].min() if player.skin is not None
                        else player.skeleton.rest_joints[:,1].min())
        source_hip_height = float(player.skeleton.rest_pelvis[1]-source_floor)
        self.height_scale = rig.position('Hips')[2]/source_hip_height
        if not .5 < self.height_scale < 1.5:
            raise ValueError('Unexpected source/target pelvis scale')
        self.max_rotation_error = 0.

    def set_time(self,time):
        pos, quat = self.player.pose(time)
        rotations = quats_to_R(quat[0])
        rig = self.rig
        rig.local = rig.rest_local.copy()
        rig.fk()
        for index, target, correction in self.mapping:
            parent = rig.parents[target]
            parent_world = rig.world[parent] if parent >= 0 else rig.base
            desired = rotations[index]@correction
            rig.local[target,:3,:3] = np.linalg.solve(parent_world[:3,:3],desired)
            if index == 0:
                pelvis = pos[0,0].copy()
                pelvis[2] *= self.height_scale
                rig.local[target,:3,3] = (np.linalg.inv(parent_world)@np.r_[pelvis,1.])[:3]
            rig.fk()
            self.max_rotation_error = max(self.max_rotation_error,
                float(np.max(abs(rig.world[target,:3,:3]-desired))))
        return rig.position('Hips')


class RetargetedHuman:
    def __init__(self,stage,args,device,player):
        from pathlib import Path
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'isaac_kitchen'))
        from avatar import AvatarController
        self.avatar = AvatarController(stage,args.legacy_avatar_glb,args.output_dir/'avatar_textures',
                                       position=(0.,0.,0.),yaw=0.,device=device)
        self.retarget = Retargeter(self.avatar.rig,player)

    @property
    def max_tracking_error(self):
        return self.avatar.max_proxy_error

    def set_time(self,time,render=True):
        import torch
        self.retarget.set_time(time)
        if render:
            self.avatar._render()
        centers = {s:self.avatar.get_hand_pos(s) for s in ('Left','Right')}
        centers['Torso'] = (self.avatar.rig.position('Hips')+self.avatar.rig.position('Neck'))/2
        for name,body in self.avatar.proxies.items():
            pose = torch.tensor([list(centers[name])+[0.,0.,0.,1.]],dtype=torch.float32,device=self.avatar.device)
            body.write_root_pose_to_sim(pose)
            body.write_root_velocity_to_sim(torch.zeros((1,6),device=self.avatar.device))

    def update(self,dt):
        self.avatar.update(dt)
        return None

    def clearance(self,point,time):
        return None
