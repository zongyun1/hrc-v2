"""Small kinematic, skinned human controller for the legacy Mixamo GLB assets.

No Genesis import or motion pickle is needed. glTF column-vector transforms,
linear blend skinning and a two-bone arm IK drive ordinary USD meshes.
The skin is visual; torso and palms have separate kinematic collision proxies.
"""

import json
import math
import struct
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from walking import WalkMotion


def rotation_between(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    cross, dot = np.cross(a, b), float(np.clip(np.dot(a, b), -1, 1))
    if dot < -0.999999:
        axis = np.cross(a, np.eye(3)[np.argmin(np.abs(a))])
        return Rotation.from_rotvec(axis / np.linalg.norm(axis) * math.pi).as_matrix()
    if np.linalg.norm(cross) < 1e-9:
        return np.eye(3)
    return Rotation.from_rotvec(cross / np.linalg.norm(cross) * math.acos(dot)).as_matrix()


class AvatarRig:
    """GLB geometry and skeleton, independent of the simulator for CPU checks."""

    def __init__(self, path):
        raw = Path(path).read_bytes()
        magic, version, size = struct.unpack_from('<4sII', raw)
        if magic != b'glTF' or version != 2 or size != len(raw):
            raise ValueError('Expected a complete glTF 2 GLB')
        offset = 12
        while offset < size:
            length, kind = struct.unpack_from('<II', raw, offset)
            chunk = raw[offset + 8:offset + 8 + length]
            if kind == 0x4E4F534A:
                self.gltf = json.loads(chunk)
            elif kind == 0x004E4942:
                self.binary = chunk
            offset += length + 8
        self.nodes = self.gltf['nodes']
        self.names = {node.get('name', str(i)): i for i, node in enumerate(self.nodes)}
        self.parents = np.full(len(self.nodes), -1, dtype=int)
        self.local = np.tile(np.eye(4), (len(self.nodes), 1, 1))
        for i, node in enumerate(self.nodes):
            if 'matrix' in node:
                self.local[i] = np.asarray(node['matrix']).reshape(4, 4).T
            else:
                self.local[i, :3, :3] = Rotation.from_quat(node.get('rotation', [0, 0, 0, 1])).as_matrix() @ np.diag(node.get('scale', [1, 1, 1]))
                self.local[i, :3, 3] = node.get('translation', [0, 0, 0])
            for child in node.get('children', []):
                self.parents[child] = i
        self.rest_local = self.local.copy()
        self.base = np.eye(4)
        self.fk()
        self.parts = []
        for node_id, node in enumerate(self.nodes):
            if 'mesh' not in node:
                continue
            skin = self.gltf['skins'][node['skin']]
            for prim in self.gltf['meshes'][node['mesh']]['primitives']:
                if prim.get('mode', 4) != 4:
                    raise ValueError('Only triangle meshes are supported')
                attrs = prim['attributes']
                vertices = self.accessor(attrs['POSITION']).astype(float)
                self.parts.append(dict(
                    node=node_id, positions=np.c_[vertices, np.ones(len(vertices))],
                    normals=self.accessor(attrs['NORMAL']).astype(float),
                    uv=self.accessor(attrs['TEXCOORD_0']),
                    indices=self.accessor(prim['indices']).ravel().astype(np.int32),
                    joints=self.accessor(attrs['JOINTS_0']).astype(int),
                    weights=self.accessor(attrs['WEIGHTS_0']).astype(float),
                    skin_joints=np.asarray(skin['joints']),
                    inverse_bind=self.accessor(skin['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1),
                    material=prim.get('material', 0)))
        points = np.concatenate([self.deform(part)[0] for part in self.parts])
        self.floor_z = float(points[:, 2].min())
        self.height = float(np.ptp(points[:, 2]))
        if not 1.2 < self.height < 2.2:
            raise ValueError(f'Expected a meter-scale Z-up legacy avatar, got height {self.height}')

    def accessor(self, index):
        a = self.gltf['accessors'][index]
        if 'sparse' in a:
            raise ValueError('Sparse base geometry is not supported')
        view = self.gltf['bufferViews'][a['bufferView']]
        dtype = np.dtype({5120: '<i1', 5121: '<u1', 5122: '<i2', 5123: '<u2', 5125: '<u4', 5126: '<f4'}[a['componentType']])
        count = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}[a['type']]
        array = np.ndarray((a['count'], count), dtype=dtype, buffer=self.binary,
                           offset=view.get('byteOffset', 0) + a.get('byteOffset', 0),
                           strides=(view.get('byteStride', dtype.itemsize * count), dtype.itemsize)).copy()
        if a.get('normalized') and dtype.kind in 'iu':
            array = np.maximum(array.astype(float) / np.iinfo(dtype).max, -1)
        return array

    def fk(self):
        self.world = np.zeros_like(self.local)
        def visit(i, parent):
            self.world[i] = parent @ self.local[i]
            for child in self.nodes[i].get('children', []):
                visit(child, self.world[i])
        for i in np.flatnonzero(self.parents < 0):
            visit(i, self.base)

    def position(self, name):
        return self.world[self.names[name], :3, 3].copy()

    def palm(self, side):
        return (self.position(side + 'Hand') + self.position(side + 'HandIndex1') + self.position(side + 'HandPinky1')) / 3

    def aim(self, joint, child, target):
        j, c = self.names[joint], self.names[child]
        delta = rotation_between(self.world[c, :3, 3] - self.world[j, :3, 3], target - self.world[j, :3, 3])
        parent = self.world[self.parents[j], :3, :3]
        self.local[j, :3, :3] = np.linalg.solve(parent, delta @ self.world[j, :3, :3])
        self.fk()

    def solve_hand(self, side, palm_target, palm_up=False):
        # Solve the wrist, refine against the actual palm center. The elbow pole
        # points downward/outward so the arm bends alongside, not into, the torso.
        for _ in range(4):
            shoulder = self.position(side + 'Arm')
            elbow, wrist = self.position(side + 'ForeArm'), self.position(side + 'Hand')
            target = np.asarray(palm_target) - (self.palm(side) - wrist)
            upper, lower = np.linalg.norm(elbow - shoulder), np.linalg.norm(wrist - elbow)
            direction = target - shoulder
            distance = np.linalg.norm(direction)
            direction /= max(distance, 1e-8)
            distance = np.clip(distance, abs(upper - lower) + 1e-4, upper + lower - 1e-4)
            target = shoulder + direction * distance
            pole = self.base[:3, :3] @ np.array([0.35 if side == 'Left' else -0.35, 0.0, -1.0])
            pole -= np.dot(pole, direction) * direction
            if np.linalg.norm(pole) < 1e-6:
                pole = np.cross(direction, [1, 0, 0])
            pole /= np.linalg.norm(pole)
            along = (upper * upper - lower * lower + distance * distance) / (2 * distance)
            bend = math.sqrt(max(upper * upper - along * along, 0))
            desired_elbow = shoulder + direction * along + pole * bend
            self.aim(side + 'Arm', side + 'ForeArm', desired_elbow)
            self.aim(side + 'ForeArm', side + 'Hand', target)
            if palm_up:
                wrist = self.position(side + 'Hand')
                normal = np.cross(self.position(side + 'HandIndex1') - wrist,
                                  self.position(side + 'HandPinky1') - wrist)
                if side == 'Left':
                    normal = -normal
                delta = rotation_between(normal, [0, 0, 1])
                j = self.names[side + 'Hand']
                parent = self.world[self.parents[j], :3, :3]
                self.local[j, :3, :3] = np.linalg.solve(parent, delta @ self.world[j, :3, :3])
                self.fk()

    def curl_fingers(self, side, amount):
        for finger in ('Index', 'Middle', 'Ring', 'Pinky', 'Thumb'):
            for segment in (2, 3):
                name = f'{side}Hand{finger}{segment}'
                if name not in self.names:
                    continue
                j = self.names[name]
                axis, direction = self.world[j, :3, 0], self.world[j, :3, 1]
                sign = 1 if np.cross(axis, direction)[2] >= 0 else -1
                angle = (20 if finger == 'Thumb' else 35) * amount * sign
                self.local[j, :3, :3] = self.local[j, :3, :3] @ Rotation.from_euler('x', angle, degrees=True).as_matrix()
                self.fk()

    def deform(self, part):
        matrices = self.world[part['skin_joints']] @ part['inverse_bind']
        blended = np.einsum('vk,vkij->vij', part['weights'], matrices[part['joints']])
        points = np.einsum('vij,vj->vi', blended[:, :3], part['positions'])
        normals = np.einsum('vij,vj->vi', blended[:, :3, :3], part['normals'])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
        return points.astype(np.float32), normals.astype(np.float32)


class AvatarController:
    """Base pose + timed palm IK. All public positions are world XYZ in meters."""

    def __init__(self, stage, glb_path, cache_dir, position=(1.25, -0.95, 0), yaw=-math.pi / 2, device='cuda:0', receiving_hand=False):
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg
        self.rig = AvatarRig(glb_path)
        self.stage, self.device = stage, device
        self._Vt = Vt
        self.position, self.yaw = np.asarray(position, float), yaw
        self.motions, self.meshes, self.proxies = {}, [], {}
        self._command_stamp = None
        self.command_history = []
        self.max_proxy_error = 0.0
        self.receiving_hand = receiving_hand
        self.grip = 0.0
        self.initial_position, self.initial_yaw = self.position.copy(), yaw
        cache = Path(cache_dir).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        UsdGeom.Xform.Define(stage, '/World/Human')
        materials = []
        for index, spec in enumerate(self.rig.gltf['materials']):
            path = f'/World/Human/Looks/Material{index}'
            material = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(stage, path + '/Surface')
            shader.CreateIdAttr('UsdPreviewSurface')
            pbr = spec.get('pbrMetallicRoughness', {})
            shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*pbr.get('baseColorFactor', [1, 1, 1])[:3]))
            shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.8)
            shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)
            if 'baseColorTexture' in pbr:
                texture_info = self.rig.gltf['textures'][pbr['baseColorTexture']['index']]
                img = self.rig.gltf['images'][texture_info['source']]
                view = self.rig.gltf['bufferViews'][img['bufferView']]
                suffix = '.png' if img['mimeType'] == 'image/png' else '.jpg'
                image_path = cache / f'texture_{texture_info["source"]}{suffix}'
                offset = view.get('byteOffset', 0)
                image_path.write_bytes(self.rig.binary[offset:offset + view['byteLength']])
                reader = UsdShade.Shader.Define(stage, path + '/UV')
                reader.CreateIdAttr('UsdPrimvarReader_float2')
                reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
                reader.CreateOutput('result', Sdf.ValueTypeNames.Float2)
                texture = UsdShade.Shader.Define(stage, path + '/Texture')
                texture.CreateIdAttr('UsdUVTexture')
                texture.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(str(image_path))
                texture.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
                texture.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
                texture.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
                shader.GetInput('diffuseColor').ConnectToSource(texture.ConnectableAPI(), 'rgb')
                if spec.get('alphaMode') == 'BLEND':
                    texture.CreateOutput('a', Sdf.ValueTypeNames.Float)
                    shader.CreateInput('opacity', Sdf.ValueTypeNames.Float).ConnectToSource(texture.ConnectableAPI(), 'a')
            shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
            materials.append(material)
        for i, part in enumerate(self.rig.parts):
            mesh = UsdGeom.Mesh.Define(stage, f'/World/Human/Skin/Part{i}')
            mesh.CreateSubdivisionSchemeAttr('none')
            mesh.CreateDoubleSidedAttr(True)
            mesh.CreateFaceVertexCountsAttr(np.full(len(part['indices']) // 3, 3, dtype=np.int32).tolist())
            mesh.CreateFaceVertexIndicesAttr(part['indices'].tolist())
            uv = part['uv'].copy()
            uv[:, 1] = 1 - uv[:, 1]
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'vertex').Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
            mesh.SetNormalsInterpolation('vertex')
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(materials[part['material']])
            self.meshes.append(mesh)
        # Basic physical representation, deliberately simpler than skin geometry.
        for name in ('Torso', 'Left', 'Right'):
            shape = (sim_utils.CapsuleCfg(radius=0.16, height=0.62) if name == 'Torso'
                     else sim_utils.SphereCfg(radius=0.035))
            if name == 'Right' and receiving_hand:
                shape = sim_utils.CuboidCfg(size=(0.09, 0.09, 0.018))
            shape.rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
            shape.collision_props = sim_utils.CollisionPropertiesCfg()
            shape.mass_props = sim_utils.MassPropertiesCfg(mass=1.0)
            shape.visible = False
            self.proxies[name] = RigidObject(RigidObjectCfg(
                prim_path=f'/World/Human/Collision/{name}', spawn=shape,
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(self.position + [0, 0, 1]))))
        self.reset()
        print('AVATAR_CREATED', json.dumps({'asset': str(glb_path), 'height_m': self.rig.height,
                                           'parts': len(self.meshes), 'joints': len(self.rig.nodes)}), flush=True)

    def _base_transform(self):
        self.rig.base[:3, :3] = Rotation.from_euler('z', self.yaw).as_matrix()
        self.rig.base[:3, 3] = self.position - [0, 0, self.rig.floor_z]

    def reset(self):
        self.position, self.yaw = self.initial_position.copy(), self.initial_yaw
        self.motions.clear()
        self.walk_motion = None
        self.grip = 0.0
        self._base_transform()
        self.rig.local = self.rig.rest_local.copy()
        self.rig.fk()
        self.targets = {}
        for side in ('Left', 'Right'):
            # Rest the arms beside the hips, with a small forward bend.
            self.targets[side] = self.rig.position(side + 'Arm') + self.rig.base[:3, :3] @ np.array([0.04 if side == 'Left' else -0.04, -0.09, -0.46])
            self.rig.solve_hand(side, self.targets[side])
        self._render()

    def set_base_pose(self, position, yaw):
        position = np.asarray(position, float)
        if position.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(yaw):
            raise ValueError('Base pose must contain finite XYZ and yaw')
        old = self.rig.base.copy()
        self.position, self.yaw = position.copy(), float(yaw)
        self._base_transform()
        change = self.rig.base @ np.linalg.inv(old)
        self.targets = {side: (change @ np.r_[value, 1])[:3] for side, value in self.targets.items()}
        self.motions.clear()

    def walk_to(self, destination, duration=6.0):
        self.motions.clear()
        self.walk_motion = WalkMotion(self.rig, self.position, destination, duration)

    def is_walking(self):
        return self.walk_motion is not None and not self.walk_motion.finished

    def reach_hand(self, target, hand='right', duration=2.0):
        side = hand.capitalize()
        target = np.asarray(target, float)
        if side not in self.targets or target.shape != (3,) or not np.isfinite(target).all() or not np.isfinite(duration) or duration <= 0:
            raise ValueError('Use left/right, finite XYZ and positive duration')
        self.motions[side] = [self.get_hand_pos(hand), target.copy(), 0.0, float(duration)]

    def get_hand_pos(self, hand='right'):
        return self.rig.palm(hand.capitalize())

    def spare(self):
        return not self.motions and not self.is_walking()

    def poll_command(self, path):
        """Read an atomically replaced JSON command; apply each revision once."""
        path = Path(path)
        if not path.is_file():
            return
        stamp = path.stat().st_mtime_ns
        if stamp == self._command_stamp:
            return
        try:
            cmd = json.loads(path.read_text())
            action = cmd['action']
            if action == 'reach':
                self.reach_hand(cmd['target'], cmd.get('hand', 'right'), cmd.get('duration', 2.0))
            elif action == 'base':
                self.set_base_pose(cmd['position'], cmd['yaw'])
            elif action == 'reset':
                self.reset()
            else:
                raise ValueError(f'Unknown action: {action}')
            self.command_history.append(cmd)
            print('AVATAR_COMMAND', json.dumps(cmd), flush=True)
        except (ValueError, KeyError, TypeError) as exc:
            print('AVATAR_COMMAND_REJECTED', str(exc), flush=True)
        self._command_stamp = stamp

    def step(self, dt, render_skin=True):
        import torch
        if self.is_walking():
            self.set_base_pose(self.walk_motion.advance(dt), self.yaw)
        for side, motion in list(self.motions.items()):
            motion[2] += dt
            u = min(motion[2] / motion[3], 1.0)
            u = u * u * (3 - 2 * u)
            self.targets[side] = motion[0] + u * (motion[1] - motion[0])
            if motion[2] >= motion[3]:
                del self.motions[side]
        # Reseed from rest every frame to avoid accumulating arm twist.
        self.rig.local = self.rig.rest_local.copy()
        self.rig.fk()
        if self.walk_motion is not None:
            self.walk_motion.pose(self.rig)
        for side, target in self.targets.items():
            self.rig.solve_hand(side, target, palm_up=self.receiving_hand and side == 'Right')
        if self.receiving_hand:
            self.rig.curl_fingers('Right', self.grip)
        if render_skin:
            self._render()
        centers = {side: self.get_hand_pos(side) for side in ('Left', 'Right')}
        centers['Torso'] = (self.rig.position('Hips') + self.rig.position('Neck')) / 2
        for name, body in self.proxies.items():
            # Runtime uses XYZW quaternions, including rigid-body root poses.
            pose = torch.tensor([list(centers[name]) + [0, 0, 0, 1]], dtype=torch.float32, device=self.device)
            body.write_root_pose_to_sim(pose)
            body.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.device))

    def update(self, dt):
        """Refresh proxy states after the simulation step and check their tracking."""
        centers = {side: self.get_hand_pos(side) for side in ('Left', 'Right')}
        centers['Torso'] = (self.rig.position('Hips') + self.rig.position('Neck')) / 2
        for name, body in self.proxies.items():
            body.update(dt)
            position = body.data.root_pos_w[0].detach().cpu().numpy()
            self.max_proxy_error = max(self.max_proxy_error, float(np.linalg.norm(position - centers[name])))

    def _render(self):
        for mesh, part in zip(self.meshes, self.rig.parts):
            points, normals = self.rig.deform(part)
            if not np.isfinite(points).all():
                raise RuntimeError('Non-finite avatar skin')
            mesh.GetPointsAttr().Set(self._Vt.Vec3fArray.FromNumpy(points))
            mesh.GetNormalsAttr().Set(self._Vt.Vec3fArray.FromNumpy(normals))
            mesh.CreateExtentAttr().Set(self._Vt.Vec3fArray.FromNumpy(np.array([points.min(0), points.max(0)])))
