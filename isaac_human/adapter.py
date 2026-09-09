"""Isaac Lab adapter for the HARPv2 full-body motion representation.

Defaults to Lab 3 (XYZW); select quaternion_order='wxyz' for Lab 2.x.

Each bone is a prescribed kinematic capsule, with shared FK driving the visual
skin. This is collision geometry for an animated obstacle, not a dynamic human.
Pose writes preserve the existing Isaac avatar's execution model; contact
forces must not be interpreted as biomechanically valid impact measurements.
"""
import numpy as np
from .motion import point_clearance


def capsule_quaternion(axis):
    """XYZW rotation mapping a Z-axis capsule to its world centerline."""
    length = np.linalg.norm(axis)
    if length < 1e-10:
        return np.array([0., 0., 0., 1.])
    direction = axis / length
    if direction[2] < -1 + 1e-8:
        return np.array([1., 0., 0., 0.])
    q = np.r_[np.cross([0., 0., 1.], direction), 1 + direction[2]]
    return q / np.linalg.norm(q)


def capsule_pose(a, b, quaternion_order='xyzw'):
    """Return a capsule center and orientation in the selected runtime convention."""
    if quaternion_order not in ('xyzw', 'wxyz'):
        raise ValueError('quaternion_order must be xyzw or wxyz')
    q = capsule_quaternion(np.asarray(b) - np.asarray(a))
    if quaternion_order == 'wxyz':
        q = q[[3, 0, 1, 2]]
    return np.r_[(np.asarray(a) + np.asarray(b)) / 2, q]


class IsaacHuman:
    def __init__(self, stage, player, device, *, prim_path='/World/Human',
                 robot_path='/World/Franka', show_capsules=False,
                 quaternion_order='xyzw'):
        if quaternion_order not in ('xyzw', 'wxyz'):
            raise ValueError('quaternion_order must be xyzw or wxyz')
        self.quaternion_order = quaternion_order
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, Sdf
        if player.sequence.n_envs != 1:
            raise ValueError('This first Isaac scene adapter supports one human; the motion core is batched')
        self.player, self.device = player, device
        self.bodies, self.sensors, self.paths = [], [], []
        self.mesh = None
        self.expected = None
        self.max_tracking_error = 0.
        # PhysX expects each filter expression to resolve one body per environment,
        # not a wildcard expanding to all robot links in a single environment.
        robot_filters = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(robot_path))
                         if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if not robot_filters:
            raise ValueError(f'No rigid robot links found under {robot_path}')
        starts, ends, radii = player.capsules(0.)
        for index, (a, b, radius) in enumerate(zip(starts[0], ends[0], radii)):
            path = f'{prim_path}/Collision/Bone{index:02d}'
            self.paths.append(path)
            spawn = sim_utils.CapsuleCfg(
                radius=float(radius), height=float(np.linalg.norm(b-a)),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.65, .45, .3)),
                visible=show_capsules or player.skin is None)
            pose = capsule_pose(a, b, self.quaternion_order)
            body = RigidObject(RigidObjectCfg(prim_path=path, spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(pose[:3]),
                                                         rot=tuple(pose[3:]))))
            self.bodies.append(body)
            PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0.)
            self.sensors.append(ContactSensor(ContactSensorCfg(
                prim_path=path, update_period=0., history_length=1,
                filter_prim_paths_expr=robot_filters)))
        # Capsules overlap by design. Keep world/robot contacts, filter all self-pairs.
        for path in self.paths:
            UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(path)).CreateFilteredPairsRel().SetTargets(
                [Sdf.Path(p) for p in self.paths if p != path])
        if player.skin is not None:
            self.mesh = UsdGeom.Mesh.Define(stage, prim_path + '/Skin')
            self.mesh.CreateSubdivisionSchemeAttr('none')
            self.mesh.CreateDoubleSidedAttr(True)
            self.mesh.CreateFaceVertexCountsAttr([3]*len(player.skin.faces))
            self.mesh.CreateFaceVertexIndicesAttr(player.skin.faces.ravel().tolist())
            material = UsdShade.Material.Define(stage, prim_path + '/Material')
            shader = UsdShade.Shader.Define(stage, prim_path + '/Material/Shader')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set((.65, .45, .3))
            shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.8)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
            UsdShade.MaterialBindingAPI.Apply(self.mesh.GetPrim()).Bind(material)
            self.render(0.)
        self.pose_tensor = torch.zeros((1, 7), device=device)

    def render(self, time):
        if self.mesh is not None:
            from pxr import Vt
            self.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(
                self.player.vertices(time)[0].astype(np.float32)))

    def set_time(self, time, render=True):
        import torch
        starts, ends, _ = self.player.capsules(time)
        self.expected = (starts[0]+ends[0])/2
        for body, a, b, center in zip(self.bodies, starts[0], ends[0], self.expected):
            self.pose_tensor.copy_(torch.as_tensor(capsule_pose(a, b, self.quaternion_order)[None],
                                                  dtype=torch.float32, device=self.device))
            if self.quaternion_order == 'wxyz':
                body.write_root_pose_to_sim(root_pose=self.pose_tensor)
            else:
                body.write_root_pose_to_sim_index(root_pose=self.pose_tensor)
        if render:
            self.render(time)

    def update(self, dt):
        import torch
        robot_force = 0.
        for body, sensor, expected in zip(self.bodies, self.sensors, self.expected):
            body.update(dt)
            position = body.data.root_pos_w[0].detach().cpu().numpy()
            self.max_tracking_error = max(self.max_tracking_error, float(np.linalg.norm(position-expected)))
            sensor.update(dt, force_recompute=True)
            forces = sensor.data.force_matrix_w
            if forces is None:
                raise RuntimeError('Robot-filtered human contact forces are unavailable')
            if not isinstance(forces, torch.Tensor):
                forces = forces.torch
            robot_force += float(torch.linalg.vector_norm(forces, dim=-1).sum().item())
        return robot_force

    def clearance(self, point, time):
        starts, ends, radii = self.player.capsules(time)
        return float(point_clearance(np.asarray(point), starts[0], ends[0], radii).min())
