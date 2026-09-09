"""ServeTea live objects and contact observations for the Isaac Lab scene probe.

This is a task observer, not a scripted manipulation policy. Contacts use PhysX
pair counts, never proximity or a net-force heuristic.
"""
import torch
import warp as wp
from pxr import Usd, UsdPhysics
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
import isaaclab.sim as sim_utils
from task_semantics import evaluate, upright_tilt_degrees


def tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch


class ServeTeaScene:
    def __init__(self, manifest, kitchen, grasp_contacts=False):
        self.manifest = manifest
        prims = list(Usd.PrimRange(kitchen))

        def body_path(name):
            matches = [str(p.GetPath()) for p in prims if p.GetName() == name
                       and p.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(matches) != 1:
                raise ValueError(f"Expected one rigid body for {name}: {matches}")
            return matches[0]

        self.paths = {name: body_path(spec["body"])
                      for name, spec in manifest["objects"].items()}
        self.objects = {name: RigidObject(RigidObjectCfg(prim_path=path, spawn=None))
                        for name, path in self.paths.items()}
        # Resolve the source fixture collision geoms to their owning USD bodies.
        table_geoms = set(manifest["fixtures"][manifest["target_fixture"]]["contact_geoms"])
        table_paths = set()
        found = set()
        for prim in prims:
            if prim.GetName() not in table_geoms or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            found.add(prim.GetName())
            owner = prim
            while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                owner = owner.GetParent()
            if not owner:
                raise ValueError(f"Table collider has no rigid body: {prim.GetPath()}")
            table_paths.add(str(owner.GetPath()))
        if found != table_geoms or not table_paths:
            raise ValueError(f"Unmatched table collision geoms: {table_geoms - found}")
        self.sensors = {}
        pairs = [
            ("teacup_saucer", "teacup", [self.paths["saucer"]]),
            ("saucer_table", "saucer", sorted(table_paths)),
        ]
        if grasp_contacts:
            for index in (1, 2):
                key = f"finger{index}"
                self.paths[key] = body_path(f"gripper0_right_finger_joint{index}_tip")
                pairs.append((f"{key}_cup", key, [self.paths["teacup"]]))
                # The pad is a separate fixed child body. Watching it alone
                # misses side-of-finger collisions that push the cup away.
                side_key = f"finger{index}_side"
                side_name = "leftfinger" if index == 1 else "rightfinger"
                self.paths[side_key] = body_path(f"gripper0_right_{side_name}")
                pairs.append((f"{side_key}_cup", side_key, [self.paths["teacup"]]))
        for name, source, filters in pairs:
            sim_utils.activate_contact_sensors(self.paths[source])
            self.sensors[name] = ContactSensor(ContactSensorCfg(
                prim_path=self.paths[source], filter_prim_paths_expr=filters,
                update_period=0., track_contact_points=True, max_contact_data_count_per_prim=4096))

    def reset(self):
        for obj in self.objects.values():
            obj.reset()
        for sensor in self.sensors.values():
            sensor.reset()

    def update(self, dt):
        for obj in self.objects.values():
            obj.update(dt)
        for sensor in self.sensors.values():
            sensor.update(dt, force_recompute=True)

    def construct_contact_probe(self):
        """Diagnostic only: release the cup above the settled saucer once."""
        cup, saucer = self.objects["teacup"], self.objects["saucer"]
        pose = tensor(cup.data.root_pose_w).clone()
        pose[:, :3] = tensor(saucer.data.root_pos_w)
        pose[:, 2] += .10
        pose[:, 3:] = torch.tensor([0., 0., 0., 1.], device=pose.device)
        cup.write_root_pose_to_sim(pose)
        cup.write_root_velocity_to_sim(torch.zeros((1, 6), device=pose.device))
        return {"kind": "constructed_contact_state", "cup_pose_xyzw": pose[0].cpu().tolist(),
                "note": "Direct pose write for predicate validation, not robot manipulation"}

    def state(self, articulations, dt):
        positions = {self.manifest["objects"][name]["body"]:
                     tensor(obj.data.root_pos_w)[0].cpu().tolist()
                     for name, obj in self.objects.items()}
        site = self.manifest["gripper_site"]
        gripper = None
        for articulation in articulations:
            if site["body"] not in articulation.body_names:
                continue
            index = articulation.body_names.index(site["body"])
            # Lab 3's body_pose_w uses xyzw quaternions (same as mobile_navigation).
            pose = tensor(articulation.data.body_pose_w)[0, index]
            local = torch.tensor(site["local_pos"], device=pose.device, dtype=pose.dtype)
            xyz, w = pose[3:6], pose[6]
            rotated = local + 2 * torch.cross(xyz, torch.cross(xyz, local, dim=0) + w * local, dim=0)
            gripper = (pose[:3] + rotated).cpu().tolist()
        if gripper is None:
            raise RuntimeError("No live articulation contains the exported gripper site body")
        counts = {}
        for name, sensor in self.sensors.items():
            # PhysX returns forces, points, normals, separations, pair counts,
            # and start indices. Fail explicitly if this backend lacks the API.
            data = sensor.contact_view.get_contact_data(dt=dt)
            pair_counts = data[4] if isinstance(data[4], torch.Tensor) else wp.to_torch(data[4])
            counts[name] = int(pair_counts.sum().item())
        contacts = {name: count > 0 for name, count in counts.items()}
        quaternions = {self.manifest["objects"][name]["body"]:
                       tensor(obj.data.root_pose_w)[0, 3:7].cpu().tolist()
                       for name, obj in self.objects.items()}
        tilt = upright_tilt_degrees(quaternions[self.manifest["objects"]["teacup"]["body"]])
        source_success = evaluate(self.manifest, positions, {}, gripper, contacts)
        self.latest_state = {"object_positions": positions, "gripper_position": gripper,
                "object_quaternions_xyzw": quaternions, "cup_tilt_degrees": tilt,
                "upright_success": source_success and tilt <= 10.,
                "contacts": contacts, "contact_counts": counts,
                "success": source_success}
        return self.latest_state
