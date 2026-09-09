"""Restore inferred source robot inertia and remove the mismapped joint friction coefficient."""
import json
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf


def anchor_reference_root(kitchen):
    """Fix the MJCF world child, leaving the three chassis DOFs driven.

    The importer targets the scene container in its fixed joint. Explicitly
    target world instead and mark that joint as the reduced-coordinate root.
    A root API on the rigid body can otherwise select a floating articulation.
    """
    roots = [p for p in Usd.PrimRange(kitchen) if p.GetName() == "robot0_base"
             and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(roots) != 1:
        raise ValueError(f"Expected one robot0_base rigid body, found {len(roots)}")
    root = roots[0]
    self_collision = None
    for name in ("newton:selfCollisionEnabled", "physxArticulation:enabledSelfCollisions"):
        attribute = root.GetAttribute(name)
        if attribute and attribute.HasAuthoredValueOpinion():
            self_collision = attribute.Get()
            break
    joints = [UsdPhysics.FixedJoint(p) for p in Usd.PrimRange(kitchen)
              if p.IsA(UsdPhysics.FixedJoint)
              and UsdPhysics.Joint(p).GetBody1Rel().GetTargets() == [root.GetPath()]]
    if len(joints) != 1:
        raise ValueError(f"Expected one reference-root fixed joint, found {len(joints)}")
    joint = joints[0]
    # Rebuild both frames from the composed world pose (including scene placement).
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(root)
    rotation = transform.ExtractRotationQuat()
    joint.CreateBody0Rel().SetTargets([])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*transform.ExtractTranslation()))
    joint.CreateLocalRot0Attr(Gf.Quatf(rotation.GetReal(), Gf.Vec3f(*rotation.GetImaginary())))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateJointEnabledAttr(True)
    joint.CreateExcludeFromArticulationAttr(False)
    for prim in Usd.PrimRange(root):
        # Newton's schema includes PhysicsArticulationRootAPI in the Lab 3
        # container. Removing only the standard API leaves an inherited root.
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAppliedSchema("NewtonArticulationRootAPI")
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    UsdPhysics.ArticulationRootAPI.Apply(joint.GetPrim())
    if self_collision is not None:
        # Articulation settings must follow the API when its root is moved.
        # Otherwise PhysX's default self-collision setting changes the model.
        joint.GetPrim().AddAppliedSchema("PhysxArticulationAPI")
        joint.GetPrim().CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(self_collision)
    return {"reference_body": str(root.GetPath()), "articulation_root": str(joint.GetPath()),
            "world_position": list(transform.ExtractTranslation()), "preserved_self_collision": self_collision}


def apply_mobile_physics(kitchen, dynamics_file, restore_gripper_friction=False, source_contact_friction=False):
    anchor = anchor_reference_root(kitchen)
    source = json.loads(dynamics_file.read_text())
    masses, friction, materials = {}, {}, {}
    gripper_materials = set()
    if source_contact_friction:
        for prim in Usd.PrimRange(kitchen):
            if prim.GetName() in ('gripper0_right_finger1_pad_collision', 'gripper0_right_finger2_pad_collision'):
                gripper_materials.update(prim.GetRelationship('material:binding:physics').GetTargets())
        if not gripper_materials:
            raise ValueError('No gripper pad physics materials found')
    for p in Usd.PrimRange(kitchen):
        name = p.GetName()
        if p.GetPath() in gripper_materials and p.HasAPI(UsdPhysics.MaterialAPI):
            material = UsdPhysics.MaterialAPI(p)
            dynamic = material.GetDynamicFrictionAttr().Get()
            if dynamic is not None:
                materials[str(p.GetPath())] = {'old_static': material.GetStaticFrictionAttr().Get(), 'source_sliding': dynamic}
                material.CreateStaticFrictionAttr(dynamic)
                p.AddAppliedSchema('PhysxMaterialAPI')
                p.CreateAttribute('physxMaterial:frictionCombineMode', Sdf.ValueTypeNames.Token).Set('max')
        if name in source and p.HasAPI(UsdPhysics.RigidBodyAPI):
            s = source[name]
            api = UsdPhysics.MassAPI.Apply(p)
            api.CreateMassAttr(max(s["mass"], .001))
            api.CreateCenterOfMassAttr(Gf.Vec3f(*s["com"]))
            api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[max(v, 1e-6) for v in s["inertia"]]))
            w, x, y, z = s["inertia_quat_wxyz"]
            api.CreatePrincipalAxesAttr(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
            masses[name] = max(s["mass"], .001)
        if (name.startswith("mobilebase0_joint_") or
            (restore_gripper_friction and name.startswith("gripper0_right_finger_joint"))) and p.IsA(UsdPhysics.Joint):
            attr = p.GetAttribute("physxJoint:jointFriction")
            if attr:
                friction[name] = attr.Get()
                attr.Set(0.)
    return {"reference_anchor": anchor, "robot_masses_kg": masses, "removed_legacy_joint_friction": friction,
            "source_contact_materials": materials,
            "gripper_friction_restored_by_controller": restore_gripper_friction,
            "friction_model": "Source mobile frictionloss approximated by smooth opposing generalized force in controller."}
