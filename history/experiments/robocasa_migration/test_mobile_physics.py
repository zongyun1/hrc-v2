"""USD-only regression tests; run with a Python environment containing usd-core."""
import unittest
import tempfile
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from mobile_physics import anchor_reference_root, apply_mobile_physics


class ReferenceAnchorTests(unittest.TestCase):
    def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()
        kitchen = UsdGeom.Xform.Define(self.stage, "/World/Kitchen")
        kitchen.AddTranslateOp().Set(Gf.Vec3d(2, -3, 1))
        kitchen.AddRotateZOp().Set(30)
        self.kitchen = kitchen.GetPrim()
        root = UsdGeom.Xform.Define(self.stage, "/World/Kitchen/robot0_base")
        root.AddTranslateOp().Set(Gf.Vec3d(10, 10, 0))
        root.AddRotateZOp().Set(90)
        self.root = root.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(self.root)
        UsdPhysics.ArticulationRootAPI.Apply(self.root)
        self.root.AddAppliedSchema("NewtonArticulationRootAPI")
        self.root.CreateAttribute("newton:selfCollisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        self.anchor = UsdPhysics.FixedJoint.Define(self.stage, self.root.GetPath().AppendChild("PhysicsFixedJoint"))
        self.anchor.CreateBody0Rel().SetTargets([self.kitchen.GetPath()])
        self.anchor.CreateBody1Rel().SetTargets([self.root.GetPath()])
        self.planar = []
        parent = self.root
        for name, schema, axis in (("forward", UsdPhysics.PrismaticJoint, "X"),
                                   ("side", UsdPhysics.PrismaticJoint, "Y"),
                                   ("yaw", UsdPhysics.RevoluteJoint, "Z")):
            link = UsdGeom.Xform.Define(self.stage, parent.GetPath().AppendChild(name)).GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(link)
            joint = schema.Define(self.stage, link.GetPath().AppendChild("joint"))
            joint.CreateBody0Rel().SetTargets([parent.GetPath()])
            joint.CreateBody1Rel().SetTargets([link.GetPath()])
            joint.CreateAxisAttr(axis)
            self.planar.append(joint)
            parent = link

    def test_world_anchor_preserves_composed_pose(self):
        before = UsdGeom.XformCache().GetLocalToWorldTransform(self.root)
        report = anchor_reference_root(self.kitchen)
        self.assertEqual(self.anchor.GetBody0Rel().GetTargets(), [])
        self.assertEqual(self.anchor.GetBody1Rel().GetTargets(), [self.root.GetPath()])
        self.assertTrue(Gf.IsClose(Gf.Vec3d(self.anchor.GetLocalPos0Attr().Get()),
                                   before.ExtractTranslation(), 1e-5))
        expected = Gf.Matrix3d(before.ExtractRotationQuat())
        actual = Gf.Matrix3d(Gf.Quatd(self.anchor.GetLocalRot0Attr().Get()))
        self.assertTrue(Gf.IsClose(actual, expected, 1e-6))
        self.assertEqual(self.anchor.GetLocalPos1Attr().Get(), Gf.Vec3f(0))
        self.assertEqual(self.anchor.GetLocalRot1Attr().Get(), Gf.Quatf(1))
        self.assertEqual(report["articulation_root"], str(self.anchor.GetPath()))
        self.assertIs(report["preserved_self_collision"], False)
        self.assertIs(self.anchor.GetPrim().GetAttribute("physxArticulation:enabledSelfCollisions").Get(), False)
        self.assertFalse(self.root.HasAPI(UsdPhysics.ArticulationRootAPI))
        self.assertNotIn("NewtonArticulationRootAPI", self.root.GetMetadata("apiSchemas").GetAppliedItems())
        self.assertTrue(self.anchor.GetPrim().HasAPI(UsdPhysics.ArticulationRootAPI))

    def test_repair_is_idempotent_and_preserves_planar_dofs(self):
        relations = [(j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets(), j.GetAxisAttr().Get())
                     for j in self.planar]
        first = anchor_reference_root(self.kitchen)
        self.assertEqual(anchor_reference_root(self.kitchen), first)
        self.assertEqual(relations, [(j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets(), j.GetAxisAttr().Get())
                                     for j in self.planar])
        self.assertEqual(sum(p.HasAPI(UsdPhysics.ArticulationRootAPI)
                             for p in Usd.PrimRange(self.kitchen)), 1)
        self.assertTrue(self.anchor.GetJointEnabledAttr().Get())
        self.assertFalse(self.anchor.GetExcludeFromArticulationAttr().Get())

    def test_missing_anchor_fails_before_changing_root(self):
        self.stage.RemovePrim(self.anchor.GetPath())
        with self.assertRaisesRegex(ValueError, "one reference-root fixed joint"):
            anchor_reference_root(self.kitchen)
        self.assertTrue(self.root.HasAPI(UsdPhysics.ArticulationRootAPI))

    def test_duplicate_anchor_is_rejected(self):
        duplicate = UsdPhysics.FixedJoint.Define(self.stage, "/World/Kitchen/duplicate")
        duplicate.CreateBody1Rel().SetTargets([self.root.GetPath()])
        with self.assertRaisesRegex(ValueError, "found 2"):
            anchor_reference_root(self.kitchen)

    def test_enabled_self_collision_is_preserved_too(self):
        self.root.GetAttribute("newton:selfCollisionEnabled").Set(True)
        report = anchor_reference_root(self.kitchen)
        self.assertIs(report["preserved_self_collision"], True)
        self.assertIs(self.anchor.GetPrim().GetAttribute("physxArticulation:enabledSelfCollisions").Get(), True)

    def test_gripper_friction_repair_is_opt_in_and_preserves_armature(self):
        finger = UsdPhysics.PrismaticJoint.Define(self.stage, self.root.GetPath().AppendChild("gripper0_right_finger_joint1")).GetPrim()
        friction = finger.CreateAttribute("physxJoint:jointFriction", Sdf.ValueTypeNames.Float)
        friction.Set(1.)
        armature = finger.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float)
        armature.Set(1.)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "dynamics.json"
            source.write_text("{}")
            apply_mobile_physics(self.kitchen, source)
            self.assertEqual(friction.Get(), 1.)
            report = apply_mobile_physics(self.kitchen, source, restore_gripper_friction=True)
        self.assertEqual(friction.Get(), 0.)
        self.assertEqual(armature.Get(), 1.)
        self.assertEqual(report["removed_legacy_joint_friction"][finger.GetName()], 1.)

    def test_source_contact_friction_restores_static_and_max_combination(self):
        prim = self.stage.DefinePrim('/World/Kitchen/material', 'Material')
        material = UsdPhysics.MaterialAPI.Apply(prim)
        material.CreateDynamicFrictionAttr(2.)
        pad = self.stage.DefinePrim('/World/Kitchen/gripper0_right_finger1_pad_collision', 'Cube')
        pad.CreateRelationship('material:binding:physics').SetTargets([prim.GetPath()])
        other = UsdPhysics.MaterialAPI.Apply(self.stage.DefinePrim('/World/Kitchen/table_material', 'Material'))
        other.CreateDynamicFrictionAttr(100.)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'dynamics.json'
            source.write_text('{}')
            apply_mobile_physics(self.kitchen, source)
            self.assertFalse(material.GetStaticFrictionAttr().HasAuthoredValueOpinion())
            apply_mobile_physics(self.kitchen, source, source_contact_friction=True)
        self.assertEqual(material.GetStaticFrictionAttr().Get(), 2.)
        self.assertEqual(prim.GetAttribute('physxMaterial:frictionCombineMode').Get(), 'max')
        self.assertFalse(other.GetStaticFrictionAttr().HasAuthoredValueOpinion())


if __name__ == "__main__":
    unittest.main()
