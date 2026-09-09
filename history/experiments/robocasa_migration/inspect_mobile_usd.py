"""Offline audit of the converted robot's mass, constraints, and material friction."""
import json
from pathlib import Path
from pxr import Usd, UsdPhysics

root = Path("outputs/robocasa_migration")
report = json.loads((root / "usd/NavigateKitchen/conversion.json").read_text())
stage = Usd.Stage.Open(report["usd_path"])
robot = stage.GetPrimAtPath("/base/Geometry/robot0_base")
for p in Usd.PrimRange(robot):
    if p.HasAPI(UsdPhysics.RigidBodyAPI):
        print("BODY", p.GetName(), "mass", UsdPhysics.MassAPI(p).GetMassAttr().Get(),
              "kinematic", UsdPhysics.RigidBodyAPI(p).GetKinematicEnabledAttr().Get())
    if p.IsA(UsdPhysics.Joint) and "mobile" in p.GetName():
        print("JOINT", p.GetPath(), [(a.GetName(), str(a.Get())) for a in p.GetAttributes() if a.HasAuthoredValue()])
