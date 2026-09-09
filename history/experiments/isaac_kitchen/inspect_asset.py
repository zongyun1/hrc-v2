"""Inspect kitchen units and furniture bounds without starting the simulator."""
import sys
from pxr import Usd, UsdGeom

stage = Usd.Stage.Open(sys.argv[1])
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
print("units", UsdGeom.GetStageMetersPerUnit(stage), "up", UsdGeom.GetStageUpAxis(stage))
print("default", stage.GetDefaultPrim().GetPath())
print("bounds", cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange())
for prim in stage.Traverse():
    if prim.GetPath().pathElementCount <= 3 and prim.IsA(UsdGeom.Xform):
        print(prim.GetPath(), cache.ComputeWorldBound(prim).ComputeAlignedRange())
