"""Persist verified geometry/texture fixes in previously converted USD files."""
import json
from pathlib import Path
from pxr import Usd, UsdGeom
from geometry_rules import apply_geometry_rules, repair_material_textures

root = Path("outputs/robocasa_migration")
for task in ("PickPlaceCounterToSink", "OpenCabinet", "OpenMicrowave"):
    report_path = root / "usd" / task / "conversion.json"
    report = json.loads(report_path.read_text())
    if report["status"] != "converted":
        raise RuntimeError(f"Cannot repair unconverted task: {task}")
    stage = Usd.Stage.Open(report["usd_path"])
    kitchen = stage.GetDefaultPrim()
    while instances := [p for p in Usd.PrimRange(kitchen) if p.IsInstance()]:
        for p in instances:
            p.SetInstanceable(False)
    xml = root / "source" / task / "scene.xml"
    report["geometry_rules"] = apply_geometry_rules(kitchen, xml)
    report["repaired_material_count"] = len(repair_material_textures(
        kitchen, xml, Path(report["usd_path"]).parent / "repaired_textures"))
    report["meshes"] = [str(p.GetPath()) for p in Usd.PrimRange(kitchen) if p.IsA(UsdGeom.Mesh)]
    stage.GetRootLayer().Save()
    report["repairs_persisted"] = True
    report_path.write_text(json.dumps(report, indent=2))
    print(task, report["geometry_rules"], "materials", report["repaired_material_count"], flush=True)
