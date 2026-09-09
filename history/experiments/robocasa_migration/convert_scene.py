"""Run the installed Isaac Lab MJCF converter and audit the resulting USD."""
import argparse
import json
import traceback
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--source", default="outputs/robocasa_migration/source")
parser.add_argument("--output", default="outputs/robocasa_migration/usd")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
from pxr import Usd, UsdGeom, UsdPhysics
from geometry_rules import apply_geometry_rules, repair_material_textures, prepare_converter_materials

output = Path(args.output).resolve() / args.task
output.mkdir(parents=True, exist_ok=True)
try:
    converter_input, material_changes = prepare_converter_materials(Path(args.source) / args.task / "scene.xml")
    cfg = MjcfConverterCfg(asset_path=str(converter_input),
                           usd_dir=str(output), fix_base=False, force_usd_conversion=True)
    # These options exist on the installed 3.0 converter; keep 2.x diagnostic use possible.
    for key, value in {"collision_from_visuals": False, "run_asset_transformer": True,
                       "run_multi_physics_conversion": True, "self_collision": False}.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    converted = MjcfConverter(cfg)
    stage = Usd.Stage.Open(converted.usd_path)
    if stage is None:
        raise RuntimeError("Converter produced no readable USD")
    root_prim = stage.GetDefaultPrim()
    while instances := [p for p in Usd.PrimRange(root_prim) if p.IsInstance()]:
        for prim in instances:
            prim.SetInstanceable(False)
    source_xml = Path(args.source) / args.task / "scene.xml"
    geometry_report = apply_geometry_rules(root_prim, source_xml)
    texture_report = repair_material_textures(root_prim, source_xml,
                                              Path(converted.usd_path).parent / "repaired_textures")
    stage.GetRootLayer().Save()
    meshes, collisions, rigid, articulations, joints = [], [], [], [], []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if prim.IsA(UsdGeom.Mesh): meshes.append(path)
        if prim.HasAPI(UsdPhysics.CollisionAPI): collisions.append(path)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI): rigid.append(path)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI): articulations.append(path)
        if prim.IsA(UsdPhysics.Joint):
            joints.append({"path": path, "type": prim.GetTypeName(),
                           "body0": [str(x) for x in UsdPhysics.Joint(prim).GetBody0Rel().GetTargets()],
                           "body1": [str(x) for x in UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()]})
    report = {"status": "converted", "task": args.task, "usd_path": converted.usd_path,
              "meshes": meshes, "collisions": collisions, "rigid_bodies": rigid,
              "articulations": articulations, "joints": joints,
              "default_prim": str(stage.GetDefaultPrim().GetPath())}
    report["geometry_rules"] = geometry_report
    report["repaired_material_count"] = len(texture_report)
    report["visual_material_clamps"] = material_changes
    (output / "conversion.json").write_text(json.dumps(report, indent=2))
    print("CONVERSION_RESULT", args.task, len(meshes), len(collisions), len(rigid), len(joints), flush=True)
except Exception:
    error = traceback.format_exc()
    print(error, flush=True)
    (output / "conversion.json").write_text(json.dumps({"status": "failed", "task": args.task, "error": error}, indent=2))
    raise
finally:
    app.close()
