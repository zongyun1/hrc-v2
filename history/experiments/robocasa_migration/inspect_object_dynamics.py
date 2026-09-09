"""Audit inferred MuJoCo object inertia against authored USD mass properties."""
import argparse
import json
from pathlib import Path
import mujoco
from pxr import Usd, UsdPhysics, Gf

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--repair", action="store_true", help="Restore exact source mass, COM and inertia to the isolated USD")
args = parser.parse_args()
model = mujoco.MjModel.from_xml_path(str(args.root / "source/ServeTea/original.xml"))
conversion = json.loads((args.root / "usd/ServeTea/conversion.json").read_text())
stage = Usd.Stage.Open(conversion["usd_path"])
rows = []
for prim in Usd.PrimRange(stage.GetPseudoRoot()):
    if prim.GetName() not in ("teacup_main", "saucer_main"):
        continue
    bid = model.body(prim.GetName()).id
    mass = UsdPhysics.MassAPI(prim)
    rows.append({"body": prim.GetName(), "source_mass": float(model.body_mass[bid]),
                 "source_inertia": model.body_inertia[bid].tolist(), "source_com": model.body_ipos[bid].tolist(),
                 "source_axes": model.body_iquat[bid].tolist(), "usd_mass": mass.GetMassAttr().Get(),
                 "usd_inertia": list(mass.GetDiagonalInertiaAttr().Get()) if mass.GetDiagonalInertiaAttr().Get() is not None else None,
                 "usd_com": list(mass.GetCenterOfMassAttr().Get()) if mass.GetCenterOfMassAttr().Get() is not None else None,
                 "path": str(prim.GetPath()), "schemas": prim.GetAppliedSchemas(),
                 "usd_axes": str(mass.GetPrincipalAxesAttr().Get())})
    if args.repair:
        if model.body_mass[bid] <= 0 or min(model.body_inertia[bid]) <= 0:
            raise ValueError(f"Invalid source mass or inertia: {prim.GetName()}")
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(float(model.body_mass[bid]))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*model.body_inertia[bid]))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*model.body_ipos[bid]))
        w, x, y, z = model.body_iquat[bid]
        mass.CreatePrincipalAxesAttr(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
if args.repair:
    stage.GetRootLayer().Save()
filename = "object_dynamics_repair.json" if args.repair else "object_dynamics_audit.json"
(args.root / filename).write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
