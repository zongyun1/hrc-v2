"""Compare source mesh bounds to converted collision bounds in their rigid-body frames."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf
from scipy.spatial import cKDTree


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--usd", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--repair", action="store_true", help="Correct verified 180-degree mesh-frame errors only")
args = parser.parse_args()
model = mujoco.MjModel.from_xml_path(str(args.source))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
stage = Usd.Stage.Open(str(args.usd))
cache = UsdGeom.XformCache()
result = []
for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
    name = prim.GetName()
    if not name.startswith(("gripper0_right_finger", "teacup_g", "saucer")) or not prim.IsA(UsdGeom.Mesh):
        continue
    owner = prim.GetParent()
    while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
        owner = owner.GetParent()
    if not owner:
        continue
    # Instanced meshes can retain another geom's leaf name. Resolve the
    # source geom whose owning rigid body matches, walking through wrappers.
    candidate = prim
    gid = -1
    while candidate and candidate != owner:
        candidate_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, candidate.GetName())
        if candidate_id >= 0 and model.body(int(model.geom_bodyid[candidate_id])).name == owner.GetName():
            gid = candidate_id
            name = candidate.GetName()
            break
        candidate = candidate.GetParent()
    if gid < 0 or model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
        continue
    mid = model.geom_dataid[gid]
    vertices = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]].astype(float)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, model.geom_quat[gid])
    source = vertices @ rotation.reshape(3, 3).T + model.geom_pos[gid]
    relative = cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(owner).GetInverse()
    converted = np.array([relative.Transform(Gf.Vec3d(*p)) for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()])
    bid = int(model.geom_bodyid[gid])
    source_world = source @ data.xmat[bid].reshape(3, 3).T + data.xpos[bid]
    owner_transform = cache.GetLocalToWorldTransform(owner)
    usd_world = np.array([owner_transform.Transform(Gf.Vec3d(*p)) for p in converted])
    lower, upper = source.min(axis=0), source.max(axis=0)
    lo, hi = converted.min(axis=0), converted.max(axis=0)
    flipped = converted * [-1., -1., 1.]
    error = max(cKDTree(source).query(converted)[0].max(), cKDTree(converted).query(source)[0].max())
    corrected_error = max(cKDTree(source).query(flipped)[0].max(), cKDTree(flipped).query(source)[0].max())
    repaired = False
    if args.repair and error > 1e-5:
        if corrected_error > 1e-6:
            raise ValueError(f"{name}: mismatch is not a verified rigid Z-180 rotation: {corrected_error}")
        correction = Gf.Matrix4d().SetScale(Gf.Vec3d(-1., -1., 1.))
        parent_to_body = cache.GetLocalToWorldTransform(prim.GetParent()) * owner_transform.GetInverse()
        new_local = relative * correction * parent_to_body.GetInverse()
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(new_local)
        repaired = True
    result.append({"geom": name, "source_body": model.body(int(model.geom_bodyid[gid])).name,
                   "usd_body": owner.GetName(), "source_bounds": [lower.tolist(), upper.tolist()],
                   "usd_bounds": [lo.tolist(), hi.tolist()],
                   "max_bound_error_m": float(max(np.max(abs(lo-lower)), np.max(abs(hi-upper)))),
                   "world_bound_error_m": float(max(np.max(abs(source_world.min(0)-usd_world.min(0))), np.max(abs(source_world.max(0)-usd_world.max(0))))),
                   "source_body_quat_wxyz": data.xquat[bid].tolist(),
                   "usd_body_quat_wxyz": [owner_transform.ExtractRotationQuat().GetReal(), *owner_transform.ExtractRotationQuat().GetImaginary()],
                   "usd_approximation": prim.GetAttribute("physics:approximation").Get()})
    result[-1].update(vertex_error_m=float(error), corrected_vertex_error_m=float(corrected_error), repaired=repaired)
if args.repair:
    stage.GetRootLayer().Save()
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2))
for r in sorted(result, key=lambda r:r["max_bound_error_m"], reverse=True):
    print(r["geom"], "local", r["max_bound_error_m"], "world", r["world_bound_error_m"], "quats", r["source_body_quat_wxyz"], r["usd_body_quat_wxyz"])
