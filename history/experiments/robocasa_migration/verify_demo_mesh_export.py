"""Check exported meshes against the actual native demonstration environment."""
import os
os.environ["MUJOCO_GL"] = "disable"
import json
import argparse
from pathlib import Path
import mujoco
import numpy as np
import robosuite
import robocasa
from scipy.spatial import cKDTree
from run_demonstration import load_episode, restore

root = Path("outputs/robocasa_migration/demonstrations")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episode", type=int, default=1)
parser.add_argument("--dataset", type=Path, default=root / "ServeTea/lerobot")
parser.add_argument("--migration-root", type=Path)
args = parser.parse_args()
dataset = args.dataset
migration = args.migration_root or root / f"episode_{args.episode:06d}"
states, meta, xml, _ = load_episode(dataset, args.episode)
kwargs = json.loads((dataset / "extras/dataset_meta.json").read_text())["env_args"]["env_kwargs"]
kwargs.update(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
env = robosuite.make(**kwargs)
try:
    restore(env, meta, xml, states[0])
    native = env.sim.model._model
    exported = mujoco.MjModel.from_xml_path(str(migration / "source/ServeTea/original.xml"))
    result = []
    for gid in range(native.ngeom):
        name = native.geom(gid).name
        if native.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH or not name.startswith(("gripper0_right_finger", "teacup_g")):
            continue
        clouds = []
        for model in (native, exported):
            g = model.geom(name).id
            mid = model.geom_dataid[g]
            points = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid]+model.mesh_vertnum[mid]]
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[g])
            clouds.append(points @ rotation.reshape(3, 3).T + model.geom_pos[g])
        error = max(cKDTree(clouds[0]).query(clouds[1])[0].max(), cKDTree(clouds[1]).query(clouds[0])[0].max())
        result.append({"geom": name, "native_to_export_vertex_error_m": float(error)})
    (migration / "native_mesh_export_check.json").write_text(json.dumps(result, indent=2))
    print("NATIVE_TO_EXPORT_MAX_M", max(r["native_to_export_vertex_error_m"] for r in result))
    assert max(r["native_to_export_vertex_error_m"] for r in result) < 1e-6
finally:
    env.close()
