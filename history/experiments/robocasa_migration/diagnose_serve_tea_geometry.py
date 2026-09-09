"""Offline source FK/contact checks at Isaac robot states; never a rollout."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("outputs/robocasa_migration/source/ServeTea"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.source / "original.xml"))
    data = mujoco.MjData(model)
    reset = np.load(args.source / "mujoco_reset.npz")["qpos"]
    result = json.loads(args.isaac.read_text())
    eef = model.body("gripper0_right_eef").id
    rows = []
    for entry in result["trace"]:
        robot = entry["robot"]
        data.qpos[:] = reset
        for name, value in robot["joint_positions"].items():
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
        mujoco.mj_forward(model, data)
        rotation = Rotation.from_matrix(data.xmat[eef].reshape(3, 3))
        contacts = []
        for contact in data.contact:
            names = [model.geom(int(g)).name for g in (contact.geom1, contact.geom2)]
            if any(n.startswith("gripper") for n in names) and any(n.startswith("teacup") for n in names):
                contacts.append({"geoms": names, "distance": float(contact.dist)})
        rows.append({"step": entry["step"], "phase": robot["phase"],
                     "eef_position_error_m": float(np.linalg.norm(data.xpos[eef]-robot["tcp"])),
                     "eef_orientation_error_rad": float((rotation.inv()*Rotation.from_quat(robot["eef_quat"])).magnitude()),
                     "source_contacts_cup_at_reset": contacts,
                     "isaac_contacts": entry["serve_tea"]["contact_counts"]})
    report = {"mode": "offline_fk_and_counterfactual_contact",
              "note": "Robot qpos copied for FK only; source cup stays at reset. Contact comparison is counterfactual after the real cup moves. No dynamics or task success claimed.",
              "max_eef_position_error_m": max(r["eef_position_error_m"] for r in rows),
              "max_eef_orientation_error_rad": max(r["eef_orientation_error_rad"] for r in rows),
              "trace": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != "trace"}))


if __name__ == "__main__":
    main()
