"""Summarize a named probe job without conflating conversion and robot success."""
import argparse
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("job")
parser.add_argument("--tasks", nargs="+", default=("PickPlaceCounterToSink", "OpenCabinet", "OpenMicrowave"))
args = parser.parse_args()
root = Path("outputs/robocasa_migration")
summary = {"job": args.job, "tasks": []}
for task in args.tasks:
    source = root / "source" / task
    manifest = json.loads((source / "manifest.json").read_text())
    checks = json.loads((source / "predicate_checks.json").read_text())
    conversion = json.loads((root / "usd" / task / "conversion.json").read_text())
    result_path = root / "probes" / args.job / task / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {"status": "missing"}
    trace = result.get("trace", [])
    item = {"task": task, "source_predicate_checks_passed": all(c["source"] == c["adapted"] == c["constructed_positive"] for c in checks),
            "usd_converted": conversion["status"] == "converted", "simulation_status": result["status"],
            "robot_task_success": result.get("robot_task_success", False),
            "source_predicate_at_end": result.get("source_predicate_at_end", bool(trace and trace[-1]["predicate"])),
            "mode": result.get("mode"), "geometry_rules": result.get("geometry_rules"),
            "video": str(result_path.parent / "preview.mp4") if result.get("camera_enabled", True) else None,
            "repaired_material_count": result.get("repaired_material_count")}
    if task == "NavigateKitchen":
        states = [t["robot"] for t in trace if t.get("robot")]
        if states:
            item["initial_distance_m"] = states[0]["target_distance_m"]
            item["final_distance_m"] = states[-1]["target_distance_m"]
            item["final_orientation_cos"] = states[-1]["orientation_cos"]
            item["base_displacement_m"] = math.dist(states[0]["base_pos"], states[-1]["base_pos"])
            item["base_path_length_m"] = sum(math.dist(a["base_pos"], b["base_pos"]) for a,b in zip(states, states[1:]))
            item["source_reset_position_error_m"] = math.dist(states[0]["base_pos"], manifest["bodies"]["mobilebase0_base"]["pos"])
            item["required_distance_m"] = .20
            item["required_orientation_cos"] = .98
            item["max_abs_base_joint_velocities"] = [max(abs(s["base_joint_velocities"][i]) for s in states)
                                                     for i in range(3)]
            if all("reference_root_pose" in s for s in states):
                item["max_reference_root_translation_m"] = max(
                    math.dist(s["reference_root_pose"][:3], states[0]["reference_root_pose"][:3]) for s in states)
    elif task == "PickPlaceCounterToSink":
        positions = [t["robot"]["object_pos"] for t in trace if t.get("robot") and t["robot"].get("object_pos")]
        if positions:
            item["max_lift_m"] = max(p[2] for p in positions)-positions[0][2]
            item["object_final_pos"] = positions[-1]
            item["object_displacement_m"] = math.dist(positions[0], positions[-1])
    else:
        door = manifest["fixtures"][manifest["target_fixture"]]["door_joints"][0]
        angles = [t["door_positions"][door] for t in trace if t.get("door_positions", {}).get(door) is not None]
        if angles:
            item["maximum_opening_degrees"] = math.degrees(max(abs(q) for q in angles))
            item["final_opening_degrees"] = math.degrees(abs(angles[-1]))
            item["required_opening_degrees"] = math.degrees(.9*(manifest["joints"][door]["range"][1]-manifest["joints"][door]["range"][0]))
    summary["tasks"].append(item)
path = root / "probes" / args.job / "summary.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
