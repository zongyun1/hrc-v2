"""Reproduce the 2026-09-08 navigation diagnosis from saved controlled runs."""
import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hrc-navigation-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(args):
    root = Path("outputs/robocasa_migration")
    manifest = json.loads((root / "source/NavigateKitchen/manifest.json").read_text())
    runs = {}
    for label, backend, job in (("native_old", "native", "722362"), ("native_corrected", "native", "722392"),
                                ("native_matched", "native", "722393"), ("isaac", "probes", args.isaac_job),
                                ("dense_before", "probes", "722394"), ("dense_collision_fixed", "probes", "722403"),
                                ("dense_no_friction", "probes", "722404")):
        path = root / backend / job / "NavigateKitchen/result.json"
        result = json.loads(path.read_text())
        if result["status"] != "simulated":
            raise ValueError(f"Incomplete run: {path}")
        native = backend == "native"
        trace = result["trace"]
        state = trace if native else [t["robot"] for t in trace]
        time = np.array([t["time_s"] if native else (t["step"] + 1) * result.get("physics_dt", 1/120) for t in trace])
        arms = [f"robot0_joint{i}" for i in range(1, 8)]
        deviation = np.array([max(abs(s["joint_positions"][n] - manifest["joints"][n]["qpos"][0]) for n in arms)
                              for s in state]) * 180 / np.pi
        success = [t["source_success"] if native else t["predicate"] for t in trace]
        if native:
            data = np.load(path.parent / "rollout.npz")
            names = list(data["joint_names"])
            velocity = data["qvel"][:, [data["qvel_addresses"][names.index("mobilebase0_joint_mobile_"+n)]
                                       for n in ("forward", "side", "yaw")]]
        else:
            velocity = np.array([s["base_joint_velocities"] for s in state])
        side = np.abs(velocity[[s["phase"] == "traverse_aisle" for s in state], 1])
        stable_speed = float(np.median(side[len(side)//4:3*len(side)//4])) if len(side) >= 4 else None
        runs[label] = {"result": result, "state": state, "time": time, "arm_deviation": deviation, "velocity": velocity,
                       "metrics": {"job": job, "success": result["robot_task_success"],
                                   "final_distance_m": state[-1]["target_distance_m"],
                                   "first_success_s": next((float(t) for t, passed in zip(time, success) if passed), None),
                                   "max_arm_deviation_deg": float(deviation.max()),
                                   "final_torso_m": state[-1]["joint_positions"]["mobilebase0_joint_torso_height"],
                                   "traverse_middle_median_abs_side_speed_m_s": stable_speed}}
    report = {key: value["metrics"] for key, value in runs.items()}
    report["native_reset_checks"] = runs["native_corrected"]["result"]["controller_reset_checks"]
    for name in ("dense_before", "dense_collision_fixed", "dense_no_friction"):
        run = runs[name]
        # Exactly the same zero-command, settled time window in all three runs.
        mask = (run["time"] > .25) & (run["time"] <= .5)
        velocity = run["velocity"][mask, 2]
        report[name]["zero_command_yaw_velocity_rms_rad_s"] = float(np.sqrt(np.mean(velocity**2)))
        report[name]["zero_command_yaw_velocity_sign_changes"] = int(np.sum(velocity[1:]*velocity[:-1] < 0))
    report["interpretation"] = {
        "confirmed": ["Native wrapper initialized an OSC world-frame pose as a base-frame desired goal; original source package was not modified.",
                      "Moving the articulation root failed to transfer its authored self-collision=false setting, causing an 8.2 cm torso displacement.",
                      "At zero command, removing only explicit tanh friction eliminates the 60 Hz yaw sign alternation in the diagnostic window.",
                      "Native speed .6 vs Isaac .3 m/s confounded arrival-time comparison; native matched to .3/.08 did not reach the target within 42 s."],
        "model_inference": "Ignoring other loads and saturation, v ~= command - frictionloss / damping: native .6-250/1000=.35; native .3-250/1000=.05; Isaac .3-250/1500=.133 m/s. This is a steady-motion approximation, not an exact simulator model.",
        "not_fixed": "Default explicit friction remains enabled. Its zero-speed behavior, force limiting and control timestep still differ from native MuJoCo. Disabling friction is a diagnosis, not a faithful migration fix."}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "diagnosis.json").write_text(json.dumps(report, indent=2))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    styles = [("native_corrected", "Native .6 m/s (OSC fixed)", "#007f73"),
              ("native_matched", "Native .3 m/s, .08 m waypoint", "#c76b00"),
              ("isaac", "Isaac .3 m/s", "#4c57bb")]
    for key, label, color in styles:
        run = runs[key]
        distance = [s["target_distance_m"] for s in run["state"]]
        pos = np.array([s["base_pos"] for s in run["state"]])
        axes[0, 0].plot(run["time"], distance, label=label, color=color)
        axes[0, 1].plot(pos[:, 0], pos[:, 1], label=label, color=color)
        axes[1, 2].plot(run["time"], np.array(distance)*100, label=label, color=color)
    axes[0, 0].axhline(.2, color="grey", ls=":", label="Success threshold")
    axes[0, 0].set(title="A. Arrival time depends on controller settings", xlabel="Simulation time (s)", ylabel="Distance to target (m)")
    axes[0, 0].legend(fontsize=8)
    target = manifest["success_spec"]["target_pos"]
    axes[0, 1].scatter(*target[:2], marker="*", s=130, color="black", label="Target")
    axes[0, 1].set(title="B. Same aisle route; different progress", xlabel="World X (m)", ylabel="World Y (m)", aspect="equal")
    for key, label, color in (("native_old", "Old native wrapper", "#c23b3b"), ("native_corrected", "Corrected native wrapper", "#007f73")):
        run = runs[key]
        axes[0, 2].plot(run["time"], run["arm_deviation"], label=label, color=color)
    axes[0, 2].set(title="C. OSC reset bug (not native robot behavior)", xlabel="Simulation time (s)", ylabel="Largest arm-joint offset (deg)")
    axes[0, 2].legend(fontsize=8)
    for key, label, color in (("dense_before", "Root setting omitted", "#c23b3b"), ("dense_collision_fixed", "Self-collision setting preserved", "#007f73")):
        run = runs[key]
        axes[1, 0].plot(run["time"], [100*s["joint_positions"]["mobilebase0_joint_torso_height"] for s in run["state"]], label=label, color=color)
    axes[1, 0].set(title="D. Spurious torso lift from self-collision", xlabel="Simulation time (s)", ylabel="Torso joint position (cm)")
    axes[1, 0].legend(fontsize=8)
    for key, label, color in (("dense_collision_fixed", "Explicit tanh friction ON", "#c23b3b"), ("dense_no_friction", "Friction OFF (diagnostic only)", "#007f73")):
        run = runs[key]
        axes[1, 1].plot(run["time"], run["velocity"][:, 2], label=label, color=color, marker=".", ms=3)
    axes[1, 1].set(title="E. 120 Hz sampling reveals 60 Hz chatter", xlim=(.25, .5), ylim=(-.055, .055), xlabel="Simulation time (s); zero command", ylabel="Yaw velocity (rad/s)")
    axes[1, 1].legend(fontsize=8, loc="lower right")
    axes[1, 2].set(title="F. Near-target drift is not equivalent friction", xlim=(30, 42), ylim=(0, 10), xlabel="Simulation time (s)", ylabel="Distance to target (cm)")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle("PandaOmron navigation migration: controlled diagnosis", fontsize=17)
    fig.savefig(args.output / "navigation_diagnosis.png", dpi=160)
    fig.savefig(args.output / "navigation_diagnosis.pdf")
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-job", default="722412")
    parser.add_argument("--output", type=Path, default=Path("outputs/robocasa_migration/comparisons/navigation_diagnosis"))
    main(parser.parse_args())
