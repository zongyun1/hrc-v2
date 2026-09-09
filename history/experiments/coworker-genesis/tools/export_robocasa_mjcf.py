"""Phase 0.2 (step 1) - dump one RoboCasa layout to a full composed MJCF.

Runs in the robocasa venv (third_party/robocasa-venv). Creates the base `Kitchen`
env (no task) at a given layout/style, resets to build the scene, and writes the
mujoco-composed XML via `env.model.get_xml()`.

Usage (from repo root):
  third_party/robocasa-venv/bin/python tools/export_robocasa_mjcf.py --layout 1 --style 1
"""

import argparse
import os

import robocasa  # noqa: F401  registers the Kitchen environments into robosuite's registry
import robosuite
from robosuite.controllers import load_composite_controller_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Kitchen")
    parser.add_argument("--robot", type=str, default="PandaOmron")
    parser.add_argument("--layout", type=int, default=1)
    parser.add_argument("--style", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="exports")
    args = parser.parse_args()

    config = {
        "env_name": args.task,
        "robots": args.robot,
        "controller_configs": load_composite_controller_config(robot=args.robot),
        "layout_and_style_ids": [[args.layout, args.style]],
        "translucent_robot": False,
    }

    print(f"Building {args.task} layout={args.layout} style={args.style} robot={args.robot} ...")
    env = robosuite.make(
        **config,
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera=None,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )
    env.reset()

    xml = env.model.get_xml()

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"robocasa_layout{args.layout}_style{args.style}.xml")
    with open(out, "w") as f:
        f.write(xml)
    print(f"wrote {out} ({len(xml)} bytes)")

    # Report how asset paths are referenced (relative vs absolute) - matters for Genesis import.
    n_abs = xml.count('file="/')
    n_mesh = xml.count("<mesh ")
    n_tex = xml.count("<texture ")
    print(f"meshes={n_mesh} textures={n_tex} absolute-file-refs={n_abs}")


if __name__ == "__main__":
    main()
