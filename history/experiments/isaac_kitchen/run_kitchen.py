"""Human handover with a kitchen Franka or a warehouse Ridgeback + Franka.

Robot grasps use contact and friction; the human uses a compliant grip model.
The warehouse adds a velocity-controlled planar mobile base and collidable racks.
"""

import argparse
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene", choices=("kitchen", "warehouse"), default="kitchen")
parser.add_argument("--kitchen_usd")
parser.add_argument("--mobile_usd", default=(
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd"))
parser.add_argument("--franka_usd", default=(
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"))
parser.add_argument("--output_dir", default="outputs/isaac_kitchen")
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--robot_xy", type=float, nargs=2, default=(0.0, -1.1))
parser.add_argument("--camera_eye", type=float, nargs=3, default=(2.6, -4.0, 2.3))
parser.add_argument("--camera_target", type=float, nargs=3, default=(0.6, -0.6, 1.0))
parser.add_argument("--task", choices=("handover", "grasp"), default="handover")
parser.add_argument("--no_avatar", action="store_true", help="Run the original robot-only grasp")
parser.add_argument("--avatar_glb", default="/scratch/jiabenchen_umass/yz/assets/avatars/custom_Adrian_Keller.glb")
parser.add_argument("--avatar_command_file", help="Poll an atomic JSON command file for live controls")
parser.add_argument("--avatar_mode", choices=("demo", "manual"), default="demo")
parser.add_argument("--interaction_route", choices=("legacy", "c1"), default="legacy")
parser.add_argument("--robot_delay_steps", type=int, default=0)
parser.add_argument("--robot_pause_steps", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.task == "handover" and (args.no_avatar or args.avatar_mode != "demo"):
    parser.error("handover requires the avatar in demo mode; use --task grasp otherwise")
if args.scene == "kitchen" and not args.kitchen_usd:
    parser.error("--kitchen_usd is required for the kitchen scene")
if args.scene == "warehouse" and (args.task != "handover" or args.steps < 2400):
    parser.error("warehouse requires --task handover and --steps >= 2400")
minimum_steps = 1200 if args.task == "handover" else 720 if not args.no_avatar and args.avatar_mode == "demo" else 600
if args.steps < minimum_steps:
    parser.error(f"--steps must be at least {minimum_steps} for this mode")
if min(args.robot_delay_steps, args.robot_pause_steps) < 0:
    parser.error("robot delay/pause steps must be nonnegative")
if (args.interaction_route != "legacy" or args.robot_delay_steps or args.robot_pause_steps) and (args.scene != "kitchen" or args.task != "handover"):
    parser.error("interaction experiments currently require kitchen handover")
app_launcher = AppLauncher(args)
app = app_launcher.app

import imageio.v2 as imageio
import carb
import numpy as np
import omni.replicator.core as rep
import torch
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, PhysxSchema

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_apply
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, RIDGEBACK_FRANKA_PANDA_CFG
from warehouse import build_warehouse
from avatar import AvatarController
from handover import HumanGrasp
from interaction import HandoverReceiver


def main():
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    mobile = args.scene == "warehouse"
    kitchen_path = Path(args.kitchen_usd).resolve() if not mobile else None
    if kitchen_path is not None and not kitchen_path.is_file():
        raise FileNotFoundError(kitchen_path)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=1.0 / 60.0, device=args.device,
        render=sim_utils.RenderCfg(antialiasing_mode="TAA", samples_per_pixel=32,
                                   enable_dlssg=False, enable_dl_denoiser=False)))
    sim.set_camera_view(args.camera_eye, args.camera_target)
    stage = sim.stage
    bounds = None
    if not mobile:
        kitchen_cfg = sim_utils.UsdFileCfg(usd_path=str(kitchen_path))
        kitchen_cfg.func("/World/Kitchen", kitchen_cfg)
        kitchen = stage.GetPrimAtPath("/World/Kitchen")
        # Expand instance roots so nested appliance physics is also overridden.
        # All edits are in this composed stage, never in the downloaded source USD.
        while instances := [p for p in Usd.PrimRange(kitchen) if p.IsInstance()]:
            for prim in instances:
                prim.SetInstanceable(False)
        for prim in list(Usd.PrimRange(kitchen)):
            if prim.IsA(UsdPhysics.Joint) or prim.IsA(UsdPhysics.Scene):
                prim.SetActive(False)
                continue
            schemas = prim.GetAppliedSchemas()
            visual_schemas = [s for s in schemas if not s.startswith(("Physics", "Physx"))]
            if schemas != visual_schemas:
                prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(visual_schemas))
        assert not any(p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.CollisionAPI)
                       for p in Usd.PrimRange(kitchen)), "Kitchen physics was not fully removed"
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(kitchen).ComputeAlignedRange()
        print("KITCHEN_BOUNDS", bounds, flush=True)

    def box(name, size, pos, color):
        cfg = sim_utils.CuboidCfg(size=size, collision_props=sim_utils.CollisionPropertiesCfg(),
                                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))
        cfg.func(f"/World/{name}", cfg, translation=pos)

    box("Floor", (20.0, 20.0, 0.06), (0.0, 0.0, -0.035), (0.30, 0.31, 0.32))
    light_cfg = sim_utils.DomeLightCfg(intensity=900.0, color=(1.0, 0.95, 0.9))
    light_cfg.func("/World/DemoLight", light_cfg)
    x, y = args.robot_xy
    delivery_offset = 1.5 if mobile else 0.0
    if mobile:
        build_warehouse(box, x, y)
        robot_cfg = RIDGEBACK_FRANKA_PANDA_CFG.replace(prim_path="/World/RidgebackFranka")
        robot_cfg.spawn.usd_path = args.mobile_usd
        robot_cfg.init_state.pos = (x, y, 0.0)
        robot_cfg.init_state.joint_pos.update({
            "dummy_base_prismatic_x_joint": -1.4,
            "panda_joint2": -0.7854, "panda_joint4": -1.8,
            "panda_joint6": 1.0146, "panda_joint7": 0.7854,
        })
        for actuator in ("panda_shoulder", "panda_forearm"):
            robot_cfg.actuators[actuator].stiffness = 2000.0
            robot_cfg.actuators[actuator].damping = 80.0
        robot_cfg.actuators["panda_hand"] = FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_hand"].copy()
    else:
        box("Worktop", (1.25, 0.85, 0.06), (x + 0.35, y, 0.77), (0.44, 0.28, 0.15))
        for i, (dx, dy) in enumerate(((-0.18, -0.32), (-0.18, 0.32), (0.88, -0.32), (0.88, 0.32))):
            box(f"TableLeg{i}", (0.06, 0.06, 0.74), (x + dx, y + dy, 0.37), (0.12, 0.13, 0.15))
        robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/Franka")
        robot_cfg.spawn.usd_path = args.franka_usd
        robot_cfg.init_state.pos = (x, y, 0.80)
    robot = Articulation(robot_cfg)
    if mobile:
        # The 5.1 asset leaves its virtual world link floating. Anchor that
        # reference frame; the XY/yaw joints still move the physical chassis.
        robot_prim = stage.GetPrimAtPath("/World/RidgebackFranka")
        world_links = [p for p in Usd.PrimRange(robot_prim)
                       if p.GetName() == "world" and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(world_links) != 1:
            raise RuntimeError("Expected exactly one Ridgeback virtual world link")
        for prim in Usd.PrimRange(robot_prim):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        anchor = UsdPhysics.FixedJoint.Define(stage, "/World/RidgebackFranka/WorldAnchor")
        anchor.CreateBody1Rel().SetTargets([world_links[0].GetPath()])
        UsdPhysics.ArticulationRootAPI.Apply(anchor.GetPrim())
        PhysxSchema.PhysxArticulationAPI.Apply(anchor.GetPrim()).CreateEnabledSelfCollisionsAttr(False)
    print("FRANKA_CREATED", flush=True)
    cube = RigidObject(RigidObjectCfg(
        prim_path="/World/TestCube",
        spawn=sim_utils.CuboidCfg(size=(0.055, 0.055, 0.055),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=4),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.65, 0.40, 0.18), roughness=0.8)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(x + 0.55, y + 0.20, 0.84))))
    print("CUBE_CREATED", flush=True)
    avatar = None
    if not args.no_avatar:
        avatar = AvatarController(stage, args.avatar_glb, output / "avatar_textures",
                                  position=(x + (2.10 if mobile else 1.10 if args.task == "handover" else 1.25), y + 0.15 + delivery_offset, 0.0),
                                  device=sim.device, receiving_hand=args.task == "handover")
    human_grasp = HumanGrasp() if args.task == "handover" else None
    camera = Camera(CameraCfg(
        prim_path="/World/OverviewCamera", update_period=0.0, height=720, width=960,
        data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(
            focal_length=20.0, horizontal_aperture=24.0, clipping_range=(0.05, 100.0))))
    print("CAMERA_CREATED", flush=True)
    sim.reset()
    print("SIM_RESET", flush=True)
    camera.set_world_poses_from_view(
        torch.tensor([args.camera_eye], device=sim.device),
        torch.tensor([args.camera_target], device=sim.device))
    # Use explicit path-tracing samples: this container cannot initialize NGX,
    # so its default real-time denoising leaves heavily speckled camera images.
    rep.settings.set_render_pathtraced(samples_per_pixel=64)
    robot.reset()
    cube.reset()
    print("ROBOT_JOINTS", robot.joint_names, "BODIES", robot.body_names, flush=True)
    initial_q = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(initial_q, torch.zeros_like(initial_q))
    for _ in range(24):
        if avatar:
            avatar.step(sim.get_physics_dt())
        robot.set_joint_position_target(initial_q)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        cube.update(sim.get_physics_dt())
        camera.update(sim.get_physics_dt())
    # Root orientation is identity, so the world Jacobian also uses base axes.
    arm_ids = [robot.joint_names.index(f"panda_joint{i}") for i in range(1, 8)]
    finger_ids = [robot.joint_names.index(f"panda_finger_joint{i}") for i in (1, 2)]
    hand_id = robot.body_names.index("panda_hand")
    ik = DifferentialIKController(DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=1, device=sim.device)

    def tensor(value):
        return value if isinstance(value, torch.Tensor) else value.torch

    root_pose = tensor(robot.data.root_pose_w).clone()
    print("ROBOT_KINEMATICS", "fixed", robot.is_fixed_base, "root_pose", root_pose.tolist(),
          "jacobian", tuple(tensor(robot.data.body_link_jacobian_w).shape), flush=True)
    if mobile:
        # Solve in world axes: the mobile asset's selected articulation root
        # need not coincide with the arm mount or have an identity orientation.
        root_pose.zero_()
        root_pose[:, 6] = 1.0
    start_hand = tensor(robot.data.body_pose_w)[:, hand_id, :3].clone() - root_pose[:, :3]
    cube_start = tensor(cube.data.root_pos_w).clone()
    grasp_hand = cube_start - root_pose[:, :3]
    # panda_hand -> finger grasp center is +0.107 m along local Z.
    # With the hand facing down the hand origin is above the cube center.
    grasp_hand[:, 2] += 0.107
    above = grasp_hand.clone()
    above[:, 2] += 0.20
    lifted = grasp_hand.clone()
    lifted[:, 2] += 0.18
    # Isaac Lab 3 uses XYZW quaternions: 180 degrees about X, fingers along Y.
    down = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device)
    tcp_offset = torch.tensor([[0.0, 0.0, 0.107]], device=sim.device)
    frames, trace = [], []
    previous_phase = None
    phase_index, phase_step = 0, 0
    scale = (args.steps - 900 if mobile else args.steps) / (1500 if human_grasp else 960 if avatar else 720)
    durations = [round(n * scale) for n in (120, 120, 90, 120, 60, 90, 120, 60, 45, 60, 120, 180)]
    durations[4] = max(60, durations[4])
    phases = ["approach", "descend", "close", "lift", "hold"]
    if human_grasp:
        phases += ["raise_for_transfer", "offer", "lower_to_palm", "human_grasp", "release", "withdraw", "human_carry", "handover_hold"]
    if mobile:
        phases = ["navigate_pick"] + phases[:5] + ["navigate_human"] + phases[5:]
        durations = [360] + durations[:5] + [420] + durations[5:]
    base_ids = [robot.joint_names.index(name) for name in (
        "dummy_base_prismatic_x_joint", "dummy_base_prismatic_y_joint", "dummy_base_revolute_z_joint")] if mobile else []
    transport_joints = None
    base_start = initial_q[0, base_ids].clone() if mobile else None
    human_ready = mobile or avatar is None or args.avatar_mode == "manual"
    human_ready_step, human_retract_step = None, None
    human_target = np.array([x + (0.68 if mobile else 0.74), y + 0.20 + delivery_offset, 0.97 if mobile else 1.08]) if human_grasp else np.array([x + 0.83, y + 0.20, 1.10])
    receive_pose = torch.tensor([human_target + [0, 0, 0.038 + 0.107]], dtype=torch.float32, device=sim.device) - root_pose[:, :3]
    offer_pose = receive_pose.clone()
    offer_pose[:, 2] += 0.01 if mobile else 0.05
    raised_pose = lifted.clone()
    raised_pose[:, 2] = offer_pose[:, 2]
    raised_pose[:, 1] += delivery_offset
    retreat_pose = raised_pose.clone()
    retreat_joints = None
    release_joints = None
    human_carry_target = np.array([x + (0.74 if mobile else 0.94), y + 0.20 + delivery_offset, 1.15])
    release_step = None
    avatar_trace = []
    walk_arrival_step = reach_start_step = None
    walk_destination = np.array([x + 0.90, y + 0.15 + delivery_offset, 0.0]) if mobile else None
    if avatar:
        human_idle = avatar.get_hand_pos().copy()
        if mobile:
            avatar.walk_to(walk_destination, duration=6.0)
        elif args.avatar_mode == "demo":
            avatar.reach_hand(human_target, duration=2.0)
    receiver = HandoverReceiver() if args.interaction_route == "c1" else None
    pause_remaining = args.robot_pause_steps
    delay_remaining = args.robot_delay_steps
    perturbations = []
    interaction_trace = []
    # Extra wall-clock budget does not slow or rescale the nominal actions.
    total_steps = args.steps + args.robot_delay_steps + args.robot_pause_steps
    for step in range(total_steps):
        if avatar:
            if args.avatar_command_file:
                avatar.poll_command(args.avatar_command_file)
            avatar.step(sim.get_physics_dt(), render_skin=step % 4 == 0)
            if mobile and walk_arrival_step is None and not avatar.is_walking():
                if np.linalg.norm(avatar.position - walk_destination) < 0.005:
                    walk_arrival_step = step
                    print("HUMAN_WALK_ARRIVED", step, avatar.position.tolist(), flush=True)
            if mobile and walk_arrival_step is not None and reach_start_step is None and step >= walk_arrival_step + 30:
                avatar.reach_hand(human_target, duration=2.0)
                reach_start_step = step
                human_idle = avatar.get_hand_pos().copy()
                print("HUMAN_REACH_START", step, flush=True)
            hand_error = float(np.linalg.norm(avatar.get_hand_pos() - human_target))
            if human_ready_step is None and (not mobile or reach_start_step is not None) and avatar.spare() and hand_error < 0.025:
                human_ready = True
                human_ready_step = step
                print("HUMAN_READY_ROBOT_START", step, hand_error, flush=True)
            if not human_grasp and args.avatar_mode == "demo" and phase_index == 4 and phase_step >= 90 and human_retract_step is None:
                avatar.reach_hand(human_idle, duration=2.0)
                human_retract_step = step
                print("HUMAN_RETRACT", step, flush=True)
            if step % 4 == 0:
                avatar_trace.append({"step": step, "right_palm": avatar.get_hand_pos().tolist(),
                                     "left_palm": avatar.get_hand_pos("left").tolist(),
                                     "base_position": avatar.position.tolist(), "base_yaw": avatar.yaw,
                                     "walking": avatar.is_walking(),
                                     "left_foot": avatar.rig.position("LeftFoot").tolist(),
                                     "right_foot": avatar.rig.position("RightFoot").tolist(),
                                     "reach_started": reach_start_step is not None,
                                     "target_error_m": hand_error, "motion_finished": avatar.spare()})
        phase = phases[phase_index] if human_ready else "wait_for_human"
        paused = False
        if human_ready and phase == "approach" and delay_remaining:
            delay_remaining -= 1
            paused = True
            reason = "start_delay"
        elif phase == "offer" and phase_step >= durations[phase_index] // 2 and pause_remaining:
            pause_remaining -= 1
            paused = True
            reason = "mid_offer_pause"
        if paused:
            if not perturbations or perturbations[-1]["end_step"] != step - 1 or perturbations[-1]["kind"] != reason:
                perturbations.append({"kind": reason, "start_step": step, "end_step": step})
            else:
                perturbations[-1]["end_step"] = step
        if receiver:
            observed_pose = tensor(robot.data.body_pose_w)[:, hand_id]
            observed_tcp = observed_pose[:, :3] + quat_apply(observed_pose[:, 3:7], tcp_offset)
            receiver.step(step, palm_ready=hand_error < 0.025 and avatar.spare(),
                          captured=human_grasp.captured,
                          fingers_open=tensor(robot.data.joint_pos)[0, finger_ids].min().item() > 0.035,
                          robot_distance=torch.linalg.vector_norm(observed_tcp - tensor(cube.data.root_pos_w)).item(),
                          carry_finished=human_retract_step is not None and avatar.spare()
                          and np.linalg.norm(avatar.get_hand_pos() - human_carry_target) < 0.025)
            if step % 4 == 0:
                interaction_trace.append({"step": step, "state": receiver.state, "palm_error_m": hand_error,
                                          "robot_paused": paused})
        u = min(phase_step / durations[min(phase_index, len(durations) - 1)], 1.0)
        navigation = mobile and phase in ("navigate_pick", "navigate_human")
        if navigation:
            a = b = start_hand if phase == "navigate_pick" else lifted
            finger = 0.04 if phase == "navigate_pick" else 0.0
        elif phase == "wait_for_human":
            a, b, finger = start_hand, start_hand, 0.04
        elif phase == "approach":
            a, b, finger = start_hand, above, 0.04
        elif phase == "descend":
            a, b, finger = above, grasp_hand, 0.04
        elif phase == "close":
            a, b, finger = grasp_hand, grasp_hand, 0.04 * (1.0 - min(u * 1.5, 1.0))
        elif phase == "lift":
            a, b, finger = grasp_hand, lifted, 0.0
        elif phase == "hold":
            a, b, finger = lifted, lifted, 0.0
        elif phase == "raise_for_transfer":
            a, b, finger = lifted, raised_pose, 0.0
        elif phase == "offer":
            a, b, finger = raised_pose, offer_pose, 0.0
        elif phase == "lower_to_palm":
            a, b, finger = offer_pose, receive_pose, 0.0
        elif phase == "human_grasp":
            a, b, finger = receive_pose, receive_pose, 0.0
            avatar.grip = u
            if not human_grasp.captured and u >= 0.5:
                human_grasp.capture(cube, avatar, step)
        elif phase == "release":
            a, b, finger = receive_pose, receive_pose, 0.04 * u
            if release_step is None:
                release_step = step
        elif phase == "withdraw":
            a, b, finger = receive_pose, retreat_pose, 0.04
        else:
            a, b, finger = retreat_pose, retreat_pose, 0.04
            if phase == "human_carry" and phase_step == 0:
                avatar.reach_hand(human_carry_target, duration=durations[phase_index] / 60)
                human_retract_step = step
        u = u * u * (3.0 - 2.0 * u)
        goal = a + (b - a) * u
        ik.set_command(torch.cat((goal, down), dim=-1))
        hand_pose = tensor(robot.data.body_pose_w)[:, hand_id]
        hand_pos_b, hand_quat_b = subtract_frame_transforms(
            root_pose[:, :3], root_pose[:, 3:7], hand_pose[:, :3], hand_pose[:, 3:7])
        jacobian_body = hand_id - 1 if robot.is_fixed_base else hand_id
        jacobian_joints = arm_ids if robot.is_fixed_base else [j + 6 for j in arm_ids]
        jacobian = tensor(robot.data.body_link_jacobian_w)[:, jacobian_body, :, jacobian_joints]
        q = tensor(robot.data.joint_pos)
        q_des = ik.compute(hand_pos_b, hand_quat_b, jacobian, q[:, arm_ids])
        if navigation:
            q_des = initial_q[:, arm_ids] if phase == "navigate_pick" else transport_joints
        if human_grasp and phase == "withdraw":
            q_des = release_joints + u * (retreat_joints - release_joints)
        elif human_grasp and phase in ("human_carry", "handover_hold"):
            q_des = retreat_joints
        # Limit per-step changes while the initial wrist turns toward the table.
        q_des = q[:, arm_ids] + (q_des - q[:, arm_ids]).clamp(-0.15, 0.15)
        limits = tensor(robot.data.soft_joint_pos_limits)[:, arm_ids]
        q_des = torch.maximum(torch.minimum(q_des, limits[..., 1]), limits[..., 0])
        robot.set_joint_position_target(q_des, joint_ids=arm_ids)
        robot.set_joint_position_target(torch.full((1, 2), finger, device=sim.device), joint_ids=finger_ids)
        if mobile:
            base_target = torch.tensor([[0.0, delivery_offset if phase_index >= phases.index("navigate_human") else 0.0, 0.0]], device=sim.device)
            base_error = base_target - q[:, base_ids]
            base_velocity = (base_error * 1.8).clamp(-0.28, 0.28)
            robot.set_joint_velocity_target(base_velocity, joint_ids=base_ids)
        robot.write_data_to_sim()
        if human_grasp:
            human_grasp.apply(cube, avatar, sim.get_physics_dt())
        # The recording is 15 FPS; physics and control remain at 60 Hz.
        sim.step(render=step % 4 == 0)
        robot.update(sim.get_physics_dt())
        cube.update(sim.get_physics_dt())
        if avatar:
            avatar.update(sim.get_physics_dt())
        if human_grasp:
            human_grasp.update(cube, avatar)
        camera.update(sim.get_physics_dt())
        if not torch.isfinite(tensor(robot.data.joint_pos)).all():
            raise RuntimeError("Non-finite robot joint state")
        if phase != previous_phase:
            print("GRASP_PHASE", phase, "cube", tensor(cube.data.root_pos_w).tolist(), flush=True)
            previous_phase = phase
        if mobile and step % 120 == 0:
            print("MOBILE_STATE", step, phase, "base", tensor(robot.data.joint_pos)[0, base_ids].tolist(),
                  "hand", tensor(robot.data.body_pose_w)[0, hand_id].tolist(),
                  "arm", tensor(robot.data.joint_pos)[0, arm_ids].tolist(), flush=True)
        if step % 4 == 0:
            rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
            frames.append(rgb)
            if mobile and step % 120 == 0:
                imageio.imwrite(output / "live.png", rgb)
            hand_pose = tensor(robot.data.body_pose_w)[:, hand_id]
            tcp = hand_pose[:, :3] + quat_apply(hand_pose[:, 3:7], tcp_offset)
            cube_pos = tensor(cube.data.root_pos_w)
            trace.append({"step": step, "phase": phase,
                          "base_joint_position": tensor(robot.data.joint_pos)[0, base_ids].tolist() if mobile else None,
                          "cube_position": cube_pos[0].tolist(),
                          "tcp_position": tcp[0].tolist(),
                          "hand_quaternion_xyzw": hand_pose[0, 3:7].tolist(),
                          "tcp_cube_distance_m": torch.linalg.vector_norm(tcp - cube_pos).item(),
                          "finger_positions_m": tensor(robot.data.joint_pos)[0, finger_ids].tolist(),
                          "human_palm": avatar.get_hand_pos().tolist() if avatar else None,
                          "human_grasp_active": human_grasp.captured if human_grasp else False})
        if human_ready and not paused:
            phase_step += 1
        last_phase = len(phases) - 1
        if not paused and phase_index < last_phase and phase_step >= durations[phase_index]:
            actual_pose = tensor(robot.data.body_pose_w)[:, hand_id]
            pos_error = torch.linalg.vector_norm(actual_pose[:, :3] - root_pose[:, :3] - b).item()
            aligned = abs(torch.sum(actual_pose[:, 3:7] * down).item()) > 0.999
            # Wait at the waypoint with fingers open until physically aligned.
            # In particular, elapsed time alone must never trigger closing.
            ready = phase == "close" or (pos_error < 0.008 and aligned)
            if navigation:
                ready = bool(torch.max(torch.abs(base_target - tensor(robot.data.joint_pos)[:, base_ids])) < 0.008)
                ready = ready and bool(torch.max(torch.abs(tensor(robot.data.joint_vel)[:, base_ids])) < 0.02)
                if phase == "navigate_human":
                    ready = ready and human_ready_step is not None
            if phase == "human_grasp":
                ready = ready and human_grasp.captured
            if phase == "release":
                ready = ready and tensor(robot.data.joint_pos)[0, finger_ids].min().item() > 0.035
            if phase == "withdraw" and receiver:
                ready = ready and receiver.can_carry
            if phase == "human_carry":
                ready = ready and avatar.spare() and np.linalg.norm(avatar.get_hand_pos() - human_carry_target) < 0.025
            if ready:
                if mobile and phase == "navigate_pick":
                    start_hand = tensor(robot.data.body_pose_w)[:, hand_id, :3].clone() - root_pose[:, :3]
                if mobile and phase == "navigate_human":
                    lifted[:, 1] += delivery_offset
                if mobile and phase == "hold":
                    transport_joints = tensor(robot.data.joint_pos)[:, arm_ids].clone()
                if phase == "raise_for_transfer":
                    retreat_joints = tensor(robot.data.joint_pos)[:, arm_ids].clone()
                if phase == "release":
                    release_joints = tensor(robot.data.joint_pos)[:, arm_ids].clone()
                phase_index += 1
                phase_step = 0
    # Save diagnostics even when the physical success criteria fail.
    imageio.imwrite(output / "preview.png", frames[-1])
    imageio.mimwrite(output / "preview.mp4", frames, fps=15, macro_block_size=1)
    hold = [row for row in trace if row["phase"] == "hold"]
    min_lift = min((row["cube_position"][2] - cube_start[0, 2].item() for row in hold), default=0.0)
    max_distance = max((row["tcp_cube_distance_m"] for row in hold), default=1.0)
    min_gap = min((sum(row["finger_positions_m"]) for row in hold), default=0.0)
    hold_duration = len(hold) * 4 / 60
    passed = hold_duration >= 1.0 and frames[-1].std() >= 2 and min_lift > 0.12 and max_distance < 0.045 and min_gap > 0.025
    avatar_result = None
    if avatar:
        positions = np.asarray([row["right_palm"] for row in avatar_trace])
        motion_range = float(np.linalg.norm(np.ptp(positions, axis=0)))
        avatar_ok = (human_ready_step is not None and human_retract_step is not None and motion_range > 0.15
                     and np.linalg.norm(avatar.get_hand_pos() - human_idle) < 0.035)
        if args.avatar_mode == "demo" and not human_grasp:
            passed = passed and bool(avatar_ok)
        passed = passed and avatar.max_proxy_error < 0.005
        walking_result = None
        if mobile:
            walk_error = float(np.linalg.norm(avatar.position - walk_destination))
            walking_ok = (walk_arrival_step is not None and reach_start_step is not None
                          and human_ready_step is not None and walk_arrival_step + 30 <= reach_start_step < human_ready_step
                          and walk_error < 0.005 and avatar.walk_motion.max_foot_error < 0.025)
            passed = passed and walking_ok
            walking_result = {"destination": walk_destination.tolist(), "arrival_step": walk_arrival_step,
                              "reach_start_step": reach_start_step, "arrival_error_m": walk_error,
                              "distance_m": float(np.linalg.norm(avatar.walk_motion.destination - avatar.walk_motion.start)),
                              "max_foot_ik_error_m": avatar.walk_motion.max_foot_error,
                              "model": "kinematic_alternating_planted_feet_with_leg_ik"}
        avatar_result = {"walking": walking_result,"mode": args.avatar_mode, "asset": args.avatar_glb,
                         "human_ready_step": human_ready_step, "human_retract_step": human_retract_step,
                         "right_hand_motion_range_m": motion_range,
                         "final_idle_error_m": float(np.linalg.norm(avatar.get_hand_pos() - human_idle)),
                         "max_collision_proxy_error_m": avatar.max_proxy_error,
                         "applied_commands": avatar.command_history,
                         "collision_proxies": ["torso_capsule", "left_palm_sphere", "right_palm_box" if human_grasp else "right_palm_sphere"],
                         "interaction": "robot_to_human_handover" if human_grasp else "human_reach_triggers_robot_grasp" if args.avatar_mode == "demo" else "manual_control",
                         "trace": avatar_trace}
        (output / "avatar_result.json").write_text(json.dumps(avatar_result, indent=2))
    handover_result = None
    if human_grasp:
        final_hold = [row for row in trace if row["phase"] == "handover_hold"]
        max_human_error = max((float(np.linalg.norm(np.array(row["cube_position"]) - np.array(row["human_palm"]) - human_grasp.capture_offset)) for row in final_hold), default=1.0)
        min_robot_distance = min((row["tcp_cube_distance_m"] for row in final_hold), default=0.0)
        open_gap = min((sum(row["finger_positions_m"]) for row in final_hold), default=0.0)
        min_human_height = min((row["cube_position"][2] for row in final_hold), default=0.0)
        duration = len(final_hold) * 4 / 60
        carried = float(np.linalg.norm(np.array(final_hold[-1]["human_palm"]) - human_target)) if final_hold else 0.0
        passed = bool(passed and human_grasp.captured and release_step is not None and duration >= 1.0
                      and human_grasp.capture_jump is not None and human_grasp.capture_jump < 0.01
                      and human_grasp.max_follow_error < 0.015
                      and max_human_error < 0.015 and min_robot_distance > 0.15
                      and open_gap > 0.07 and min_human_height > 1.0 and carried > 0.15)
        handover_result = {"capture_step": human_grasp.capture_step, "release_step": release_step,
                           "human_carry_step": human_retract_step, "final_hold_s": duration,
                           "capture_offset_m": human_grasp.capture_offset.tolist() if human_grasp.captured else None,
                           "capture_jump_m": human_grasp.capture_jump,
                           "max_follow_error_since_capture_m": human_grasp.max_follow_error,
                           "max_human_grasp_error_m": max_human_error,
                           "min_robot_cube_distance_m": min_robot_distance, "min_robot_finger_gap_m": open_gap,
                           "human_carry_distance_m": carried, "min_cube_height_m": min_human_height,
                           "peak_human_grip_force_n": human_grasp.max_force,
                           "human_grasp_model": "proximity_gated_6dof_spring_grasp"}
        (output / "handover_result.json").write_text(json.dumps(handover_result, indent=2))
    interaction_result = {"route": args.interaction_route,
                          "contact_model": "proximity_gated_6dof_spring_grasp",
                          "robot_delay_steps": args.robot_delay_steps,
                          "robot_pause_steps": args.robot_pause_steps,
                          "perturbations": perturbations, "total_steps": total_steps,
                          "state": receiver.state if receiver else None,
                          "events": receiver.events if receiver else [],
                          "trace": interaction_trace}
    if receiver:
        passed = passed and receiver.state == "complete"
    (output / "interaction_result.json").write_text(json.dumps(interaction_result, indent=2))
    mobile_result = None
    if mobile:
        delivery_rows = [r for r in trace if r["phase"] == "navigate_human"]
        final_base = tensor(robot.data.joint_pos)[0, base_ids]
        base_error_m = float(torch.linalg.vector_norm(final_base[:2] - torch.tensor([0.0, delivery_offset], device=sim.device)))
        transport_error = max((r["tcp_cube_distance_m"] for r in delivery_rows), default=1.0)
        mobile_result = {"robot": "Clearpath Ridgeback + Franka", "asset": args.mobile_usd,
                        "base_model": "official planar XY/yaw velocity-controlled joints; no wheel-ground dynamics",
                        "initial_base_joints": base_start.tolist(), "final_base_joints": final_base.tolist(),
                        "dock_error_m": base_error_m, "transport_max_tcp_cube_distance_m": transport_error}
        passed = bool(passed and delivery_rows and base_error_m < 0.015 and transport_error < 0.045)
    result = {"scene": args.scene, "mobile": mobile_result,"status": "passed" if passed else "failed", "task": "robot_to_human_handover" if human_grasp else "human_robot_grasp" if avatar else "physical_grasp_lift_hold",
              "handover": handover_result,
              "interaction_route": args.interaction_route,
              "steps": total_steps, "nominal_steps": args.steps, "device": str(sim.device),
              "avatar": {k: v for k, v in avatar_result.items() if k != "trace"} if avatar_result else None,
              "render_mode": carb.settings.get_settings().get("/rtx/rendermode"),
              "samples_per_pixel": carb.settings.get_settings().get("/rtx/pathtracing/spp"),
              "kitchen_usd": str(kitchen_path) if kitchen_path else None, "kitchen_fixture_physics": "disabled",
              "cube_initial_position": cube_start[0].tolist(),
              "cube_final_position": tensor(cube.data.root_pos_w)[0].tolist(),
              "hold_min_lift_m": min_lift, "hold_max_tcp_distance_m": max_distance,
              "hold_min_finger_gap_m": min_gap, "hold_duration_s": hold_duration,
              "kitchen_bounds": [list(bounds.GetMin()), list(bounds.GetMax())] if bounds else None, "trace": trace}
    (output / "result.json").write_text(json.dumps(result, indent=2))
    print("KITCHEN_GRASP_RESULT", json.dumps({k: v for k, v in result.items() if k != "trace"}), flush=True)
    if not passed:
        raise RuntimeError(f"Task {args.task} verification failed; see result.json and preview.mp4 in {output}")



try:
    main()
except BaseException:
    traceback.print_exc()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "error.txt").write_text(traceback.format_exc())
    raise
finally:
    app.close()
