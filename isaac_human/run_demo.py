"""Yield-then-reach: a Franka waits for a moving human to clear its goal region.

Run from the project root: python -m isaac_human.run_demo --help
This is a scripted state-observation baseline, not a learned safety policy.
"""
import argparse
import json
import math
from pathlib import Path
import traceback
import numpy as np
from .motion import MotionPlayer, YieldGate, sha256


def prepare(args):
    paths = [args.motion, args.skeleton] + ([args.skin] if args.skin else [])
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f'Missing input {path}; obtain HARPv2 motion/body exports. No fallback motion is used.')
    if not args.skin and not args.capsules:
        raise ValueError('Supply --skin, or explicitly select --capsules for collision-only visualization')
    player = MotionPlayer.load(args.motion, args.skeleton, args.skin)
    if player.sequence.n_envs != 1:
        raise ValueError('The first scene demo requires B=1')
    synthetic = bool(player.sequence.meta.get('synthetic_diagnostic', False))
    if synthetic and not args.allow_diagnostic:
        raise ValueError('Synthetic test motion requires --allow_diagnostic; it is not a TRUMANS demo')
    reference_error = None
    if player.sequence.joints is not None:
        if player.sequence.joints.shape[2] < 22:
            raise ValueError('Reference joint cache has fewer than 22 body joints')
        reference_error = 0.
        for frame in np.linspace(0, player.sequence.n_frames-1, 12).astype(int):
            computed = player.pose(frame/player.sequence.fps)[0]
            cached = player.sequence.joints[:, frame, :22]
            reference_error = max(reference_error, float(np.linalg.norm(computed-cached, axis=-1).max()))
        if reference_error > .01:
            raise ValueError(f'FK differs from source joint cache by {reference_error:.4f} m; verify body and coordinate frames')
    # Relocate the whole sequence by one rigid world transform. Preserve gait,
    # temporal cadence and height; never independently move individual bones.
    sample_times = np.linspace(0, player.end_time, 100)
    pelvis = np.stack([player.pose(t)[0][0, 0] for t in sample_times])
    direction = pelvis[-1, :2] - pelvis[0, :2]
    if np.linalg.norm(direction) < .7:
        raise ValueError('Yield demo needs a walking clip with at least 0.7 m net root displacement')
    yaw = np.pi/2 - np.arctan2(direction[1], direction[0])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    center = .5*(pelvis[0, :2]+pelvis[-1, :2])
    offset = np.r_[np.array([args.crossing_x, 0.])-rotation@center, 0.]
    player = MotionPlayer(player.sequence, player.skeleton, player.skin, yaw=yaw, translation=offset)
    return player, {'files': {str(Path(p).resolve()): sha256(p) for p in paths},
                    'source_meta': player.sequence.meta, 'synthetic_diagnostic': synthetic,
                    'source_joint_cache_max_error_m': reference_error,
                    'world_yaw_rad': float(yaw), 'world_translation_m': offset.tolist()}


def run(args, player, provenance, app):
    import torch
    import imageio.v2 as imageio
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
    from .adapter import IsaacHuman
    from pxr import UsdGeom

    def tensor(value):
        return value if isinstance(value, torch.Tensor) else value.torch

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dt = 1/60
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device))
    stage = sim.stage
    light = sim_utils.DomeLightCfg(intensity=1500.)
    light.func('/World/Light', light)
    floor = sim_utils.GroundPlaneCfg()
    floor.func('/World/Floor', floor)
    table = sim_utils.CuboidCfg(size=(.8, .8, .08), collision_props=sim_utils.CollisionPropertiesCfg(),
                               visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.3, .22, .14)))
    table.func('/World/Table', table, translation=(.2, 0., .76))
    cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path='/World/Franka')
    cfg.spawn.usd_path = args.franka_usd
    cfg.init_state.pos = (0., 0., .8)
    robot = Articulation(cfg)
    human = IsaacHuman(stage, player, args.device, show_capsules=args.capsules)
    goal = np.array([.55, 0., 1.25])
    marker = sim_utils.SphereCfg(radius=.025,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.1, .8, .2)))
    marker.func('/World/ReachGoal', marker, translation=goal.tolist())
    camera = Camera(CameraCfg(prim_path='/World/Camera', height=540, width=720, data_types=['rgb'],
                              spawn=sim_utils.PinholeCameraCfg(focal_length=20., horizontal_aperture=24.)))
    sim.reset()
    camera.set_world_poses_from_view(torch.tensor([[3., -4., 2.8]], device=args.device),
                                     torch.tensor([[.5, 0., .9]], device=args.device))
    import omni.replicator.core as rep
    rep.settings.set_render_pathtraced(samples_per_pixel=32)
    q_home = tensor(robot.data.default_joint_pos).clone()
    robot.write_joint_state_to_sim(q_home, torch.zeros_like(q_home))
    human.set_time(0.)
    for _ in range(24):
        robot.set_joint_position_target(q_home)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
    arm = [robot.joint_names.index(f'panda_joint{i}') for i in range(1, 8)]
    hand = robot.body_names.index('panda_hand')
    ik = DifferentialIKController(DifferentialIKControllerCfg(
        command_type='pose', use_relative_mode=False, ik_method='dls'), num_envs=1, device=args.device)
    initial_hand = tensor(robot.data.body_pose_w)[:, hand].clone()
    root = tensor(robot.data.root_pose_w).clone()
    # This scene has an identity root rotation; preserve the initial hand rotation.
    if not torch.allclose(root[:, 3:], torch.tensor([[0., 0., 0., 1.]], device=args.device), atol=1e-4):
        raise RuntimeError('Demo assumes Isaac Lab 3 XYZW and identity robot root orientation')
    gate = YieldGate()
    elapsed_move = 0.
    saw_human = False
    previous_pause = False
    hold_q = q_home[:, arm].clone()
    trace, events, checks = [], [], []
    final_hold = 0.
    peak_force = 0.
    total = player.end_time+8.
    writer = imageio.get_writer(out/'preview.mp4', fps=15)
    try:
        for step in range(math.ceil(total/dt)):
            time = step*dt
            human.set_time(time, render=step % 4 == 0)
            clearance = human.clearance(goal, time)
            paused = gate.step(clearance, dt)
            if paused:
                saw_human = True
            if paused != previous_pause:
                events.append({'time': time, 'event': 'yield' if paused else 'resume', 'clearance_m': clearance})
                if paused:
                    hold_q = tensor(robot.data.joint_pos)[:, arm].clone()
            previous_pause = paused
            moving = saw_human and not paused and gate.resumes > 0
            if moving:
                elapsed_move += dt
                u = min(elapsed_move/3., 1.)
                u = u*u*(3-2*u)
                target = initial_hand[:, :3]*(1-u)+torch.tensor(goal[None], dtype=torch.float32, device=args.device)*u
                ik.set_command(torch.cat((target-root[:, :3], initial_hand[:, 3:]), dim=-1))
                pose = tensor(robot.data.body_pose_w)[:, hand]
                jacobian = tensor(robot.data.body_link_jacobian_w)[:, hand-1, :, arm]
                current = tensor(robot.data.joint_pos)[:, arm]
                command = ik.compute(pose[:, :3]-root[:, :3], pose[:, 3:], jacobian, current)
                command = current+(command-current).clamp(-.04, .04)
                limits = tensor(robot.data.soft_joint_pos_limits)[:, arm]
                command = torch.maximum(torch.minimum(command, limits[..., 1]), limits[..., 0])
            else:
                command = hold_q
            robot.set_joint_position_target(command, joint_ids=arm)
            robot.write_data_to_sim()
            sim.step(render=step % 4 == 0)
            robot.update(dt)
            force = human.update(dt)
            peak_force = max(peak_force, force)
            pose = tensor(robot.data.body_pose_w)[:, hand]
            if not torch.isfinite(tensor(robot.data.joint_pos)).all():
                raise RuntimeError('Non-finite robot state')
            error = float(np.linalg.norm(pose[0, :3].detach().cpu().numpy()-goal))
            final_hold = final_hold+dt if moving and error < .025 and elapsed_move >= 3 else 0.
            trace.append({'time_s': time, 'human_time_s': min(time, player.end_time),
                          'goal_clearance_m': clearance, 'paused': paused,
                          'hand_error_m': error, 'robot_human_contact_force_n': force})
            if step % 4 == 0:
                camera.update(4*dt, force_recompute=True)
                rgb = tensor(camera.data.output['rgb'])[0, ..., :3].detach().cpu().numpy().astype(np.uint8)
                if not np.isfinite(rgb).all() or rgb.std() < 1:
                    raise RuntimeError('Empty camera image')
                writer.append_data(rgb)
                if step == 0 or (paused and not (out/'yield.png').exists()):
                    imageio.imwrite(out/('yield.png' if paused else 'start.png'), rgb)
                imageio.imwrite(out/'preview.png', rgb)
        checks = {'human_entered': gate.entries > 0, 'robot_resumed': gate.resumes > 0,
                  'final_goal_hold': final_hold >= 1., 'no_reported_robot_contact': peak_force < .1,
                  'capsule_tracking': human.max_tracking_error < .005}
        passed = all(checks.values())
        result = {'status': ('diagnostic_passed' if provenance['synthetic_diagnostic'] else 'passed') if passed else 'failed',
                  'task': 'yield_then_reach', 'checks': checks, 'provenance': provenance,
                  'human_model': 'prescribed_kinematic_capsules_and_optional_LBS_skin',
                  'controller': 'privileged_goal_region_yield_gate_and_differential_IK',
                  'scope': 'Single-scene scripted reach; not manipulation or general collision avoidance; forces are diagnostic only.',
                  'goal_world_m': goal.tolist(), 'peak_robot_contact_force_n': peak_force,
                  'max_capsule_tracking_error_m': human.max_tracking_error,
                  'final_hold_s': final_hold, 'events': events}
        (out/'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        if not passed:
            raise RuntimeError('Demo acceptance checks failed; see result.json and trace.json')
    finally:
        writer.close()
        (out/'trace.json').write_text(json.dumps(trace))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion', required=True)
    parser.add_argument('--skeleton', required=True)
    parser.add_argument('--skin')
    parser.add_argument('--capsules', action='store_true')
    parser.add_argument('--allow_diagnostic', action='store_true')
    parser.add_argument('--check_assets', action='store_true')
    parser.add_argument('--crossing_x', type=float, default=1.05)
    parser.add_argument('--output_dir', default='outputs/isaac_human/manual')
    parser.add_argument('--franka_usd', default='https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd')
    pre, _ = parser.parse_known_args()
    player, provenance = prepare(pre)
    if pre.check_assets:
        print(json.dumps({'status': 'assets_valid', 'duration_s': player.end_time, 'provenance': provenance}, indent=2))
        return
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        run(args, player, provenance, launcher.app)
    except Exception:
        (output/'error.txt').write_text(traceback.format_exc())
        if not (output/'result.json').exists():
            (output/'result.json').write_text(json.dumps({'status': 'error', 'provenance': provenance}))
        raise
    finally:
        launcher.app.close()


if __name__ == '__main__':
    main()
