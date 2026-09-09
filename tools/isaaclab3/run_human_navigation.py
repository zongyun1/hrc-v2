"""Native Lightwheel NavigateKitchen with real TRUMANS motion and stop/resume control."""
import argparse
import json
from pathlib import Path
import site
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
site.addsitedir(str(ROOT / 'external/isaaclab3/deps'))
sys.path[:0] = [str(ROOT), str(ROOT / 'third_party/IsaacLab-Arena')]

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--motion', required=True)
parser.add_argument('--skeleton', required=True)
parser.add_argument('--skin', required=True)
parser.add_argument('--layout', default='robocasakitchen-4-2')
parser.add_argument('--duration', type=float, default=45.)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--human-delay', type=float, default=4.)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
result = {'status': 'started', 'task': 'NavigateKitchen', 'layout': args.layout, 'seed': args.seed}
app = env = None
started = time.monotonic()
trace = []
frames = []

def phase(name):
    result['phase'] = name
    (out/'result.json').write_text(json.dumps(result, indent=2))
    print('HUMAN_NAV_PHASE', name, flush=True)

try:
    app = AppLauncher(args).app
    import numpy as np
    import torch
    import gymnasium as gym
    import imageio.v2 as imageio
    import isaaclab.sim as sim_utils
    from isaaclab.managers import EventTermCfg
    from isaaclab.sensors import CameraCfg
    from isaac_human.motion import MotionPlayer, YieldGate, sha256
    from isaac_human.adapter import IsaacHuman
    from tools.isaaclab3.navigation import aisle_scenario, clearance, base_command, yaw_xyzw, wrap
    from tools.isaaclab3.robot_materials import restore_panda_white
    from lw_benchhub.utils.env import parse_env_cfg, ExecuteMode
    from lightwheel_sdk.client import lw_client
    lw_client.base_timeout = 60
    player = MotionPlayer.load(args.motion, args.skeleton, args.skin)
    if player.sequence.meta.get('backend') != 'trumans' or player.sequence.meta.get('synthetic_diagnostic'):
        raise ValueError('This task requires real TRUMANS motion')
    result['inputs'] = {key: {'path': getattr(args, key), 'sha256': sha256(getattr(args, key))}
                        for key in ('motion', 'skeleton', 'skin')}
    result['source_meta'] = player.sequence.meta
    phase('configuration')
    cfg = parse_env_cfg(scene_backend='robocasa', task_backend='robocasa', scene_name=args.layout,
        task_name='NavigateKitchen', robot_name='PandaOmron-Rel', robot_scale=1.,
        execute_mode=ExecuteMode.EVAL, device=args.device, num_envs=1,
        use_fabric=True, enable_cameras=args.enable_cameras, headless_mode=True,
        seed=args.seed, sources=['objaverse', 'lightwheel', 'aigen_objs'], object_projects=[],
        max_scene_retry=1, max_object_placement_retry=1,
        resample_objects_placement_on_reset=False, resample_robot_placement_on_reset=False)
    cfg.seed = args.seed
    cfg.scene.replicate_physics = False
    cfg.sim.render = sim_utils.RenderCfg(antialiasing_mode='TAA', samples_per_pixel=16,
                                         enable_dlssg=False, enable_dl_denoiser=False)
    cfg.scene.overview = CameraCfg(prim_path='{ENV_REGEX_NS}/Overview', width=960, height=720,
        data_types=['rgb'], update_period=0.,
        spawn=sim_utils.PinholeCameraCfg(focal_length=22., horizontal_aperture=32., clipping_range=(.05, 100.)))
    # Spawn before the first physics reset so bodies and contact sensors initialize normally.
    def spawn_human(env, env_ids=None):
        result['robot_material'] = restore_panda_white(env.sim.stage, '/World/envs/env_0/Robot')
        env.crossing_human = IsaacHuman(env.sim.stage, player, env.device,
                                        robot_path='/World/envs/env_0/Robot')
    cfg.events.crossing_human = EventTermCfg(func=spawn_human, mode='prestartup')
    cfg.env_name = 'LW-Human-Navigation-v0'
    gym.register(cfg.env_name, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True)
    phase('environment_creation')
    env = gym.make(cfg.env_name, cfg=cfg).unwrapped
    env.reset()
    human = env.crossing_human
    def tensor(x):
        return x.torch if hasattr(x, 'torch') else x
    robot = env.scene.articulations['robot']
    task = env.cfg.isaaclab_arena_env.task
    base_id = robot.data.body_names.index(env.cfg.isaaclab_arena_env.embodiment.robot_base_link)
    def state():
        p = tensor(robot.data.body_com_pos_w)[0, base_id].detach().cpu().numpy().copy()
        q = tensor(robot.data.body_com_quat_w)[0, base_id].detach().cpu().numpy()
        return p, yaw_xyzw(q)
    p0, yaw0 = state()
    goal = np.asarray(task.target_pos).reshape(-1, 3)[0]
    goal_yaw = float(np.asarray(task.target_ori).reshape(-1)[2])
    anchor_yaw = yaw_xyzw(tensor(robot.data.root_quat_w)[0].detach().cpu().numpy())
    # Curated layout 4 aisle: navigate around the island, human walks along the rear aisle.
    if args.layout != 'robocasakitchen-4-2' or args.seed != 0:
        raise ValueError('This first aisle scenario is calibrated for layout 4-2, seed 0')
    waypoints, hyaw, translation = aisle_scenario(player, p0, goal)
    waypoint_index = 0
    result['waypoints'] = [p.tolist() for p in waypoints]
    result['human_delay_s'] = args.human_delay
    human.player = player = MotionPlayer.load(args.motion, args.skeleton, args.skin,
                                               yaw=hyaw, translation=translation)
    result.update(start=p0.tolist(), goal=goal.tolist(), goal_yaw=goal_yaw,
                  anchor_yaw=anchor_yaw, human_yaw=hyaw, human_translation=translation.tolist(),
                  source_duration_s=player.end_time, initial_task_success=task._check_success(env).tolist())
    result['fixtures'] = {name: {'position': np.asarray(f.pos).tolist(), 'size': np.asarray(f.size).tolist()} for name, f in task.fixtures.items()}
    print('NAV_GEOMETRY', json.dumps({k: result[k] for k in ('start', 'goal', 'goal_yaw', 'anchor_yaw', 'human_yaw', 'human_translation', 'waypoints')}), flush=True)
    if any(result['initial_task_success']):
        raise RuntimeError('Task already successful at reset')
    camera = env.scene.sensors['overview']
    center = (p0+goal)*.5
    center[2] = .8
    eye = center + np.array([2.8, -3.5, 3.0])
    camera.set_world_poses_from_view(torch.tensor(eye[None], device=env.device, dtype=torch.float32),
                                    torch.tensor(center[None], device=env.device, dtype=torch.float32))
    import omni.replicator.core as rep
    rep.settings.set_render_pathtraced(samples_per_pixel=32)
    def capture(t, paused):
        human.render(max(0., t-args.human_delay))
        env.sim.render()
        camera.update(env.physics_dt, force_recompute=True)
        frame = tensor(camera.data.output['rgb'])[0, ..., :3].detach().cpu().numpy().copy()
        if not np.isfinite(frame).all() or np.ptp(frame) == 0:
            raise RuntimeError('Invalid RGB image')
        # Burn in simulation time and controller status for review.
        from PIL import Image, ImageDraw
        im = Image.fromarray(frame)
        ImageDraw.Draw(im).text((20, 20), f'{t:05.2f}s  '+('YIELD' if paused else 'NAVIGATE'), fill='white', stroke_width=2, stroke_fill='black')
        return np.asarray(im)
    term = env.action_manager.get_term('base_action')
    names = [robot.data.joint_names[i] for i in term._joint_ids]
    result['base_action_joints'] = names
    gate = YieldGate(stop=.30, resume=.45, clear_time=.4)
    dt = env.physics_dt*env.cfg.decimation
    t = 0.
    peak_force = 0.
    min_clearance = float('inf')
    success_hold = 0.
    yield_motion = 0.
    first_yield = None
    events = []
    previously_paused = False
    phase('rollout')
    with torch.no_grad():
        for step in range(int(args.duration/dt)):
            pos, yaw = state()
            measured = clearance(player, max(0., t-args.human_delay), pos[:2], horizon=0.)
            predicted = clearance(player, max(0., t-args.human_delay), pos[:2])
            paused = gate.step(predicted, dt)
            if paused != previously_paused:
                events.append({'time': t, 'event': 'yield' if paused else 'resume', 'position': pos.tolist()})
                if paused:
                    first_yield = pos.copy()
                    imageio.imwrite(out/'yield.png', capture(t, True))
                else:
                    imageio.imwrite(out/'resume.png', capture(t, False))
            previously_paused = paused
            if paused and first_yield is not None:
                yield_motion = max(yield_motion, float(np.linalg.norm(pos[:2]-first_yield[:2])))
            action = torch.zeros(env.action_space.shape, device=env.device)
            action[:, 6] = 1.
            if not paused:
                if np.linalg.norm(pos[:2]-waypoints[waypoint_index][:2]) < .10 and waypoint_index < len(waypoints)-1:
                    waypoint_index += 1
                command = base_command(pos, yaw, waypoints[waypoint_index], yaw0 if waypoint_index == 0 else goal_yaw, anchor_yaw, dt)
                action[0, 7:] = torch.tensor([command[n] for n in names], device=env.device)
            env.action_manager.process_action(action)
            # Native action application and physics, without automatic terminal reset.
            # Human and contact sensors advance at every physics substep.
            for _ in range(env.cfg.decimation):
                t += env.physics_dt
                human.set_time(max(0., t-args.human_delay), render=False)
                env.action_manager.apply_action()
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(env.physics_dt)
                peak_force = max(peak_force, human.update(env.physics_dt))
            pos, yaw = state()
            if not torch.isfinite(tensor(robot.data.joint_pos)).all() or not np.isfinite(pos).all():
                raise RuntimeError('Nonfinite robot state')
            native_success = bool(task._check_success(env).all().item())
            success_hold = success_hold+dt if native_success else 0.
            min_clearance = min(min_clearance, measured)
            row = dict(time=t, position=pos.tolist(), yaw=yaw, paused=paused,
                       clearance=measured, predicted_clearance=predicted,
                       native_success=native_success, peak_contact_force=peak_force)
            trace.append(row)
            if step % max(1, round(1/(10*dt))) == 0:
                frames.append(capture(t, paused))
            if step % 100 == 0:
                print('NAV_STEP', json.dumps(row), flush=True)
                imageio.imwrite(out/'progress.png', capture(t, paused))
            if success_hold >= 1. and t >= player.end_time+args.human_delay:
                break
    checks = {'native_task_success_held_1s': success_hold >= 1.,
              'human_intrusion_observed': gate.entries > 0,
              'robot_resumed': gate.resumes > 0,
              'robot_held_during_yield': yield_motion < .05,
              'full_source_motion_played': t >= player.end_time+args.human_delay,
              'no_reported_robot_human_contact': peak_force < .1,
              'positive_conservative_clearance': min_clearance > 0.,
              'capsule_tracking': human.max_tracking_error < .005}
    result.update(status='passed' if all(checks.values()) else 'failed', checks=checks,
                  sim_seconds=t, events=events, yield_entries=gate.entries, resumes=gate.resumes,
                  min_clearance_m=min_clearance, peak_robot_human_contact_N=peak_force,
                  max_capsule_tracking_error_m=human.max_tracking_error,
                  yield_displacement_m=yield_motion, final_position=pos.tolist(),
                  position_error_m=float(np.linalg.norm(pos[:2]-goal[:2])),
                  heading_error_rad=abs(wrap(yaw-goal_yaw)),
                  note='Privileged stop/resume navigation; conservative 0.65 m robot disk, full-body capsule projection. Prescribed SMPL-X human, source time preserved. Contact forces are diagnostics.')
    phase('complete')
except Exception as exc:
    result.update(status='failed', error=str(exc), traceback=traceback.format_exc())
    traceback.print_exc()
finally:
    result['elapsed_seconds'] = time.monotonic()-started
    (out/'result.json').write_text(json.dumps(result, indent=2))
    (out/'trace.json').write_text(json.dumps(trace))
    if frames:
        imageio.imwrite(out/'start.png', frames[0])
        imageio.imwrite(out/'preview.png', frames[-1])
        imageio.mimsave(out/'preview.mp4', frames, fps=10)
    print('HUMAN_NAV_RESULT', json.dumps({k: v for k, v in result.items() if k not in ('source_meta', 'inputs', 'traceback')}), flush=True)
    if env is not None:
        env.close()
    if app is not None:
        app.close()
raise SystemExit(0 if result['status'] == 'passed' else 1)
