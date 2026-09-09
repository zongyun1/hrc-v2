"""Exercise upstream task creation, reset, stepping and RGB output; no policy claim."""
import argparse
import json
from pathlib import Path
import time
import traceback
import importlib.metadata
import faulthandler

faulthandler.enable()
faulthandler.dump_traceback_later(180, repeat=True)

versions = {name: importlib.metadata.version(name) for name in
            ('isaacsim', 'isaaclab', 'isaaclab_arena', 'lightwheel-sdk', 'torch', 'warp-lang')}
print('LW_RUNTIME_VERSIONS', json.dumps(versions), flush=True)

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='OpenDrawer')
parser.add_argument('--layout', default='robocasakitchen-4-2')
parser.add_argument('--steps', type=int, default=120)
parser.add_argument('--output', required=True)
parser.add_argument('--asset-timeout', type=int, default=60)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
result = {'task': args.task, 'layout': args.layout, 'status': 'started', 'phase': 'app_launch', 'versions': versions}
(out / 'result.json').write_text(json.dumps(result, indent=2))
app = None
env = None
started = time.monotonic()
try:
    app = AppLauncher(args).app
    result['phase'] = 'task_configuration'
    from lightwheel_sdk.client import lw_client
    lw_client.base_timeout = args.asset_timeout
    result['asset_timeout_seconds'] = args.asset_timeout
    import gymnasium as gym
    import numpy as np
    import torch
    from lw_benchhub.utils.env import parse_env_cfg, ExecuteMode
    cfg = parse_env_cfg(
        scene_backend='robocasa', task_backend='robocasa', scene_name=args.layout,
        task_name=args.task, robot_name='PandaOmron-Rel', robot_scale=1.0,
        execute_mode=ExecuteMode.EVAL, device=args.device, num_envs=1,
        use_fabric=True, enable_cameras=args.enable_cameras, headless_mode=True,
        seed=0, sources=['objaverse', 'lightwheel', 'aigen_objs'], object_projects=[],
        max_scene_retry=1, max_object_placement_retry=1,
        resample_objects_placement_on_reset=False,
        resample_robot_placement_on_reset=False,
    )
    cfg.env_name = 'LW-Source-Smoke-v0'
    gym.register(cfg.env_name, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True)
    result['phase'] = 'environment_creation'
    env = gym.make(cfg.env_name, cfg=cfg, render_mode='rgb_array' if args.enable_cameras else None).unwrapped
    result['phase'] = 'reset'
    env.reset()
    task = env.cfg.isaaclab_arena_env.task
    result['initial_task_success'] = task._check_success(env).tolist()
    result['action_shape'] = list(env.action_space.shape)
    actions = torch.zeros(env.action_space.shape, device=env.device)
    result['phase'] = 'stepping'
    frames = []
    with torch.no_grad():
        for step in range(args.steps):
            env.step(actions)
            for name, art in env.scene.articulations.items():
                for field in ('joint_pos', 'joint_vel', 'root_state_w'):
                    if not torch.isfinite(getattr(art.data, field)).all():
                        raise RuntimeError(f'Nonfinite {field}: {name}, step {step}')
            for name, obj in env.scene.rigid_objects.items():
                if not torch.isfinite(obj.data.root_state_w).all():
                    raise RuntimeError(f'Nonfinite rigid object: {name}, step {step}')
            if args.enable_cameras and step % 4 == 0:
                frame = env.render()
                if frame is not None:
                    if not np.isfinite(frame).all() or np.ptp(frame) == 0:
                        raise RuntimeError(f'Invalid or constant rendered frame at step {step}')
                    frames.append(frame.copy())
    result['steps_completed'] = args.steps
    result['phase'] = 'second_reset'
    env.reset()
    result['second_reset_task_success'] = task._check_success(env).tolist()
    if args.task == 'OpenDrawer':
        # State injection checks the predicate only, not robot manipulation.
        task.drawer.open_door(env, min=1.0, max=1.0)
        env.sim.forward()
        task.drawer.update_state(env)
        opened = task._check_success(env).tolist()
        task.drawer.close_door(env)
        env.sim.forward()
        task.drawer.update_state(env)
        closed = task._check_success(env).tolist()
        result['predicate_probe'] = {'opened': opened, 'closed': closed,
                                     'method': 'fixture state injection; not a policy rollout'}
        if not all(opened) or any(closed):
            raise RuntimeError('Drawer success predicate failed open/closed state probe')
        env.reset()
    if args.enable_cameras and not frames:
        raise RuntimeError('Cameras enabled but no rendered frames were returned')
    if frames:
        import imageio.v2 as imageio
        imageio.imwrite(out / 'preview.png', frames[-1])
        imageio.mimsave(out / 'preview.mp4', frames, fps=12)
    result.update(status='passed', phase='complete', rendered_frames=len(frames),
                  note='Initialization and stepping only; no task-solving policy was tested.')
except Exception as exc:
    result.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    traceback.print_exc()
finally:
    result['elapsed_seconds'] = time.monotonic() - started
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    print('LW_SMOKE_RESULT', json.dumps(result), flush=True)
    if env is not None:
        env.close()
    if app is not None:
        app.close()
raise SystemExit(0 if result['status'] == 'passed' else 1)
