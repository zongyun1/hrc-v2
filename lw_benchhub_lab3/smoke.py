"""Exercise upstream task creation, reset, stepping and RGB output; no policy claim."""
import argparse
import json
from pathlib import Path
import time
import traceback
import importlib.metadata
import faulthandler
import sys
import site

ROOT = Path(__file__).resolve().parent
site.addsitedir(str(ROOT / 'deps'))
sys.path[:0] = [str(ROOT / 'source'), str(ROOT / 'source/third_party/IsaacLab-Arena')]

faulthandler.enable()
faulthandler.dump_traceback_later(180, repeat=True)

versions = {}
for name in ('isaacsim', 'isaaclab', 'isaaclab_arena', 'lightwheel-sdk', 'torch', 'warp-lang'):
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = 'not registered'
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
def phase(name):
    result['phase'] = name
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    print('LW_SMOKE_PHASE', name, flush=True)


app = None
env = None
started = time.monotonic()
try:
    app = AppLauncher(args).app
    phase('task_configuration')
    from lightwheel_sdk.client import lw_client
    lw_client.base_timeout = args.asset_timeout
    result['asset_timeout_seconds'] = args.asset_timeout
    import gymnasium as gym
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    def tensor(value):
        return value.torch if hasattr(value, 'torch') else value
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
    cfg.seed = 0
    cfg.sim.render = sim_utils.RenderCfg(antialiasing_mode='TAA', samples_per_pixel=32,
                                         enable_dlssg=False, enable_dl_denoiser=False)
    cfg.scene.overview = CameraCfg(
        prim_path='{ENV_REGEX_NS}/Overview', width=960, height=720,
        data_types=['rgb'], update_period=0.,
        spawn=sim_utils.PinholeCameraCfg(focal_length=24., horizontal_aperture=32.,
                                        clipping_range=(0.05, 100.)),
    )
    cfg.env_name = 'LW-Source-Smoke-v0'
    gym.register(cfg.env_name, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True)
    phase('environment_creation')
    env = gym.make(cfg.env_name, cfg=cfg).unwrapped
    phase('reset')
    env.reset()
    # This container cannot initialize NGX; explicit path-tracing samples avoid
    # the heavy noise from the default real-time denoising path.
    import omni.replicator.core as rep
    rep.settings.set_render_pathtraced(samples_per_pixel=64)
    result['renderer'] = {'mode': 'pathtraced', 'samples_per_pixel': 64}
    task = env.cfg.isaaclab_arena_env.task
    camera = env.scene.sensors['overview']
    target = torch.as_tensor(np.asarray(task.drawer.pos).reshape(-1, 3)[0],
                             dtype=torch.float32, device=env.device) + torch.tensor([0., 0., 0.4], device=env.device)
    eye = tensor(env.scene.articulations['robot'].data.root_pos_w)[0] + torch.tensor([1.5, -1.5, 1.8], device=env.device)
    camera.set_world_poses_from_view(eye[None], target[None])

    def capture():
        env.sim.render()
        camera.update(env.physics_dt, force_recompute=True)
        frame = tensor(camera.data.output['rgb'])[0, ..., :3].detach().cpu().numpy()
        if not np.isfinite(frame).all() or np.ptp(frame) == 0:
            raise RuntimeError('Invalid or constant RGB sensor frame')
        return frame.copy()

    for _ in range(8):
        env.sim.render()
        camera.update(env.physics_dt, force_recompute=True)
    result['initial_task_success'] = task._check_success(env).tolist()
    result['camera_shape'] = list(capture().shape)
    result['action_shape'] = list(env.action_space.shape)
    actions = torch.zeros(env.action_space.shape, device=env.device)
    phase('stepping')
    frames = []
    with torch.no_grad():
        for step in range(args.steps):
            env.step(actions)
            for name, art in env.scene.articulations.items():
                for field in ('joint_pos', 'joint_vel', 'root_state_w'):
                    if not torch.isfinite(tensor(getattr(art.data, field))).all():
                        raise RuntimeError(f'Nonfinite {field}: {name}, step {step}')
            for name, obj in env.scene.rigid_objects.items():
                if not torch.isfinite(tensor(obj.data.root_state_w)).all():
                    raise RuntimeError(f'Nonfinite rigid object: {name}, step {step}')
            if args.enable_cameras and step % 4 == 0:
                frame = capture()
                if frame is not None:
                    if not np.isfinite(frame).all() or np.ptp(frame) == 0:
                        raise RuntimeError(f'Invalid or constant rendered frame at step {step}')
                    frames.append(frame.copy())
    result['steps_completed'] = args.steps
    phase('second_reset')
    env.reset()
    result['second_reset_task_success'] = task._check_success(env).tolist()
    import imageio.v2 as imageio
    imageio.imwrite(out / 'second_reset.png', capture())
    if args.task == 'OpenDrawer':
        # State injection checks the predicate only, not robot manipulation.
        task.drawer.open_door(env, min=1.0, max=1.0)
        env.sim.forward()
        task.drawer.update_state(env)
        opened = task._check_success(env).tolist()
        imageio.imwrite(out / 'probe_open.png', capture())
        task.drawer.close_door(env)
        env.sim.forward()
        task.drawer.update_state(env)
        closed = task._check_success(env).tolist()
        imageio.imwrite(out / 'probe_closed.png', capture())
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
