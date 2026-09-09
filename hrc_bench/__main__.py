"""Default Lightwheel benchmark runner; zero actions validate the environment interface."""
import argparse
from dataclasses import replace, asdict
import json
import os
from pathlib import Path
import time
import traceback

from .config import BenchmarkConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1]/'configs/lightwheel.json'))
    parser.add_argument('--lightwheel-root', default=os.environ.get('LIGHTWHEEL_ROOT'))
    parser.add_argument('--no-human', action='store_true')
    parser.add_argument('--check-assets', action='store_true')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--episode-seconds', type=float)
    parser.add_argument('--output', default='outputs/benchmark/'+time.strftime('%Y%m%d_%H%M%S'))
    known, _ = parser.parse_known_args()
    config = BenchmarkConfig.load(known.config)
    if known.no_human:
        config = replace(config, human=replace(config.human, enabled=False))
    if known.episode_seconds is not None:
        config = replace(config, episode_length_s=known.episode_seconds)
    if known.lightwheel_root:
        config = replace(config, lightwheel_root=str(Path(known.lightwheel_root).resolve()))
    if known.episodes < 1:
        parser.error('--episodes must be positive')
    from .human import TrumanHumanMotion
    checked = TrumanHumanMotion(config.human)
    if known.check_assets:
        print(json.dumps({'backend': config.backend, 'human': checked.observe(),
                          'config': asdict(config), 'provenance': checked.provenance}, indent=2))
        return 0

    from .lightwheel import configure_source
    configure_source(config.lightwheel_root)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    result = {'status': 'started', 'backend': 'lightwheel', 'policy': 'zero',
              'config': asdict(config), 'episodes': [], 'note': 'Interface smoke, not a task-solving baseline.'}
    (out/'result.json').write_text(json.dumps(result, indent=2))
    app = env = None
    frames = []
    try:
        app = AppLauncher(args).app
        import numpy as np
        import torch
        from .lightwheel import make_env
        env = make_env(config, args.device, args.enable_cameras)
        def tensor(value):
            return value.torch if hasattr(value, 'torch') else value
        import lw_benchhub
        import isaaclab_arena
        result['source_paths'] = {'lightwheel': str(Path(lw_benchhub.__file__).resolve()),
                                  'arena': str(Path(isaaclab_arena.__file__).resolve())}
        result['action_shape'] = list(env.action_space.shape)
        result['human_provenance'] = env.human_motion.provenance
        result['control_dt'] = env.step_dt
        obs, info = env.reset()
        camera = env.scene.sensors.get('overview')
        if camera is not None:
            import omni.replicator.core as rep
            rep.settings.set_render_pathtraced(samples_per_pixel=32)
            camera.set_world_poses_from_view(
                torch.tensor([[4.9, -5.4, 3.8]], device=env.device),
                torch.tensor([[2.1, -1.9, .8]], device=env.device))
        actions = torch.zeros(env.action_space.shape, device=env.device)
        total_steps = 0
        reset_times = [env.human_motion.elapsed_s]
        with torch.no_grad():
            for _ in range(args.episodes):
                episode_steps = 0
                while True:
                    obs, reward, terminated, truncated, info = env.step(actions)
                    total_steps += 1
                    episode_steps += 1
                    robot = env.scene.articulations['robot']
                    for field in ('joint_pos', 'joint_vel', 'root_state_w'):
                        if not torch.isfinite(tensor(getattr(robot.data, field))).all():
                            raise RuntimeError(f'Nonfinite robot {field}')
                    if camera is not None and total_steps % max(1, round(.1/env.step_dt)) == 0:
                        env.sim.render()
                        camera.update(env.physics_dt, force_recompute=True)
                        frame = tensor(camera.data.output['rgb'])[0, ..., :3].cpu().numpy().copy()
                        if not np.isfinite(frame).all() or np.ptp(frame) == 0:
                            raise RuntimeError('Invalid RGB frame')
                        frames.append(frame)
                    done, timeout = bool(terminated[0]), bool(truncated[0])
                    if done or timeout:
                        # hrc_capture stores the final human state before native auto-reset.
                        result['episodes'].append({'task_success': env.hrc_step_snapshot['task_success'],
                            'terminated': done, 'timeout': timeout,
                            'control_steps': episode_steps, 'human_terminal': env.hrc_step_snapshot})
                        reset_times.append(env.human_motion.elapsed_s)
                        break
                    if episode_steps > env.max_episode_length+2:
                        raise RuntimeError('Native episode timeout did not fire')
        result.update(status='passed', total_steps=total_steps, human_reset_times=reset_times)
        if any(t != 0. for t in reset_times):
            raise RuntimeError('Human timeline did not reset with the native episode')
        if frames:
            import imageio.v2 as imageio
            imageio.imwrite(out/'preview.png', frames[-1])
            imageio.mimsave(out/'preview.mp4', frames, fps=10)
    except Exception as exc:
        result.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        (out/'result.json').write_text(json.dumps(result, indent=2))
        print('BENCHMARK_RESULT', json.dumps({k: v for k, v in result.items() if k != 'human_provenance'}), flush=True)
        if env is not None:
            env.close()
        if app is not None:
            app.close()
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
