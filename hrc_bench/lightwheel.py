"""Compose human playback with native Lightwheel/Isaac Lab managers."""
import sys
import site
from pathlib import Path


def configure_source(root):
    root = Path(root).resolve()
    if not (root/'lw_benchhub').is_dir():
        raise FileNotFoundError(f'Lightwheel source missing: {root}; see README setup')
    site.addsitedir(str(root.parent/'deps'))
    sys.path[:0] = [str(root), str(root/'third_party/IsaacLab-Arena')]


def make_env(config, device='cuda:0', cameras=False):
    """Call after AppLauncher. Returns the native env; actions/reset/step stay native."""
    import torch
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    from isaaclab.managers import ActionTerm, ActionTermCfg, EventTermCfg, TerminationTermCfg
    from isaaclab.utils import configclass
    from isaaclab.sensors import CameraCfg
    from lw_benchhub.utils.env import parse_env_cfg, ExecuteMode
    from lightwheel_sdk.client import lw_client
    from .human import TrumanHumanMotion
    from lw_benchhub_lab3.robot_materials import restore_panda_white

    lw_client.base_timeout = 60
    cfg = parse_env_cfg(scene_backend='robocasa', task_backend='robocasa',
        scene_name=config.scene, task_name=config.task, robot_name=config.robot,
        robot_scale=1., execute_mode=ExecuteMode.EVAL, device=device, num_envs=1,
        use_fabric=True, enable_cameras=cameras, headless_mode=True, seed=config.seed,
        sources=['objaverse', 'lightwheel', 'aigen_objs'], object_projects=[],
        max_scene_retry=1, max_object_placement_retry=1,
        resample_objects_placement_on_reset=False, resample_robot_placement_on_reset=False)
    cfg.seed = config.seed
    cfg.episode_length_s = config.episode_length_s
    cfg.scene.replicate_physics = False
    cfg.sim.render = sim_utils.RenderCfg(antialiasing_mode='TAA', samples_per_pixel=16,
                                         enable_dlssg=False, enable_dl_denoiser=False)
    if cameras:
        cfg.scene.overview = CameraCfg(prim_path='{ENV_REGEX_NS}/Overview', width=960, height=720,
            data_types=['rgb'], update_period=0., spawn=sim_utils.PinholeCameraCfg(
                focal_length=22., horizontal_aperture=32., clipping_range=(.05, 100.)))

    def initialize(env, env_ids=None):
        env.human_motion = TrumanHumanMotion(config.human)
        env.human_motion.attach(env.sim.stage, env.device)
        if config.robot.startswith('PandaOmron'):
            env.hrc_material = restore_panda_white(env.sim.stage, '/World/envs/env_0/Robot')
    cfg.events.hrc_initialize = EventTermCfg(func=initialize, mode='prestartup')

    class HumanClockAction(ActionTerm):
        """Zero-dimensional action term updates the human at each physics substep."""
        def __init__(self, term_cfg, env):
            super().__init__(term_cfg, env)
            if getattr(env, '_physics_handles_decimation', False):
                raise RuntimeError('Human clock requires external physics substeps')
            self._empty = torch.zeros((1, 0), device=env.device)
        @property
        def action_dim(self):
            return 0
        @property
        def raw_actions(self):
            return self._empty
        @property
        def processed_actions(self):
            return self._empty
        def process_actions(self, actions):
            pass
        def apply_actions(self):
            env = self._env
            env.human_motion.before_physics(env.physics_dt,
                render=env._sim_step_counter % env.cfg.sim.render_interval == 0)
        def reset(self, env_ids=None):
            if env_ids is None or len(env_ids):
                self._env.human_motion.reset()

    @configclass
    class HumanClockActionCfg(ActionTermCfg):
        class_type = HumanClockAction
        asset_name = 'robot'
    cfg.actions.hrc_human_clock = HumanClockActionCfg()

    def capture_step(env):
        # Runs after the final substep and before native auto-reset. This term
        # never terminates an episode and leaves task success/timeout unchanged.
        env.human_motion.after_physics(env.physics_dt)
        env.hrc_step_snapshot = env.human_motion.observe()
        env.hrc_step_snapshot['task_success'] = bool(env.termination_manager.get_term('success')[0])
        return torch.zeros((1,), device=env.device, dtype=torch.bool)
    cfg.terminations.hrc_capture = TerminationTermCfg(func=capture_step)
    name = 'HRC-Lightwheel-v0'
    if name not in gym.registry:
        gym.register(name, entry_point='isaaclab.envs:ManagerBasedRLEnv', disable_env_checker=True)
    return gym.make(name, cfg=cfg).unwrapped
