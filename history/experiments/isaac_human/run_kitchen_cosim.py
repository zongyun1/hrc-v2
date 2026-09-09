"""HARP Phase 1 analogue: RoboCasa kitchen, parked PandaOmron, recorded human.

Run with python -m isaac_human.run_kitchen_cosim. Real source motions retain
their world coordinates unless an explicit transform is supplied.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
import numpy as np
from .motion import MotionPlayer, sha256


def prepare(args):
    legacy = getattr(args, 'legacy_avatar_glb', None)
    retarget = getattr(args, 'retarget_trumans', False)
    if retarget and (not legacy or not args.motion or not args.skeleton or not args.skin or args.capsules):
        raise ValueError('TRUMANS retargeting requires GLB, motion, skeleton and source skin for sole-height calibration; no capsules')
    if legacy and not retarget:
        if args.motion or args.skeleton or args.skin or args.capsules:
            raise ValueError('Legacy GLB walking and SMPL motion inputs are separate modes')
        if not math.isfinite(args.walk_duration) or args.walk_duration <= 0:
            raise ValueError('walk_duration must be positive and finite')
        player = SimpleNamespace(end_time=args.walk_duration)
        synthetic = False
    else:
        if not args.motion or not args.skeleton:
            raise ValueError('Provide motion and skeleton, or --legacy_avatar_glb')
        player = MotionPlayer.load(args.motion, args.skeleton, args.skin,
                              yaw=args.human_yaw, translation=args.human_translation)
        if player.sequence.n_envs != 1:
            raise ValueError('Kitchen co-simulation supports B=1')
        synthetic = bool(player.sequence.meta.get('synthetic_diagnostic', False))
        if synthetic and not args.allow_diagnostic:
            raise ValueError('Synthetic motion requires --allow_diagnostic')
        if retarget and (synthetic or player.sequence.meta.get('backend') != 'trumans'):
            raise ValueError('TRUMANS retargeting requires a genuine TRUMANS export')
        if not args.skin and not args.capsules and not retarget:
            raise ValueError('Provide --skin or explicitly select --capsules')
    source = args.migration_root / 'source' / 'NavigateKitchen'
    conversion_path = args.migration_root / 'usd' / 'NavigateKitchen' / 'conversion.json'
    conversion = json.loads(conversion_path.read_text())
    usd = Path(conversion['usd_path'])
    paths = ([Path(legacy)] if legacy else [Path(args.motion), Path(args.skeleton)]) + [source/'manifest.json',
             source/'scene.xml', source/'mobile_dynamics.json', conversion_path, usd]
    if args.skin:
        paths.append(Path(args.skin))
    if retarget:
        paths.extend([Path(args.motion),Path(args.skeleton)])
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    provenance = {'synthetic_diagnostic': synthetic,
                  'files': {str(p): sha256(p) for p in paths},
                  'human_yaw_rad': args.human_yaw,
                  'human_translation_m': args.human_translation,
                  'source_meta': {'asset': 'legacy Genesis Adrian Keller GLB',
                                  'motion': 'procedural planted-foot IK, not TRUMANS'} if legacy and not retarget else player.sequence.meta,
                  'scene': 'existing RoboCasa NavigateKitchen layout1/style1 export',
                  'coworker_reference': '6c7c467:tools/phase1_cosim_demo.py'}
    if retarget and player.sequence.joints is not None:
        errors = []
        for frame in np.linspace(0,player.sequence.n_frames-1,12,dtype=int):
            expected = player.sequence.joints[:,frame,:22]@player.rotation.T+player.translation
            errors.append(float(np.max(np.linalg.norm(player.pose(frame/player.sequence.fps)[0]-expected,axis=-1))))
        provenance['source_joint_cache_max_error_m'] = max(errors)
        if max(errors) > .01:
            raise ValueError('Source SMPL-X joint cache disagrees with motion/template FK')
    return player, source, usd, provenance


def run(args, player, source, usd, provenance):
    import torch
    import imageio.v2 as imageio
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sensors import Camera, CameraCfg
    import omni.replicator.core as rep
    # Existing migration scripts use sibling imports; retain their proven helpers.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'robocasa_migration'))
    from geometry_rules import apply_geometry_rules, repair_material_textures
    from mobile_physics import apply_mobile_physics
    from mobile_navigation import tensor, yaw_xyzw
    from .adapter import IsaacHuman

    dt = 1/120
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device=args.device,
        render=sim_utils.RenderCfg(antialiasing_mode='TAA', samples_per_pixel=32,
                                   enable_dlssg=False, enable_dl_denoiser=False)))
    cfg = sim_utils.UsdFileCfg(usd_path=str(usd))
    cfg.func('/World/Kitchen', cfg)
    kitchen = sim.stage.GetPrimAtPath('/World/Kitchen')
    while instances := [p for p in Usd.PrimRange(kitchen) if p.IsInstance()]:
        for prim in instances:
            prim.SetInstanceable(False)
    geometry = apply_geometry_rules(kitchen, source/'scene.xml')
    textures = repair_material_textures(kitchen, source/'scene.xml', usd.parent/'repaired_textures')
    physics = apply_mobile_physics(kitchen, source/'mobile_dynamics.json')
    # The shared helper's description assumes its navigation controller. Here the
    # chassis is position-held, with no additional generalized friction forces.
    physics['friction_model'] = ('Legacy planar/torso friction coefficients removed; '
        'this parked demo uses PD position holds without explicit friction forces.')
    for prim in Usd.PrimRange(kitchen):
        if prim.GetName().startswith(('wall_front', 'ceiling')) and prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(prim).MakeInvisible()
        if prim.GetName().endswith('_eef_target'):
            prim.SetActive(False)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and 'hinge' in prim.GetName():
            mass = UsdPhysics.MassAPI.Apply(prim)
            if not mass.GetMassAttr().Get() or mass.GetMassAttr().Get() <= 0:
                mass.CreateMassAttr(.001)
                mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-6))
    manifest = json.loads((source/'manifest.json').read_text())
    robot_root = physics['reference_anchor']['articulation_root']
    robot_body = physics['reference_anchor']['reference_body']
    reset = {name: spec['qpos'][0] for name, spec in manifest['joints'].items()
             if name.startswith(('robot0_', 'mobilebase0_', 'gripper0_'))}
    robot = Articulation(ArticulationCfg(prim_path=robot_root, spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(joint_pos=reset), actuators={
        'base': ImplicitActuatorCfg(joint_names_expr=['mobilebase0_joint_mobile_.*'],
            stiffness=10000., damping=1500., effort_limit_sim=600.),
        'torso': ImplicitActuatorCfg(joint_names_expr=['mobilebase0_joint_torso_height'],
            stiffness=20000., damping=2000., effort_limit_sim=100000.),
        'arm': ImplicitActuatorCfg(joint_names_expr=['robot0_joint.*'],
            stiffness=1000., damping=100., effort_limit_sim=300.),
        'gripper': ImplicitActuatorCfg(joint_names_expr=['gripper0_.*'],
            stiffness=1000., damping=100., effort_limit_sim=100.)}))
    furniture = [Articulation(ArticulationCfg(prim_path=str(p.GetPath()), spawn=None,
        actuators={'passive': ImplicitActuatorCfg(joint_names_expr=['.*'], stiffness=0., damping=0.)}))
        for p in Usd.PrimRange(kitchen) if p.HasAPI(UsdPhysics.ArticulationRootAPI)
        and str(p.GetPath()) != robot_root]
    if args.retarget_trumans:
        from .retarget import RetargetedHuman
        human = RetargetedHuman(sim.stage,args,sim.device,player)
    elif args.legacy_avatar_glb:
        from .legacy_walk import LegacyWalk
        human = LegacyWalk(sim.stage, args, sim.device)
    else:
        human = IsaacHuman(sim.stage, player, sim.device, robot_path=robot_body,
                           show_capsules=args.capsules)
    light = sim_utils.DomeLightCfg(intensity=1500.)
    light.func('/World/Light', light)
    camera = Camera(CameraCfg(prim_path='/World/Camera', width=720, height=544,
        data_types=['rgb'], spawn=sim_utils.PinholeCameraCfg(focal_length=18., horizontal_aperture=24.)))
    sim.reset()
    rep.settings.set_render_pathtraced(samples_per_pixel=32)
    camera.set_world_poses_from_view(torch.tensor([args.camera_eye], device=sim.device),
                                     torch.tensor([args.camera_target], device=sim.device))
    if not robot.is_fixed_base:
        raise RuntimeError('PandaOmron must have a fixed reference root and driven chassis joints')
    q0 = tensor(robot.data.default_joint_pos).clone()
    arm = [robot.joint_names.index(f'robot0_joint{i}') for i in range(1, 8)]
    base = [robot.joint_names.index('mobilebase0_joint_mobile_'+n) for n in ('forward','side','yaw')]
    limits = tensor(robot.data.soft_joint_pos_limits)
    # Coworker's neutral arm is the midpoint of each bounded joint range.
    q0[:, arm] = limits[:, arm].mean(-1)
    # Relocate only at reset through the original planar DOFs, never the fixed root.
    w, x, y, z = manifest['bodies']['robot0_base']['quat_wxyz']
    yaw = yaw_xyzw([x, y, z, w])
    c, s = math.cos(yaw), math.sin(yaw)
    start = np.array(manifest['bodies']['mobilebase0_base']['pos'])
    delta = np.array([[c,s],[-s,c]]) @ (np.array(args.robot_xy)-start[:2])
    q0[:, base[:2]] += torch.tensor(delta, device=sim.device, dtype=q0.dtype)
    w, x, y, z = manifest['bodies']['mobilebase0_base']['quat_wxyz']
    q0[:, base[2]] += args.robot_yaw-yaw_xyzw([x,y,z,w])
    robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
    base_id = robot.body_names.index('mobilebase0_base')
    eef_id = robot.body_names.index('gripper0_right_right_gripper')
    for a in furniture:
        q = tensor(a.data.default_joint_pos).clone()
        for i, name in enumerate(a.joint_names):
            if name in manifest['joints']:
                q[:, i] = manifest['joints'][name]['qpos'][0]
        a.write_joint_state_to_sim(q, torch.zeros_like(q))
    human.set_time(0.)
    for _ in range(60):
        robot.set_joint_position_target(q0)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(dt)
    base_start = tensor(robot.data.body_pos_w)[0, base_id].cpu().numpy().copy()
    trace, arm_states = [], []
    max_base_drift = 0.
    peak_force = None if args.legacy_avatar_glb else 0.
    writer = imageio.get_writer(args.output_dir/'preview.mp4', fps=15)
    try:
        # Include the last motion frame and use physical elapsed time for both actors.
        for step in range(math.ceil(player.end_time/dt)+1):
            time = min(step*dt, player.end_time)
            phase = time/player.end_time
            q = q0.clone()
            for joint, amplitude, offset in ((1,.55,0.),(3,.45,1.),(5,.60,2.)):
                q[:, arm[joint]] += amplitude*math.sin(2*math.pi*phase+offset)
            q = torch.maximum(torch.minimum(q, limits[...,1]), limits[...,0])
            robot.set_joint_position_target(q)
            robot.write_data_to_sim()
            human.set_time(time, render=step%8 == 0)
            sim.step(render=step%8 == 0)
            robot.update(dt)
            for a in furniture:
                a.update(dt)
            force = human.update(dt)
            if force is not None:
                peak_force = max(peak_force, force)
            joints = tensor(robot.data.joint_pos)
            if not torch.isfinite(joints).all():
                raise RuntimeError('Non-finite robot state')
            bp = tensor(robot.data.body_pos_w)[0,base_id].cpu().numpy()
            max_base_drift = max(max_base_drift,float(np.linalg.norm(bp-base_start)))
            if step%8 == 0:
                camera.update(8*dt, force_recompute=True)
                rgb = tensor(camera.data.output['rgb'])[0,...,:3].cpu().numpy().astype(np.uint8)
                if rgb.std() < 1:
                    raise RuntimeError('Empty render')
                writer.append_data(rgb)
                if step == 0:
                    imageio.imwrite(args.output_dir/'start.png',rgb)
                imageio.imwrite(args.output_dir/'preview.png',rgb)
                eef = tensor(robot.data.body_pos_w)[0,eef_id].cpu().numpy()
                arm_states.append(joints[0,arm].cpu().tolist())
                trace.append({'time_s':time,'base_pos':bp.tolist(),
                    'eef_capsule_clearance_m':human.clearance(eef,time),
                    'robot_human_contact_force_n':force,'arm_q':arm_states[-1]})
    finally:
        writer.close()
        (args.output_dir/'trace.json').write_text(json.dumps(trace))
    arm_excursion = np.ptp(np.array(arm_states),axis=0).max()
    checks = {'motion_completed':time >= player.end_time,
              'human_tracked':human.max_tracking_error < .005,
              'base_held':max_base_drift < .05, 'arm_moved':bool(arm_excursion > .2)}
    if args.legacy_avatar_glb and not args.retarget_trumans:
        checks['walk_arrived'] = bool(np.linalg.norm(human.avatar.position-np.array(args.walk_end)) < .01)
        checks['foot_ik'] = human.avatar.walk_motion.max_foot_error < .025
    if args.retarget_trumans:
        checks['joint_rotation_mapping'] = human.retarget.max_rotation_error < 1e-5
    result = {'status':('diagnostic_passed' if provenance['synthetic_diagnostic'] else 'passed')
              if all(checks.values()) else 'failed', 'checks':checks,
              'mode':'kitchen_pandaomron_human_cosim', 'provenance':provenance,
              'scope':'Independent prescribed human and arm sweep; no navigation, avoidance or manipulation success claim.',
              'robot_xy':args.robot_xy,'robot_yaw_rad':args.robot_yaw,
              'duration_s':player.end_time,'max_base_drift_m':max_base_drift,
              'arm_excursion_rad':float(arm_excursion),
              'max_capsule_tracking_error_m':human.max_tracking_error,
              'peak_reported_robot_contact_force_n':peak_force,
              'geometry_rules':geometry,'repaired_material_count':len(textures),
              'mobile_physics':physics}
    if args.legacy_avatar_glb:
        result['max_proxy_tracking_error_m'] = result.pop('max_capsule_tracking_error_m')
        result['human_model'] = 'textured_legacy_GLB_with_TRUMANS_retargeting' if args.retarget_trumans else 'textured_legacy_GLB_with_procedural_leg_IK'
        result['collision_scope'] = 'Torso and palms only; no full-body capsule model or contact-force measurement'
        if args.retarget_trumans:
            result['retargeting'] = {'root_height_scale':human.retarget.height_scale,
                'max_rotation_matrix_error':human.retarget.max_rotation_error,
                'method':'Global joint rotations with bind-frame corrections; source root XY/time unchanged; no procedural gait or foot IK',
                'limitation':'Different body proportions may introduce foot sliding; fingers retain bind pose.'}
        else:
            result['walk'] = {'start':args.walk_start,'end':args.walk_end,
                          'duration_s':args.walk_duration,
                          'max_foot_ik_error_m':human.avatar.walk_motion.max_foot_error}
    (args.output_dir/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    if not all(checks.values()):
        raise RuntimeError('Co-simulation checks failed; see result.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion')
    parser.add_argument('--skeleton')
    parser.add_argument('--legacy_avatar_glb',type=Path)
    parser.add_argument('--retarget_trumans',action='store_true')
    parser.add_argument('--walk_start',type=float,nargs=3,default=[4.1,-2.1,0.])
    parser.add_argument('--walk_end',type=float,nargs=3,default=[.9,-2.1,0.])
    parser.add_argument('--walk_duration',type=float,default=10.)
    parser.add_argument('--skin')
    parser.add_argument('--capsules',action='store_true')
    parser.add_argument('--allow_diagnostic',action='store_true')
    parser.add_argument('--check_assets',action='store_true')
    parser.add_argument('--human_yaw',type=float,default=0.)
    parser.add_argument('--human_translation',type=float,nargs=3,default=[0.,0.,0.])
    parser.add_argument('--robot_xy',type=float,nargs=2,default=[2.,-1.15])
    parser.add_argument('--robot_yaw',type=float,default=math.pi/2)
    parser.add_argument('--camera_eye',type=float,nargs=3,default=[4.,-5.5,3.8])
    parser.add_argument('--camera_target',type=float,nargs=3,default=[2.5,-1.3,1.])
    parser.add_argument('--migration_root',type=Path,default=Path('outputs/robocasa_migration'))
    parser.add_argument('--output_dir',type=Path,default=Path('outputs/isaac_human/kitchen_manual'))
    pre, _ = parser.parse_known_args()
    prepared = prepare(pre)
    if pre.check_assets:
        parser.parse_args()
        print(json.dumps(prepared[-1],indent=2))
        return
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    launcher = AppLauncher(args)
    try:
        run(args,*prepared)
    except Exception:
        (args.output_dir/'error.txt').write_text(traceback.format_exc())
        if not (args.output_dir/'result.json').exists():
            (args.output_dir/'result.json').write_text(json.dumps({'status':'error'}))
        raise
    finally:
        launcher.app.close()


if __name__ == '__main__':
    main()
