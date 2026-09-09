"""Apply reviewed Lab 3 edits to an isolated b2bcb2d LW-BenchHub checkout.

Run with the checkout path. Re-running accepts already-applied replacements.
"""
import argparse
import ast
from pathlib import Path

ARRAY_FIELDS = set('''body_com_pos_w body_com_pose_w body_com_quat_w body_com_vel_w
body_lin_vel_w body_link_ang_vel_w body_link_pos_w body_link_pose_w body_link_quat_w
body_link_state_w body_pos_w body_quat_w default_joint_pos default_joint_vel
default_mass default_root_state joint_acc joint_effort_target joint_pos joint_pos_limits
joint_pos_target joint_vel joint_vel_target root_com_pos_w root_com_pose_w root_com_quat_w
root_lin_vel_w root_link_pos_w root_link_pose_w root_link_vel_w root_pos_w root_pose_w
root_quat_w root_state_w target_pos_w target_quat_w target_pos_source target_quat_source
source_pos_w source_quat_w net_forces_w net_forces_w_history force_matrix_w
force_matrix_w_history'''.split())


def port_array_access(path):
    source = path.read_text()
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and node.attr in ARRAY_FIELDS
                and ((isinstance(node.value, ast.Attribute) and node.value.attr == 'data')
                     or (isinstance(node.value, ast.Name) and node.value.id == 'ee_tf_data'))
                and isinstance(node.ctx, ast.Load)):
            parent = parents.get(node)
            if isinstance(parent, ast.Attribute) and parent.attr in ('torch', 'warp'):
                continue
            # AST column offsets are UTF-8 bytes, not Unicode character offsets.
            end = offsets[node.end_lineno-1] + len(lines[node.end_lineno-1].encode()[:node.end_col_offset].decode())
            edits.append(end)
    for end in sorted(edits, reverse=True):
        source = source[:end] + '.torch' + source[end:]
    if edits:
        path.write_text(source)


def port_literal_rotations(path):
    source = path.read_text()
    marker = '# LW Lab 3: authored rot= quaternion literals use XYZW.\n'
    if marker in source:
        return
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.keyword) or node.arg != 'rot':
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            continue
        if not all(isinstance(x, (int, float)) for x in value):
            continue
        v = node.value
        start = offsets[v.lineno-1] + len(lines[v.lineno-1].encode()[:v.col_offset].decode())
        end = offsets[v.end_lineno-1] + len(lines[v.end_lineno-1].encode()[:v.end_col_offset].decode())
        edits.append((start, end, repr(tuple(value[1:]) + (value[0],))))
    for start, end, replacement in sorted(edits, reverse=True):
        source = source[:start] + replacement + source[end:]
    if edits:
        path.write_text(source + '\n' + marker)


def replace(root, name, old, new):
    path = root / name
    source = path.read_text()
    if old in new:
        # Protect already-expanded insertions, but still migrate other copies.
        parts = source.split(new)
        if any(old in part for part in parts):
            path.write_text(new.join(part.replace(old, new) for part in parts))
        elif len(parts) == 1:
            raise RuntimeError(f'Unexpected source version: {name}: {old[:80]}')
    elif old in source:
        path.write_text(source.replace(old, new))
    elif new not in source:
        raise RuntimeError(f'Unexpected source version: {name}: {old[:80]}')


def migrate(root):
    replace(root, 'lw_benchhub_tasks/lightwheel_robocasa_tasks/single_stage/kitchen_navigate.py',
            'w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]',
            'x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]  # Lab 3 XYZW')
    replace(root, 'lw_benchhub/core/models/fixtures/fixture.py',
            'env.scene.articulations[self.name].write_joint_position_to_sim(\n'
            '                torch.tensor([[self.rng.uniform(float(desired_min), float(desired_max))]]).to(env.device),\n'
            '                torch.tensor([joint_idx]).to(env.device),\n'
            '                torch.as_tensor(env_ids).to(env.device) if env_ids is not None else None\n'
            '            )',
            'env.scene.articulations[self.name].write_joint_position_to_sim_index(\n'
            '                position=torch.full((env.num_envs if env_ids is None else len(env_ids), 1),\n'
            '                    self.rng.uniform(float(desired_min), float(desired_max)),\n'
            '                    dtype=torch.float32, device=env.device),\n'
            '                joint_ids=torch.tensor([joint_idx], dtype=torch.int32, device=env.device),\n'
            '                env_ids=torch.as_tensor(env_ids, dtype=torch.int32, device=env.device) if env_ids is not None else None\n'
            '            )')
    replace(root, 'lw_benchhub/core/models/fixtures/fixture.py',
            '._root_physx_view.prim_paths', '.root_physx_view.prim_paths')
    # Placement consumes net_forces_w only. The obsolete floor wildcard matches
    # multiple bodies per environment, which the new PhysX filter API rejects.
    replace(root, 'lw_benchhub/core/robots/compositional/pandaomron.py',
            'filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Scene/floor.*"],',
            'filter_prim_paths_expr=[],  # Lab 3: base placement uses total contact force.')
    for name in ('core/tasks/base.py', 'core/rl/base.py', 'core/scenes/kitchen/kitchen.py'):
        path = root / 'lw_benchhub' / name
        source = path.read_text()
        if 'env_cfg.sim.physx.' in source:
            source = source.replace('        env_cfg.sim.physx.bounce_threshold_velocity = 0.2',
                '        from isaaclab_physx.physics import PhysxCfg\n'
                '        if env_cfg.sim.physics is None:\n'
                '            env_cfg.sim.physics = PhysxCfg()\n'
                '        env_cfg.sim.physics.bounce_threshold_velocity = 0.2')
            path.write_text(source.replace('env_cfg.sim.physx.', 'env_cfg.sim.physics.'))
    replace(root, 'lw_benchhub/core/mdp/__init__.py',
            'from isaaclab.envs.mdp import *  # noqa: F401, F403',
            'from isaaclab.envs.mdp import *  # noqa: F401, F403\n'
            'from isaaclab.managers import ActionTerm, ActionTermCfg\n'
            'from isaaclab.controllers import DifferentialIKControllerCfg')
    replace(root, 'lw_benchhub/core/robots/robot_arena_base.py',
            'from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg, RecorderTerm, RecorderTermCfg',
            'from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg\n'
            'from isaaclab.managers import RecorderTerm, RecorderTermCfg')
    replace(root, 'lw_benchhub/utils/usd_utils.py', 'from turtle import st',
            '# Removed unused turtle import: headless runtime does not include Tk.')
    replace(root, 'lw_benchhub/core/context.py',
            'from isaaclab.utils import dataclass', 'from dataclasses import dataclass')
    replace(root, 'lw_benchhub/core/tasks/base.py',
            'from lightwheel_sdk.loader import ENDPOINT',
            'from lightwheel_sdk.client import ENDPOINT')
    # Keep native Lab 3 stepping, reset, recorder, reward and termination managers.
    # Old replacements copy private Lab 2 internals and bypass ProxyArray updates.
    for patch in ('reset', 'recorder_manager_ep_meta', 'recorder_manager_joint_targets',
                  'step', 'reward_manager', 'create_teleop_device',
                  'isaaclab_tasks_mdp', 'termination_manager'):
        replace(root, 'lw_benchhub/utils/monkey_patch.py',
                f'\npatch_{patch}()\n',
                f'\n# Lab 3: use native implementation instead of patch_{patch}().\n')
    for package in ('lw_benchhub', 'lw_benchhub_tasks', 'lw_benchhub_rl',
                    'third_party/IsaacLab-Arena/isaaclab_arena'):
        for path in (root / package).rglob('*.py'):
            port_array_access(path)
            port_literal_rotations(path)
    replace(root, 'lw_benchhub/core/robots/robot_arena_base.py',
            'Tn.convert_quat(Tn.mat2quat(Tn.euler2mat(self.init_robot_base_ori_anchor)), to="wxyz")',
            'Tn.mat2quat(Tn.euler2mat(self.init_robot_base_ori_anchor))')
    replace(root, 'lw_benchhub/core/orchestrate/orchestrate.py',
            'Tt.convert_quat(torch.tensor(obj_quat_xyzw, device=self.context.device, dtype=torch.float32), to="wxyz")',
            'torch.tensor(obj_quat_xyzw, device=self.context.device, dtype=torch.float32)')
    replace(root, 'lw_benchhub/utils/place_utils/env_utils.py',
            'T.convert_quat(T.mat2quat(T.euler2mat(global_ori)), "wxyz")',
            'T.mat2quat(T.euler2mat(global_ori))')
    # Arena's serialized Pose keeps WXYZ; convert only at the Lab 3 boundary.
    replace(root, 'lw_benchhub/core/tasks/base.py',
            'obj_quat_wxyz = tuple(obj_quat)',
            'obj_quat_wxyz = tuple(Tn.convert_quat(obj_quat, to="wxyz"))')
    arena = 'third_party/IsaacLab-Arena/isaaclab_arena/'
    replace(root, arena + 'utils/pose.py', '    def __post_init__(self):',
            '    @property\n    def rotation_xyzw(self):\n'
            '        return self.rotation_wxyz[1:] + self.rotation_wxyz[:1]\n\n'
            '    def __post_init__(self):')
    for name in ('assets/object.py', 'assets/object_reference.py', 'embodiments/embodiment_base.py'):
        path = root / arena / name
        source = path.read_text()
        path.write_text(source.replace('.rotation_wxyz', '.rotation_xyzw'))
    replace(root, 'lw_benchhub/core/robots/compositional/pandaomron.py',
            'scene_config.stand.init_state.rot = pose.rotation_wxyz',
            'scene_config.stand.init_state.rot = pose.rotation_xyzw')
    for name in ('T_B_A', 'T_C_B'):
        replace(root, arena + 'utils/pose.py',
                f'matrix_from_quat(torch.tensor({name}.rotation_wxyz))',
                f'matrix_from_quat(torch.tensor({name}.rotation_xyzw))')
    replace(root, arena + 'utils/pose.py',
            'rotation_wxyz=tuple(q_C_A.tolist())',
            'rotation_wxyz=tuple(q_C_A[[3, 0, 1, 2]].tolist())')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    migrate(parser.parse_args().source)
