#!/usr/bin/env python3
"""
Export motion from Blender Armature directly to motion.pkl format.
Bypasses fbx-extract - reads bone transforms from Blender's Python API.

Scene structure: Armature (extract motion from this, Mixamo bones) + SMPLX-{...} (used only for motion name).
  - Armature: object named "Armature" with Mixamo bone hierarchy for animation extraction
  - SMPLX: object named like SMPLX-lh-neutral_phone_pass_1_stageii (motion name extracted from this)
  - Output is merged into existing motion.pkl by default (no overwrite)

Usage (run inside Blender):
  1. Open Blender, load scene with Armature + SMPLX-* armature
  2. Or: blender yourfile.blend -b -P export_motion_from_blender.py -- --output motion.pkl --merge motion.pkl

Output: motion.pkl in the same format as convert_fbx.py
  - trans: (T, 3) root position, meters, Y-up
  - rot: (T, 4) root quaternion wxyz
  - joint: (T, 64, 4) joint quaternions wxyz
  - mat: (T, 65, 3, 3) identity (no basis correction, like convert_fbx2)

Coordinate conversion: Blender Z-up -> target Y-up
  Position: (x, z, y)  [Blender's Z becomes Y, Y becomes Z]
  Rotation: apply axis transform to quaternion
"""

import bpy
import mathutils
import numpy as np
import pickle
import os

# Mixamo hierarchy: parent index for each bone (matches Taunt_hierarchy.txt)
HIERARCHY_PARENT = [
    -1, 0, 1, 2, 3, 4, 5, 6, 3, 7, 8, 9, 10, 11, 12, 13, 14, 10, 15, 16, 17, 18, 10, 19, 20, 21, 22, 10, 23, 24, 25, 26, 10, 27, 28, 29, 30, 3, 31, 32, 33, 34, 35, 36, 37, 38, 34, 39, 40, 41, 42, 34, 43, 44, 45, 46, 34, 47, 48, 49, 50, 34, 51, 52, 53, 54, 0, 55, 56, 57, 58, 59, 0, 60, 61, 62, 63, 64,
]

# Mixamo hierarchy order (65 bones) - must match convert_fbx / Genesis skin
MIXAMO_BONE_NAMES = [
    "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head", "HeadTop_End",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumb4",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumb4",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4",
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky4",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "LeftToe_End",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "RightToe_End",
]


def _find_bone_by_name(armature, name):
    """Find bone by name, with or without mixamorig prefix."""
    for b in armature.pose.bones:
        base = b.name.split(":")[-1] if ":" in b.name else b.name
        if base == name:
            return b
    return None




def _blender_to_yup_pos(v):
    """Blender (Z-up, Y-forward) -> Y-up (Z-forward). x'=x, y'=z, z'=-y."""
    return np.array([v.x, v.z, -v.y], dtype=np.float64)


def _blender_to_yup_quat(q):
    """Blender quat (Z-up) -> Y-up quat (wxyz) matching convert_fbx/Mixamo format.
    After axis transform, quat components need remap: [w, x, z, -y].
    """
    R_swap = mathutils.Matrix(((1, 0, 0), (0, 0, -1), (0, 1, 0)))
    q_swap = R_swap.to_quaternion()
    q_new = q_swap @ q @ q_swap.inverted()
    arr = np.array([q_new.w, q_new.x, q_new.y, q_new.z], dtype=np.float64)
    # Remap to match convert_fbx output: swap y,z and negate y
    return np.array([arr[0], arr[1], arr[3], -arr[2]], dtype=np.float64)


def _quat_to_R(q):
    """Quat wxyz -> 3x3 rotation matrix (pure numpy, no scipy)."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _R_to_quat(R):
    """3x3 rotation matrix -> quat wxyz (pure numpy)."""
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1)
        w, x, y, z = 0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def _compute_coord_mat_from_rest_pose(armature):
    """
    Compute coord_mat (per joint) from Blender's rest pose, same formula as convert_fbx.
    Returns (65, 3, 3) - one 3x3 matrix per bone. No external reference needed.
    """
    static_quats = []
    for name in MIXAMO_BONE_NAMES:
        pose_bone = _find_bone_by_name(armature, name)
        if pose_bone is not None:
            rest_rot = pose_bone.bone.matrix_local.to_3x3().to_quaternion()
            q_yup = _blender_to_yup_quat(rest_rot)
        else:
            q_yup = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        static_quats.append(q_yup)
    static_quats = np.array(static_quats, dtype=np.float64)

    coord_mats = np.zeros((65, 3, 3), dtype=np.float64)
    for i in range(65):
        p = HIERARCHY_PARENT[i]
        if p >= 0:
            coord_mat = _quat_to_R(static_quats[p]) @ _quat_to_R(static_quats[i])
        else:
            coord_mat = _quat_to_R(static_quats[i])
        coord_mats[i] = coord_mat
    return coord_mats


def _extract_motion_name_from_smplx(smplx_obj_name):
    """
    Extract motion name from SMPLX object name.
    E.g. SMPLX-lh-neutral_phone_pass_1_stageii -> neutral_phone_pass_1
    """
    name = smplx_obj_name
    if name.startswith("SMPLX-"):
        name = name[6:]
    for suffix in ("_stageii", "_stagei"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    for prefix in ("lh-", "rh-"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name if name else "motion"


def export_motion_from_blender(
    armature_name=None,
    motion_name=None,
    output_path=None,
    merge_existing=None,
):
    """
    Export Armature's animation to motion.pkl format.
    Scene structure: Armature (extract motion from this, Mixamo bones) + SMPLX-{...} (used only for motion name).
    Motion name is extracted from SMPLX object name, e.g. SMPLX-lh-neutral_phone_pass_1_stageii -> neutral_phone_pass_1.
    Output is merged into existing motion.pkl by default.
    """
    if armature_name:
        armature = bpy.data.objects.get(armature_name)
    else:
        armature = bpy.data.objects.get("Armature")
        if armature is None:
            armature = bpy.context.active_object
            if armature is None or armature.type != "ARMATURE":
                for obj in bpy.data.objects:
                    if obj.type == "ARMATURE":
                        armature = obj
                        break
        if armature is not None:
            bpy.context.view_layer.objects.active = armature
        if armature is None or armature.type != "ARMATURE":
            raise RuntimeError("No Armature found. Use object named 'Armature' (Mixamo bones).")

    if motion_name is None:
        for obj in bpy.data.objects:
            if obj.type == "ARMATURE" and obj.name.startswith("SMPLX-"):
                motion_name = _extract_motion_name_from_smplx(obj.name)
                print(f"Extracted motion name from SMPLX: {obj.name} -> {motion_name}")
                break
    if motion_name is None:
        motion_name = "Taunt"

    # Build bone order (match Mixamo hierarchy)
    bone_order = []
    for name in MIXAMO_BONE_NAMES:
        b = _find_bone_by_name(armature, name)
        if b is None:
            print(f"Warning: bone {name} not found, skipping")
            continue
        bone_order.append((name, b))

    if len(bone_order) < 65:
        print(f"Warning: found {len(bone_order)} bones, expected 65. Missing bones will use identity rotation.")

    # Get frame range from animation
    scene = bpy.context.scene
    if not armature.animation_data or not armature.animation_data.action:
        raise RuntimeError("Armature has no animation action. Assign one (e.g. Armature|mixa)")

    action = armature.animation_data.action
    frame_start = int(action.frame_range[0])
    frame_end = int(action.frame_range[1])
    n_frames = frame_end - frame_start + 1

    # Compute coord_mat from Blender rest pose (no external reference)
    coord_mats = _compute_coord_mat_from_rest_pose(armature)
    mat = np.tile(coord_mats[None, :, :, :], (n_frames, 1, 1, 1))

    trans = np.zeros((n_frames, 3), dtype=np.float64)
    rot = np.zeros((n_frames, 4), dtype=np.float64)
    joint = np.zeros((n_frames, 64, 4), dtype=np.float64)  # 64 joints (skip root)
    joint[:, :, 0] = 1.0  # identity quaternion (wxyz) for missing bones

    for i, frame in enumerate(range(frame_start, frame_end + 1)):
        scene.frame_set(frame)
        bpy.context.view_layer.update()

        for j, (name, bone) in enumerate(bone_order):
            # bone.matrix_basis is the local transform relative to rest pose
            # For pose, we want the rotation part
            loc, rot_q, scale = bone.matrix_basis.decompose()
            q_yup = _blender_to_yup_quat(rot_q)

            if j == 0:
                # Root: trans + rot
                world_loc = armature.matrix_world @ bone.matrix @ mathutils.Vector((0, 0, 0, 1))
                trans[i] = _blender_to_yup_pos(world_loc.to_3d())
                rot[i] = q_yup
            else:
                # Joint (index j-1 in joint array, since we skip root)
                joint[i, j - 1] = q_yup

    # Apply coord_mat to rot and joint so output matches ground truth (convert_fbx format)
    # quat_out = R_to_quat(mat @ quat_to_R(quat) @ mat_inv)
    for i in range(n_frames):
        # Root rotation: mat[:, 0]
        M = mat[i, 0]
        M_inv = np.linalg.inv(M)
        R = _quat_to_R(rot[i])
        R_new = M @ R @ M_inv
        rot[i] = _R_to_quat(R_new)
        # Joint rotations: mat[:, 1:65] for joints 0..63
        for j in range(64):
            M = mat[i, j + 1]
            M_inv = np.linalg.inv(M)
            R = _quat_to_R(joint[i, j])
            R_new = M @ R @ M_inv
            joint[i, j] = _R_to_quat(R_new)

    # Canonical quaternion sign (w >= 0) - no external reference
    for i in range(n_frames):
        if rot[i, 0] < 0:
            rot[i] = -rot[i]
        for j in range(64):
            if joint[i, j, 0] < 0:
                joint[i, j] = -joint[i, j]

    one_motion = {
        "trans": trans,
        "rot": rot,
        "joint": joint,
        "mat": mat,
    }

    if output_path is None:
        blend_path = bpy.data.filepath or ""
        if blend_path:
            output_path = os.path.join(os.path.dirname(blend_path), "motion.pkl")
        else:
            output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motion.pkl")

    # Merge with existing motion.pkl (default behavior)
    merge_path = merge_existing or output_path
    if merge_path and os.path.isfile(merge_path):
        with open(merge_path, "rb") as f:
            motion_data = pickle.load(f)
        motion_data[motion_name] = one_motion
    else:
        motion_data = {motion_name: one_motion}

    if output_path:
        d = os.path.dirname(output_path)
        if d:
            os.makedirs(d, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(motion_data, f)

    print(f"Exported {n_frames} frames to {output_path}")
    print(f"  trans: {trans.shape}, rot: {rot.shape}, joint: {joint.shape}")
    return output_path


if __name__ == "__main__":
    # When run from Blender: blender file.blend -P this_script.py
    # Or: blender file.blend -P this_script.py -- --output /path/to/motion.pkl --merge /path/to/motion.pkl
    # Or: run this script in Blender's Scripting workspace (selected Armature)
    import sys
    kwargs = {"armature_name": None, "motion_name": None, "output_path": None, "merge_existing": None}
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        args = sys.argv[idx + 1:]
        i = 0
        while i < len(args):
            if args[i] == "--output" and i + 1 < len(args):
                kwargs["output_path"] = args[i + 1]
                i += 2
            elif args[i] == "--merge" and i + 1 < len(args):
                kwargs["merge_existing"] = args[i + 1]
                i += 2
            elif args[i] == "--name" and i + 1 < len(args):
                kwargs["motion_name"] = args[i + 1]
                i += 2
            else:
                i += 1
    try:
        export_motion_from_blender(**kwargs)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
