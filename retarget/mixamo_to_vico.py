import os
import glob
import pickle
import argparse
import subprocess
import numpy as np
from tqdm import tqdm
import genesis as gs
import genesis.utils.geom as geom_utils
import torch
from scipy.spatial.transform import Rotation

def euler_to_quat(euler_xyz):
    # added for backward compatibility
    if isinstance(euler_xyz, tuple):
        euler_xyz = np.array(euler_xyz)
    if isinstance(euler_xyz, list):
        euler_xyz = np.array(euler_xyz)
    return xyz_to_quat(euler_xyz)

def xyz_to_quat(euler_xyz, rpy=False, degrees=False):
    if isinstance(euler_xyz, torch.Tensor):
        if degrees:
            euler_xyz *= torch.pi / 180.0
        roll, pitch, yaw = euler_xyz.unbind(-1)
        cosr = (roll * 0.5).cos()
        sinr = (roll * 0.5).sin()
        cosp = (pitch * 0.5).cos()
        sinp = (pitch * 0.5).sin()
        cosy = (yaw * 0.5).cos()
        siny = (yaw * 0.5).sin()
        sign = 1.0 if rpy else -1.0
        qw = cosr * cosp * cosy + sign * sinr * sinp * siny
        qx = sinr * cosp * cosy - sign * cosr * sinp * siny
        qy = cosr * sinp * cosy + sign * sinr * cosp * siny
        qz = cosr * cosp * siny - sign * sinr * sinp * cosy
        return torch.stack([qw, qx, qy, qz], dim=-1)
    elif isinstance(euler_xyz, np.ndarray):
        if rpy:
            rot = Rotation.from_euler("xyz", euler_xyz, degrees=degrees)
        else:
            rot = Rotation.from_euler("zyx", euler_xyz[::-1], degrees=degrees)
        return rot.as_quat(scalar_first=True)
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(euler_xyz)=}")

def xyzw_to_wxyz(xyzw):
    if xyzw.ndim == 1:
        return xyzw[[3, 0, 1, 2]]
    elif xyzw.ndim == 2:
        return xyzw[:, [3, 0, 1, 2]]
    else:
        gs.raise_exception(f"ndim expected to be 1 or 2, but got {xyzw.ndim=}")

def R_to_quat(R):
    if isinstance(R, torch.Tensor):
        batch = R.shape[:-2]  # Support batch dimension
        quat_xyzw = torch.zeros((*batch, 4), dtype=R.dtype, device=R.device)

        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

        # Compute quaternion based on the trace of the matrix
        mask1 = trace > 0
        mask2 = ~mask1 & (R[..., 0, 0] >= R[..., 1, 1]) & (R[..., 0, 0] >= R[..., 2, 2])
        mask3 = ~mask1 & ~mask2 & (R[..., 1, 1] >= R[..., 2, 2])
        mask4 = ~mask1 & ~mask2 & ~mask3

        S = torch.zeros_like(trace)

        S[mask1] = torch.sqrt(trace[mask1] + 1.0) * 2
        quat_xyzw[mask1, 0] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / S[mask1]
        quat_xyzw[mask1, 1] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / S[mask1]
        quat_xyzw[mask1, 2] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / S[mask1]
        quat_xyzw[mask1, 3] = 0.25 * S[mask1]

        S[mask2] = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
        quat_xyzw[mask2, 0] = 0.25 * S[mask2]
        quat_xyzw[mask2, 1] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / S[mask2]
        quat_xyzw[mask2, 2] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / S[mask2]
        quat_xyzw[mask2, 3] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / S[mask2]

        S[mask3] = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
        quat_xyzw[mask3, 0] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / S[mask3]
        quat_xyzw[mask3, 1] = 0.25 * S[mask3]
        quat_xyzw[mask3, 2] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / S[mask3]
        quat_xyzw[mask3, 3] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / S[mask3]

        S[mask4] = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
        quat_xyzw[mask4, 0] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / S[mask4]
        quat_xyzw[mask4, 1] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / S[mask4]
        quat_xyzw[mask4, 2] = 0.25 * S[mask4]
        quat_xyzw[mask4, 3] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / S[mask4]

        return xyzw_to_wxyz(quat_xyzw)
    elif isinstance(R, np.ndarray):
        quat_xyzw = Rotation.from_matrix(R).as_quat().astype(R.dtype)
        return xyzw_to_wxyz(quat_xyzw)
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(R)=}")

def quat_to_R(quat):
    if isinstance(quat, torch.Tensor):
        qw, qx, qy, qz = torch.unbind(quat, -1)
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        two_s = 2.0 / (quat * quat).sum(-1)
        return torch.stack(
            (
                1 - two_s * (qy * qy + qz * qz),
                two_s * (qx * qy - qz * qw),
                two_s * (qx * qz + qy * qw),
                two_s * (qx * qy + qz * qw),
                1 - two_s * (qx * qx + qz * qz),
                two_s * (qy * qz - qx * qw),
                two_s * (qx * qz - qy * qw),
                two_s * (qy * qz + qx * qw),
                1 - two_s * (qx * qx + qy * qy),
            ),
            -1,
        ).reshape(quat.shape[:-1] + (3, 3))
    elif isinstance(quat, np.ndarray):
        return Rotation.from_quat(quat, scalar_first=True).as_matrix().astype(quat.dtype)
    else:
        gs.raise_exception(f"the input must be either torch.Tensor or np.ndarray. got: {type(quat)=}")

crt_path = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument("--extract_root", "-e", type=str, required=True)
parser.add_argument("--output_dir", "-o", type=str, required=True)
parser.add_argument("--incremental", "-i", action="store_true", default=False)
parser.add_argument("--motion_name", "-n", type=str, default=None)
parser.add_argument("--motion_frag", "-f", type=str, default=None, help="usage: -f 0,27,2")
parser.add_argument("--bin_path", "-b", type=str, default="/work/pi_chuangg_umass_edu/qinhongzhou/fbx-extract/build/fbx-extract")
args = parser.parse_args()

if args.incremental:
    assert (args.motion_name is not None) and (args.motion_frag is not None)
    motions = [(args.motion_name, list(map(int, args.motion_frag.split(","))))]
    output_file = os.path.join(crt_path, args.output_dir, f"motion.pkl")
    print(f"increasing motion: {args.motion_name}, {args.motion_frag}")
    print(f"loading motion data from {output_file}...")
    motion_data = pickle.load(open(output_file, "rb"))
    assert args.motion_name not in motion_data
else:
    motions = []
    motion_data = {}
    for root, dirs, files in os.walk(args.extract_root):
        for file in files:
            if file.endswith(".fbx"):
                motion_name = os.path.splitext(file)[0]
                motions.append((motion_name, None))

os.chdir(f"{args.extract_root}")

for (motion_name, clip_range) in tqdm(motions):
    input_file = os.path.join(args.extract_root, motion_name + ".fbx")
    fbx_file_name = os.path.basename(f"{input_file}").split(".")[0]
    txt_files = glob.glob(f"{fbx_file_name}*.txt")

    print(txt_files)

    skeletal_joints_file = f"{fbx_file_name}_skel_local.txt"
    if not os.path.exists(skeletal_joints_file):
        subprocess.call(
            [f"{args.bin_path}", f"{os.path.join(crt_path, input_file)}", f"{os.path.join(crt_path, args.extract_root)}"]
        )

    if not os.path.isfile(skeletal_joints_file):
        continue
    with open(skeletal_joints_file) as fo:
        data = np.asarray(
            [list(map(float, l.strip().split(" "))) for l in fo.readlines() if len(l.split(" ")) > 2 and not l.startswith("#")]
        )

    print("animation length: {} (length, skeleton dim)".format(data.shape))

    base_trans = data[:,3:6] / 100
    euler_angles = np.concatenate([data[:,:3], data[:,6:]], axis=1).reshape(data.shape[0], -1, 3) * 180 / np.pi # deg used in genesis geom utils
    quaternions = np.zeros((euler_angles.shape[0], euler_angles.shape[1], 4))
    for i in range(euler_angles.shape[0]):
        for j in range(euler_angles.shape[1]):
            quaternions[i, j] = geom_utils.euler_to_quat(euler_angles[i, j]) # wxyz format

    static_transform_file = os.path.join(crt_path, args.extract_root, motion_name + "_static_transforms.txt")
    hier_file = os.path.join(crt_path, args.extract_root, motion_name + "_hierarchy.txt")

    with open(static_transform_file) as fo:
        static_transform = np.asarray(
            [list(map(float, l.strip().split(" "))) for l in fo.readlines() if len(l.split(" ")) > 2 and not l.startswith("#")]
        )

    with open(hier_file) as fo:
        hier = np.asarray(
            [list(map(int, l.strip().split(" ")[:2])) for l in fo.readlines() if len(l.split(" ")) > 2 and not l.startswith("#")]
        )

    static_transform = static_transform[:,21:25] # xyzw format
    local_coord_mats = np.zeros((quaternions.shape[0],static_transform.shape[0],3,3))

    for i in range(static_transform.shape[0]):
        p = hier[i, 1]
        static_transform[i] = xyzw_to_wxyz(static_transform[i])
        if p >= 0:
            coord_mat = quat_to_R(static_transform[p]) @ quat_to_R(static_transform[i])
        else:
            coord_mat = quat_to_R(static_transform[i])
        coord_mat_inv = np.linalg.inv(coord_mat)
        static_transform[i] = R_to_quat(coord_mat)
        for frame in range(quaternions.shape[0]):
            new_quat_mat = coord_mat @ quat_to_R(quaternions[frame, i]) @ coord_mat_inv
            quaternions[frame, i] = R_to_quat(new_quat_mat)
            local_coord_mats[frame, i] = coord_mat
    
    if clip_range is not None:
        base_trans = base_trans[clip_range[0]: clip_range[1]: clip_range[2]]
        quaternions = quaternions[clip_range[0]: clip_range[1]: clip_range[2]]

    motion_data[motion_name] = {
        "trans": base_trans,
        "rot": quaternions[:, 0, :],
        "joint": quaternions[:, 1:, :],
        "mat": local_coord_mats
    }

output_file = os.path.join(crt_path, args.output_dir, f"motion.pkl")

print(f"total {len(motion_data)} motions in motion.pkl:")
print("\n".join(motion_data.keys()))
print(f"saving motion data to {output_file}...")
pickle.dump(motion_data, open(output_file, "wb"))

os.chdir(crt_path)