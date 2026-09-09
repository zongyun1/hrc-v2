"""Palm-orientation post-process shared by the avatar pick/place motions.

After a position-only IK solve (FABRIK), the palm ends up in whatever
orientation the chain solve implies — for tabletop reaches it often faces
up or sideways.  ``rotate_hand_subtree_palm_down`` rigidly rotates the
hand subtree (hand + fingers) about the palm centre so the palm normal
points toward world −Z, leaving the reached palm position unchanged.

Modes:
  - ``"reference"`` (default) — set the hand's FULL world orientation to a
    palm-down posture harvested from an authored motion clip (see
    ``scripts/probe_palm_ref_scan.py`` → ``palm_ref_pose.json``), yawed
    about world Z so the fingers follow the current reach direction.
    Unlike the delta modes below, this reproduces a naturally-articulated
    wrist instead of inheriting FABRIK's arbitrary twist.
  - ``"forearm_roll"`` — roll only about the forearm axis (pronation).
    Most natural single-DOF correction, but cannot reach full palm-down
    when the forearm is steep.
  - ``"align"`` — single shortest-arc rotation of the palm normal onto
    −Z.  Exact, but can twist the fingers unnaturally.
  - ``"hybrid"`` — forearm roll first (natural pronation), then the
    minimal residual bend to finish the alignment.  Exact and keeps the
    fingers close to the forearm direction.
"""
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

# World = _M_W_I @ internal for skin global-transform positions/vectors
# (see KinematicAvatarSkin.get_global_translation: swap xy, negate x).
_M_W_I = np.array([[0.0, -1.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)

_REF_POSE_PATH = os.path.join(os.path.dirname(__file__), "palm_ref_pose.json")
_ref_pose_cache = None


def _load_ref_pose():
    """Load (and cache) the authored palm-down reference postures.

    Returns {"left": {...}, "right": {...}} or {} if the file is missing.
    If one side is absent it is synthesised by mirroring the other across
    the world X axis (thumb maps to thumb, palm normal flips sign).
    """
    global _ref_pose_cache
    if _ref_pose_cache is not None:
        return _ref_pose_cache
    try:
        with open(_REF_POSE_PATH) as f:
            raw = json.load(f)
    except Exception:
        _ref_pose_cache = {}
        return _ref_pose_cache
    out = {}
    for side in ("left", "right"):
        if side in raw and "R_hand_world" in raw[side]:
            entry = {
                "R": np.asarray(raw[side]["R_hand_world"], dtype=np.float64),
                "az": float(raw[side]["reach_az"]),
                "fore": None,
            }
            if "forearm_dir_world" in raw[side]:
                fore = np.asarray(
                    raw[side]["forearm_dir_world"], dtype=np.float64
                )
                entry["fore"] = fore / (np.linalg.norm(fore) + 1e-12)
            # Optional full-arm bone frames (world) for the arm retarget.
            if "R_bones_world" in raw[side]:
                entry["bones"] = {
                    k: np.asarray(v, dtype=np.float64)
                    for k, v in raw[side]["R_bones_world"].items()
                }
            if "dir_upper_world" in raw[side]:
                up = np.asarray(raw[side]["dir_upper_world"], dtype=np.float64)
                entry["dir_upper"] = up / (np.linalg.norm(up) + 1e-12)
            # Optional shoulder-twist envelope measured from the clips
            # (scripts/probe_shoulder_twist_scan.py).
            if ("R_arm_rest_world" in raw[side]
                    and "rest_upper_dir_world" in raw[side]):
                rd = np.asarray(
                    raw[side]["rest_upper_dir_world"], dtype=np.float64
                )
                entry["arm_rest"] = {
                    "R": np.asarray(
                        raw[side]["R_arm_rest_world"], dtype=np.float64
                    ),
                    "dir": rd / (np.linalg.norm(rd) + 1e-12),
                    "lo": np.deg2rad(
                        float(raw[side].get("shoulder_twist_lo_deg", -180.0))
                    ),
                    "hi": np.deg2rad(
                        float(raw[side].get("shoulder_twist_hi_deg", 180.0))
                    ),
                }
            out[side] = entry
    flip = np.diag([-1.0, 1.0, 1.0])
    for side, other in (("left", "right"), ("right", "left")):
        if side not in out and other in out:
            R_o = out[other]["R"]
            # Mirror across world X: e1/e2 reflect, e3 = e1 x e2 flips too.
            R_m = flip @ R_o @ np.diag([1.0, 1.0, -1.0])
            fore_o = out[other]["fore"]
            out[side] = {
                "R": R_m,
                "az": float(np.pi - out[other]["az"]),
                "fore": None if fore_o is None else flip @ fore_o,
            }
    _ref_pose_cache = out
    return _ref_pose_cache


def rotation_between(src, dst):
    """Shortest-arc rotation matrix taking unit-ish vector src to dst."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    src = src / (np.linalg.norm(src) + 1e-9)
    dst = dst / (np.linalg.norm(dst) + 1e-9)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if dot > 0.9995:
        return np.eye(3)
    if dot < -0.9995:
        axis = np.cross(src, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(src, np.array([0.0, 1.0, 0.0]))
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        return R.from_rotvec(np.pi * axis).as_matrix()
    axis = np.cross(src, dst)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    angle = float(np.arccos(dot))
    return R.from_rotvec(angle * axis).as_matrix()


def roll_about_axis_toward(src, dst, axis):
    """Rotation about ``axis`` that best aligns src with dst (projections)."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    src = np.asarray(src, dtype=np.float64)
    src = src / (np.linalg.norm(src) + 1e-9)
    dst = np.asarray(dst, dtype=np.float64)
    dst = dst / (np.linalg.norm(dst) + 1e-9)

    src_p = src - np.dot(src, axis) * axis
    dst_p = dst - np.dot(dst, axis) * axis
    if np.linalg.norm(src_p) < 1e-7 or np.linalg.norm(dst_p) < 1e-7:
        return np.eye(3)
    src_p = src_p / np.linalg.norm(src_p)
    dst_p = dst_p / np.linalg.norm(dst_p)
    sin_a = float(np.dot(axis, np.cross(src_p, dst_p)))
    cos_a = float(np.clip(np.dot(src_p, dst_p), -1.0, 1.0))
    angle = float(np.arctan2(sin_a, cos_a))
    return R.from_rotvec(angle * axis).as_matrix()


def descendant_indices(skin, root_idx):
    """All node indices in the subtree rooted at root_idx (inclusive)."""
    nodes = skin._nodes[1]
    out = []
    stack = [int(root_idx)]
    while stack:
        idx = stack.pop()
        out.append(idx)
        try:
            stack.extend(int(c) for c in nodes[idx].children)
        except Exception:
            pass
    return out


def rotate_hand_subtree_palm_down(robot, node_trans, hand_id, weight=1.0,
                                  mode="forearm_roll"):
    """Rotate hand/finger nodes so the palm normal points toward world -Z.

    This is a local visual post-process after position IK.  It rotates the
    hand subtree around the palm centre in the avatar's internal node_trans
    coordinates, so the reached palm position stays effectively fixed while
    the fingers/palm face downward.  Returns the edited node_trans copy
    (or the input unchanged on any lookup/degeneracy failure).
    """
    if weight <= 0.0:
        return np.asarray(node_trans, dtype=np.float64).copy()

    skin = robot.skin
    hand_id = 1 if hand_id is None else int(hand_id)
    side = "Left" if hand_id == 0 else "Right"
    try:
        hand_idx = int(skin.node_findup[f"{side}Hand"])
        forearm_idx = int(skin.node_findup[f"{side}ForeArm"])
        thumb_idx = int(skin.node_findup[f"{side}HandThumb1"])
        pinky_idx = int(skin.node_findup[f"{side}HandPinky1"])
        mid_idx = int(skin.node_findup[f"{side}HandMiddle1"])
    except Exception:
        return node_trans

    nt = np.asarray(node_trans, dtype=np.float64).copy()
    if max(hand_idx, forearm_idx, thumb_idx, pinky_idx, mid_idx) >= nt.shape[0]:
        return nt

    saved_node = robot.node_trans
    robot.node_trans = nt
    robot.update()
    global_t = np.asarray(skin.global_transforms, dtype=np.float64).copy()
    if max(hand_idx, forearm_idx, thumb_idx, pinky_idx, mid_idx) >= global_t.shape[0]:
        robot.node_trans = saved_node
        robot.update()
        return nt

    # ``global_transforms = node_trans @ C`` for the current root.  Recover
    # C once, edit the hand subtree in global-transform coordinates, then
    # convert the edited rows back to node_trans.
    try:
        root_tail = np.linalg.inv(nt[hand_idx]) @ global_t[hand_idx]
        root_tail_inv = np.linalg.inv(root_tail)
    except np.linalg.LinAlgError:
        robot.node_trans = saved_node
        robot.update()
        return nt

    def _restore_and_return():
        robot.node_trans = saved_node
        robot.update()
        return nt

    hand_pos = global_t[hand_idx][3, :3].copy()
    thumb = global_t[thumb_idx][3, :3]
    pinky = global_t[pinky_idx][3, :3]
    mid = global_t[mid_idx][3, :3]

    e1 = thumb - pinky
    e1_norm = np.linalg.norm(e1)
    if e1_norm < 1e-8:
        return _restore_and_return()
    e1 = e1 / e1_norm
    v_finger = mid - hand_pos
    v2 = v_finger - np.dot(v_finger, e1) * e1
    v2_norm = np.linalg.norm(v2)
    if v2_norm < 1e-8:
        return _restore_and_return()
    e2 = v2 / v2_norm
    palm_normal = np.cross(e1, e2)
    pn_norm = np.linalg.norm(palm_normal)
    if pn_norm < 1e-8:
        return _restore_and_return()
    palm_normal = palm_normal / pn_norm

    # The internal global-transform frame differs from world by a pure
    # z-rotation + z-offset, so world -Z is also internal -Z.
    # The e1×e2 "normal" flips side between hands (the thumb mirrors):
    # palm-down is e3 = -Z for the RIGHT hand but e3 = +Z for the LEFT.
    desired = np.array(
        [0.0, 0.0, -1.0 if hand_id != 0 else 1.0], dtype=np.float64
    )
    delta = None
    if mode == "reference":
        ref = _load_ref_pose().get("left" if hand_id == 0 else "right")
        if ref is None:
            mode = "hybrid"  # no harvested posture available — fall back
        else:
            # Full-orientation retarget: take the authored palm-down wrist
            # posture and yaw it so the fingers follow the current reach
            # direction; then rotate the subtree from current to that.
            wrist_w, R_cur_w = robot._get_hand_frame(hand_id)
            elbow_w = np.asarray(
                skin.get_global_translation(f"{side}ForeArm")[0],
                dtype=np.float64,
            ).ravel()[:3]
            fore_w = (np.asarray(wrist_w, dtype=np.float64).ravel()[:3]
                      - elbow_w)
            if np.linalg.norm(fore_w[:2]) < 1e-6:
                mode = "hybrid"  # vertical forearm — azimuth undefined
            else:
                az_cur = float(np.arctan2(fore_w[1], fore_w[0]))
                yaw = az_cur - ref["az"]
                cz, sz = np.cos(yaw), np.sin(yaw)
                Rz = np.array([[cz, -sz, 0.0],
                               [sz, cz, 0.0],
                               [0.0, 0.0, 1.0]], dtype=np.float64)
                fore_cur = fore_w / (np.linalg.norm(fore_w) + 1e-12)
                # Carry the reference's WRIST-LOCAL rotation instead of its
                # absolute hand orientation: align the whole reference
                # (forearm + hand rigidly) onto the current IK forearm, so
                # the wrist bend stays exactly as animated no matter how
                # steep the reach is.
                A = np.eye(3)
                if ref.get("fore") is not None:
                    A = rotation_between(Rz @ ref["fore"], fore_cur)
                R_des_w = A @ Rz @ ref["R"]
                # Palm-down correction that cannot bend the wrist: roll
                # about the forearm axis only...
                desired_w = np.array(
                    [0.0, 0.0, -1.0 if hand_id != 0 else 1.0],
                    dtype=np.float64,
                )
                droll = roll_about_axis_toward(
                    R_des_w[:, 2], desired_w, fore_cur
                )
                R_des_w = droll @ R_des_w
                # ...then at most a small capped bend for the remainder.
                resid = rotation_between(R_des_w[:, 2], desired_w)
                rv = R.from_matrix(resid).as_rotvec()
                ang = float(np.linalg.norm(rv))
                cap = np.deg2rad(15.0)
                if ang > cap:
                    resid = R.from_rotvec(rv * (cap / ang)).as_matrix()
                R_des_w = resid @ R_des_w
                delta_w = R_des_w @ np.asarray(R_cur_w, dtype=np.float64).T
                delta = _M_W_I.T @ delta_w @ _M_W_I
    if mode == "forearm_roll":
        forearm_pos = global_t[forearm_idx][3, :3]
        axis = hand_pos - forearm_pos
        delta = roll_about_axis_toward(palm_normal, desired, axis)
    elif mode == "hybrid":
        forearm_pos = global_t[forearm_idx][3, :3]
        axis = hand_pos - forearm_pos
        delta = roll_about_axis_toward(palm_normal, desired, axis)
        rolled = delta @ palm_normal
        delta = rotation_between(rolled, desired) @ delta
    elif delta is None:  # "align"
        delta = rotation_between(palm_normal, desired)
    if delta is None:
        return _restore_and_return()
    if weight < 1.0:
        rotvec = R.from_matrix(delta).as_rotvec() * max(0.0, float(weight))
        delta = R.from_rotvec(rotvec).as_matrix()

    palm_center = np.mean(
        [
            global_t[int(skin.node_findup[f"{side}HandIndex1"])][3, :3],
            global_t[int(skin.node_findup[f"{side}HandMiddle1"])][3, :3],
            global_t[int(skin.node_findup[f"{side}HandRing1"])][3, :3],
            global_t[int(skin.node_findup[f"{side}HandPinky1"])][3, :3],
        ],
        axis=0,
    )

    # Distribute the forearm-axis twist component across the rig's forearm
    # twist bones (ForeArm1/ForeArm2 — the authored clips spread pronation
    # along them).  Dumping the whole roll into the wrist joint
    # candy-wrappers the skin.  Two rig layouts exist:
    #   - chain: Hand is a descendant of the twist bones → give each twist
    #     bone 1/3 of the roll and the hand only the residual;
    #   - sibling helpers (this benchmark's Mixamo rigs): the twist bones
    #     hang off the forearm NEXT to the hand, so the hand must receive
    #     the FULL correction while the helpers take 1/3 and 2/3 of the
    #     roll purely to blend the forearm skin elbow → FA1 → FA2 → hand.
    stages = []
    fa_idx = []
    for nm in (f"{side}ForeArm1", f"{side}ForeArm2"):
        idx = skin.node_findup.get(nm)
        if idx is None or int(idx) >= min(nt.shape[0], global_t.shape[0]):
            fa_idx = []
            break
        fa_idx.append(int(idx))
    axis = hand_pos - global_t[forearm_idx][3, :3]
    a_norm = np.linalg.norm(axis)
    if fa_idx and a_norm > 1e-8:
        axis = axis / a_norm
        q = R.from_matrix(delta).as_quat()  # xyzw
        twist = 2.0 * np.arctan2(float(np.dot(q[:3], axis)), float(q[3]))
        twist = (twist + np.pi) % (2.0 * np.pi) - np.pi
        if abs(twist) > 1e-6:
            if hand_idx in set(descendant_indices(skin, fa_idx[0])):
                # Chain layout: twist bones carry the hand with them.
                Rt = R.from_rotvec(axis * (twist / 3.0)).as_matrix()
                stages.append((fa_idx[0], global_t[fa_idx[0]][3, :3].copy(), Rt))
                stages.append((fa_idx[1], global_t[fa_idx[1]][3, :3].copy(), Rt))
                # Residual for the hand stage: total = residual @ Rt @ Rt.
                pre = R.from_rotvec(axis * (2.0 * twist / 3.0)).as_matrix()
                delta = delta @ pre.T
                # Both twist pivots lie on the forearm axis, so their
                # composition is one rotation about that line; move the
                # hand-stage pivot with it.
                c0 = global_t[fa_idx[0]][3, :3]
                palm_center = c0 + pre @ (palm_center - c0)
            else:
                # Sibling-helper layout: helpers only blend the skin.
                stages.append((
                    fa_idx[0], global_t[fa_idx[0]][3, :3].copy(),
                    R.from_rotvec(axis * (twist / 3.0)).as_matrix(),
                ))
                stages.append((
                    fa_idx[1], global_t[fa_idx[1]][3, :3].copy(),
                    R.from_rotvec(axis * (2.0 * twist / 3.0)).as_matrix(),
                ))
    stages.append((hand_idx, palm_center, delta))

    # Row-vector convention (p' = p @ M, rotation block stores the column
    # matrix TRANSPOSED): composing a world rotation D is a RIGHT-multiply
    # by D.T on the rotation block.  Left-multiplying keeps bone positions
    # right but scrambles the skinning frames — hand mesh twists at every
    # joint.
    for root_i, pivot, D in stages:
        row_rot = D.T
        for idx in descendant_indices(skin, root_i):
            if idx >= nt.shape[0] or idx >= global_t.shape[0]:
                continue
            edited = global_t[idx]
            edited[3, :3] = pivot + (edited[3, :3] - pivot) @ row_rot
            edited[:3, :3] = edited[:3, :3] @ row_rot
            global_t[idx] = edited
            nt[idx] = edited @ root_tail_inv
    robot.node_trans = saved_node
    robot.update()
    return nt


def _scale_rot(delta, weight):
    """Fractional rotation: delta^weight via rotvec scaling."""
    if weight >= 1.0:
        return delta
    rotvec = R.from_matrix(delta).as_rotvec() * max(0.0, float(weight))
    return R.from_rotvec(rotvec).as_matrix()


def _block_world(global_t, idx):
    """World (column-convention) rotation of a bone row.

    Rows store the column matrix transposed with possible per-row scale;
    normalise rows before transposing.  World = _M_W_I @ internal.
    """
    block = np.asarray(global_t[idx][:3, :3], dtype=np.float64)
    norms = np.linalg.norm(block, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return _M_W_I @ (block / norms).T, norms


def _compose_block_world(global_t, nt, idx, delta_w, root_tail_inv):
    """Compose world rotation delta_w onto a bone row's ROTATION ONLY
    (position untouched — used for interior arm bones whose joint
    positions the IK already placed)."""
    R_cur_w, norms = _block_world(global_t, idx)
    R_new_int = _M_W_I.T @ (delta_w @ R_cur_w)
    global_t[idx][:3, :3] = R_new_int.T * norms
    nt[idx] = global_t[idx] @ root_tail_inv


def _rotate_subtree(skin, global_t, nt, root_idx, pivot, delta_int,
                    root_tail_inv):
    """Rigidly rotate a subtree (positions + rotation blocks) about pivot
    by delta_int (internal-frame, column convention)."""
    row_rot = delta_int.T
    for idx in descendant_indices(skin, root_idx):
        if idx >= nt.shape[0] or idx >= global_t.shape[0]:
            continue
        edited = global_t[idx]
        edited[3, :3] = pivot + (edited[3, :3] - pivot) @ row_rot
        edited[:3, :3] = edited[:3, :3] @ row_rot
        global_t[idx] = edited
        nt[idx] = edited @ root_tail_inv


def _palm_frame_internal(skin, global_t, side):
    """(palm_normal, palm_center, ok) from joint POSITIONS, internal frame."""
    try:
        hand_pos = global_t[int(skin.node_findup[f"{side}Hand"])][3, :3]
        thumb = global_t[int(skin.node_findup[f"{side}HandThumb1"])][3, :3]
        pinky = global_t[int(skin.node_findup[f"{side}HandPinky1"])][3, :3]
        mid = global_t[int(skin.node_findup[f"{side}HandMiddle1"])][3, :3]
    except Exception:
        return None, None, False
    e1 = thumb - pinky
    n1 = np.linalg.norm(e1)
    if n1 < 1e-8:
        return None, None, False
    e1 = e1 / n1
    v2 = (mid - hand_pos) - np.dot(mid - hand_pos, e1) * e1
    n2 = np.linalg.norm(v2)
    if n2 < 1e-8:
        return None, None, False
    e2 = v2 / n2
    pn = np.cross(e1, e2)
    npn = np.linalg.norm(pn)
    if npn < 1e-8:
        return None, None, False
    return pn / npn, None, True


def capture_arm_pose(robot, hand_id):
    """Capture the CURRENT arm bone frames as a rest reference.

    Called at motion start (live stance, correct body yaw) so the
    approach can ramp the orientation from — and the retract back to —
    the avatar's actual resting arm instead of FABRIK's arbitrary frame.
    Returns {"bones": {...}, "dir_upper", "fore", "az"} or None.
    """
    skin = robot.skin
    hand_id = 1 if hand_id is None else int(hand_id)
    side = "Left" if hand_id == 0 else "Right"
    robot.update()
    try:
        global_t = np.asarray(skin.global_transforms, dtype=np.float64)
        idxs = {
            k: int(skin.node_findup[f"{side}{b}"])
            for k, b in (("arm", "Arm"), ("forearm", "ForeArm"),
                         ("forearm1", "ForeArm1"), ("forearm2", "ForeArm2"),
                         ("hand", "Hand"))
        }
    except Exception:
        return None
    if max(idxs.values()) >= global_t.shape[0]:
        return None
    bones = {k: _block_world(global_t, i)[0] for k, i in idxs.items()}
    S = global_t[idxs["arm"]][3, :3]
    E = global_t[idxs["forearm"]][3, :3]
    W = global_t[idxs["hand"]][3, :3]
    up = _M_W_I @ (E - S)
    fo = _M_W_I @ (W - E)
    n_up, n_fo = np.linalg.norm(up), np.linalg.norm(fo)
    if n_up < 1e-8 or n_fo < 1e-8:
        return None
    fo = fo / n_fo
    return {
        "bones": bones,
        "dir_upper": up / n_up,
        "fore": fo,
        "az": float(np.arctan2(fo[1], fo[0])),
    }


def apply_reference_arm_pose(robot, node_trans, hand_id, weight=1.0,
                             bend_cap_deg=15.0, rest_pose=None,
                             rest_blend=0.0):
    """Retarget the WHOLE ARM's bone orientations from the authored
    reference posture onto the FABRIK-solved joint positions.

    FABRIK aims each bone with a shortest-arc rotation — zero twist
    control — so upper arm, forearm and the twist helpers come out rolled
    arbitrarily.  Here the IK contributes ONLY joint positions; every
    bone's rotation frame is rebuilt as (swing to the IK bone direction)
    @ (yawed authored bone frame):

      - upper arm  : swing(ref upper dir -> IK upper dir)
      - forearm/FA1/FA2/hand: swing(ref forearm dir -> IK forearm dir)

    so all twist along the chain comes from the authored clip.  The hand
    subtree is rotated rigidly about the WRIST (it stays glued to the
    forearm tip; palm-position error from this is compensated by the
    caller's IK refinement loop).  Palm-down is then restored by forearm
    roll (pronation, distributed 1/3 / 2/3 / full across FA1 / FA2 /
    hand) plus a small capped residual bend.

    rest_pose / rest_blend: blend the DESIRED frames between the
    palm-down reference (rest_blend=0) and a live rest capture from
    ``capture_arm_pose`` (rest_blend=1), both swung onto the current IK
    geometry.  This is how the approach ramps the orientation in and the
    retract ramps it out: interpolating between two authored/live
    postures instead of toward raw FABRIK output — whose arbitrary frame
    made the palm visibly flip outward-then-inward as the hand came back
    toward the body.  The pronation/bend palm-down restore is scaled by
    (1 − rest_blend) so the palm relaxes as the arm returns to rest.

    Returns the edited node_trans, or None when the reference JSON lacks
    the full-arm fields (caller should fall back)."""
    if weight <= 0.0:
        return np.asarray(node_trans, dtype=np.float64).copy()
    hand_id = 1 if hand_id is None else int(hand_id)
    ref = _load_ref_pose().get("left" if hand_id == 0 else "right")
    if not ref or ref.get("bones") is None or ref.get("fore") is None \
            or ref.get("dir_upper") is None:
        return None
    bones_ref = ref["bones"]
    if any(k not in bones_ref for k in
           ("arm", "forearm", "forearm1", "forearm2", "hand")):
        return None

    skin = robot.skin
    side = "Left" if hand_id == 0 else "Right"
    try:
        idx_arm = int(skin.node_findup[f"{side}Arm"])
        idx_fore = int(skin.node_findup[f"{side}ForeArm"])
        idx_fa1 = int(skin.node_findup[f"{side}ForeArm1"])
        idx_fa2 = int(skin.node_findup[f"{side}ForeArm2"])
        idx_hand = int(skin.node_findup[f"{side}Hand"])
    except Exception:
        return None

    nt = np.asarray(node_trans, dtype=np.float64).copy()
    saved_node = robot.node_trans
    robot.node_trans = nt
    robot.update()
    global_t = np.asarray(skin.global_transforms, dtype=np.float64).copy()
    hi = max(idx_arm, idx_fore, idx_fa1, idx_fa2, idx_hand)
    if hi >= nt.shape[0] or hi >= global_t.shape[0]:
        robot.node_trans = saved_node
        robot.update()
        return None
    try:
        root_tail = np.linalg.inv(nt[idx_hand]) @ global_t[idx_hand]
        root_tail_inv = np.linalg.inv(root_tail)
    except np.linalg.LinAlgError:
        robot.node_trans = saved_node
        robot.update()
        return None

    def _restore():
        robot.node_trans = saved_node
        robot.update()

    # IK joint positions (internal frame) and bone directions (world).
    S = global_t[idx_arm][3, :3].copy()
    E = global_t[idx_fore][3, :3].copy()
    W = global_t[idx_hand][3, :3].copy()
    up_cur_w = _M_W_I @ (E - S)
    fo_cur_w = _M_W_I @ (W - E)
    n_up = np.linalg.norm(up_cur_w)
    n_fo = np.linalg.norm(fo_cur_w)
    if n_up < 1e-8 or n_fo < 1e-8:
        _restore()
        return None
    up_cur_w /= n_up
    fo_cur_w /= n_fo

    # Yaw the whole reference to the current reach azimuth.  Damp the
    # yaw-follow as the forearm approaches vertical: there the azimuth
    # estimate is noisy and the yaw mostly spins the hand about the
    # forearm axis — the visible "palm flips outward" artifact near the
    # body.  fo_cur_w is unit, so its horizontal magnitude is the natural
    # damping signal.
    az_cur = float(np.arctan2(fo_cur_w[1], fo_cur_w[0]))
    horiz = float(np.linalg.norm(fo_cur_w[:2]))
    yaw_damp = min(1.0, horiz / 0.25)
    yaw_damp = yaw_damp * yaw_damp * (3.0 - 2.0 * yaw_damp)

    def desired_from(src):
        """Desired world frames for all 5 bones from one reference-like
        posture (palm-down ref or live rest capture), swung onto the
        current IK bone directions with damped yaw-follow."""
        yw = az_cur - src["az"]
        yw = (yw + np.pi) % (2.0 * np.pi) - np.pi
        yw *= yaw_damp
        cz, sz = np.cos(yw), np.sin(yw)
        Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        s_up = rotation_between(Rz @ src["dir_upper"], up_cur_w)
        s_fo = rotation_between(Rz @ src["fore"], fo_cur_w)
        b = src["bones"]
        return {
            "arm": s_up @ Rz @ b["arm"],
            "forearm": s_fo @ Rz @ b["forearm"],
            "forearm1": s_fo @ Rz @ b["forearm1"],
            "forearm2": s_fo @ Rz @ b["forearm2"],
            "hand": s_fo @ Rz @ b["hand"],
        }

    des = desired_from({"bones": bones_ref, "dir_upper": ref["dir_upper"],
                        "fore": ref["fore"], "az": ref["az"]})
    rb = float(np.clip(rest_blend, 0.0, 1.0))
    if rb > 1e-3 and rest_pose is not None:
        try:
            des_rest = desired_from(rest_pose)
            # Per-bone slerp from the palm-down frames toward the rest
            # frames by rb (rb=1 → pure rest).
            des = {
                k: _scale_rot(des_rest[k] @ des[k].T, rb) @ des[k]
                for k in des
            }
        except Exception:
            pass

    R_arm_des = des["arm"]
    # Clamp the shoulder twist into the envelope the authored clips use.
    # Twist = swing-twist residual about the CURRENT upper-arm axis,
    # measured against the rest pose's Arm frame swung onto that axis —
    # the same measure probe_shoulder_twist_scan.py ran over the clips.
    # Swinging the reference onto steep IK reach directions can compose
    # into shoulder roll outside anything the clips contain; project it
    # back.  Forearm/hand frames are set independently below, so this
    # touches only the shoulder region's skinning.
    arm_rest = ref.get("arm_rest")
    if arm_rest is not None:
        R_zero = rotation_between(arm_rest["dir"], up_cur_w) @ arm_rest["R"]
        q = R.from_matrix(R_arm_des @ R_zero.T).as_quat()  # xyzw
        t = 2.0 * np.arctan2(float(np.dot(q[:3], up_cur_w)), float(q[3]))
        t = float((t + np.pi) % (2.0 * np.pi) - np.pi)
        t_cl = float(np.clip(t, arm_rest["lo"], arm_rest["hi"]))
        if abs(t_cl - t) > 1e-6:
            R_arm_des = (
                R.from_rotvec(up_cur_w * (t_cl - t)).as_matrix() @ R_arm_des
            )
    des_w = {
        idx_arm: R_arm_des,
        idx_fore: des["forearm"],
        idx_fa1: des["forearm1"],
        idx_fa2: des["forearm2"],
    }
    # Interior bones: rotation-only edits (positions are IK's).
    for idx, R_des in des_w.items():
        R_cur_w, _ = _block_world(global_t, idx)
        delta_w = _scale_rot(R_des @ R_cur_w.T, weight)
        _compose_block_world(global_t, nt, idx, delta_w, root_tail_inv)

    # Hand subtree: rigid rotation about the WRIST so the hand stays on
    # the forearm tip.  Carries the authored wrist-local rotation because
    # forearm and hand receive the same swing/yaw composition.
    R_hand_cur_w, _ = _block_world(global_t, idx_hand)
    delta_w = _scale_rot(des["hand"] @ R_hand_cur_w.T, weight)
    delta_int = _M_W_I.T @ delta_w @ _M_W_I
    _rotate_subtree(skin, global_t, nt, idx_hand, W, delta_int,
                    root_tail_inv)

    # Palm-down restore: forearm roll (pronation) — measured from joint
    # positions, applied about the forearm axis THROUGH the wrist so the
    # wrist stays put.  Distribute 1/3 / 2/3 of the roll onto the twist
    # helpers' rotation blocks for a smooth skin gradient.
    desired_int = np.array([0.0, 0.0, -1.0 if hand_id != 0 else 1.0])
    axis_int = W - E
    axis_int = axis_int / (np.linalg.norm(axis_int) + 1e-12)
    pn, _, ok = _palm_frame_internal(skin, global_t, side)
    pd_weight = weight * (1.0 - rb)
    if ok and pd_weight > 1e-3:
        droll = _scale_rot(
            roll_about_axis_toward(pn, desired_int, axis_int), pd_weight
        )
        rv = R.from_matrix(droll).as_rotvec()
        _compose_block_world(
            global_t, nt, idx_fa1,
            _M_W_I @ R.from_rotvec(rv / 3.0).as_matrix() @ _M_W_I.T,
            root_tail_inv,
        )
        _compose_block_world(
            global_t, nt, idx_fa2,
            _M_W_I @ R.from_rotvec(rv * (2.0 / 3.0)).as_matrix() @ _M_W_I.T,
            root_tail_inv,
        )
        _rotate_subtree(skin, global_t, nt, idx_hand, W, droll,
                        root_tail_inv)
        # Small capped residual bend for what pronation can't reach.
        pn2, _, ok2 = _palm_frame_internal(skin, global_t, side)
        if ok2:
            resid = rotation_between(pn2, desired_int)
            rv2 = R.from_matrix(resid).as_rotvec()
            ang = float(np.linalg.norm(rv2))
            cap = np.deg2rad(float(bend_cap_deg))
            if ang > 1e-6:
                if ang > cap:
                    rv2 = rv2 * (cap / ang)
                _rotate_subtree(
                    skin, global_t, nt, idx_hand, W,
                    _scale_rot(R.from_rotvec(rv2).as_matrix(), pd_weight),
                    root_tail_inv,
                )

    _restore()
    return nt
