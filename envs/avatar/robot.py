"""Avatar robot: skin + box entities, pose update. Uses scene directly (Option B)."""
import numpy as np
import genesis as gs
import genesis.utils.geom as geom_utils
from ..genesis_compat import avatar_or_kinematic_material, has_avatar_material, mesh_frame_kwargs
from .utils import (
    AvatarState,
    ActionStatus,
    SMPLX_JOINT_NUM,
)
from .kinematic_skin import KinematicAvatarSkin

from scipy.spatial.transform import Rotation as R


def _to_numpy(x):
    """Convert tensor/array to numpy."""
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def _R_to_ypr(rot):
    fn = getattr(geom_utils, "R_to_ypr", None)
    if fn is not None:
        return fn(rot)
    return R.from_matrix(rot).as_euler("zyx").astype(np.asarray(rot).dtype)


class AvatarRobot:
    """Avatar representation: collision box + skin mesh. Accepts scene directly."""

    def __init__(self, scene, skin_options, name):
        has_native_avatar = has_avatar_material()
        mat_avatar = avatar_or_kinematic_material()
        self.box = scene.add_entity(
            material=mat_avatar,
            morph=gs.morphs.Box(
                lower=(-0.3, -0.3, 0.5),
                upper=(0.3, 0.3, 1.5),
                fixed=False,
                visualization=False,
            ),
            surface=gs.surfaces.Default(color=(1.0, 1.0, 1.0)),
        )
        self.joint_num = SMPLX_JOINT_NUM

        if skin_options is not None:
            morph_kwargs = dict(
                file=skin_options["glb_path"],
                euler=skin_options["euler"],
                pos=skin_options["pos"],
                decimate=False,
                convexify=False,
                collision=False,
                group_by_material=False,
            )
            morph_kwargs.update(mesh_frame_kwargs(skin_options["glb_path"], align=False))
            if not has_native_avatar:
                morph_kwargs["enable_custom_vverts"] = True
            self.skin = scene.add_entity(
                material=mat_avatar,
                morph=gs.morphs.Mesh(**morph_kwargs),
            )
            if not has_native_avatar:
                self.skin = KinematicAvatarSkin(
                    self.skin,
                    glb_path=skin_options["glb_path"],
                    group_by_material=morph_kwargs["group_by_material"],
                    scale=morph_kwargs.get("scale", 1.0),
                )
            self.skin_rot = geom_utils.euler_to_R(skin_options["euler"])
        else:
            self.skin = None

        self.base_state = AvatarState.NO_STATE
        self.action_state = AvatarState.NO_ACTION
        self.action_status = ActionStatus.INIT
        # attached_object[hand_id] = (entity, v2_local, R_obj_local, R_obj_init) or None
        # v2_local: object offset from hand_pos expressed in hand's 3-D local frame
        # R_obj_local: object rotation matrix expressed in hand's 3-D local frame
        self.attached_object = [None, None]
        self.two_hand_attached_object = None

        self.base_rot = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        self.global_trans = np.zeros(3, dtype=np.float64)
        self.global_rot = np.eye(3, dtype=np.float64)
        self.pose = np.zeros(7 + self.joint_num * 4)
        self.stop_pose = np.zeros_like(self.pose)
        self.stop_node = []
        self.stop_mat = np.zeros((65, 3, 3), dtype=np.float64)
        self.stop_mat_inv = np.zeros((65, 3, 3), dtype=np.float64)
        self.sit_mat = np.zeros((65, 3, 3), dtype=np.float64)
        self.sit_mat_inv = np.zeros((65, 3, 3), dtype=np.float64)
        self.global_mat = np.zeros_like(self.stop_mat)
        self.sit_pose = np.zeros_like(self.pose)
        self.sit_node = []
        self._h_attach_to = None

    def reset(
        self,
        global_trans: np.ndarray = np.zeros(3, dtype=np.float64),
        global_rot: np.ndarray = np.eye(3, dtype=np.float64),
        update_mesh: bool = True,
    ):
        self.base_state = AvatarState.STANDING
        self.action_state = AvatarState.NO_ACTION
        self.action_status = ActionStatus.INIT
        self.attached_object = [None, None]
        self.two_hand_attached_object = None
        self.global_trans = global_trans
        self.global_rot = global_rot
        self.pose = self.stop_pose
        self.node_trans = self.stop_node
        self.global_mat = self.stop_mat
        self.global_mat_inv = self.stop_mat_inv
        self._h_attach_to = None
        if update_mesh:
            self.update()

    def get_global_xy(self):
        if self.action_state == "walk":
            total_trans = self.global_rot @ self.base_rot @ self.pose[:3] + self.global_trans
        else:
            total_trans = self.global_trans.copy()
        return total_trans[0], total_trans[1]

    def get_global_height(self):
        return self.global_trans[2]

    def get_global_pose(self):
        x, y = self.get_global_xy()
        z = self.get_global_height()
        ypr = _R_to_ypr(self.global_rot)
        return np.array([x, y, z, ypr[2], ypr[1], ypr[0]], dtype=self.global_trans.dtype)

    def _get_hand_ref(self, hand_id):
        """Return (hand_pos, v1) where v1 = Thumb1 - Pinky1 (kept for external callers)."""
        if hand_id == 0:
            hand_pos = self.skin.get_global_translation("LeftHand")[0]
            thumb_pos = self.skin.get_global_translation("LeftHandThumb1")[0]
            pinky_pos = self.skin.get_global_translation("LeftHandPinky1")[0]
        else:
            hand_pos = self.skin.get_global_translation("RightHand")[0]
            thumb_pos = self.skin.get_global_translation("RightHandThumb1")[0]
            pinky_pos = self.skin.get_global_translation("RightHandPinky1")[0]
        hand_pos = _to_numpy(hand_pos).ravel()[:3]
        thumb_pos = _to_numpy(thumb_pos).ravel()[:3]
        pinky_pos = _to_numpy(pinky_pos).ravel()[:3]
        v1 = thumb_pos - pinky_pos
        return hand_pos, v1

    def get_palm_center(self, hand_id):
        """Return the center of the palm — average of the four finger knuckle positions."""
        side = "Left" if hand_id == 0 else "Right"
        knuckles = []
        for finger in ("Index", "Middle", "Ring", "Pinky"):
            bone = f"{side}Hand{finger}1"
            pos = _to_numpy(self.skin.get_global_translation(bone)[0]).ravel()[:3]
            knuckles.append(pos)
        return np.mean(knuckles, axis=0)

    def _get_two_hand_frame(self, center_offset=np.zeros(3, dtype=np.float64)):
        """Build an object frame between both palms for two-handed holds."""
        left = self.get_palm_center(0)
        right = self.get_palm_center(1)
        center = 0.5 * (left + right) + np.asarray(center_offset, dtype=np.float64)

        x_axis = right - left
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-8:
            x_axis = np.array([1.0, 0.0, 0.0])
        else:
            x_axis = x_axis / x_norm

        up = np.array([0.0, 0.0, 1.0])
        z_axis = up - np.dot(up, x_axis) * x_axis
        if np.linalg.norm(z_axis) < 1e-8:
            z_axis = np.array([0.0, 1.0, 0.0]) - x_axis[1] * x_axis
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)

        return center, np.column_stack([x_axis, y_axis, z_axis])

    def _get_hand_frame(self, hand_id):
        """Build a proper 3-D orthonormal frame for the hand.

        Axes:
          e1 = normalize(Thumb1 − Pinky1)        (across palm)
          e2 = finger direction ⊥ e1             (wrist → Middle1, Gram-Schmidt)
          e3 = e1 × e2                            (palm normal)

        Returns (hand_pos, R_hand) where R_hand columns are [e1, e2, e3].
        """
        hand_pos, v1 = self._get_hand_ref(hand_id)
        e1 = v1 / (np.linalg.norm(v1) + 1e-8)

        # Second reference vector: wrist → first middle-finger knuckle
        mid1_bone = "LeftHandMiddle1" if hand_id == 0 else "RightHandMiddle1"
        try:
            mid1 = _to_numpy(self.skin.get_global_translation(mid1_bone)[0]).ravel()[:3]
            v_finger = mid1 - hand_pos
        except Exception:
            v_finger = None

        if v_finger is not None and np.linalg.norm(v_finger) > 1e-6:
            raw = v_finger
        else:
            raw = np.array([0.0, 0.0, 1.0])  # world-up fallback

        # Gram-Schmidt: orthogonalize raw against e1
        v2 = raw - np.dot(raw, e1) * e1
        norm_v2 = np.linalg.norm(v2)
        if norm_v2 < 1e-6:
            # raw was collinear with e1; try another fallback
            raw2 = np.array([0.0, 1.0, 0.0])
            v2 = raw2 - np.dot(raw2, e1) * e1
            norm_v2 = np.linalg.norm(v2) + 1e-8
        e2 = v2 / norm_v2

        e3 = np.cross(e1, e2)
        norm_e3 = np.linalg.norm(e3)
        if norm_e3 < 1e-6:
            e3 = np.cross(e1, np.array([0.0, 0.0, 1.0]))
            norm_e3 = np.linalg.norm(e3) + 1e-8
        e3 /= norm_e3

        # Final re-orthogonalization to guarantee numerical orthogonality
        e2 = np.cross(e3, e1)
        e2 /= np.linalg.norm(e2) + 1e-8

        R_hand = np.column_stack([e1, e2, e3])  # 3×3, columns are basis vectors
        return hand_pos, R_hand

    def attach_object_to_hand(self, hand_id, entity):
        """Attach entity to hand.

        Both position and rotation are tracked in the hand's local frame so
        the object follows the hand's full rigid-body motion (translation + rotation).
        """
        if self.skin is None:
            return
        for i, attached in enumerate(self.attached_object):
            if i != int(hand_id) and attached is not None and attached[0] is entity:
                self.attached_object[i] = None
        hand_pos, R_hand = self._get_hand_frame(hand_id)

        obj_pos = _to_numpy(entity.get_pos()).ravel()[:3]
        quat_wxyz = _to_numpy(entity.get_quat()).ravel()[:4]
        obj_rot = geom_utils.quat_to_R(quat_wxyz)

        v2_local = R_hand.T @ (obj_pos - hand_pos)
        R_obj_local = R_hand.T @ obj_rot

        self.attached_object[hand_id] = (
            entity, v2_local.astype(np.float64), R_obj_local.astype(np.float64),
        )

    def detach_object(self, hand_id):
        """Detach object from hand."""
        self.attached_object[hand_id] = None

    def attach_object_to_two_hands(
        self,
        entity,
        center_offset=np.zeros(3, dtype=np.float64),
        snap_to_frame=False,
    ):
        """Attach entity to a frame defined by both palms."""
        if self.skin is None:
            return
        frame_pos, frame_rot = self._get_two_hand_frame(center_offset)
        if snap_to_frame:
            v_local = np.zeros(3, dtype=np.float64)
            R_obj_local = np.eye(3, dtype=np.float64)
            entity.set_pos(frame_pos.astype(float))
            entity.set_quat(geom_utils.R_to_quat(frame_rot).astype(float))
        else:
            obj_pos = _to_numpy(entity.get_pos()).ravel()[:3]
            quat_wxyz = _to_numpy(entity.get_quat()).ravel()[:4]
            obj_rot = geom_utils.quat_to_R(quat_wxyz)
            v_local = frame_rot.T @ (obj_pos - frame_pos)
            R_obj_local = frame_rot.T @ obj_rot
        self.two_hand_attached_object = (
            entity,
            np.asarray(center_offset, dtype=np.float64),
            v_local.astype(np.float64),
            R_obj_local.astype(np.float64),
        )

    def detach_object_from_two_hands(self):
        """Detach the two-hand attached object, if any."""
        self.two_hand_attached_object = None

    def update(self):
        motion_trans = self.pose[:3]
        motion_rot = geom_utils.quat_to_R(self.pose[3:7])
        total_trans = self.global_rot @ self.base_rot @ motion_trans + self.global_trans
        self.box.set_pos(self.get_global_pose()[:3] + np.array([0, 0, 0.959008030]))
        self.box.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))

        if self.skin is not None:
            skin_base_rot = geom_utils.R_to_quat(
                geom_utils.euler_to_R(
                    R.from_matrix(self.global_rot).as_euler("xyz", degrees=True)[[0, 2, 1]]
                )
                @ motion_rot
            )
            self.skin.update_mesh(
                total_trans[[1, 2, 0]],
                self.global_mat,
                self.global_mat_inv,
                skin_base_rot,
                self.node_trans,
            )

        # Update attached objects: both position and rotation follow hand frame
        for hand_id in (0, 1):
            if self.attached_object[hand_id] is None:
                continue
            entity, v2_local, R_obj_local = self.attached_object[hand_id]
            hand_pos, R_hand = self._get_hand_frame(hand_id)
            obj_pos = hand_pos + R_hand @ v2_local
            obj_rot = R_hand @ R_obj_local
            entity.set_pos(obj_pos.astype(float))
            quat_wxyz = geom_utils.R_to_quat(obj_rot).astype(float)
            entity.set_quat(quat_wxyz)

        if self.two_hand_attached_object is not None:
            entity, center_offset, v_local, R_obj_local = self.two_hand_attached_object
            frame_pos, frame_rot = self._get_two_hand_frame(center_offset)
            obj_pos = frame_pos + frame_rot @ v_local
            obj_rot = frame_rot @ R_obj_local
            entity.set_pos(obj_pos.astype(float))
            entity.set_quat(geom_utils.R_to_quat(obj_rot).astype(float))
